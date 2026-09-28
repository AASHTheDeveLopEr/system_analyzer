#!/usr/bin/env python3
"""System One decision layer for System Analyzer.

System One models are decisions-only: you hand them a ``state`` (the facts to
judge) plus a map of typed ``questions``, and they hand back typed ``answers``
with probabilities instead of generated text. Three question types exist:

    noul    yes/no        -> probability the answer is yes (0..1)
    choice  one of a set  -> chosen option + full probability distribution
    score   ordered level -> probability-weighted level + legend + distribution

Two endpoints speak that wire format:

    jev-free   OpenCode Zen's free Jev tier, works with no account
    typesafe   TypeSafe Jev -- paid, needs TYPESAFE_API_KEY and cloud opt-in

The paid one is opt-in (spec FR-010): it only joins the chain when the user set
both the key and the opt-in flag. The free one needs neither today, but its key
is read too (OPENCODE_ZEN_API_KEY) so the tool keeps working if the open tier
ever closes to anonymous callers.

Run it directly to check the wiring::

    python system_one.py --selftest     # offline: the test suite, no network
    python system_one.py --probe        # live: one real Jev call, prints answers
    python system_one.py --probe -n 5   # live: repeat, to see how much it moves
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping

import requests

# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
# Decision: the default is OpenCode Zen's free `jev-1.13-free`, and there is no
# local model. "Jev free" means that hosted endpoint, not a model on this
# machine -- so every rating makes a network call, and the app has no offline
# mode yet. Do not add a local provider here without re-arguing it: the seam
# exists (it is a list), but shipping it was deferred on purpose.

FREE_URL = "https://opencode.ai/zen/v1/systemone"
FREE_MODEL = "jev-1.13-free"

PAID_URL = "https://api.typesafe.ai/v1/systemone"
PAID_MODEL = "jev-latest"

# The paid key is read from the environment only -- never from a file that
# could be committed, and never from the command line (shell history).
PAID_KEY_ENV = "TYPESAFE_API_KEY"
CLOUD_ENV = "SYSTEM_ANALYZER_CLOUD_FALLBACK"

# The free endpoint answers anonymous callers today, so this key is optional.
# It exists for the day the open tier is withdrawn: without it the app has no
# way to keep talking to Zen at all, because the paid key does not work there.
FREE_KEY_ENV = "OPENCODE_ZEN_API_KEY"

QUESTION_TYPES = ("noul", "choice", "score")
TIMEOUT_SECONDS = 30
# Never sit on a Retry-After longer than this; a stale header must not hang the CLI.
RETRY_AFTER_CAP = 5.0


class SystemOneError(RuntimeError):
    """No endpoint could answer, or the request/answer was not usable."""


def _paid_provider() -> tuple[dict | None, str]:
    """The paid TypeSafe endpoint, plus a reason when it is not opted in.

    Cloud escalation is off by default (spec FR-010), so this needs both the
    key and the opt-in flag. The reason is returned so it can be shown to the
    user instead of silently dropping the backup.
    """
    api_key = os.environ.get(PAID_KEY_ENV, "").strip()
    if not api_key:
        return None, f"{PAID_KEY_ENV} is not set"
    if os.environ.get(CLOUD_ENV, "").strip().lower() not in ("1", "true", "yes", "on"):
        return None, f"cloud escalation is off (set {CLOUD_ENV}=1 to enable it)"

    return {
        "name": "typesafe",
        "url": PAID_URL,
        "model": PAID_MODEL,
        "source": "paid",
        "auth": f"Bearer {api_key}",
    }, ""


def _providers() -> tuple[list[dict], list[str]]:
    """Endpoints to try in order (free first) and the reasons the paid one is out.

    The free entry is the whole default configuration: OpenCode Zen's
    ``jev-1.13-free``, reached anonymously unless a key is set. It is remote
    (see the note above the constants) and deliberately ungated -- the paid
    escalation is the part that needs opting in to.
    """
    free_key = os.environ.get(FREE_KEY_ENV, "").strip()
    free = {
        "name": "jev-free",
        "url": FREE_URL,
        "model": FREE_MODEL,
        "source": "free",
        # Anonymous callers are accepted, so a missing key is not an error.
        # Note that the probe prints these dicts: never put the key in one.
        "auth": f"Bearer {free_key}" if free_key else "Bearer public",
    }
    paid, reason = _paid_provider()
    providers = [free] + ([paid] if paid else [])
    return providers, ([f"paid backup (TypeSafe) skipped: {reason}"] if not paid else [])


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------
# Catching a bad body here saves a round-trip and names the offending question.


def validate_request(state: Any, questions: Any) -> str | None:
    """Return ``None`` when the body is valid, else a human-readable reason.

    Exception-free on purpose: the caller prints the returned string verbatim.

    The shape rules below are not guesses. They were checked against the live
    endpoint, and getting them wrong is expensive: a choice whose ``criteria``
    is a *list* passes a naive check and then comes back as
    ``HTTP 422: Endpoint is unavailable``, which names the wrong problem.
    """
    if state is None or (isinstance(state, str) and not state.strip()):
        return "Missing state: the content Jev should evaluate"
    if not isinstance(state, (str, dict, list)):
        return "Invalid state: must be a string, object or array"

    if not isinstance(questions, dict) or not questions:
        return "Missing questions: ask at least one noul, choice or score question"

    for qid, question in questions.items():
        if not isinstance(question, dict):
            return f'Invalid question "{qid}": must be an object with type and instructions'
        if question.get("type") not in QUESTION_TYPES:
            return f'Invalid question "{qid}": type must be one of {", ".join(QUESTION_TYPES)}'
        if not question.get("instructions"):
            return f'Invalid question "{qid}": instructions are required'

        criteria = question.get("criteria")
        kind = question["type"]
        if kind == "choice":
            if not criteria:
                return f'Invalid question "{qid}": choice needs criteria'
            if not isinstance(criteria, dict):
                return (f'Invalid question "{qid}": choice criteria must be an object '
                        f'of option name to description, got {type(criteria).__name__}')
        elif kind == "score":
            if not isinstance(criteria, list):
                return (f'Invalid question "{qid}": score criteria must be a list of '
                        f'ordered levels, got {type(criteria).__name__}')
            if len(criteria) < 2:
                return f'Invalid question "{qid}": score needs at least 2 ordered levels'

    return None


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _post_json(provider: Mapping[str, Any], state: Any, questions: Mapping[str, Any]) -> dict:
    """POST one request to one endpoint and return the decoded answer.

    Every failure becomes a ``SystemOneError`` carrying the status or reason, so
    the caller can report *why* an endpoint was skipped.
    """
    payload = {"model": provider["model"], "state": state, "questions": dict(questions)}
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "system-analyzer/0.1",
        "Authorization": provider["auth"],
    }

    for attempt in range(2):  # one retry, for 429 only
        try:
            response = requests.post(
                provider["url"], json=payload, headers=headers, timeout=TIMEOUT_SECONDS
            )
        except requests.RequestException as error:
            raise SystemOneError(f"network error: {error}") from error

        if response.status_code == 429 and attempt == 0:
            # The free tier is rate limited per caller. Waiting the interval the
            # server asked for beats failing a question over a second of queue.
            delay = _retry_after(response)
            if delay:
                time.sleep(min(delay, RETRY_AFTER_CAP))
                continue
        break

    if response.status_code >= 400:
        raise SystemOneError(f"HTTP {response.status_code}: {_error_message(response)}")

    try:
        decoded = response.json()
    except ValueError as error:
        raise SystemOneError(f"unreadable JSON response: {_trim(response.text)}") from error

    if not isinstance(decoded, dict):
        raise SystemOneError("unexpected response shape (not a JSON object)")
    return decoded


def _retry_after(response: Any) -> float:
    """Seconds to wait, from Retry-After as seconds or as an HTTP date."""
    raw = (response.headers.get("Retry-After") or "").strip()
    if raw.isdigit():
        return float(raw)
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return 0.0
    if when is None:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _error_message(response: Any) -> str:
    """One readable line from an error body.

    The endpoints do not agree on the error envelope, and they are worth
    telling apart: a 400 names the question we got wrong, a 401 means the key,
    a 422 from the free gateway means the *model* rejected the body even though
    our validator accepted it. Raw JSON hides that difference.
    """
    try:
        body = response.json()
    except ValueError:
        return _trim(response.text) or "no details"

    if not isinstance(body, dict):
        return _trim(str(body))

    detail = body.get("detail")
    if isinstance(detail, str) and detail:
        return _trim(detail)

    # {"error": {"type": ..., "message": ...}} and
    # {"type": "error", "error": {"type": ..., "message": ...}}
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        kind = error.get("type")
        return _trim(f"{message} ({kind})" if kind and message else (message or kind or "error"))
    if isinstance(error, str) and error:
        return _trim(error)
    return _trim(str(body))


def _trim(text: str, limit: int = 300) -> str:
    """Collapse whitespace and cap length so error text stays one readable line."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


@dataclass
class Decision:
    """One answered request, with provenance attached.

    ``answers`` is the upstream map verbatim, keyed by the question ids we
    sent. The accessors below read it; keeping the raw map lets a later feature
    (the audit trail in FR-011) show exactly what Jev returned.
    """

    model: str
    answers: dict[str, Any]
    provider: str
    source: str  # "free" | "paid" -- shown to the user for provenance (FR-011)
    usage: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None

    def _answer(self, question_id: str, kind: str) -> dict[str, Any]:
        answer = self.answers.get(question_id)
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise SystemOneError(f'answer "{question_id}" is not a {kind} answer')
        return answer

    @staticmethod
    def _number(answer: Mapping[str, Any], key: str, question_id: str) -> float:
        value = answer.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SystemOneError(f'answer "{question_id}" has no numeric {key} value')
        return float(value)

    @staticmethod
    def _percent(answer: Mapping[str, Any], question_id: str) -> float:
        """The answer's ``confidence`` as 0..100, whatever scale it arrived in.

        The live endpoint sends a fraction (``0.86``); a hand-written or cached
        answer may already be a percentage (``86``). Values of 1.0 and below are
        read as fractions, because that is how the model expresses "fully
        confident" -- it sends ``1.0``, not ``100``. Zero is the same number
        either way, so the ambiguous case costs nothing.

        An answer with no ``confidence`` at all is reported as 0.0, which reads
        as "no confidence claimed" rather than a measurement.
        """
        value = answer.get("confidence")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return 0.0
        value = float(value)
        if value <= 1.0:
            value *= 100.0
        return max(0.0, min(100.0, value))

    def noul(self, question_id: str) -> float:
        """Probability the yes/no question is true (0..1)."""
        return self._number(self._answer(question_id, "noul"), "noul", question_id)

    def choice(self, question_id: str) -> tuple[str, dict[str, float], float]:
        """Return ``(chosen, probabilities, confidence 0..100)`` for a choice answer."""
        answer = self._answer(question_id, "choice")
        chosen = answer.get("choice")
        if not isinstance(chosen, str):
            raise SystemOneError(f'answer "{question_id}" has no chosen option')
        probabilities = dict(answer.get("probabilities") or {})
        return chosen, probabilities, self._percent(answer, question_id)

    def score(self, question_id: str) -> tuple[float, dict[str, str], dict[str, float], float]:
        """Return ``(score, legend, probabilities, confidence 0..100)`` for a score answer.

        ``score`` can land between levels -- it is the probability-weighted
        position, not an index into ``legend``.
        """
        answer = self._answer(question_id, "score")
        score = self._number(answer, "score", question_id)
        legend = dict(answer.get("legend") or {})
        probabilities = dict(answer.get("probabilities") or {})
        return score, legend, probabilities, self._percent(answer, question_id)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class SystemOneClient:
    """Ask typed questions, trying each endpoint in order until one answers.

    The first endpoint to return a usable response wins; every earlier failure
    is collected so the final error explains what each one said.
    """

    def __init__(self, providers: list[dict] | None = None) -> None:
        if providers is None:
            providers, self.skipped = _providers()
        else:
            self.skipped = []
        self.providers = providers

    def ask(self, state: Any, questions: Mapping[str, Any]) -> Decision:
        """Evaluate ``questions`` against ``state``; return the first success."""
        error = validate_request(state, questions)
        if error:
            # A malformed body fails for every endpoint; report the real problem
            # instead of a pile of HTTP 400s.
            raise SystemOneError(error)

        failures = []
        for provider in self.providers:
            try:
                decoded = _post_json(provider, state, questions)
            except SystemOneError as failure:
                failures.append(f"{provider['name']}: {failure}")
                continue

            return Decision(
                model=decoded.get("model") or provider["model"],
                answers=decoded.get("answers") or {},
                provider=provider["name"],
                source=provider["source"],
                usage=decoded.get("usage") or {},
                # The free endpoint sends this as the *string* "0", not a
                # number, so read it as a number or report nothing at all.
                cost=_as_float(decoded.get("cost")),
            )

        raise SystemOneError("every System One endpoint failed: " + " | ".join(failures))


def _as_float(value: Any) -> float | None:
    """Parse a JSON number that may have arrived as a string, else ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Composition layer
# --------------------------------------------------------------------------
# Everything above returns typed answers. Everything below turns them into the
# two numbers a user actually reads: a 0-100 rating and a banded verdict with a
# confidence.
#
# The design rule throughout: the model supplies judgement, the local code
# supplies arithmetic and honesty. Every verdict names a component that was
# really detected, an unknown reading is never quietly scored as zero, and
# when the model cannot be reached the rating still comes out -- flagged as
# rule-only rather than dressed up as a model answer (FR-004, FR-014).

GIB = 1024 ** 3


@dataclass(frozen=True)
class ComponentScore:
    """One component's deterministic 0-100 sub-score.

    ``score`` is ``None`` when the component could not be measured. It is not
    zero: an unreadable GPU is a gap in the evidence, and scoring it as zero
    would drag the rating down for a reason we cannot prove (FR-002).
    """

    name: str
    score: float | None
    detail: str

    @property
    def known(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": self.score, "detail": self.detail}


@dataclass(frozen=True)
class Factor:
    """One thing that pushed the rating up or down."""

    text: str
    effect: str  # "up" | "down" | "unknown"

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "effect": self.effect}


@dataclass(frozen=True)
class Rating:
    """A 0-100 rating with the working shown (FR-003, FR-004, FR-011)."""

    total: int
    subscores: list[ComponentScore]
    factors: list[Factor]
    method: str  # "model+rules" or "rules"
    source: str  # "cloud", or "rules" when no model answered
    model: str | None = None
    provider: str | None = None
    tier: str | None = None
    model_total: float | None = None
    rules_total: float | None = None
    estimated: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "estimated": self.estimated,
            "method": self.method,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "model_total": _round(self.model_total),
            "rules_total": _round(self.rules_total),
            "subscores": [c.to_dict() for c in self.subscores],
            "factors": [f.to_dict() for f in self.factors],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CapabilityVerdict:
    """A banded answer with an honest confidence (FR-005, FR-006, FR-011)."""

    verdict: str
    confidence: int
    explanation: str
    question: str
    probability: float
    source: str
    model: str | None = None
    provider: str | None = None
    grounded: bool = True
    uncertain: bool = False
    bottlenecks: list[str] = field(default_factory=list)
    sub_scores: list[ComponentScore] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "question": self.question,
            "probability": _round(self.probability),
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "grounded": self.grounded,
            "uncertain": self.uncertain,
            "bottlenecks": list(self.bottlenecks),
            "sub_scores": [c.to_dict() for c in self.sub_scores],
            "notes": list(self.notes),
        }


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None else None


# --------------------------------------------------------------------------
# Facts and curves
# --------------------------------------------------------------------------


def _gib(num_bytes: int | None) -> float | None:
    return round(num_bytes / GIB, 2) if num_bytes is not None else None


def _numbers(values: list[Any]) -> list[float]:
    """The real numbers out of a list that may be full of ``None``."""
    return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _measured(inventory: Any) -> dict[str, float | None]:
    """The handful of numbers the curves need, from either input shape.

    Accepts ``menu.Inventory`` (real units, preferred) or the scrubbed state
    mapping, so this layer can be tested without psutil installed.
    """
    if isinstance(inventory, Mapping):
        cpu = inventory.get("cpu") or {}
        memory = inventory.get("memory") or {}
        gpus = [g for g in (inventory.get("gpus") or []) if isinstance(g, Mapping)]
        storage = [d for d in (inventory.get("storage") or []) if isinstance(d, Mapping)]
        return {
            "cores": cpu.get("logical_cores"),
            "ram_gib": memory.get("total_gib"),
            "vram_gib": max(_numbers([g.get("dedicated_vram_gib") for g in gpus]), default=None),
            "disk_gib": max(_numbers([d.get("total_gib") for d in storage]), default=None),
            "gpu_count": len(gpus),
        }

    gpus = list(getattr(inventory, "gpus", ()) or ())
    disks = list(getattr(inventory, "disks", ()) or ())
    memory = getattr(inventory, "memory", None)
    return {
        "cores": getattr(inventory, "cpu_cores", None),
        "ram_gib": _gib(getattr(memory, "total_bytes", None)),
        "vram_gib": max(
            _numbers([_gib(getattr(g, "vram_bytes", None)) for g in gpus]), default=None
        ),
        "disk_gib": max(_numbers([_gib(getattr(d, "total_bytes", None)) for d in disks]), default=None),
        "gpu_count": len(gpus),
    }


def _curve_score(curve: tuple[tuple[float, float], ...], value: float | None) -> float | None:
    """Map a measured value onto 0-100 along the declared anchors."""
    if value is None:
        return None
    value = float(value)
    if value <= curve[0][0]:
        return curve[0][1]
    if value >= curve[-1][0]:
        return curve[-1][1]
    for (low_x, low_y), (high_x, high_y) in zip(curve, curve[1:]):
        if low_x <= value <= high_x:
            if high_x == low_x:
                return high_y
            return low_y + (value - low_x) / (high_x - low_x) * (high_y - low_y)
    return curve[-1][1]


def component_scores(inventory: Any, config: Any = None) -> list[ComponentScore]:
    """Score each measurable component on its own, before any weighting."""
    config = config or _config()
    facts = _measured(inventory)

    ram = facts["ram_gib"]
    vram = facts["vram_gib"]
    disk = facts["disk_gib"]
    cores = facts["cores"]
    gpu_count = facts["gpu_count"]

    if gpu_count == 0:
        gpu_detail = "no display adapter was detected"
    elif vram is None:
        gpu_detail = "an adapter was found but reported no dedicated VRAM"
    else:
        gpu_detail = f"{vram:g} GiB dedicated VRAM"

    return [
        ComponentScore(
            "CPU",
            _curve_score(config.curve("cpu"), cores),
            f"{cores} logical cores" if cores else "core count unknown",
        ),
        ComponentScore(
            "Memory",
            _curve_score(config.curve("memory"), ram),
            f"{ram:g} GiB installed" if ram else "total RAM unknown",
        ),
        ComponentScore(
            "GPU",
            _curve_score(config.curve("gpu"), vram),
            gpu_detail,
        ),
        ComponentScore(
            "Storage",
            _curve_score(config.curve("storage"), disk),
            f"{disk:g} GiB in the largest volume" if disk else "no volume size reported",
        ),
    ]


def rules_rating(
    subscores: list[ComponentScore], config: Any = None
) -> tuple[float, list[ComponentScore]]:
    """The deterministic rating: weighted mean of the components we could measure.

    Unknown components are dropped and their weight is shared out across the
    rest, so one unreadable adapter cannot drag a good machine down. The caller
    reports the drop separately, as an "estimated" flag.
    """
    config = config or _config()
    weights = config.weights()
    known = [c for c in subscores if c.known and weights.get(c.name.lower())]
    if not known:
        return 0.0, []

    total_weight = sum(weights[c.name.lower()] for c in known)
    total = sum(c.score * weights[c.name.lower()] for c in known) / total_weight
    return total, known


def _config(config: Any = None) -> Any:
    if config is not None:
        return config
    import config as config_module

    return config_module.load()


def _state_for(inventory: Any) -> dict[str, Any]:
    """The state object to send: a dict already, or an Inventory's scrubbed view."""
    if isinstance(inventory, Mapping):
        return dict(inventory)
    to_state = getattr(inventory, "to_state", None)
    if callable(to_state):
        return to_state()
    raise TypeError(f"cannot build a System One state from {type(inventory).__name__}")


# --------------------------------------------------------------------------
# rate_system (FR-003, FR-004)
# --------------------------------------------------------------------------


def _rating_questions(config: Any) -> dict[str, Any]:
    return {
        "overall_tier": {
            "type": "score",
            "instructions": (
                "Judge only from the hardware in this computer's detected specs. How "
                "capable is this machine at demanding desktop work such as modern "
                "games, video editing, and running AI models locally?"
            ),
            "criteria": list(config.rating_tiers),
        }
    }


def _tier_label(legend: Mapping[str, Any], levels: tuple[str, ...]) -> str | None:
    """The level the weighted score actually landed on."""
    if not legend:
        return None
    key = min(legend, key=lambda k: float(k)) if all(str(k).lstrip("-").isdigit() for k in legend) else None
    if key is None:
        return str(next(iter(legend.values())))
    index = int(float(key))
    if 0 <= index < len(levels):
        return levels[index].split(":", 1)[0].strip()
    return str(legend[key])


def rate_system(inventory: Any, client: Any, config: Any = None) -> Rating:
    """Turn an inventory into a 0-100 rating with its working shown.

    The model's tier and the deterministic component scores are blended
    (FR-004: the result stays explainable and reproducible). If no endpoint
    answers, the deterministic score is returned on its own with
    ``method="rules"`` -- a rating is always available, and it is always clear
    which one it came from.
    """
    config = _config(config)
    subscores = component_scores(inventory, config)
    rules_total, known = rules_rating(subscores, config)
    estimated = len(known) < len(subscores)

    notes: list[str] = []
    model_total: float | None = None
    tier: str | None = None
    model: str | None = None
    provider: str | None = None
    decision_source: str = "rules"
    levels = config.rating_tiers

    try:
        decision = client.ask(_state_for(inventory), _rating_questions(config))
        score, legend, _probabilities, _confidence = decision.score("overall_tier")
        # The endpoint returns a probability-weighted position across the
        # ordered levels, i.e. 0 at the first level and 1 at the last.
        model_total = max(0.0, min(1.0, score)) * 100.0
        tier = _tier_label(legend, levels)
        model = decision.model
        provider = decision.provider
        decision_source = decision.source
    except SystemOneError as error:
        notes.append(f"no System One endpoint answered, so this is the rule-only score: {error}")

    if model_total is None:
        total, method, source = rules_total, "rules", "rules"
    else:
        blend = config.model_blend
        total = model_total * blend + rules_total * (1.0 - blend)
        method, source = "model+rules", decision_source

    if estimated:
        unknown = [c for c in subscores if not c.known]
        notes.append(
            "estimated: no reading for "
            + ", ".join(f"{c.name} ({c.detail})" for c in unknown)
        )

    factors = _rating_factors(subscores, total, tier, estimated, method, config)
    return Rating(
        total=int(round(max(0.0, min(100.0, total)))),
        subscores=subscores,
        factors=factors,
        method=method,
        source=source,
        model=model,
        provider=provider,
        tier=tier,
        model_total=model_total,
        rules_total=rules_total,
        estimated=estimated,
        notes=notes,
    )


def _rating_factors(
    subscores: list[ComponentScore],
    total: float,
    tier: str | None,
    estimated: bool,
    method: str,
    config: Any,
) -> list[Factor]:
    """The pulls on the number: the tier, the components, and the caveats.

    SC-002 wants at least three factors. Two of those slots are reserved for
    the tier and for the caveats -- "we could not read the GPU" and "no model
    answered" are the factors a reader most needs, and truncating the list from
    the end would throw away exactly those.
    """
    scored: list[tuple[float, float, str]] = []
    for component in subscores:
        if not component.known:
            continue
        deviation = component.score - total
        scored.append(
            (
                abs(deviation),
                deviation,
                f"{component.name} {component.score:.0f}/100 ({component.detail}) "
                f"pulls the score {'up' if deviation > 0 else 'down'} "
                f"by {abs(deviation):.0f}",
            )
        )
    # Biggest pull first. The sign rides along separately, because sorting on
    # the magnitude and then reading the direction off it would label every
    # factor "up" -- the one a component drags down is the same size as the one
    # that lifts it.
    scored.sort(key=lambda entry: entry[0], reverse=True)

    head: list[Factor] = []
    if tier:
        head.append(
            Factor(f"Model placed this machine in the {tier} band", "up" if total >= 50 else "down")
        )

    caveats: list[Factor] = []
    if estimated:
        unknown = next((c for c in subscores if not c.known), None)
        if unknown is not None:
            caveats.append(
                Factor(
                    f"{unknown.name} was not readable ({unknown.detail}), so it is not counted",
                    "unknown",
                )
            )
    if method == "rules":
        caveats.append(
            Factor("No model was available, so this is the deterministic estimate alone", "unknown")
        )

    limit = max(config.min_factors, 3)
    room = max(0, limit - len(head) - len(caveats))
    pulls = [Factor(text, "up" if deviation > 0 else "down") for _size, deviation, text in scored[:room]]
    return head + pulls + caveats


# --------------------------------------------------------------------------
# answer_capability (FR-005, FR-006, FR-007)
# --------------------------------------------------------------------------

GROUNDED_YES = "specs_decide_it"
GROUNDED_NO = "specs_do_not_decide_it"


def _capability_questions(question: str) -> dict[str, Any]:
    """Three questions, asked in one call.

    They look redundant on purpose. The noul gives the probability the verdict
    is built from; the two choices give a real ``confidence`` field (noul
    answers carry none) and, more importantly, two ways to catch a question
    that the specs simply cannot answer -- which is what stops a 90%-confident
    nonsense answer when someone types "can my PC run the internet?".
    """
    return {
        "verdict": {
            "type": "noul",
            "instructions": (
                f"Considering only this computer's detected CPU, GPU, memory and storage, "
                f"is it realistic to do the following on it: {question}?"
            ),
        },
        "grounding": {
            "type": "choice",
            "instructions": (
                f"Does the detected hardware in this computer's specs actually determine "
                f"the answer to this request: {question}?"
            ),
            "criteria": {
                GROUNDED_YES: (
                    "The CPU, GPU, memory or storage listed here is what decides "
                    "whether this request can be done on this machine"
                ),
                GROUNDED_NO: (
                    "This request depends on things these specs do not show, such as "
                    "availability, price, permissions, or the physical world"
                ),
            },
        },
        "second_opinion": {
            "type": "choice",
            "instructions": (
                f"Taking this computer's detected hardware into account, which is closer "
                f"to the truth for this request: {question}?"
            ),
            "criteria": {
                "yes": "This machine can do it well",
                "no": "This machine cannot do it well",
            },
        },
    }


def _band_for(probability: float, config: Any) -> str:
    for threshold, label in config.bands():
        if probability >= threshold:
            return label
    return "No"


def answer_capability(inventory: Any, question: str, client: Any, config: Any = None) -> CapabilityVerdict:
    """Turn "can I run X?" into a verdict and a confidence (FR-005, FR-006).

    Confidence is built from the model's own ``confidence`` field and then
    pushed *down* wherever the evidence is weak: a probability near a coin
    flip, a second opinion that disagrees, or a question the hardware does not
    determine. There is no floor and no rounding up -- if the evidence does not
    support a number, the answer says so instead of inventing one.
    """
    config = _config(config)
    question = (question or "").strip()
    if not question:
        raise ValueError("ask what you want to do, in words")

    subscores = component_scores(inventory, config)
    decision = client.ask(_state_for(inventory), _capability_questions(question))

    probability = decision.noul("verdict")
    grounding, _ground_probs, ground_conf = decision.choice("grounding")
    second, second_probs, second_conf = decision.choice("second_opinion")

    grounded = grounding == GROUNDED_YES
    uncertain = abs(probability - 0.5) < config.uncertain_margin
    verdict = _band_for(probability, config)

    # How far from a coin flip, scaled to 0..1: an answer sitting on 0.5 has
    # earned less than half the credit no matter what confidence says.
    strength = min(1.0, abs(probability - 0.5) / 0.4)
    confidence = second_conf * (0.5 + 0.5 * strength)

    second_yes = _numbers([second_probs.get("yes"), 1.0 if second == "yes" else 0.0])
    second_leans_yes = (second_yes[0] if second_yes else 0.5) >= 0.5
    if second_leans_yes != (probability >= 0.5):
        confidence -= config.disagreement_penalty
        notes = "the two ways of asking disagreed, so the confidence is reduced"
    else:
        notes = ""

    if uncertain:
        confidence = min(confidence, config.uncertain_confidence_cap)
    if not grounded:
        verdict = "Cannot tell"
        confidence = min(confidence, config.ungrounded_confidence_cap, ground_conf)

    confidence = int(round(max(0.0, min(100.0, confidence))))

    bottlenecks = [c for c in subscores if c.known and c.score <= config.bottleneck_score]
    if not bottlenecks:
        measured = [c for c in subscores if c.known]
        bottlenecks = [min(measured, key=lambda c: c.score)] if measured else []
    bottleneck_names = [f"{c.name} ({c.detail})" for c in bottlenecks]

    explanation = _explain(verdict, question, bottlenecks, probability, grounded, uncertain)
    all_notes = [n for n in (notes,) if n] + (
        [f"not counted in the rating: {c.name} ({c.detail})" for c in subscores if not c.known]
    )

    return CapabilityVerdict(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        question=question,
        probability=probability,
        source=decision.source,
        model=decision.model,
        provider=decision.provider,
        grounded=grounded,
        uncertain=uncertain,
        bottlenecks=bottleneck_names,
        sub_scores=subscores,
        notes=all_notes,
    )


def _explain(
    verdict: str,
    question: str,
    bottlenecks: list[ComponentScore],
    probability: float,
    grounded: bool,
    uncertain: bool,
) -> str:
    """One line, naming a component that was really detected (FR-005)."""
    if not grounded:
        return (
            f"The detected specs do not determine this one, so I will not put a "
            f"number on it. Give me the requirements for {question} and I will check "
            f"them against your CPU, GPU and memory."
        )
    if not bottlenecks:
        return (
            f"{verdict} for {question}. Every component I could measure is well above "
            f"the low end, so nothing in the detected specs stands out as a limit."
        )

    weakest = bottlenecks[0]
    limit = (
        f"the tightest part is {weakest.name} at {weakest.detail} "
        f"({weakest.score:.0f}/100)"
    )
    others = len(bottlenecks) - 1
    if others > 0:
        limit += f", along with {', '.join(c.name for c in bottlenecks[1:])}"
    tail = " (the model was close to a coin flip, so treat this as a maybe)"
    return f"{verdict} for {question}: {limit}{tail if uncertain else ''}."



# --------------------------------------------------------------------------
# Self-test (offline) and probe (live)
# --------------------------------------------------------------------------


def _selftest() -> int:
    """Run the offline test suite in-process, with no network.

    Deliberately not a second copy of the checks: ``tests/`` is the only place
    the rules are written down, so this flag and
    ``python -m unittest discover -s tests`` can never disagree.
    """
    import unittest

    root = Path(__file__).resolve().parent
    suite = unittest.TestLoader().discover(
        start_dir=str(root / "tests"), top_level_dir=str(root)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

def _probe(repeats: int = 1) -> int:
    """One real Jev call: confirms the wiring reaches the endpoint.

    The state is this machine's own inventory, scrubbed of paths and names, so
    the probe exercises the payload the rating feature will actually send. With
    ``repeats > 1`` it re-asks the same questions and reports how far the
    answers move -- the cheapest way to see whether a verdict would be stable
    enough to show a confidence percent (spec FR-007).
    """
    client = SystemOneClient()
    print("endpoints in the chain:")
    for provider in client.providers:
        print(f"  - {provider['name']} ({provider['source']}) -> {provider['url']} [{provider['model']}]")
    for reason in client.skipped:
        print(f"  - skipped: {reason}")

    state = _probe_state()
    questions = _probe_questions()

    print(f"\nstate sent to Jev:\n{json.dumps(state, indent=2)}")
    print(f"\nPOST {client.providers[0]['url']} x{repeats} ...")

    nouls, scores, confidences = [], [], []
    decision = None
    for attempt in range(1, repeats + 1):
        try:
            decision = client.ask(state, questions)
        except SystemOneError as error:
            print(f"probe failed on run {attempt}/{repeats}: {error}")
            return 1

        nouls.append(decision.noul("can_run_local_ai"))
        score, _legend, _probabilities, confidence = decision.score("tier")
        scores.append(score)
        confidences.append(confidence)
        print(f"  run {attempt}/{repeats}: noul={nouls[-1]:.3f}  score={score:.2f}  confidence={confidence:.2f}")

    print(f"\nanswered by  : {decision.provider} ({decision.source})")
    print(f"model        : {decision.model}")
    print(f"usage        : {decision.usage}")
    print(f"cost         : {decision.cost}")
    print("\nanswers:")
    print(json.dumps(decision.answers, indent=2))
    print(f"\nnoul(can_run_local_ai) = {nouls[-1]:.3f}")
    print(f"score(tier)            = {scores[-1]:.2f} (confidence {confidences[-1]:.2f})")
    if repeats > 1:
        print(f"spread over {repeats} runs: noul {min(nouls):.3f}-{max(nouls):.3f}, "
              f"score {min(scores):.2f}-{max(scores):.2f}")
    return 0


def _probe_state() -> dict[str, Any]:
    """The real inventory, as the scrubbed state object. Falls back if needed."""
    try:
        from menu import detect_inventory
    except ImportError:
        return {"description": "A desktop with 6 CPU cores, 16 GB RAM, 8 GB dedicated VRAM."}
    return detect_inventory().to_state()


def _probe_questions() -> dict[str, Any]:
    """Questions whose instructions stand on their own.

    Question ids are not sent to the model, so the wording has to carry the
    whole question, and score levels are read as definitions -- bare tier names
    like "Enthusiast" give the model nothing to anchor to.
    """
    return {
        "can_run_local_ai": {
            "type": "noul",
            "instructions": (
                "Taking this computer's CPU, RAM and GPU into account, is it realistic "
                "to run a 7B-parameter language model locally and interact with it?"
            ),
        },
        "tier": {
            "type": "score",
            "instructions": (
                "Judging only by this computer's detected hardware, how capable is it "
                "at demanding desktop workloads such as modern games, video editing "
                "and local AI models?"
            ),
            "criteria": [
                "Entry: integrated graphics and 4 GB of system RAM; handles web and office work only",
                "Mainstream: 6 or more CPU cores and 16 GB RAM; runs everyday games and apps",
                "Enthusiast: 8 or more CPU cores, 32 GB RAM and a discrete GPU with 8 GB or more VRAM",
                "Workstation: 16 or more CPU cores, 64 GB RAM and a professional GPU with 16 GB or more VRAM",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="System One decision layer.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true", help="offline validation checks")
    group.add_argument("--probe", action="store_true", help="one live Jev call")
    parser.add_argument(
        "-n", "--repeat", type=int, default=1, metavar="N",
        help="with --probe, ask N times and report how much the answers move",
    )
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return _selftest() if args.selftest else _probe(repeats=args.repeat)


if __name__ == "__main__":
    sys.exit(main())
