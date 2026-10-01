#!/usr/bin/env python3
"""The System One wire client for System Analyzer.

System One models are decisions-only: you hand them a ``state`` (the facts to
judge) plus a map of typed ``questions``, and they hand back typed ``answers``
with probabilities instead of generated text. Three question types exist:

    noul    yes/no        -> probability the answer is yes (0..1)
    choice  one of a set  -> chosen option + full probability distribution
    score   ordered level -> probability-weighted level + legend + distribution

This module owns the wire: request validation, transport, answer decoding and
the failover client. *Who* the endpoints are -- local Ollama, OpenCode Zen's
free tier, the paid TypeSafe tier -- and in what order they are tried is the
provider chain, owned by ``providers.py`` and held in settings.

Run it directly to check the wiring::

    python system_one.py --selftest     # offline: the test suite, no network
    python system_one.py --probe        # live: one real Jev call, prints answers
    python system_one.py --probe -n 5   # live: repeat, to see how much it moves
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping

import requests

import providers as providers_module

QUESTION_TYPES = ("noul", "choice", "score")
TIMEOUT_SECONDS = 30
# Never sit on a Retry-After longer than this; a stale header must not hang the CLI.
RETRY_AFTER_CAP = 5.0


class SystemOneError(RuntimeError):
    """No endpoint could answer, or the request/answer was not usable."""


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
    }
    # The local Ollama endpoint takes no auth; an empty auth means no header.
    if provider.get("auth"):
        headers["Authorization"] = provider["auth"]

    for attempt in range(2):  # one retry, for 429 only
        try:
            response = requests.post(
                provider["url"], json=payload, headers=headers,
                timeout=provider.get("timeout", TIMEOUT_SECONDS),
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
    sent. The accessors below read it; keeping the raw map lets a later audit
    trail show exactly what Jev returned.
    """

    model: str
    answers: dict[str, Any]
    provider: str
    source: str  # shown to the user for provenance
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
            providers, self.skipped = providers_module.chain()
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
    enough to show a confidence percent.
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
        from inventory import detect_inventory
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
