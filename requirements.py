"""Requirement checklists: the ask-back that turns "Cannot tell" into an answer.

A decision model -- local or cloud -- cannot be trusted to know what a given
program requires, and a small local model additionally cannot judge an
open-ended "can this machine run X" question at all (it answers coin flips).
What the user can often supply is the requirements text; what the app can
then do is check it, each requirement in the role that fits:

* a numeric requirement ("8 GB RAM") is compared against the measured machine
  directly -- deterministic, offline, no model call;
* a named part ("GTX 970 or better") is checked against the curated
  capability tables first: when both the required part and the machine's
  best part have a score, the comparison is arithmetic and offline. Only
  when either side is unknown does it become one atomic noul question for
  the provider, gated by the usual coin-flip rule. A coin flip counts as
  *unknown*, never as a pass or a fail.

Anything that parses to neither is reported as unchecked. Nothing is guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import rankings
from system_one import SystemOneClient, SystemOneError

#: How many atomic questions go into one provider call. The local endpoint
#: scores every question against the full state inside a small context
#: window, so batches stay small on purpose.
QUESTIONS_PER_CALL = 8

_GB = re.compile(r"(\d+(?:[.,]\d+)?)\s*(gb|mb)\b", re.IGNORECASE)
_GHZ = re.compile(r"(\d+(?:[.,]\d+)?)\s*ghz\b", re.IGNORECASE)
_CORES = re.compile(r"(\d+)\s*cores?\b", re.IGNORECASE)
_RAM = re.compile(r"\b(ram|memory)\b", re.IGNORECASE)
_VRAM = re.compile(r"\b(vram|graphics|video|gpu|gtx|rtx|gts|geforce|radeon|arc)\b", re.IGNORECASE)
_STORAGE = re.compile(r"\b(storage|disk|hdd|ssd|drive|space)\b", re.IGNORECASE)
_GPU_NAMED = re.compile(
    r"\b(gtx|rtx|gts|gt\s*\d|radeon|geforce|arc|iris|uhd|hd\s+graphics)\b", re.IGNORECASE
)
_CPU_NAMED = re.compile(
    r"\b(core\s*i\d|ryzen|xeon|athlon|pentium|celeron|processor|cpu)\b", re.IGNORECASE
)
#: Lines about platforms and peripherals are not checkable hardware facts,
#: and pretending they are "unchecked requirements" would just be noise.
_IGNORED = re.compile(
    r"\b(directx|opengl|vulkan|windows|macos|linux|steam\s*os|internet|broadband|"
    r"network|keyboard|mouse|sound|audio|display|resolution|screen|os)\b",
    re.IGNORECASE,
)

_NUMERIC = ("ram", "vram", "storage_free", "cpu_ghz", "cpu_cores")

_QUESTION = {
    "ram": "Is this computer's RAM at least {value:g} GB?",
    "vram": "Does this computer have a GPU with at least {value:g} GB of dedicated VRAM?",
    "storage_free": "Does this computer have at least {value:g} GB of free storage?",
    "cpu_ghz": "Is this computer's processor speed at least {value:g} GHz?",
    "cpu_cores": "Does this computer's CPU have at least {value:g} cores?",
}


class RequirementError(ValueError):
    """The requirements text cannot be used at all."""


@dataclass(frozen=True)
class Requirement:
    """One parsed requirement clause."""

    component: str  # a _NUMERIC member, or "gpu_model" / "cpu_model"
    raw: str
    value: float | None = None
    unit: str = ""


@dataclass(frozen=True)
class CheckItem:
    """One requirement's outcome, with the working shown."""

    tier: str  # "minimum" | "recommended"
    component: str
    raw: str
    state: str  # "passed" | "failed" | "unknown"
    required: float | None = None
    unit: str = ""
    probability: float | None = None  # model-judged items only
    detail: str = ""
    path: str = ""  # "tier-table" | "model" | "" (numeric checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "component": self.component,
            "requirement": self.raw,
            "state": self.state,
            "required": self.required,
            "unit": self.unit,
            "probability": self.probability,
            "detail": self.detail,
            "path": self.path,
        }


@dataclass(frozen=True)
class CheckResult:
    """The composed checklist verdict with every item's working."""

    task: str
    verdict: str
    confidence: int
    items: list[CheckItem]
    unchecked: list[str]
    notes: list[str]
    model_unavailable: bool = False
    low_confidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            "items": [i.to_dict() for i in self.items],
            "unchecked": list(self.unchecked),
            "notes": list(self.notes),
        }


def parse_requirements(text: str) -> tuple[list[Requirement], list[str], list[str]]:
    """Split requirement text into clauses and classify each one.

    Returns the parsed requirements plus two honest leftovers: clauses that
    look like requirements but cannot be checked, and lines that were never
    checkable hardware facts to begin with.
    """
    requirements: list[Requirement] = []
    unchecked: list[str] = []
    ignored: list[str] = []
    for clause in re.split(r"[,;\n]| and ", text or "", flags=re.IGNORECASE):
        clause = clause.strip(" .-\t")
        if not clause:
            continue
        requirement = _parse_clause(clause)
        if requirement is not None:
            requirements.append(requirement)
        elif _IGNORED.search(clause):
            ignored.append(clause)
        else:
            unchecked.append(clause)

    # The same requirement twice adds nothing but noise.
    unique: list[Requirement] = []
    seen: set[tuple[str, float | None, str]] = set()
    for requirement in requirements:
        key = (requirement.component, requirement.value, requirement.unit)
        if key not in seen:
            seen.add(key)
            unique.append(requirement)
    return unique, unchecked, ignored


def _parse_clause(clause: str) -> Requirement | None:
    ghz = _GHZ.search(clause)
    if ghz:
        return Requirement("cpu_ghz", clause, _number(ghz.group(1)), "GHz")
    cores = _CORES.search(clause)
    if cores:
        return Requirement("cpu_cores", clause, float(cores.group(1)), "cores")
    size = _GB.search(clause)
    if size:
        value = _number(size.group(1))
        if size.group(2).lower() == "mb":
            value = round(value / 1024, 3)
        if _VRAM.search(clause):
            return Requirement("vram", clause, value, "GB")
        if _RAM.search(clause):
            return Requirement("ram", clause, value, "GB")
        if _STORAGE.search(clause):
            # A storage requirement means space to install into, so it is
            # checked against free storage, not the size of the largest disk.
            return Requirement("storage_free", clause, value, "GB")
    if _GPU_NAMED.search(clause):
        return Requirement("gpu_model", clause)
    if _CPU_NAMED.search(clause):
        return Requirement("cpu_model", clause)
    return None


def _number(text: str) -> float:
    return float(text.replace(",", "."))


def _machine_values(state: Mapping[str, Any]) -> dict[str, float | None]:
    """The measured numbers a numeric requirement can be compared against."""
    def clean(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    def best(entries: Any, key: str) -> float | None:
        values = [
            clean(entry.get(key))
            for entry in (entries or [])
            if isinstance(entry, Mapping)
        ]
        real = [v for v in values if v is not None]
        return max(real) if real else None

    cpu = state.get("cpu") if isinstance(state.get("cpu"), Mapping) else {}
    memory = state.get("memory") if isinstance(state.get("memory"), Mapping) else {}
    return {
        "ram": clean(memory.get("total_gib")),
        "vram": best(state.get("gpus"), "dedicated_vram_gib"),
        "storage_free": best(state.get("storage"), "free_gib"),
        "cpu_ghz": clean(cpu.get("clock_ghz")),
        "cpu_cores": clean(cpu.get("logical_cores")),
    }


def _capability_values(state: Mapping[str, Any]) -> dict[str, tuple[str, float] | None]:
    """The machine's best GPU and CPU, with their curated capability scores.

    The scores ride in the state (``capability_score``), filled by
    detection, so what a tier comparison sees is exactly what the model
    would have seen. A state without the key -- a hand-built payload --
    has no known tier, and the named-part check falls back to the model
    rather than inventing one.
    """
    def best(entries: Any) -> tuple[str, float] | None:
        scored = [
            (str(entry.get("name")), float(entry.get("capability_score")))
            for entry in (entries or [])
            if isinstance(entry, Mapping)
            and not isinstance(entry.get("capability_score"), bool)
            and isinstance(entry.get("capability_score"), (int, float))
        ]
        return max(scored, key=lambda entry: entry[1]) if scored else None

    cpu = state.get("cpu") if isinstance(state.get("cpu"), Mapping) else {}
    cpu_score = cpu.get("capability_score")
    cpu_best = None
    if not isinstance(cpu_score, bool) and isinstance(cpu_score, (int, float)):
        cpu_best = (str(cpu.get("name")), float(cpu_score))
    return {"gpu_model": best(state.get("gpus")), "cpu_model": cpu_best}


def check_requirements(
    state: Mapping[str, Any],
    task: str,
    minimum_text: str,
    recommended_text: str | None,
    client: SystemOneClient | None,
    config: Any = None,
    progress: Any = None,
) -> CheckResult:
    """Check pasted requirements against the machine and compose the verdict.

    Numeric requirements never touch the model. Named-part requirements
    are settled by the curated capability tables when both sides are
    known -- a deterministic comparison, offline. Only a named part the
    tables cannot place (an unknown required part, or a machine whose
    state carries no capability score) becomes an atomic noul question,
    and a coin flip or a dead endpoint leaves it unknown, which can
    soften the verdict but never flips it. ``progress``,
    when given, is called once right before the model is asked, so the UI can
    explain the wait.
    """
    minimum, unchecked_min, ignored_min = parse_requirements(minimum_text)
    recommended, unchecked_rec, ignored_rec = parse_requirements(recommended_text or "")
    if not minimum and not recommended:
        raise RequirementError(
            "no requirement could be parsed from the text; write lines like "
            "'8 GB RAM' or 'GTX 970'"
        )

    machine = _machine_values(state)
    capabilities = _capability_values(state)
    items: list[CheckItem] = []
    notes: list[str] = []

    for tier, requirements in (("minimum", minimum), ("recommended", recommended)):
        for requirement in requirements:
            if requirement.component in _NUMERIC:
                items.append(_check_numeric(requirement, tier, machine))
                continue
            # Named part: the tier table answers when it can.
            deterministic = _check_named_tier(requirement, tier, capabilities, config)
            if deterministic is not None:
                items.append(deterministic)
            else:
                items.append(CheckItem(
                    tier=tier, component=requirement.component, raw=requirement.raw,
                    state="unknown",
                    detail="waiting for the model's judgment" if client is not None
                    else "no model was available to judge it",
                ))

    # Only the named parts the tables could not settle go to the model,
    # so a checklist of known parts never spends a model call.
    model_indexes = [
        i for i, item in enumerate(items)
        if item.component in ("gpu_model", "cpu_model")
        and item.path != "tier-table"
    ]
    model_unavailable = False
    if model_indexes and client is not None:
        if progress is not None:
            progress()
        every_batch_failed = _judge_named(items, model_indexes, state, client, config, notes)
        if every_batch_failed:
            # The client exists but no endpoint answered, which is the
            # same outage as no client at all: the named parts are
            # unknown and the user must be told why.
            model_unavailable = True
            notes.append("no System One endpoint answered, so the named parts could not be judged")
    elif model_indexes:
        model_unavailable = True
        notes.append("no System One endpoint answered, so the named parts could not be judged")

    return _compose(task, items, notes, unchecked_min + unchecked_rec,
                    ignored_min + ignored_rec, model_unavailable, config)


def _check_named_tier(
    requirement: Requirement, tier: str,
    capabilities: dict[str, tuple[str, float] | None], config: Any,
) -> CheckItem | None:
    """The deterministic tier comparison for a named part, if possible.

    Both sides must be known: the required part must have a score in
    the curated table, and the machine's state must carry a capability
    score for its best GPU (or CPU). When either is missing, ``None``
    comes back and the caller falls back to the model -- an unknown
    tier is a gap in the evidence, not a zero.
    """
    lookup = rankings.gpu_entry if requirement.component == "gpu_model" else rankings.cpu_entry
    required = lookup(requirement.raw, config)
    if required is None:
        return None
    machine = capabilities.get(requirement.component)
    if machine is None:
        return None
    required_name, required_score = required
    machine_name, machine_score = machine
    passed = machine_score + 1e-9 >= required_score
    return CheckItem(
        tier=tier, component=requirement.component, raw=requirement.raw,
        state="passed" if passed else "failed", path="tier-table",
        detail=(
            f"{required_name} ~ {required_score:g}/100 vs this machine's "
            f"{machine_name} ~ {machine_score:g}/100 "
            f"(curated tier table v"
            f"{getattr(config, 'rankings_version', rankings.RANKINGS_VERSION)})"
        ),
    )


def _check_numeric(requirement: Requirement, tier: str,
                   machine: dict[str, float | None]) -> CheckItem:
    have = machine.get(requirement.component)
    if have is None:
        return CheckItem(
            tier=tier, component=requirement.component, raw=requirement.raw,
            state="unknown", required=requirement.value, unit=requirement.unit,
            detail=f"the machine's {requirement.component.replace('_', ' ')} is not readable",
        )
    passed = have + 1e-9 >= (requirement.value or 0.0)
    return CheckItem(
        tier=tier, component=requirement.component, raw=requirement.raw,
        state="passed" if passed else "failed", required=requirement.value,
        unit=requirement.unit,
        detail=(f"needs {requirement.value:g} {requirement.unit}; "
                f"this machine reports {have:g} {requirement.unit}"),
    )


def _judge_named(items: list[CheckItem], indexes: list[int], state: Mapping[str, Any],
                 client: SystemOneClient, config: Any, notes: list[str]) -> bool:
    """Ask the provider one atomic noul per named part, in small batches.

    Every item this settles is marked ``path="model"`` so a reader can
    tell a model judgment from a tier-table one at a glance.

    Returns True when *every* batch failed, meaning no System One
    endpoint answered at all. Partial success (some batches answered)
    returns False: the items those batches covered carry real answers,
    so the model was available even though other items stayed unknown.
    """
    margin = getattr(config, "uncertain_margin", 0.10)
    all_failed = True
    for start in range(0, len(indexes), QUESTIONS_PER_CALL):
        batch = indexes[start:start + QUESTIONS_PER_CALL]
        questions = {
            f"req_{index}": {
                "type": "noul",
                "instructions": _named_question(items[index]),
            }
            for index in batch
        }
        try:
            decision = client.ask(state, questions)
        except SystemOneError as error:
            notes.append(f"the named parts could not be judged: {error}")
            for index in batch:
                items[index] = _unknown_item(items[index], "no model answer")
            continue
        all_failed = False
        for index in batch:
            probability = decision.noul(f"req_{index}")
            if abs(probability - 0.5) < margin:
                items[index] = CheckItem(
                    tier=items[index].tier, component=items[index].component,
                    raw=items[index].raw, state="unknown", path="model",
                    probability=probability,
                    detail=f"the model was close to a coin flip (P={probability:.2f})",
                )
            else:
                passed = probability >= 0.5
                items[index] = CheckItem(
                    tier=items[index].tier, component=items[index].component,
                    raw=items[index].raw, path="model",
                    state="passed" if passed else "failed",
                    probability=probability,
                    detail=f"the model judged this with P={probability:.2f}",
                )
    return all_failed


def _named_question(item: CheckItem) -> str:
    if item.component == "gpu_model":
        return (f"Is this computer's GPU at least as capable as the one in "
                f"this requirement: {item.raw}?")
    return (f"Is this computer's CPU at least as capable as the one in "
            f"this requirement: {item.raw}?")


def _unknown_item(item: CheckItem, reason: str) -> CheckItem:
    return CheckItem(
        tier=item.tier, component=item.component, raw=item.raw, state="unknown",
        required=item.required, unit=item.unit, probability=item.probability,
        detail=reason, path=item.path,
    )


def _item_confidence(item: CheckItem) -> float | None:
    """How confident one outcome is, as 0-100, or ``None``.

    A passed or failed item is as confident as its
    probability backs the verdict. An unknown item carries
    a probability only when the model was unsure (a coin
    flip), so it can never be more than ``min(p, 1-p)``
    -- 50 at best.
    """
    if item.probability is None:
        return None
    probability = item.probability
    if item.state == "passed":
        return probability * 100.0
    if item.state == "failed":
        return (1.0 - probability) * 100.0
    return min(probability, 1.0 - probability) * 100.0


def _compose(task: str, items: list[CheckItem], notes: list[str], unchecked: list[str],
             ignored: list[str], model_unavailable: bool,
             config: Any = None) -> CheckResult:
    fails_min = [i for i in items if i.tier == "minimum" and i.state == "failed"]
    fails_rec = [i for i in items if i.tier == "recommended" and i.state == "failed"]
    passes_min = [i for i in items if i.tier == "minimum" and i.state == "passed"]
    recommended_items = [i for i in items if i.tier == "recommended"]

    if fails_min:
        verdict = "Below the minimum"
    elif fails_rec:
        verdict = "Meets the minimum -- it will run, but tightly"
    elif passes_min:
        rec_all_pass = bool(recommended_items) and all(
            i.state == "passed" for i in recommended_items
        )
        verdict = ("Meets the recommended specs -- it should run well"
                   if rec_all_pass else "Meets the minimum")
    else:
        verdict = "Cannot tell"

    model_confidences = [
        confidence
        for confidence in (_item_confidence(item) for item in items)
        if confidence is not None
    ]
    if model_confidences:
        confidence = int(round(min(model_confidences)))
    elif any(item.state == "unknown" for item in items) or model_unavailable:
        # Nothing was judged: an all-unknown checklist (a
        # full model outage) must not inherit the
        # deterministic score.
        confidence = 0
    else:
        # A purely deterministic checklist: every item was
        # settled without a model, so nothing was guessed.
        confidence = 100

    # The reportable floor labels, it does not hide: a checklist
    # below the floor keeps its verdict and its number, but the
    # user is told to treat it as a guess instead of a plain,
    # confident-looking answer. getattr, so a config object that
    # predates the setting still composes.
    floor = getattr(config, "min_reportable_confidence", 20)
    low_confidence = confidence < floor

    notes = list(notes)
    if unchecked:
        notes.append("not checked, because no known requirement could be read from them: "
                     + "; ".join(unchecked))
    if ignored:
        notes.append(f"{len(ignored)} line(s) were platform or peripheral notes, "
                     "not hardware requirements")
    if low_confidence:
        notes.append(
            f"confidence is below the reportable floor "
            f"({floor}%); treat this verdict as a guess"
        )

    return CheckResult(
        task=task, verdict=verdict, confidence=confidence,
        items=items, unchecked=list(unchecked), notes=notes,
        model_unavailable=model_unavailable, low_confidence=low_confidence,
    )
