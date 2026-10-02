"""The judgment layer: turning model answers and measured facts into verdicts.

The model supplies judgement; the local code supplies arithmetic and honesty.
Everything here takes either an ``inventory.Inventory`` (real units) or the
scrubbed state mapping, plus a client that answers typed questions, and
composes the two things a user actually reads:

* :func:`rate_system` -- a 0-100 rating with its working shown. The model's
  tier and the deterministic component scores are blended; when no endpoint
  answers, the deterministic score is returned on its own, flagged, never
  dressed up as a model answer.
* :func:`answer_capability` -- a banded verdict for "can I run X?" with an
  honest confidence.

The honesty rules live here and are non-negotiable: an unknown reading is
never quietly scored as zero, a coin flip is capped and named uncertain, and
a question the specs cannot determine gets "Cannot tell", not an invention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from system_one import SystemOneError

GIB = 1024 ** 3


@dataclass(frozen=True)
class ComponentScore:
    """One component's deterministic 0-100 sub-score.

    ``score`` is ``None`` when the component could not be measured. It is not
    zero: an unreadable GPU is a gap in the evidence, and scoring it as zero
    would drag the rating down for a reason we cannot prove.
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
    """A 0-100 rating with the working shown."""

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
    """A banded answer with an honest confidence."""

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

    Accepts ``inventory.Inventory`` (real units, preferred) or the scrubbed
    state mapping, so this layer can be tested without psutil installed.
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
# rate_system
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

    The model's tier and the deterministic component scores are blended, so
    the result stays explainable and reproducible. If no endpoint answers, the
    deterministic score is returned on its own with ``method="rules"`` -- a
    rating is always available, and it is always clear which one it came from.
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

    At least three factors are reported. Two of those slots are reserved for
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
# answer_capability
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
    """Turn "can I run X?" into a verdict and a confidence.

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
    """One line, naming a component that was really detected."""
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
