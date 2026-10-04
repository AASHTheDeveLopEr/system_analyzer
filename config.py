"""Declared tunables for rating and verdicts.

Every number that decides what the user is told lives here, named and
documented, rather than inside the logic that uses it. The verdict bands are
deliberately "derived from model probabilities using thresholds declared in
configuration, not hard-coded ad hoc" -- so if you want a different
split between "Yes" and "Likely", this is the file, not the question builder.

Override any of it with a JSON file, either by pointing ``--config`` at one or
by setting ``SYSTEM_ANALYZER_CONFIG``::

    {"verdict_bands": {"yes_at": 0.8}, "rating_weights": {"cpu": 0.4}}

Unknown keys are rejected rather than ignored, so a typo in a config file
fails loudly instead of silently keeping a default you thought you changed.

These are *starting points*, not blessed product values. The weights in
particular encode a judgement about what a general-purpose 0-100 rating should
reward, and that judgement has not been made by anybody yet. They are in a file so that arguing about them is a
diff, not a code change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_ENV = "SYSTEM_ANALYZER_CONFIG"

#: Where the model may answer "no" at all. Below this we say we cannot tell
#: rather than reporting a very confident "No".
MIN_PLAUSIBLE_PROBABILITY = 0.05

# --------------------------------------------------------------------------
# Verdict bands
# --------------------------------------------------------------------------
# Ordered high to low; the first threshold the probability reaches wins. The
# bands are deliberately not symmetric about 0.5: "Likely" occupies a wider
# slice than "Probably not" because a machine that *just* misses a requirement
# and one that is nowhere near it deserve different words.
VERDICT_BANDS: dict[str, float] = {
    "yes_at": 0.80,
    "likely_at": 0.55,
    "probably_not_at": 0.30,
}

#: How close to 0.5 counts as "the model is not sure". Inside this margin the
#: verdict is capped at "Probably not"-or-better reporting and the confidence
#: is clamped down, because a coin flip is not evidence.
UNCERTAIN_MARGIN = 0.10

#: Confidence ceiling applied when the question could not be grounded in the
#: detected hardware at all. Reporting 80% on a question the specs cannot
#: answer would be exactly the kind of confident-looking number that must not
#: happen.
UNGROUNDED_CONFIDENCE_CAP = 30

#: Confidence ceiling when the probability sits inside UNCERTAIN_MARGIN.
UNCERTAIN_CONFIDENCE_CAP = 45

#: Penalty applied to confidence when the two independent questions about the
#: same request disagree. Disagreement is evidence of a shaky answer.
DISAGREEMENT_PENALTY = 15

#: Minimum confidence a verdict needs to be shown as a plain verdict;
#: below this the verdict is labelled a guess, not hidden -- the band
#: stays the same, and a note says to treat it as a guess.
MIN_REPORTABLE_CONFIDENCE = 20

#: Allowed confidence drift across repeated identical questions, in points
#:.
STABILITY_TOLERANCE = 5

# --------------------------------------------------------------------------
# Rating composition
# --------------------------------------------------------------------------
# Weights over the deterministic component sub-scores. They must sum to 1.0;
# :func:`Config.validate` enforces it, because weights that do not sum to 1
# would silently rescale every rating.
RATING_WEIGHTS: dict[str, float] = {
    "cpu": 0.30,
    "memory": 0.25,
    "gpu": 0.35,
    "storage": 0.10,
}

#: How much of the final number comes from the model rather than the
#: deterministic sub-scores. 0.5 means equal blend. The sub-scores are always
#: shown separately, so this changes the headline number, never the audit.
MODEL_BLEND = 0.5

#: Contributing factors reported with every rating (at least 3).
MIN_FACTORS = 3

#: A component scoring below this is called out as a bottleneck in verdicts.
BOTTLENECK_SCORE = 45

# --------------------------------------------------------------------------
# Component scoring curves
# --------------------------------------------------------------------------
# Piecewise-linear anchors, in the component's own unit, mapped onto 0-100.
# Anchors are (value, score) pairs and the score is linearly interpolated
# between them; a value above the last anchor is clamped to 100 and below the
# first to 0. The anchors below are the *shape* of the judgement; the exact
# numbers are the part to argue about.
CPU_CURVE: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 4.0),
    (2.0, 10.0),
    (4.0, 25.0),
    (6.0, 40.0),
    (8.0, 52.0),
    (12.0, 68.0),
    (16.0, 80.0),
    (24.0, 92.0),
    (32.0, 100.0),
)

MEMORY_CURVE: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 3.0),
    (2.0, 8.0),
    (4.0, 18.0),
    (8.0, 35.0),
    (16.0, 55.0),
    (32.0, 75.0),
    (64.0, 92.0),
    (128.0, 100.0),
)

#: Scored on dedicated VRAM only. An integrated adapter with no dedicated
#: memory is *unknown*, not zero -- see inventory.Gpu.vram_bytes.
GPU_VRAM_CURVE: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 12.0),
    (2.0, 26.0),
    (4.0, 45.0),
    (6.0, 60.0),
    (8.0, 72.0),
    (12.0, 86.0),
    (16.0, 94.0),
    (24.0, 100.0),
)

STORAGE_CURVE: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (64.0, 30.0),
    (256.0, 55.0),
    (512.0, 72.0),
    (1024.0, 86.0),
    (2048.0, 100.0),
)

#: The tiers the model is asked to place the machine in. Each level has to
#: stand on its own, because the model never sees a neighbouring level -- it
#: sees the whole list, and levels that only make sense relative to each other
#: give it nothing to anchor to.
RATING_TIERS: tuple[str, ...] = (
    "Entry level: 4 GB of RAM and no dedicated GPU. Web browsing, office work "
    "and very light games.",
    "Mainstream: 8 GB to 16 GB of RAM and 4 GB of dedicated VRAM. Everyday "
    "games, photo editing, many apps at once.",
    "Performance: 16 GB to 32 GB of RAM and 8 GB of dedicated VRAM. Recent "
    "games at high settings, video editing, running local AI models.",
    "Enthusiast: 32 GB of RAM and 12 GB or more of dedicated VRAM. 4K video, "
    "3D rendering, large local AI models.",
    "Workstation: 64 GB of RAM and 16 GB or more of professional VRAM. "
    "Machine learning, large simulations, sustained heavy rendering.",
)


class ConfigError(ValueError):
    """The config file says something that cannot be used."""


class Config:
    """Effective settings: defaults, overlaid with the user's JSON file."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._values = values or {}

    # -- access ---------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        value = self._values.get(key, default)
        return dict(value) if isinstance(value, dict) else value

    def bands(self) -> list[tuple[float, str]]:
        """``[(threshold, verdict), ...]`` high to low, from the declared bands.

        An override may set any one threshold and inherit the rest, because
        "I only want Yes to mean 95%" is a reasonable thing to ask for and it
        should not require restating the whole scale.
        """
        bands = {**VERDICT_BANDS, **self.get("verdict_bands", {})}
        return [
            (float(bands["yes_at"]), "Yes"),
            (float(bands["likely_at"]), "Likely"),
            (float(bands["probably_not_at"]), "Probably not"),
            (0.0, "No"),
        ]

    def to_dict(self) -> dict[str, Any]:
        """Everything that affects an answer -- for ``config`` and for --json."""
        return {
            "verdict_bands": self.get("verdict_bands", VERDICT_BANDS),
            "uncertain_margin": self.uncertain_margin,
            "ungrounded_confidence_cap": self.ungrounded_confidence_cap,
            "uncertain_confidence_cap": self.uncertain_confidence_cap,
            "disagreement_penalty": self.disagreement_penalty,
            "min_reportable_confidence": self.min_reportable_confidence,
            "stability_tolerance": self.stability_tolerance,
            "rating_weights": self.get("rating_weights", RATING_WEIGHTS),
            "model_blend": self.model_blend,
            "min_factors": self.min_factors,
            "bottleneck_score": self.bottleneck_score,
            "rating_tiers": list(self.rating_tiers),
        }

    # Derived accessors, so the rest of the code never indexes a dict directly.
    @property
    def uncertain_margin(self) -> float:
        return float(self.get("uncertain_margin", UNCERTAIN_MARGIN))

    @property
    def ungrounded_confidence_cap(self) -> int:
        return int(self.get("ungrounded_confidence_cap", UNGROUNDED_CONFIDENCE_CAP))

    @property
    def uncertain_confidence_cap(self) -> int:
        return int(self.get("uncertain_confidence_cap", UNCERTAIN_CONFIDENCE_CAP))

    @property
    def disagreement_penalty(self) -> int:
        return int(self.get("disagreement_penalty", DISAGREEMENT_PENALTY))

    @property
    def min_reportable_confidence(self) -> int:
        return int(self.get("min_reportable_confidence", MIN_REPORTABLE_CONFIDENCE))

    @property
    def stability_tolerance(self) -> int:
        return int(self.get("stability_tolerance", STABILITY_TOLERANCE))

    @property
    def model_blend(self) -> float:
        return float(self.get("model_blend", MODEL_BLEND))

    @property
    def min_factors(self) -> int:
        return int(self.get("min_factors", MIN_FACTORS))

    @property
    def bottleneck_score(self) -> float:
        return float(self.get("bottleneck_score", BOTTLENECK_SCORE))

    @property
    def rating_tiers(self) -> tuple[str, ...]:
        return tuple(self.get("rating_tiers", RATING_TIERS))

    def weights(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.get("rating_weights", RATING_WEIGHTS).items()}

    def curve(self, name: str) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(x), float(y))
            for x, y in self.get(f"{name}_curve", _DEFAULT_CURVES[name])
        )

    # -- checking -------------------------------------------------------
    def validate(self) -> None:
        """Reject settings that would produce nonsense, loudly."""
        bands = self.bands()
        ordered = [threshold for threshold, _ in bands]
        if ordered != sorted(ordered, reverse=True):
            raise ConfigError(
                f"verdict_bands must run high to low, got {ordered}"
            )
        if any(not 0.0 <= threshold <= 1.0 for threshold, _ in bands):
            raise ConfigError("verdict_bands must sit between 0 and 1")

        weights = self.weights()
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"rating_weights must sum to 1.0 so ratings stay on the 0-100 "
                f"scale, got {total:.4f} from {weights}"
            )
        if not 0.0 <= self.model_blend <= 1.0:
            raise ConfigError("model_blend must be between 0 and 1")
        if not 0.0 <= self.uncertain_margin <= 0.5:
            raise ConfigError("uncertain_margin must be between 0 and 0.5")

        # The floor is a count of percentage points, so it has to be
        # a whole number on the 0-100 scale. Checking it here turns
        # "twenty" into a config error (exit 2) instead of a traceback
        # the first time the property is read, and stops -5 (which
        # would flag nothing) and 500 (which would flag everything)
        # from silently changing what every verdict looks like.
        floor = self.get("min_reportable_confidence", MIN_REPORTABLE_CONFIDENCE)
        if isinstance(floor, bool) or not isinstance(floor, int):
            raise ConfigError(
                f"min_reportable_confidence must be an integer between "
                f"0 and 100, got {floor!r}"
            )
        if not 0 <= floor <= 100:
            raise ConfigError(
                f"min_reportable_confidence must be between 0 and 100, "
                f"got {floor}"
            )

        for name, curve in (
            ("cpu", self.curve("cpu")),
            ("memory", self.curve("memory")),
            ("gpu", self.curve("gpu")),
            ("storage", self.curve("storage")),
        ):
            values = [x for x, _ in curve]
            if values != sorted(values):
                raise ConfigError(f"{name} curve anchors must ascend")
            if not curve:
                raise ConfigError(f"{name} curve has no anchors")

        tiers = self.rating_tiers
        if len(tiers) < 2:
            raise ConfigError("rating_tiers needs at least 2 levels to be a scale")


_DEFAULT_CURVES = {
    "cpu": CPU_CURVE,
    "memory": MEMORY_CURVE,
    "gpu": GPU_VRAM_CURVE,
    "storage": STORAGE_CURVE,
}


def load(path: str | os.PathLike[str] | None = None) -> Config:
    """Defaults, overlaid with a JSON file from ``path`` or the environment."""
    if path is None:
        path = os.environ.get(CONFIG_ENV) or None
    if not path:
        return Config()

    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"config file not found: {file}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"{file} is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError(f"{file} must contain a JSON object, got {type(raw).__name__}")

    known = _known_keys()
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"{file} has unknown key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(known))}"
        )

    config = Config(raw)
    config.validate()
    return config


def _known_keys() -> set[str]:
    return {
        "verdict_bands",
        "uncertain_margin",
        "ungrounded_confidence_cap",
        "uncertain_confidence_cap",
        "disagreement_penalty",
        "min_reportable_confidence",
        "stability_tolerance",
        "rating_weights",
        "model_blend",
        "min_factors",
        "bottleneck_score",
        "rating_tiers",
        "cpu_curve",
        "memory_curve",
        "gpu_curve",
        "storage_curve",
    }
