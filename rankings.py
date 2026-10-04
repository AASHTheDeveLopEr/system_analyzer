"""Curated, offline capability rankings for common GPUs and CPUs.

The decision model receives raw specs ("GeForce 610M, 2 GiB VRAM") but
has no idea how hardware compares across generations: it cannot know
whether an RTX 3050 satisfies a "GTX 970 or better" requirement. This
module is the grounding layer that fills that gap, in two roles:

* the scores ride along in the state payload (``capability_score``), so
  every model call gets a relative-capability anchor for each detected
  part;
* requirement checks use them for a deterministic tier comparison, so a
  named part ("GTX 970 or better") is settled by arithmetic against a
  known machine instead of a coin-flip-prone model question.

Provenance: the ordering is informed by publicly available GPU/CPU tier
lists and generation ordering (which card beat which, which generation
came after which). The scores are RELATIVE ESTIMATES on a 0-100 scale,
not benchmarks -- they say "roughly how capable", never "how many
frames". A part that matches no entry is ``None`` (unknown), and no
score is ever invented.

Anchors (so the scale can be reasoned about): the current flagship
discrete GPU is ~100 (RTX 5090 class), an RTX 4060 is ~55-60, a GTX
1060 is ~35, a GTX 970 is ~28, a GeForce 610M is ~8. On the CPU side:
the current flagship is ~100, a Ryzen 5 3600 is ~45, an i5-2450M is ~10.

Mobile and laptop GPU variants are mapped to their desktop equivalent
with a documented penalty (x0.8, rounded to one decimal): a laptop
card is the same silicon at a lower clock, so it is the same tier
slightly discounted, not a different part. A few laptop-only parts
(the GeForce 610M class) have entries of their own because no desktop
equivalent exists.

The tables can be overridden through the config file (``gpu_rankings``
/ ``cpu_rankings`` as lists of ``[pattern, score]``), which is how a
user corrects a score without a code change. The version string dates
the ordering; an override may declare its own.
"""

from __future__ import annotations

from typing import Any

#: Dates the ordering in the tables below. Bump it (YYYY-MM) whenever
#: the tables change, so a state payload says which ordering produced
#: its capability scores.
RANKINGS_VERSION = "2026-10"

UNKNOWN = "unknown"

#: Penalty applied to a mobile/laptop GPU relative to its desktop
#: equivalent. Documented here, not where it is applied, because it is
#: a judgement about the gap between desktop and laptop clocks, and
#: that judgement is what a reader should argue with.
MOBILE_PENALTY = 0.8

# Tokens that carry no tier information once the model family is known.
# "NVIDIA GeForce RTX 3060", "GeForce RTX 3060" and "RTX 3060" are
# the same card, and "Intel(R) UHD Graphics 620" is the same adapter
# as "UHD Graphics 620"; the vendor/brand markers are folded away so
# the normalization below cannot tell them apart.
_STRIPPED_TOKENS = frozenset({
    "nvidia", "geforce", "radeon", "intel", "amd", "r", "tm",
})

# Tokens that mark a laptop variant of a desktop card. "M" is included
# because old NVIDIA mobile parts are named with a bare M suffix
# ("GeForce 610M" keeps its M glued to the number, so only a separate
# "m" token counts). "Max-Q" folds to "maxq" during normalization.
_MOBILE_TOKENS = frozenset({"mobile", "laptop", "notebook", "maxq", "m"})

# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------
# Each entry is (name pattern, capability score 0-100). Ordering within
# the table is irrelevant; matching is normalization-based, so a pattern
# only has to normalize to the same tokens as the name being looked up.

#: Desktop and mobile discrete GPUs, 2013-2026, plus the integrated
#: adapters common enough to anchor the low end.
GPU_RANKINGS: tuple[tuple[str, float], ...] = (
    # NVIDIA GeForce RTX 50 series (2025)
    ("RTX 5090", 100.0),
    ("RTX 5080", 92.0),
    ("RTX 5070 Ti", 86.0),
    ("RTX 5070", 80.0),
    ("RTX 5060", 62.0),
    # NVIDIA GeForce RTX 40 series (2022-2023)
    ("RTX 4090", 98.0),
    ("RTX 4080", 88.0),
    ("RTX 4070 Ti Super", 82.0),
    ("RTX 4070", 70.0),
    ("RTX 4060 Ti", 64.0),
    ("RTX 4060", 58.0),
    # NVIDIA GeForce RTX 30 series (2020-2022)
    ("RTX 3090", 86.0),
    ("RTX 3080", 76.0),
    ("RTX 3070", 66.0),
    ("RTX 3060 Ti", 60.0),
    ("RTX 3060", 55.0),
    ("RTX 3050", 44.0),
    # NVIDIA GeForce RTX 20 series (2018-2019)
    ("RTX 2080 Ti", 66.0),
    ("RTX 2080", 60.0),
    ("RTX 2070", 52.0),
    ("RTX 2060", 42.0),
    # NVIDIA GeForce GTX 16 series (2019)
    ("GTX 1660 Super", 38.0),
    ("GTX 1660", 34.0),
    ("GTX 1650", 26.0),
    # NVIDIA GeForce GTX 10 series (2016-2017)
    ("GTX 1080 Ti", 46.0),
    ("GTX 1080", 40.0),
    ("GTX 1070", 36.0),
    ("GTX 1060", 35.0),
    ("GTX 1050 Ti", 28.0),
    # NVIDIA GeForce GTX 900 series (2014-2015)
    ("GTX 980 Ti", 30.0),
    ("GTX 980", 29.0),
    ("GTX 970", 28.0),
    # NVIDIA GeForce GTX 700 series (2013-2014)
    ("GTX 780 Ti", 27.0),
    ("GTX 770", 23.0),
    ("GTX 750 Ti", 15.0),
    # Legacy low-end discrete
    ("GeForce 610M", 8.0),
    ("GT 1030", 16.0),
    # AMD Radeon RX 9000 series (2025)
    ("RX 9070 XT", 90.0),
    ("RX 9070", 84.0),
    # AMD Radeon RX 7000 series (2022-2023)
    ("RX 7900 XTX", 84.0),
    ("RX 7800 XT", 70.0),
    ("RX 7600", 50.0),
    # AMD Radeon RX 6000 series (2020-2021)
    ("RX 6900 XT", 74.0),
    ("RX 6800 XT", 70.0),
    ("RX 6700 XT", 58.0),
    # AMD Radeon RX 5000 series (2019)
    ("RX 590", 31.0),
    ("RX 580", 29.0),
    # AMD Radeon RX 4000 series (2016)
    ("RX 480", 30.0),
    ("RX 460", 17.0),
    # Intel Arc (2022-2024)
    ("Arc A770", 55.0),
    ("Arc A750", 50.0),
    ("Arc A380", 32.0),
    # Apple silicon GPU tiers (reported as the chip name on macOS)
    ("Apple M4", 58.0),
    ("Apple M3", 50.0),
    ("Apple M2", 40.0),
    ("Apple M1", 32.0),
    # Common Intel integrated graphics, the low anchors
    ("Iris Xe Graphics", 24.0),
    ("UHD Graphics 770", 20.0),
    ("UHD Graphics 630", 14.0),
    ("UHD Graphics 620", 13.0),
    ("HD Graphics 630", 12.0),
    ("HD Graphics 3000", 5.0),
)

#: CPUs of the last ~8 years plus the low anchors. Mobile-only parts
#: (the i5-2450M class) are entries in their own right because no
#: desktop equivalent exists to penalize.
CPU_RANKINGS: tuple[tuple[str, float], ...] = (
    # Intel Core 14th generation (2023)
    ("Core i9-14900K", 96.0),
    ("Core i7-14700K", 88.0),
    ("Core i5-14600K", 78.0),
    # Intel Core 13th generation (2022-2023)
    ("Core i9-13900K", 94.0),
    ("Core i7-13700K", 86.0),
    ("Core i5-13600K", 76.0),
    # Intel Core 12th generation (2021-2022)
    ("Core i9-12900K", 90.0),
    ("Core i7-12700K", 82.0),
    ("Core i5-12600K", 72.0),
    # Intel Core 11th generation (2020-2021)
    ("Core i9-11900K", 80.0),
    ("Core i7-11700K", 74.0),
    ("Core i5-11600K", 64.0),
    ("Core i5-1135G7", 62.0),
    # Intel Core 10th generation (2019-2020)
    ("Core i9-10900K", 74.0),
    ("Core i7-10700K", 68.0),
    ("Core i5-10400", 60.0),
    # Intel Core 9th generation (2018-2019)
    ("Core i9-9900K", 62.0),
    ("Core i7-9700K", 58.0),
    ("Core i5-9400", 52.0),
    # Intel Core 8th generation (2017-2018)
    ("Core i7-8700", 54.0),
    ("Core i5-8400", 48.0),
    # Intel Core 7th generation (2016-2017)
    ("Core i7-7700", 46.0),
    ("Core i5-7400", 40.0),
    # Intel Core 6th generation (2015-2016)
    ("Core i7-6700", 44.0),
    ("Core i5-6400", 38.0),
    # Intel Core 4th generation (2013-2014)
    ("Core i7-4790", 36.0),
    ("Core i5-4460", 30.0),
    # Intel Core 2nd generation, mobile-only (2011-2012)
    ("Core i7-2670QM", 12.0),
    ("Core i5-2450M", 10.0),
    # Intel Core i3 (the low end of each generation)
    ("Core i3-12100", 56.0),
    ("Core i3-10100", 44.0),
    ("Core i3-9100", 40.0),
    ("Core i3-8100", 38.0),
    # Intel Celeron / Pentium / Atom, the floor
    ("Pentium Gold G7400", 10.0),
    ("Celeron G5905", 6.0),
    ("Atom x5-Z8350", 4.0),
    # AMD Ryzen 9000 series (2024)
    ("Ryzen 9 9950X", 100.0),
    ("Ryzen 7 9800X", 92.0),
    # AMD Ryzen 7000 series (2022-2023)
    ("Ryzen 9 7950X", 94.0),
    ("Ryzen 7 7700X", 86.0),
    ("Ryzen 5 7600", 80.0),
    # AMD Ryzen 5000 series (2020-2021)
    ("Ryzen 9 5950X", 82.0),
    ("Ryzen 9 5900X", 80.0),
    ("Ryzen 7 5800X", 76.0),
    ("Ryzen 7 5700X", 74.0),
    ("Ryzen 5 5600X", 70.0),
    ("Ryzen 5 5600", 68.0),
    # AMD Ryzen 3000 series (2019-2020)
    ("Ryzen 9 3950X", 72.0),
    ("Ryzen 9 3900X", 68.0),
    ("Ryzen 7 3700X", 62.0),
    ("Ryzen 5 3600X", 47.0),
    ("Ryzen 5 3600", 45.0),
    ("Ryzen 3 3200G", 30.0),
    # AMD Ryzen 2000 series (2018)
    ("Ryzen 7 2700X", 48.0),
    ("Ryzen 5 2600", 40.0),
    ("Ryzen 3 2200G", 26.0),
    # AMD Ryzen 1000 series (2017)
    ("Ryzen 5 1600", 34.0),
    # Apple M-series (the chip name is what the platform reports)
    ("Apple M4", 88.0),
    ("Apple M3", 76.0),
    ("Apple M2", 62.0),
    ("Apple M1", 52.0),
)

#: Capability bands, high to low. The word describes where a part sits
#: relative to everything else in its table, today, not in its own era.
_TIERS: tuple[tuple[float, str], ...] = (
    (90.0, "flagship"),
    (70.0, "high-end"),
    (50.0, "mid-range"),
    (30.0, "entry-level"),
    (0.0, "low-end"),
)

# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def gpu_score(name: str, config: Any = None) -> float | None:
    """The capability score of a GPU name, or ``None`` when unknown.

    ``config``, when given, is anything with a ``get`` method (the app's
    ``Config``); a ``gpu_rankings`` override in it replaces the built-in
    table.
    """
    entry = gpu_entry(name, config)
    return entry[1] if entry else None


def gpu_entry(
    name: str, config: Any = None
) -> tuple[str, float] | None:
    """The matched GPU table entry as ``(pattern, score)``, or ``None``.

    The pattern is the table's spelling of the part, which is what a
    comparison should name: the requirement text may say "NVIDIA GTX
    970 or better", but the table matched "GTX 970".
    """
    return _lookup_entry(name, effective_table("gpu_rankings", GPU_RANKINGS, config))


def cpu_score(name: str, config: Any = None) -> float | None:
    """The capability score of a CPU name, or ``None`` when unknown."""
    entry = cpu_entry(name, config)
    return entry[1] if entry else None


def cpu_entry(
    name: str, config: Any = None
) -> tuple[str, float] | None:
    """The matched CPU table entry as ``(pattern, score)``, or ``None``."""
    return _lookup_entry(name, effective_table("cpu_rankings", CPU_RANKINGS, config))


def describe(name: str) -> str | None:
    """A short human label for a GPU or CPU name, or ``None`` if unknown.

    The label names the nearest table anchor, so it can be checked
    against the tables above instead of trusted on faith: a returned
    ``"mid-range, ~55/100 (comparable to a RTX 3060)"`` is a claim
    about the table, not about the world. The GPU table is searched
    first because GPU names are the ones usually asked about.
    """
    for table in (GPU_RANKINGS, CPU_RANKINGS):
        entry = _lookup_entry(name, table)
        if entry is not None:
            score = entry[1]
            anchor, _anchor_score = _nearest_anchor(score, table)
            article = "an" if anchor[:1].lower() in "aeiou" else "a"
            return (
                f"{tier_word(score)}, ~{score:g}/100 "
                f"(comparable to {article} {anchor})"
            )
    return None


def tier_word(score: float) -> str:
    """The capability band of a score: "flagship" ... "low-end"."""
    for threshold, word in _TIERS:
        if score >= threshold:
            return word
    return "low-end"


def tier_text(score: float | None) -> str:
    """The tier as it rides along in a display line.

    Returns ``"mid-range 55/100"`` for a known score and ``"unknown"``
    for ``None`` -- the honest string, not a guessed zero.
    """
    if score is None:
        return UNKNOWN
    return f"{tier_word(score)} {score:g}/100"


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def _lookup_entry(
    name: str, table: tuple[tuple[str, float], ...]
) -> tuple[str, float] | None:
    """Find *name* in *table*, or ``None`` when nothing matches.

    Matching is deliberately conservative: normalization (vendor
    prefixes, suffixes, punctuation) plus at most a contiguous-token
    containment, longest pattern first. Anything fuzzier would cross
    the boundary between a card and its Ti/Super variant, which is
    exactly the distinction a requirement check is asked about.
    """
    tokens = _tokens(name)
    if not tokens:
        return None

    mobile = any(token in _MOBILE_TOKENS for token in tokens)
    if mobile:
        tokens = [token for token in tokens if token not in _MOBILE_TOKENS]
        if not tokens:
            return None

    # Exact match first. This is the common case, and it is the only
    # match that can never confuse "RTX 3060" with "RTX 3060 Ti".
    for pattern, score in table:
        if _tokens(pattern) == tokens:
            return pattern, _laptop_adjusted(score, mobile)

    # Fallback: the longest pattern whose tokens appear contiguously in
    # the name. This is what absorbs generation prefixes ("11th Gen
    # Intel Core i5-1135G7") and requirement prose ("GTX 970 or
    # better") without crossing a suffix boundary, because a longer
    # pattern (the Ti variant of the same card) always wins over a
    # shorter one.
    best_length = 0
    best: tuple[str, float] | None = None
    for pattern, score in table:
        pattern_tokens = _tokens(pattern)
        if len(pattern_tokens) <= best_length:
            continue
        if _contains_contiguous(tokens, pattern_tokens):
            best_length = len(pattern_tokens)
            best = (pattern, score)

    if best is None:
        return None
    return best[0], _laptop_adjusted(best[1], mobile)


def _tokens(name: str) -> list[str]:
    """The comparison form of a hardware name.

    Lowercase, vendor/brand markers removed, punctuation folded to
    spaces, "Max-Q" folded to "maxq", and the redundant "gpu" token
    dropped (everything in the GPU table is a GPU).
    """
    lowered = (name or "").lower()
    for marker in ("(r)", "(tm)"):
        lowered = lowered.replace(marker, " ")
    folded = "".join(
        char if char.isalnum() or char.isspace() else " " for char in lowered
    )
    tokens = [
        token for token in folded.split() if token not in _STRIPPED_TOKENS
    ]
    result: list[str] = []
    for token in tokens:
        if token == "gpu":
            continue
        if result and result[-1] == "max" and token == "q":
            result[-1] = "maxq"
        else:
            result.append(token)
    return result


def _contains_contiguous(haystack: list[str], needle: list[str]) -> bool:
    """True if *needle* appears in *haystack* as one unbroken run."""
    if not needle or len(needle) > len(haystack):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start:start + len(needle)] == needle:
            return True
    return False


def _laptop_adjusted(score: float, mobile: bool) -> float:
    """A laptop card is its desktop equivalent at a lower clock.

    The penalty is applied, not a re-tiering: the card keeps its place
    in the ordering and is discounted by ``MOBILE_PENALTY``, rounded to
    one decimal so the table's precision is kept.
    """
    return round(score * MOBILE_PENALTY, 1) if mobile else score


def _nearest_anchor(
    score: float, table: tuple[tuple[str, float], ...]
) -> tuple[str, float]:
    """The table entry whose score is closest to *score*."""
    best_pattern = ""
    best_score = 0.0
    best_distance = float("inf")
    for pattern, table_score in table:
        distance = abs(table_score - score)
        if distance < best_distance:
            best_pattern, best_score, best_distance = pattern, table_score, distance
    return best_pattern, best_score


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def effective_table(
    key: str, default: tuple[tuple[str, float], ...], config: Any
) -> tuple[tuple[str, float], ...]:
    """The table in force: the config's override, else the built-in.

    A ranking is context for a judgment, never worth a crash, so a
    config without a ``get`` or with an unusable override silently gets
    the built-in table. ``config.load`` has already rejected a malformed
    override loudly at startup; this is only the belt to that
    suspenders.
    """
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if not callable(getter):
        return default
    try:
        raw = getter(key, default)
        table = _coerce_table(raw)
    except Exception:  # noqa: BLE001 - a bad override means "use the default"
        return default
    return table if table else default


def _coerce_table(raw: Any) -> tuple[tuple[str, float], ...]:
    """A config override (lists of ``[pattern, score]``) as table form."""
    table = []
    for entry in raw or ():
        pattern, score = entry
        table.append((str(pattern), float(score)))
    return tuple(table)


def validate_table(raw: Any) -> None:
    """Check a config-file override, loudly.

    Raises ``ValueError`` (which ``config.load`` turns into a
    ``ConfigError``) unless *raw* is a non-empty list of
    ``[pattern, score]`` pairs with non-empty string patterns and scores
    on the 0-100 scale.
    """
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            "a rankings override must be a non-empty list of "
            "[pattern, score] pairs"
        )
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"each rankings entry must be [pattern, score], got {entry!r}"
            )
        pattern, score = entry
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(
                f"rankings patterns must be non-empty strings, got {pattern!r}"
            )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(
                f"rankings scores must be numbers, got {score!r} for {pattern!r}"
            )
        if not 0.0 <= float(score) <= 100.0:
            raise ValueError(
                f"rankings scores must sit between 0 and 100, "
                f"got {score!r} for {pattern!r}"
            )


def rankings_version(config: Any = None) -> str:
    """The version of the ordering in force.

    An override may declare its own version -- the string dates the
    tables, so a user shipping a corrected table should date it too.
    """
    if config is None:
        return RANKINGS_VERSION
    getter = getattr(config, "get", None)
    if not callable(getter):
        return RANKINGS_VERSION
    version = getter("rankings_version", RANKINGS_VERSION)
    return str(version) if version else RANKINGS_VERSION
