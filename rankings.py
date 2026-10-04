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

Provenance: the ordering is calibrated against published
hierarchies -- Tom's Hardware's GPU benchmarks hierarchy
(2026, which ranks cards as a percentage of the RTX 4090)
and CPU benchmarks hierarchy (2026, gaming and
single-threaded), UL Benchmarks' ranked GPU/CPU lists, and
PassMark single-thread ratios for the older generations the
modern hierarchies do not cover. The scores are RELATIVE
ESTIMATES on a 0-100 scale, not benchmarks -- they say
"roughly how capable", never "how many frames". A part that
matches no entry is ``None`` (unknown), and no score is
ever invented.

Anchors (so the scale can be reasoned about): the current
flagship discrete GPU is 100 (RTX 5090), an RTX 4060 is
~50, a GTX 1060 is ~25, a GTX 970 is ~18, a GeForce 610M
is ~6. On the CPU side: the current flagship is ~100 (the
Ryzen 7 9850X3D/9950X3D class), a Ryzen 5 3600 is ~57, an
i5-2450M is ~21.

Modern CPUs are scored as a blend of gaming, single-thread
and multithreaded capability; older ones by single-thread
ratio with a multithread adjustment, so a 16-core chip is
not rated like a 4-core at the same clock. Apple silicon
entries (both tables) are approximate, derived from
single/multi-thread and Metal ratios rather than a direct
hierarchy.

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
# the normalization below cannot tell them apart. The family word
# folds away too ("Core i7-13700KF", "i7-13700KF" and "13700KF"
# name one chip) -- symmetrically on patterns and on names, so
# dropping it cannot make one part match its neighbour.
_STRIPPED_TOKENS = frozenset({
    "nvidia", "geforce", "radeon", "intel", "amd", "r", "tm",
    "core", "ryzen", "pentium", "celeron", "atom", "ultra",
})

# Tokens that mark a laptop variant of a desktop card. "M" is included
# because old NVIDIA mobile parts are named with a bare M suffix
# ("GeForce 610M" keeps its M glued to the number, so only a separate
# "m" token counts). "Max-Q" folds to "maxq" during normalization.
_MOBILE_TOKENS = frozenset({"mobile", "laptop", "notebook", "maxq", "m"})

# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------
# Each entry is (name pattern, capability score 0-100). Matching is
# normalization-based, so a pattern only has to normalize to the same
# tokens as the name being looked up. Ordering is not irrelevant: it
# breaks ties -- _nearest_anchor anchors a score to the first entry
# at that score, and equal-length contiguous matches keep the earlier
# pattern.

#: Desktop discrete GPUs, 2013-2026, plus the integrated
#: adapters common enough to anchor the low end. Scores follow
#: Tom's Hardware's 2026 GPU hierarchy (a card's measured
#: percentage of an RTX 4090), rescaled so the RTX 5090 is
#: 100; generations the hierarchy does not cover (RTX 50
#: series, RX 9000 series, Arc Battlemage, Apple silicon)
#: are placed from UL Benchmarks' rankings and review-class
#: comparisons. Mobile coverage is partial on purpose: a
#: laptop variant of a desktop card is the desktop entry at
#: the documented penalty, and only the laptop-only parts
#: with no desktop equivalent (the "GeForce 610M" class)
#: have entries of their own.
GPU_RANKINGS: tuple[tuple[str, float], ...] = (
    # NVIDIA GeForce RTX 50 series (2025-2026)
    ("RTX 5090", 100.0),
    ("RTX 5080", 94.0),
    ("RTX 5070 Ti", 88.0),
    ("RTX 5070", 83.0),
    ("RTX 5060 Ti", 68.0),
    ("RTX 5060", 52.0),
    ("RTX 5050", 46.0),
    # NVIDIA GeForce RTX 40 series (2022-2023)
    ("RTX 4090", 93.0),
    ("RTX 4080 Super", 90.0),
    ("RTX 4080", 89.0),
    ("RTX 4070 Ti Super", 86.0),
    ("RTX 4070 Super", 83.0),
    ("RTX 4070 Ti", 81.0),
    ("RTX 4070", 74.0),
    ("RTX 4060 Ti", 61.0),
    ("RTX 4060", 50.0),
    ("RTX 4050", 34.0),
    # NVIDIA GeForce RTX 30 series (2020-2022)
    ("RTX 3090 Ti", 81.0),
    ("RTX 3090", 79.0),
    ("RTX 3080 Ti", 76.0),
    ("RTX 3080 12GB", 75.0),
    ("RTX 3080", 71.0),
    ("RTX 3070 Ti", 63.0),
    ("RTX 3070", 58.0),
    ("RTX 3060 Ti", 52.0),
    ("RTX 3060", 43.0),
    ("RTX 3050 Ti", 40.0),
    ("RTX 3050", 38.0),
    # NVIDIA GeForce RTX 20 series (2018-2019)
    ("RTX 2080 Ti", 60.0),
    ("RTX 2080 Super", 57.0),
    ("RTX 2080", 56.0),
    ("RTX 2070 Super", 51.0),
    ("RTX 2070", 49.0),
    ("RTX 2060 Super", 46.0),
    ("RTX 2060", 42.0),
    ("RTX 2050", 30.0),
    # NVIDIA GeForce GTX 16 series (2019)
    ("GTX 1660 Super", 37.0),
    ("GTX 1660 Ti", 36.0),
    ("GTX 1660", 33.0),
    ("GTX 1650 Super", 24.0),
    ("GTX 1650", 23.0),
    ("GTX 1630", 14.0),
    # NVIDIA GeForce GTX 10 series (2016-2017)
    ("GTX 1080 Ti", 39.0),
    ("GTX 1080", 35.0),
    ("GTX 1070 Ti", 33.0),
    ("GTX 1070", 31.0),
    ("GTX 1060", 25.0),
    ("GTX 1050 Ti", 18.0),
    ("GTX 1050", 15.0),
    ("GT 1030", 12.0),
    # NVIDIA GeForce GTX 900 series (2014-2015)
    ("GTX 980 Ti", 22.0),
    ("GTX 980", 20.0),
    ("GTX 970", 18.0),
    ("GTX 960", 15.0),
    ("GTX 950", 13.0),
    # NVIDIA GeForce GTX 700 series (2013-2014)
    ("GTX 780 Ti", 16.0),
    ("GTX 780", 14.0),
    ("GTX 770", 13.0),
    ("GTX 760", 12.0),
    ("GTX 750 Ti", 8.0),
    ("GTX 750", 7.0),
    ("GTX 745", 6.0),
    # NVIDIA mobile-only parts (no desktop equivalent)
    ("GeForce 610M", 6.0),
    ("GT 750M", 10.0),
    ("GT 740M", 9.0),
    ("GT 730M", 7.0),
    ("GT 720M", 6.0),
    ("GT 635M", 7.0),
    ("GT 630M", 6.0),
    ("GT 620M", 5.0),
    ("GT 610M", 5.0),
    ("GTX 960M", 12.0),
    ("GTX 950M", 9.0),
    ("GTX 940M", 7.0),
    ("GTX 860M", 11.0),
    ("GTX 850M", 8.0),
    ("GTX 770M", 11.0),
    ("GTX 760M", 10.0),
    ("GTX 765M", 9.0),
    ("MX570", 18.0),
    ("MX550", 16.0),
    ("MX450", 14.0),
    ("MX350", 11.0),
    ("MX250", 8.0),
    ("MX150", 8.0),
    ("MX130", 6.0),
    ("MX110", 5.0),
    # AMD Radeon RX 9000 series (2025)
    ("RX 9070 XT", 89.0),
    ("RX 9070", 85.0),
    ("RX 9070 GRE", 78.0),
    ("RX 9060 XT", 68.0),
    ("RX 9060", 59.0),
    # AMD Radeon RX 7000 series (2022-2023)
    ("RX 7900 XTX", 90.0),
    ("RX 7900 XT", 87.0),
    ("RX 7900 GRE", 82.0),
    ("RX 7800 XT", 78.0),
    ("RX 7700 XT", 70.0),
    ("RX 7600 XT", 48.0),
    ("RX 7600", 47.0),
    # AMD Radeon RX 6000 series (2020-2021)
    ("RX 6950 XT", 79.0),
    ("RX 6900 XT", 75.0),
    ("RX 6800 XT", 74.0),
    ("RX 6800", 69.0),
    ("RX 6750 XT", 62.0),
    ("RX 6700 XT", 60.0),
    ("RX 6650 XT", 55.0),
    ("RX 6600 XT", 52.0),
    ("RX 6600", 39.0),
    ("RX 6500 XT", 34.0),
    ("RX 6400", 26.0),
    # AMD Radeon RX 5000 series (2019)
    ("RX 5700 XT", 44.0),
    ("RX 5700", 40.0),
    ("RX 5600 XT", 36.0),
    ("RX 5500 XT", 28.0),
    # AMD Radeon RX 500/400 series (2016-2017)
    ("RX 590", 29.0),
    ("RX 580", 27.0),
    ("RX 570", 25.0),
    ("RX 560", 19.0),
    ("RX 550", 14.0),
    ("RX 480", 26.0),
    ("RX 470", 25.0),
    ("RX 460", 16.0),
    # AMD Radeon Vega (2017) and R9 (2013-2014)
    ("RX Vega 64", 40.0),
    ("RX Vega 56", 36.0),
    ("R9 390X", 27.0),
    ("R9 390", 26.0),
    ("R9 380", 21.0),
    ("R9 290X", 24.0),
    ("R9 290", 21.0),
    ("R9 280", 19.0),
    # Intel Arc Alchemist (2022-2024) and Battlemage (2025)
    ("Arc B580", 52.0),
    ("Arc B570", 48.0),
    ("Arc B390", 40.0),
    ("Arc A770", 45.0),
    ("Arc A750", 42.0),
    ("Arc A580", 39.0),
    ("Arc A530", 30.0),
    ("Arc A380", 20.0),
    ("Arc A310", 13.0),
    # Apple silicon GPU tiers (reported as the chip name on
    # macOS). Approximate: Metal-benchmark ratios against the
    # discrete cards above, not a measured hierarchy.
    ("Apple M4 Ultra", 85.0),
    ("Apple M4 Max", 72.0),
    ("Apple M4 Pro", 50.0),
    ("Apple M4", 38.0),
    ("Apple M3 Max", 56.0),
    ("Apple M3 Pro", 45.0),
    ("Apple M3", 32.0),
    ("Apple M2 Ultra", 76.0),
    ("Apple M2 Max", 43.0),
    ("Apple M2 Pro", 42.0),
    ("Apple M2", 26.0),
    ("Apple M1 Ultra", 71.0),
    ("Apple M1 Max", 44.0),
    ("Apple M1 Pro", 30.0),
    ("Apple M1", 18.0),
    # Common Intel integrated graphics, the low anchors
    ("Iris Xe Graphics", 16.0),
    ("Iris Plus 655", 13.0),
    ("Iris Plus 640", 12.0),
    ("Iris Plus 630", 11.0),
    ("Iris Pro 5200", 11.0),
    ("UHD Graphics 770", 12.0),
    ("UHD Graphics 750", 11.0),
    ("UHD Graphics 740", 10.0),
    ("UHD Graphics 730", 10.0),
    ("UHD Graphics 710", 9.0),
    ("UHD Graphics 630", 9.0),
    ("UHD Graphics 620", 8.0),
    ("UHD Graphics 610", 6.0),
    ("UHD Graphics 600", 5.0),
    ("UHD Graphics 520", 7.0),
    ("UHD Graphics 510", 5.0),
    ("HD Graphics 630", 9.0),
    ("HD Graphics 620", 7.0),
    ("HD Graphics 610", 5.0),
    ("HD Graphics 530", 7.0),
    ("HD Graphics 520", 6.0),
    ("HD Graphics 510", 5.0),
    ("HD Graphics 4600", 4.0),
    ("HD Graphics 4400", 3.0),
    ("HD Graphics 3000", 3.0),
    ("HD Graphics 2500", 2.0),
)

#: Desktop and mobile CPUs, 2011-2026, plus the low anchors.
#: Modern chips (2022+) are a blend of Tom's Hardware's 2026
#: gaming and single-thread hierarchies with a multithread
#: term; older ones are PassMark single-thread ratios with a
#: multithread adjustment, so core count still counts. A few
#: mobile-only parts (the i5-2450M class) are entries in
#: their own right because no desktop equivalent exists.
CPU_RANKINGS: tuple[tuple[str, float], ...] = (
    # AMD Ryzen 9000 series, Zen 5 (2024-2025) -- the top
    ("Ryzen 9 9950X3D", 100.0),
    ("Ryzen 7 9850X3D", 99.0),
    ("Ryzen 9 9900X3D", 97.0),
    ("Ryzen 9 9950X", 94.0),
    ("Ryzen 7 9800X3D", 98.0),
    ("Ryzen 9 9900X", 93.0),
    ("Ryzen 7 9700X", 86.0),
    ("Ryzen 5 9600X", 83.0),
    # Intel Core Ultra 200 series, Arrow Lake (2024)
    ("Core Ultra 7 270K Plus", 97.0),
    ("Core Ultra 5 250K Plus", 94.0),
    ("Core Ultra 7 270K", 94.0),    ("Core Ultra 9 285K", 94.0),    ("Core Ultra 7 265K", 89.0),
    ("Core Ultra 5 250K", 91.0),
    ("Core Ultra 5 225", 84.0),
    ("Core Ultra 5 245K", 86.0),
    ("Core Ultra 5 235K", 85.0),
    ("Core Ultra 7 255K", 92.0),
    ("Core Ultra 9 275K", 95.0),
    ("Core Ultra 5 225K", 84.0),
    # Intel Core 14th generation (2023)
    ("Core i9-14900KS", 95.0),
    ("Core i9-14900K", 90.0),
    ("Core i7-14700K", 88.0),
    ("Core i7-14700", 87.0),
    ("Core i5-14600K", 84.0),
    ("Core i5-14500", 76.0),
    ("Core i5-14400", 73.0),
    ("Core i3-14100", 51.0),
    # Intel Core 13th generation (2022-2023)
    ("Core i9-13900KS", 93.0),
    ("Core i9-13900K", 89.0),
    ("Core i9-13900", 89.0),
    ("Core i7-13700K", 85.0),
    ("Core i7-13700", 83.0),
    ("Core i5-13600K", 84.0),
    ("Core i5-13500", 79.0),
    ("Core i5-13400", 76.0),
    ("Core i3-13100", 50.0),
    # Intel Core 12th generation (2021-2022)
    ("Core i9-12900K", 90.0),
    ("Core i9-12900", 86.0),
    ("Core i7-12700K", 82.0),
    ("Core i7-12700", 80.0),
    ("Core i5-12600K", 73.0),
    ("Core i5-12500", 74.0),
    ("Core i5-12400", 69.0),
    ("Core i3-12100", 49.0),
    # AMD Ryzen 7000 series, Zen 4 (2022-2023)
    ("Ryzen 9 7950X3D", 95.0),
    ("Ryzen 9 7900X3D", 93.0),
    ("Ryzen 7 7800X3D", 85.0),
    ("Ryzen 9 7950X", 87.0),
    ("Ryzen 9 7900X", 88.0),
    ("Ryzen 9 7900", 85.0),
    ("Ryzen 7 7700X", 86.0),
    ("Ryzen 7 7700", 84.0),
    ("Ryzen 5 7600X", 80.0),
    ("Ryzen 5 7600", 78.0),
    ("Ryzen 5 7500F", 74.0),
    ("Ryzen 5 7500X", 76.0),
    ("Ryzen 5 7600X3D", 84.0),
    # Intel Core 11th generation (2020-2021)
    ("Core i9-11900K", 78.0),
    ("Core i7-11700K", 74.0),
    ("Core i7-11700", 71.0),
    ("Core i5-11600K", 66.0),
    ("Core i5-11400", 64.0),
    # Common Intel laptop chips (Tiger Lake and successors)
    ("Core i7-1185G7", 70.0),
    ("Core i7-11800H", 72.0),
    ("Core i7-1165G7", 68.0),
    ("Core i5-1135G7", 62.0),
    ("Core i5-1130G7", 60.0),
    # Intel Core 10th generation (2019-2020)
    ("Core i9-10980XE", 80.0),
    ("Core i9-10940X", 78.0),
    ("Core i9-10900K", 75.0),
    ("Core i9-10900", 73.0),
    ("Core i7-10700K", 71.0),
    ("Core i7-10700", 69.0),
    ("Core i5-10600K", 69.0),
    ("Core i5-10500", 65.0),
    ("Core i5-10400", 62.0),
    ("Core i3-10100", 52.0),
    ("Ryzen 9 5950X", 87.0),
    ("Ryzen 9 5900X", 85.0),
    ("Ryzen 7 5800X3D", 72.0),
    ("Ryzen 7 5800X", 71.0),
    ("Ryzen 7 5700X", 73.0),
    ("Ryzen 7 5700G", 68.0),
    ("Ryzen 7 5700GT", 69.0),
    ("Ryzen 5 5600X", 65.0),
    ("Ryzen 5 5600G", 62.0),
    ("Ryzen 5 5600GT", 63.0),
    ("Ryzen 5 5600", 63.0),
    ("Ryzen 5 5500GT", 60.0),
    ("Ryzen 5 5500", 62.0),
    ("Ryzen 5 5500GE", 58.0),
    # Intel Core 9th generation (2018-2019)
    ("Core i9-9900K", 70.0),
    ("Core i9-9900", 67.0),
    ("Core i7-9700K", 67.0),
    ("Core i7-9700", 65.0),
    ("Core i5-9600K", 64.0),
    ("Core i5-9400", 62.0),
    ("Core i3-9100", 55.0),
    ("Core i7-8700K", 65.0),
    ("Core i7-8700", 64.0),
    ("Core i5-8600K", 62.0),
    ("Core i5-8500", 60.0),
    ("Core i5-8400", 58.0),
    ("Core i3-8300", 56.0),
    ("Core i3-8100", 54.0),
    # AMD Ryzen 3000 series, Zen 2 (2019-2020)
    ("Ryzen 9 3950X", 81.0),
    ("Ryzen 9 3900X", 79.0),
    ("Ryzen 7 3800X", 75.0),
    ("Ryzen 7 3700X", 71.0),
    ("Ryzen 5 3600X", 58.0),
    ("Ryzen 5 3600", 57.0),
    ("Ryzen 3 3300X", 49.0),
    ("Ryzen 3 3200G", 38.0),
    # Intel Core 7th generation (2016-2017)
    ("Core i7-7700K", 56.0),
    ("Core i7-7700", 61.0),
    ("Core i5-7600K", 60.0),
    ("Core i5-7500", 56.0),
    ("Core i5-7400", 54.0),
    ("Core i3-7300", 52.0),
    ("Core i3-7100", 48.0),
    # Intel Core 6th generation (2015-2016)
    ("Core i7-6700K", 55.0),
    ("Core i7-6700", 57.0),
    ("Core i5-6600K", 56.0),
    ("Core i5-6500", 54.0),
    ("Core i5-6400", 52.0),
    ("Core i3-6100", 48.0),
    # Intel Core 4th generation (2013-2014)
    ("Core i7-4790K", 47.0),
    ("Core i7-4790", 46.0),
    ("Core i7-4770", 45.0),
    ("Core i5-4690", 44.0),
    ("Core i5-4590", 42.0),
    ("Core i5-4460", 42.0),
    ("Core i3-4130", 35.0),
    ("Core i3-4160", 35.0),
    # Intel Core 3rd generation (2012)
    ("Core i7-3770K", 43.0),
    ("Core i7-3770", 42.0),
    ("Core i5-3570K", 40.0),
    ("Core i5-3570", 40.0),
    ("Core i5-3470", 38.0),
    ("Core i3-3220", 31.0),
    ("Core i3-3210", 30.0),
    # Intel Core 2nd generation, the Sandy Bridge era (2011)
    ("Core i7-2600K", 37.0),
    ("Core i7-2600", 36.0),
    ("Core i7-2670QM", 23.0),
    ("Core i5-2500K", 35.0),
    ("Core i5-2500", 34.0),
    ("Core i5-2400", 34.0),
    ("Core i5-2450M", 21.0),
    ("Core i3-2120", 27.0),
    ("Core i3-2100", 26.0),
    # AMD Ryzen 2000 series, Zen+ (2018)
    ("Ryzen 7 2700X", 36.0),
    ("Ryzen 5 2600X", 40.0),
    ("Ryzen 5 2600", 38.0),
    ("Ryzen 5 2500X", 37.0),
    ("Ryzen 5 2400G", 33.0),
    ("Ryzen 3 2200G", 30.0),
    # AMD Ryzen 1000 series, Zen 1 (2017)
    ("Ryzen 7 1800X", 33.0),
    ("Ryzen 7 1700X", 32.0),
    ("Ryzen 7 1700", 31.0),
    ("Ryzen 5 1600X", 33.0),
    ("Ryzen 5 1600", 31.0),
    ("Ryzen 5 1500X", 30.0),
    ("Ryzen 5 1400", 29.0),
    ("Ryzen 3 1300X", 26.0),
    ("Ryzen 3 1200", 24.0),
    # Common Intel laptop chips (Alder Lake and newer)
    ("Core Ultra 9 185H", 82.0),
    ("Core Ultra 7 165H", 76.0),
    ("Core Ultra 7 155H", 74.0),
    ("Core Ultra 5 135H", 70.0),
    ("Core Ultra 5 125H", 68.0),
    ("Core Ultra 7 145H", 72.0),
    ("Core i7-1355U", 70.0),
    ("Core i7-1365U", 72.0),
    ("Core i7-13700H", 82.0),
    ("Core i7-14700H", 85.0),
    ("Core i9-13900H", 88.0),
    ("Core i9-14900H", 90.0),
    ("Core i7-1255U", 68.0),
    ("Core i7-1265U", 70.0),
    ("Core i7-12700H", 78.0),
    ("Core i5-1335U", 66.0),
    ("Core i5-1345U", 68.0),
    ("Core i5-13500H", 74.0),
    ("Core i5-1235U", 64.0),
    ("Core i5-1245U", 66.0),
    ("Core i5-12500H", 70.0),
    ("Core i5-1435U", 66.0),
    ("Core i5-1445U", 68.0),
    # Intel HX laptop chips: desktop-class silicon in a
    # laptop body. Estimates, calibrated ~= their K
    # counterparts, a touch lower for the mobile platform.
    ("Core Ultra 9 285HX", 93.0),
    ("Core i9-14900HX", 89.0),
    ("Core i7-14700HX", 87.0),
    ("Core i9-13900HX", 88.0),
    ("Core i7-13700HX", 84.0),
    # Common Intel laptop chips (Comet Lake; estimate:
    # ~= Core i7-10700, slightly lower)
    ("Core i7-10750H", 68.0),
    # Common AMD laptop chips (Zen 3+ and newer)
    # AMD HX laptop chips: desktop-class Zen 4.
    # Estimates, ~= the 7950X class.
    ("Ryzen 9 7945HX", 87.0),
    ("Ryzen 9 7940HX", 86.0),
    ("Ryzen 9 7940HS", 88.0),
    ("Ryzen 9 8945HS", 88.0),
    ("Ryzen 7 8845HS", 82.0),
    ("Ryzen 7 7840HS", 82.0),
    ("Ryzen 7 7840U", 78.0),
    ("Ryzen 7 7730U", 70.0),
    ("Ryzen 7 7735U", 70.0),
    ("Ryzen 5 7640U", 72.0),
    ("Ryzen 5 7530U", 62.0),
    ("Ryzen 5 7535U", 62.0),
    ("Ryzen 7 7700S", 84.0),
    ("Ryzen 5 7600GE", 74.0),
    ("Ryzen 5 8600G", 66.0),
    ("Ryzen 5 8500G", 62.0),
    ("Ryzen 5 8300G", 58.0),
    ("Ryzen 7 8700G", 72.0),
    # Old Intel ULV laptop chips (estimates from the
    # nearest mobile anchors: the i5-2450M at 21, the
    # i3-6100 class at 48)
    ("Core i5-8250U", 45.0),
    ("Core i5-7200U", 40.0),
    ("Core i5-4200U", 23.0),
    # The pre-Zen and pre-Core-i7 floor (estimates:
    # the FX-8350 is Bulldozer, weak single-thread; the
    # Core 2 Duo is a decade and a half old)
    ("AMD FX-8350", 20.0),
    ("Core 2 Duo E8400", 12.0),
    # Intel Celeron / Pentium / Atom, the floor
    ("Core i3-N305", 40.0),
    ("Pentium Gold G7400", 20.0),
    ("Pentium Gold G6400", 25.0),
    # Kaby Lake dual-core with hyper-threading
    # (estimate: ~30)
    ("Pentium G4560", 30.0),
    ("Pentium Silver N6000", 21.0),
    ("Pentium Silver N5100", 19.0),
    ("Celeron G5905", 17.0),
    ("Celeron G5900", 16.0),
    ("Celeron G4900", 16.0),
    ("Celeron N4500", 16.0),
    ("Celeron N4020", 13.0),
    ("Atom x7-Z8750", 11.0),
    ("Atom x5-Z8350", 9.0),
    # Apple M-series (the chip name is what the platform
    # reports). Approximate: single/multi-thread ratios
    # against the Intel chips above, not a measured hierarchy.
    ("Apple M4 Ultra", 94.0),
    ("Apple M4 Max", 78.0),
    ("Apple M4 Pro", 71.0),
    ("Apple M4", 63.0),
    ("Apple M3 Max", 62.0),
    ("Apple M3 Pro", 57.0),
    ("Apple M3", 52.0),
    ("Apple M2 Ultra", 60.0),
    ("Apple M2 Max", 53.0),
    ("Apple M2 Pro", 49.0),
    ("Apple M2", 43.0),
    ("Apple M1 Ultra", 43.0),
    ("Apple M1 Max", 41.0),
    ("Apple M1 Pro", 38.0),
    ("Apple M1", 33.0),
)

#: Capability bands, high to low. The word describes where a part sits
#: relative to everything else in its table, today, not in its own
#: era. The thresholds sit between the tiers of the current tables:
#: the RTX 3060 class is the middle of the pack, the GTX 1060
#: class is entry-level, and the decade-old chips are low-end.
_TIERS: tuple[tuple[float, str], ...] = (
    (90.0, "flagship"),
    (70.0, "high-end"),
    (40.0, "mid-range"),
    (10.0, "entry-level"),
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


def describe(name: str, config: Any = None) -> str | None:
    """A short human label for a GPU or CPU name, or ``None`` if unknown.

    The label names the part itself when the name is a table
    entry (``"GTX 970"`` anchors to the GTX 970, not to the
    first entry at the same score), so it can be checked
    against the tables above instead of trusted on faith. A
    name that only matches by containment ("GTX 970 or
    better") is not a table entry, so it is labelled against
    the nearest anchor at its score. A mobile variant names
    the desktop twin it was matched to, at its discounted
    laptop score. ``config``, when given, is anything with a
    ``get`` method (the app's ``Config``); a ``gpu_rankings``
    or ``cpu_rankings`` override in it replaces the built-in
    table. The GPU table is searched first because GPU names
    are the ones usually asked about.
    """
    for key, default in (("gpu_rankings", GPU_RANKINGS),
                         ("cpu_rankings", CPU_RANKINGS)):
        table = effective_table(key, default, config)
        detail = _lookup_detail(name, table)
        if detail is None:
            continue
        pattern, score, mobile = detail
        if _exact_entry(name, table) is None:
            anchor, _anchor_score = _nearest_anchor(score, table)
        else:
            anchor = pattern
        article = "an" if anchor[:1].lower() in "aeiou" else "a"
        if mobile:
            return (
                f"{tier_word(score)}, ~{score:g}/100 "
                f"({article} {anchor} at laptop clocks)"
            )
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
    """Find *name* in *table*, or ``None`` when nothing matches."""
    detail = _lookup_detail(name, table)
    return None if detail is None else (detail[0], detail[1])


def _lookup_detail(
    name: str, table: tuple[tuple[str, float], ...]
) -> tuple[str, float, bool] | None:
    """The matched entry as ``(pattern, score, was_mobile)``.

    Four passes, each more permissive than the last, and each
    reached only when the previous one failed:

    1. exact: the name normalizes to the pattern's tokens --
       the only match that can never confuse "RTX 3060"
       with "RTX 3060 Ti";
    2. contiguous: the longest pattern whose tokens appear as
       one unbroken run in the name, which absorbs generation
       prefixes ("11th Gen Intel Core i5-1135G7") and
       requirement prose ("GTX 970 or better") without
       crossing a suffix boundary, because a longer pattern
       (the Ti variant of the same card) always wins;
    3. glued mobile: a token like "4080m" is a model number
       with the mobile marker glued to it, so it is split
       into base + "m" and retried as mobile -- "RTX 4080M"
       is an RTX 4080 at laptop clocks. Only reached when
       the normal passes failed, so laptop-only parts with
       entries of their own ("GeForce 610M", "GT 740M")
       keep their exact, unpenalized first-pass match;
    4. bare model number: a token such as "13700K" that
       names exactly one pattern is that pattern. A token
       several patterns share ("4060" is both the RTX 4060
       and the RTX 4060 Ti) is ambiguous, and ambiguity
       stays unknown rather than picking a favourite.
    """
    prepared = _prepared_tokens(name)
    if prepared is None:
        return None
    mobile, tokens = prepared
    entry = _match_tokens(tokens, table)
    if entry is not None:
        return entry[0], _laptop_adjusted(entry[1], mobile), mobile

    split = _split_glued_mobile(tokens)
    if split is not None:
        base = [token for token in split if token != "m"]
        if base:
            entry = _match_tokens(base, table)
            if entry is not None:
                return entry[0], _laptop_adjusted(entry[1], True), True

    return _model_token_entry(tokens, table)


def _exact_entry(
    name: str, table: tuple[tuple[str, float], ...]
) -> tuple[str, float] | None:
    """The entry when *name* is itself a table pattern.

    An exact normalization match, or the glued-mobile
    split -- but not the contiguous fallback, which is for
    prose around a part ("GTX 970 or better"), and not
    the bare-model-number fallback, which is for names
    too short to carry the family word.
    """
    prepared = _prepared_tokens(name)
    if prepared is None:
        return None
    mobile, tokens = prepared
    entry = _exact_tokens(tokens, table)
    if entry is not None:
        return entry[0], _laptop_adjusted(entry[1], mobile)
    split = _split_glued_mobile(tokens)
    if split is None:
        return None
    base = [token for token in split if token != "m"]
    if not base:
        return None
    entry = _exact_tokens(base, table)
    if entry is None:
        return None
    return entry[0], _laptop_adjusted(entry[1], True)


def _prepared_tokens(name: str) -> tuple[bool, list[str]] | None:
    """``(is_mobile, tokens)`` for a lookup, or ``None`` if unusable.

    A separate mobile marker ("Laptop", "Max-Q", a bare
    "M") is dropped from the tokens and remembered, so the
    lookup compares the part against its desktop twin and
    the caller can apply the penalty.
    """
    tokens = _tokens(name)
    if not tokens:
        return None
    mobile = any(token in _MOBILE_TOKENS for token in tokens)
    if mobile:
        tokens = [token for token in tokens if token not in _MOBILE_TOKENS]
        if not tokens:
            return None
    return mobile, tokens


def _exact_tokens(
    tokens: list[str], table: tuple[tuple[str, float], ...]
) -> tuple[str, float] | None:
    """The entry whose tokens are exactly *tokens*."""
    for pattern, score in table:
        if _tokens(pattern) == tokens:
            return pattern, score
    return None


def _match_tokens(
    tokens: list[str], table: tuple[tuple[str, float], ...]
) -> tuple[str, float] | None:
    """The best entry for already-normalized *tokens*.

    Exact first, then the longest pattern whose tokens appear
    contiguously in *tokens*; equal-length contenders keep
    the earlier pattern, so table order breaks the tie.
    """
    entry = _exact_tokens(tokens, table)
    if entry is not None:
        return entry
    best_length = 0
    best: tuple[str, float] | None = None
    for pattern, score in table:
        pattern_tokens = _tokens(pattern)
        if len(pattern_tokens) <= best_length:
            continue
        if _contains_contiguous(tokens, pattern_tokens):
            best_length = len(pattern_tokens)
            best = (pattern, score)
    return best


def _split_glued_mobile(tokens: list[str]) -> list[str] | None:
    """Split glued mobile markers ("4080m" -> "4080" + "m").

    A token that ends in "m" with a digit before it is a
    model number plus the mobile marker. Returns ``None``
    when no token carries a glued marker, so the caller
    knows this pass does not apply.
    """
    split: list[str] = []
    glued = False
    for token in tokens:
        if (len(token) > 1 and token.endswith("m")
                and token[-2].isdigit()):
            split.extend((token[:-1], "m"))
            glued = True
        else:
            split.append(token)
    return split if glued else None


def _model_token_entry(
    tokens: list[str], table: tuple[tuple[str, float], ...]
) -> tuple[str, float] | None:
    """The entry a bare model number ("13700K") names, if it is one.

    Patterns are indexed by their model tokens (the tokens
    carrying 3+ digits). A name token that maps to exactly
    one pattern is that pattern; one that several patterns
    share is ambiguous, and ambiguity stays unknown.
    """
    by_token: dict[str, set[tuple[str, ...]]] = {}
    for pattern, _score in table:
        pattern_tokens = _tokens(pattern)
        for token in pattern_tokens:
            if _is_model_token(token):
                by_token.setdefault(token, set()).add(
                    tuple(pattern_tokens)
                )
    matched: set[tuple[str, ...]] = set()
    for token in tokens:
        if _is_model_token(token):
            matched.update(by_token.get(token, ()))
    if len(matched) != 1:
        return None
    pattern_tokens = matched.pop()
    for pattern, score in table:
        if tuple(_tokens(pattern)) == pattern_tokens:
            return pattern, score
    return None  # unreachable: the set came from this table


def _is_model_token(token: str) -> bool:
    """True for a token carrying a model number (3+ digits)."""
    return sum(1 for char in token if char.isdigit()) >= 3


def _tokens(name: str) -> list[str]:
    """The comparison form of a hardware name.

    Lowercase, vendor/brand markers removed, punctuation folded to
    spaces, "Max-Q" folded to "maxq", the redundant "gpu" token
    dropped (everything in the GPU table is a GPU), and a
    trailing "F"/"D" on a model number folded away (the same
    chip).
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
        elif (token.endswith(("f", "d")) and len(token) > 1
              and token[0].isdigit()):
            # An "F" SKU is the same chip without integrated
            # graphics ("Core i5-12400F" vs "Core i5-12400",
            # "Core i7-13700KF" vs "Core i7-13700K"), and a
            # "D" variant is the same silicon in another market
            # ("RTX 4090D" vs "RTX 4090"): the same
            # capability, so the suffix folds away.
            result.append(token[:-1])
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
    """The table entry whose score is closest to *score*.

    Ties (two entries at the same score) go to the earlier
    entry in the table, so an anchor that sits at the head of
    its class ("GTX 1060" above the "GTX 1650 Super") wins
    over one listed later.
    """
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
