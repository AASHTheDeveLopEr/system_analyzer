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

#: Desktop discrete GPUs, 2011-2026, plus the integrated
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
    ("RTX 5090", 100.0),
    ("RTX 5080", 94.0),
    ("RTX 5070 Ti", 88.0),
    ("RTX 5070", 83.0),
    ("RTX 5060 Ti", 68.0),
    ("RTX 5060", 52.0),
    ("RTX 5050", 46.0),
    ("RTX 4090", 93.0),
    ("RTX 4090D", 91.0),
    ("RTX 4080 Super", 90.0),
    ("RTX 4080", 89.0),
    ("RTX 4070 Ti Super", 86.0),
    ("RTX 4070 Super", 83.0),
    ("RTX 4070 Ti", 81.0),
    ("RTX 4070", 74.0),
    ("RTX 4060 Ti", 61.0),
    ("RTX 4060", 50.0),
    ("RTX 4050", 34.0),
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
    ("RTX 2080 Ti", 60.0),
    ("RTX 2080 Super", 57.0),
    ("RTX 2080", 56.0),
    ("RTX 2070 Super", 51.0),
    ("RTX 2070", 49.0),
    ("RTX 2060 Super", 46.0),
    ("RTX 2060", 42.0),
    ("RTX 2050", 30.0),
    ("GTX 1660 Super", 37.0),
    ("GTX 1660 Ti", 36.0),
    ("GTX 1660", 33.0),
    ("GTX 1650 Super", 24.0),
    ("GTX 1650", 23.0),
    ("GTX 1630", 14.0),
    ("GTX 1080 Ti", 39.0),
    ("GTX 1080", 35.0),
    ("GTX 1070 Ti", 33.0),
    ("GTX 1070", 31.0),
    ("GTX 1060", 25.0),
    ("GTX 1050 Ti", 18.0),
    ("GTX 1050", 15.0),
    ("GT 1030", 12.0),
    ("GTX 980 Ti", 22.0),
    ("GTX 980", 20.0),
    ("GTX 970", 18.0),
    ("GTX 960", 15.0),
    ("GTX 950", 13.0),
    ("GTX 780 Ti", 16.0),
    ("GTX 780", 14.0),
    ("GTX 770", 13.0),
    ("GTX 760", 12.0),
    ("GTX 750 Ti", 8.0),
    ("GTX 750", 7.0),
    ("GTX 745", 6.0),
    ("GTX 690", 15.0),
    ("GTX 680", 14.0),
    ("GTX 670", 13.0),
    ("GTX 660 Ti", 12.0),
    ("GTX 660", 11.0),
    ("GTX 650 Ti", 9.0),
    ("GTX 650", 8.0),
    ("GT 645", 6.0),
    ("GT 640", 5.0),
    ("GT 630", 4.0),
    ("GT 620", 3.0),
    ("GT 610", 2.0),
    ("GTX 590", 14.0),
    ("GTX 580", 13.0),
    ("GTX 570", 12.0),
    ("GTX 560 Ti", 11.0),
    ("GTX 560", 10.0),
    ("GTX 550 Ti", 8.0),
    ("GT 545", 6.0),
    ("GT 530", 4.0),
    ("GT 520", 3.0),
    ("GT 510", 2.0),
    ("GTX 480", 12.0),
    ("GTX 470", 11.0),
    ("GTX 465", 9.0),
    ("GTX 460", 8.0),
    ("GTS 450", 6.0),
    ("GT 440", 4.0),
    ("GT 430", 3.0),
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
    ("RX 9070 XT", 89.0),
    ("RX 9070", 85.0),
    ("RX 9070 GRE", 78.0),
    ("RX 9060 XT", 68.0),
    ("RX 9060", 59.0),
    ("RX 7900 XTX", 90.0),
    ("RX 7900 XT", 87.0),
    ("RX 7900 GRE", 82.0),
    ("RX 7800 XT", 78.0),
    ("RX 7700 XT", 70.0),
    ("RX 7600 XT", 48.0),
    ("RX 7600", 47.0),
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
    ("RX 5700 XT", 44.0),
    ("RX 5700", 40.0),
    ("RX 5600 XT", 36.0),
    ("RX 5500 XT", 28.0),
    ("RX 590", 29.0),
    ("RX 580", 27.0),
    ("RX 570", 25.0),
    ("RX 560", 19.0),
    ("RX 550", 14.0),
    ("RX 480", 26.0),
    ("RX 470", 25.0),
    ("RX 460", 16.0),
    ("RX Vega 64", 40.0),
    ("RX Vega 56", 36.0),
    ("R9 390X", 27.0),
    ("R9 390", 26.0),
    ("R9 380X", 23.0),
    ("R9 380", 21.0),
    ("R9 290X", 24.0),
    ("R9 290", 21.0),
    ("R9 280X", 20.0),
    ("R9 280", 19.0),
    ("R9 270X", 18.0),
    ("R9 270", 16.0),
    ("R9 260X", 14.0),
    ("R9 260", 12.0),
    ("HD 7970", 18.0),
    ("HD 7950", 16.0),
    ("HD 7870", 14.0),
    ("HD 7850", 12.0),
    ("HD 7790", 11.0),
    ("HD 7770", 10.0),
    ("HD 7750", 8.0),
    ("HD 7730", 6.0),
    ("HD 7670", 5.0),
    ("HD 7570", 4.0),
    ("HD 7470", 3.0),
    ("HD 7450", 2.0),
    ("HD 7350", 1.0),
    ("HD 6970", 14.0),
    ("HD 6950", 13.0),
    ("HD 6870", 12.0),
    ("HD 6850", 11.0),
    ("HD 6790", 10.0),
    ("HD 6770", 9.0),
    ("HD 6750", 8.0),
    ("HD 6670", 7.0),
    ("HD 6570", 5.0),
    ("HD 6450", 3.0),
    ("Arc B580", 52.0),
    ("Arc B570", 48.0),
    ("Arc B390", 40.0),
    ("Arc A770", 45.0),
    ("Arc A750", 42.0),
    ("Arc A580", 39.0),
    ("Arc A530", 30.0),
    ("Arc A380", 20.0),
    ("Arc A310", 13.0),
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
    ("Iris Xe Graphics", 16.0),
    ("Iris Xe Graphics G7", 18.0),
    ("Iris Xe Graphics G4", 14.0),
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
    ("UHD Graphics 500", 4.0),
    ("UHD Graphics 505", 4.0),
    ("UHD Graphics 515", 5.0),
    ("UHD Graphics 530", 7.0),
    ("UHD Graphics 615", 6.0),
    ("UHD Graphics 617", 7.0),
    ("UHD Graphics 605", 5.0),
    ("Iris Plus 655", 13.0),
    ("Iris Plus 650", 13.0),
    ("Iris Plus 645", 12.0),
    ("Iris Plus 640", 12.0),
    ("Iris Plus 630", 11.0),
    ("Iris Plus 620", 10.0),
    ("Iris Plus 610", 9.0),
    ("Iris Pro 5200", 11.0),
    ("Iris Pro 580", 14.0),
    ("Iris Pro 6200", 15.0),
    ("Iris Pro P555", 12.0),
    ("Iris Pro P580", 15.0),
    ("HD Graphics 630", 9.0),
    ("HD Graphics 620", 7.0),
    ("HD Graphics 610", 5.0),
    ("HD Graphics 530", 7.0),
    ("HD Graphics 520", 6.0),
    ("HD Graphics 510", 5.0),
    ("HD Graphics 500", 4.0),
    ("HD Graphics 505", 4.0),
    ("HD Graphics 515", 5.0),
    ("HD Graphics 5300", 6.0),
    ("HD Graphics 5500", 7.0),
    ("HD Graphics 5600", 8.0),
    ("HD Graphics 6000", 9.0),
    ("HD Graphics 4600", 4.0),
    ("HD Graphics 4400", 3.0),
    ("HD Graphics 4200", 3.0),
    ("HD Graphics 4000", 3.0),
    ("HD Graphics 3000", 3.0),
    ("HD Graphics 2500", 2.0),
    ("HD Graphics 2000", 2.0),
    ("HD Graphics 1000", 1.0),
    ("Radeon 780M", 22.0),
    ("Radeon 760M", 18.0),
    ("Radeon 740M", 14.0),
    ("Radeon 680M", 20.0),
    ("Radeon 660M", 16.0),
    ("Radeon 610M", 10.0),
    ("Radeon Vega 11", 12.0),
    ("Radeon Vega 10", 11.0),
    ("Radeon Vega 9", 10.0),
    ("Radeon Vega 8", 10.0),
    ("Radeon Vega 7", 9.0),
    ("Radeon Vega 6", 8.0),
    ("Radeon Vega 3", 5.0),
    ("Radeon R7 Graphics", 8.0),
    ("Radeon R6 Graphics", 6.0),
    ("Radeon R5 Graphics", 5.0),
    ("Radeon R4 Graphics", 4.0),
    ("Radeon R3 Graphics", 3.0),
    ("Radeon HD 8670D", 6.0),
    ("Radeon HD 8570D", 5.0),
    ("Radeon HD 8470D", 4.0),
    ("Radeon HD 8370D", 3.0),
    ("Radeon HD 7660D", 5.0),
    ("Radeon HD 7560D", 4.0),
    ("Radeon HD 7480D", 3.0),
    ("Radeon HD 6550D", 4.0),
    ("Radeon HD 6530D", 3.0),
    ("Radeon HD 6410D", 2.0),
)

#: Desktop and mobile CPUs, 2011-2026, plus the low anchors.
#: Modern chips (2022+) are a blend of Tom's Hardware's 2026
#: gaming and single-thread hierarchies with a multithread
#: term; older ones are PassMark single-thread ratios with a
#: multithread adjustment, so core count still counts. A few
#: mobile-only parts (the i5-2450M class) are entries in
#: their own right because no desktop equivalent exists.
CPU_RANKINGS: tuple[tuple[str, float], ...] = (
    ("Ryzen 9 9950X3D", 100.0),
    ("Ryzen 7 9850X3D", 99.0),
    ("Ryzen 9 9900X3D", 97.0),
    ("Ryzen 9 9950X", 94.0),
    ("Ryzen 7 9800X3D", 98.0),
    ("Ryzen 9 9900X", 93.0),
    ("Ryzen 7 9700X", 86.0),
    ("Ryzen 5 9600X", 83.0),
    ("Ryzen 5 9600", 81.0),
    ("Ryzen 5 9500", 78.0),
    ("Ryzen 5 9600X3D", 85.0),
    ("Ryzen 7 9700F", 84.0),
    ("Ryzen 5 9600F", 80.0),
    ("Ryzen 9 9950X3D2", 100.0),
    ("Core Ultra 9 285K", 94.0),
    ("Core Ultra 7 270K Plus", 97.0),
    ("Core Ultra 7 270K", 94.0),
    ("Core Ultra 7 265K", 89.0),
    ("Core Ultra 7 255K", 92.0),
    ("Core Ultra 5 250K Plus", 94.0),
    ("Core Ultra 5 250K", 91.0),
    ("Core Ultra 5 245K", 86.0),
    ("Core Ultra 5 235K", 85.0),
    ("Core Ultra 5 225K", 84.0),
    ("Core Ultra 5 225", 84.0),
    ("Core Ultra 9 275K", 95.0),
    ("Core Ultra 9 290HX Plus", 96.0),
    ("Core i9-14900KS", 95.0),
    ("Core i9-14900K", 90.0),
    ("Core i9-14900", 88.0),
    ("Core i9-14900F", 88.0),
    ("Core i7-14700K", 88.0),
    ("Core i7-14700", 87.0),
    ("Core i7-14700F", 87.0),
    ("Core i5-14600K", 84.0),
    ("Core i5-14600", 82.0),
    ("Core i5-14500", 76.0),
    ("Core i5-14400", 73.0),
    ("Core i5-14400F", 73.0),
    ("Core i3-14100", 51.0),
    ("Core i3-14100F", 51.0),
    ("Core i3-14300", 55.0),
    ("Core i9-13900KS", 93.0),
    ("Core i9-13900K", 89.0),
    ("Core i9-13900", 89.0),
    ("Core i9-13900F", 89.0),
    ("Core i7-13700K", 85.0),
    ("Core i7-13700", 83.0),
    ("Core i7-13700F", 83.0),
    ("Core i5-13600K", 84.0),
    ("Core i5-13600", 82.0),
    ("Core i5-13500", 79.0),
    ("Core i5-13400", 76.0),
    ("Core i5-13400F", 76.0),
    ("Core i3-13100", 50.0),
    ("Core i3-13100F", 50.0),
    ("Core i3-13300", 53.0),
    ("Core i9-12900KS", 92.0),
    ("Core i9-12900K", 90.0),
    ("Core i9-12900", 86.0),
    ("Core i9-12900F", 86.0),
    ("Core i7-12700K", 82.0),
    ("Core i7-12700", 80.0),
    ("Core i7-12700F", 80.0),
    ("Core i5-12600K", 73.0),
    ("Core i5-12600", 71.0),
    ("Core i5-12500", 74.0),
    ("Core i5-12400", 69.0),
    ("Core i5-12400F", 69.0),
    ("Core i3-12100", 49.0),
    ("Core i3-12100F", 49.0),
    ("Core i3-12300", 52.0),
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
    ("Ryzen 7 7700F", 83.0),
    ("Ryzen 5 7600F", 77.0),
    ("Ryzen 9 7950X3D", 95.0),
    ("Core i9-11900K", 78.0),
    ("Core i9-11900", 76.0),
    ("Core i9-11900F", 76.0),
    ("Core i7-11700K", 74.0),
    ("Core i7-11700", 71.0),
    ("Core i7-11700F", 71.0),
    ("Core i5-11600K", 66.0),
    ("Core i5-11600", 64.0),
    ("Core i5-11500", 62.0),
    ("Core i5-11400", 64.0),
    ("Core i5-11400F", 64.0),
    ("Core i3-11100", 50.0),
    ("Core i3-11300", 52.0),
    ("Core i9-10980XE", 80.0),
    ("Core i9-10940X", 78.0),
    ("Core i9-10920X", 76.0),
    ("Core i9-10900X", 75.0),
    ("Core i9-10900K", 75.0),
    ("Core i9-10900", 73.0),
    ("Core i9-10900F", 73.0),
    ("Core i9-10850K", 74.0),
    ("Core i7-10700K", 71.0),
    ("Core i7-10700", 69.0),
    ("Core i7-10700F", 69.0),
    ("Core i5-10600K", 69.0),
    ("Core i5-10600", 67.0),
    ("Core i5-10500", 65.0),
    ("Core i5-10400", 62.0),
    ("Core i5-10400F", 62.0),
    ("Core i3-10320", 54.0),
    ("Core i3-10300", 52.0),
    ("Core i3-10100", 52.0),
    ("Core i3-10100F", 52.0),
    ("Ryzen 9 5950X", 87.0),
    ("Ryzen 9 5900X", 85.0),
    ("Ryzen 9 5900", 83.0),
    ("Ryzen 7 5800X3D", 72.0),
    ("Ryzen 7 5800X", 71.0),
    ("Ryzen 7 5800", 69.0),
    ("Ryzen 7 5700X", 73.0),
    ("Ryzen 7 5700G", 68.0),
    ("Ryzen 7 5700GT", 69.0),
    ("Ryzen 7 5700GE", 66.0),
    ("Ryzen 5 5600X", 65.0),
    ("Ryzen 5 5600G", 62.0),
    ("Ryzen 5 5600GT", 63.0),
    ("Ryzen 5 5600", 63.0),
    ("Ryzen 5 5600F", 63.0),
    ("Ryzen 5 5500GT", 60.0),
    ("Ryzen 5 5500", 62.0),
    ("Ryzen 5 5500GE", 58.0),
    ("Ryzen 5 5600X3D", 68.0),
    ("Core i9-9900KS", 72.0),
    ("Core i9-9900K", 70.0),
    ("Core i9-9900", 67.0),
    ("Core i9-9900F", 67.0),
    ("Core i7-9700K", 67.0),
    ("Core i7-9700", 65.0),
    ("Core i7-9700F", 65.0),
    ("Core i5-9600K", 64.0),
    ("Core i5-9600", 62.0),
    ("Core i5-9500", 60.0),
    ("Core i5-9400", 62.0),
    ("Core i5-9400F", 62.0),
    ("Core i3-9350KF", 58.0),
    ("Core i3-9320", 56.0),
    ("Core i3-9300", 55.0),
    ("Core i3-9100", 55.0),
    ("Core i3-9100F", 55.0),
    ("Core i7-8700K", 65.0),
    ("Core i7-8700", 64.0),
    ("Core i7-8086K", 66.0),
    ("Core i5-8600K", 62.0),
    ("Core i5-8600", 60.0),
    ("Core i5-8500", 60.0),
    ("Core i5-8400", 58.0),
    ("Core i3-8350K", 58.0),
    ("Core i3-8300", 56.0),
    ("Core i3-8100", 54.0),
    ("Ryzen 9 3950X", 81.0),
    ("Ryzen 9 3900XT", 80.0),
    ("Ryzen 9 3900X", 79.0),
    ("Ryzen 9 3900", 77.0),
    ("Ryzen 7 3800XT", 76.0),
    ("Ryzen 7 3800X", 75.0),
    ("Ryzen 7 3700X", 71.0),
    ("Ryzen 5 3600XT", 60.0),
    ("Ryzen 5 3600X", 58.0),
    ("Ryzen 5 3600", 57.0),
    ("Ryzen 5 3500X", 55.0),
    ("Ryzen 5 3500", 54.0),
    ("Ryzen 3 3300X", 49.0),
    ("Ryzen 3 3100", 47.0),
    ("Ryzen 3 3200G", 38.0),
    ("Core i7-7700K", 56.0),
    ("Core i7-7700", 61.0),
    ("Core i7-7700T", 54.0),
    ("Core i5-7600K", 60.0),
    ("Core i5-7600", 58.0),
    ("Core i5-7500", 56.0),
    ("Core i5-7400", 54.0),
    ("Core i3-7350K", 54.0),
    ("Core i3-7320", 53.0),
    ("Core i3-7300", 52.0),
    ("Core i3-7100", 48.0),
    ("Core i3-7101TE", 46.0),
    ("Core i7-6700K", 55.0),
    ("Core i7-6700", 57.0),
    ("Core i7-6700T", 52.0),
    ("Core i5-6600K", 56.0),
    ("Core i5-6600", 55.0),
    ("Core i5-6500", 54.0),
    ("Core i5-6400", 52.0),
    ("Core i5-6402P", 50.0),
    ("Core i3-6320", 50.0),
    ("Core i3-6300", 49.0),
    ("Core i3-6100", 48.0),
    ("Core i3-6100T", 46.0),
    ("Core i3-6098P", 47.0),
    ("Core i7-5775C", 52.0),
    ("Core i5-5675C", 50.0),
    ("Core i5-5575R", 48.0),
    ("Core i3-5157U", 42.0),
    ("Core i7-4790K", 47.0),
    ("Core i7-4790", 46.0),
    ("Core i7-4790S", 45.0),
    ("Core i7-4790T", 42.0),
    ("Core i7-4771", 45.0),
    ("Core i7-4770K", 45.0),
    ("Core i7-4770", 44.0),
    ("Core i7-4770S", 43.0),
    ("Core i7-4770T", 40.0),
    ("Core i5-4690K", 44.0),
    ("Core i5-4690", 43.0),
    ("Core i5-4690S", 42.0),
    ("Core i5-4690T", 39.0),
    ("Core i5-4670K", 43.0),
    ("Core i5-4670", 42.0),
    ("Core i5-4670S", 41.0),
    ("Core i5-4670T", 38.0),
    ("Core i5-4590", 42.0),
    ("Core i5-4590S", 41.0),
    ("Core i5-4590T", 38.0),
    ("Core i5-4570", 41.0),
    ("Core i5-4570S", 40.0),
    ("Core i5-4570T", 37.0),
    ("Core i5-4460", 42.0),
    ("Core i5-4460S", 41.0),
    ("Core i5-4460T", 38.0),
    ("Core i5-4440", 40.0),
    ("Core i5-4440S", 39.0),
    ("Core i5-4430", 39.0),
    ("Core i5-4430S", 38.0),
    ("Core i3-4370", 37.0),
    ("Core i3-4360", 36.0),
    ("Core i3-4350", 35.0),
    ("Core i3-4340", 35.0),
    ("Core i3-4330", 35.0),
    ("Core i3-4170", 35.0),
    ("Core i3-4160", 35.0),
    ("Core i3-4150", 34.0),
    ("Core i3-4130", 35.0),
    ("Core i7-3770K", 43.0),
    ("Core i7-3770", 42.0),
    ("Core i7-3770S", 41.0),
    ("Core i7-3770T", 38.0),
    ("Core i5-3570K", 40.0),
    ("Core i5-3570", 40.0),
    ("Core i5-3570S", 39.0),
    ("Core i5-3570T", 36.0),
    ("Core i5-3550", 39.0),
    ("Core i5-3550S", 38.0),
    ("Core i5-3470", 38.0),
    ("Core i5-3470S", 37.0),
    ("Core i5-3470T", 34.0),
    ("Core i5-3450", 37.0),
    ("Core i5-3450S", 36.0),
    ("Core i5-3350P", 36.0),
    ("Core i5-3340", 37.0),
    ("Core i5-3330", 36.0),
    ("Core i5-3330S", 35.0),
    ("Core i3-3250", 33.0),
    ("Core i3-3240", 32.0),
    ("Core i3-3225", 32.0),
    ("Core i3-3220", 31.0),
    ("Core i3-3210", 30.0),
    ("Core i7-2700K", 38.0),
    ("Core i7-2600K", 37.0),
    ("Core i7-2600", 36.0),
    ("Core i7-2600S", 34.0),
    ("Core i5-2550K", 36.0),
    ("Core i5-2500K", 35.0),
    ("Core i5-2500", 34.0),
    ("Core i5-2500S", 33.0),
    ("Core i5-2500T", 30.0),
    ("Core i5-2450P", 33.0),
    ("Core i5-2400", 34.0),
    ("Core i5-2400S", 32.0),
    ("Core i5-2405S", 32.0),
    ("Core i5-2390T", 30.0),
    ("Core i5-2320", 33.0),
    ("Core i5-2310", 32.0),
    ("Core i5-2300", 31.0),
    ("Core i3-2130", 29.0),
    ("Core i3-2125", 28.0),
    ("Core i3-2120", 27.0),
    ("Core i3-2105", 27.0),
    ("Core i3-2100", 26.0),
    ("Core i3-2100T", 24.0),
    ("Core i7-980X", 35.0),
    ("Core i7-980", 34.0),
    ("Core i7-975", 33.0),
    ("Core i7-970", 33.0),
    ("Core i7-965", 32.0),
    ("Core i7-960", 32.0),
    ("Core i7-950", 31.0),
    ("Core i7-940", 30.0),
    ("Core i7-930", 30.0),
    ("Core i7-920", 29.0),
    ("Core i5-760", 30.0),
    ("Core i5-750", 29.0),
    ("Core i5-680", 29.0),
    ("Core i5-670", 28.0),
    ("Core i5-661", 28.0),
    ("Core i5-660", 28.0),
    ("Core i5-655K", 28.0),
    ("Core i5-650", 27.0),
    ("Core i3-560", 27.0),
    ("Core i3-550", 26.0),
    ("Core i3-540", 25.0),
    ("Core i3-530", 24.0),
    ("Core 2 Extreme QX9775", 22.0),
    ("Core 2 Extreme QX9770", 22.0),
    ("Core 2 Extreme QX9650", 20.0),
    ("Core 2 Extreme QX6850", 18.0),
    ("Core 2 Extreme QX6800", 17.0),
    ("Core 2 Extreme QX6700", 15.0),
    ("Core 2 Extreme X6800", 14.0),
    ("Core 2 Quad Q9650", 16.0),
    ("Core 2 Quad Q9550", 15.0),
    ("Core 2 Quad Q9505", 14.0),
    ("Core 2 Quad Q9500", 14.0),
    ("Core 2 Quad Q9450", 14.0),
    ("Core 2 Quad Q9400", 13.0),
    ("Core 2 Quad Q9300", 12.0),
    ("Core 2 Quad Q8400", 12.0),
    ("Core 2 Quad Q8300", 11.0),
    ("Core 2 Quad Q8200", 10.0),
    ("Core 2 Quad Q6700", 10.0),
    ("Core 2 Quad Q6600", 9.0),
    ("Core 2 Duo E8600", 10.0),
    ("Core 2 Duo E8500", 10.0),
    ("Core 2 Duo E8400", 12.0),
    ("Core 2 Duo E8300", 9.0),
    ("Core 2 Duo E8200", 9.0),
    ("Core 2 Duo E8190", 9.0),
    ("Core 2 Duo E7600", 9.0),
    ("Core 2 Duo E7500", 9.0),
    ("Core 2 Duo E7400", 8.0),
    ("Core 2 Duo E7300", 8.0),
    ("Core 2 Duo E7200", 7.0),
    ("Core 2 Duo E6850", 8.0),
    ("Core 2 Duo E6750", 8.0),
    ("Core 2 Duo E6700", 7.0),
    ("Core 2 Duo E6600", 7.0),
    ("Core 2 Duo E6550", 7.0),
    ("Core 2 Duo E6540", 7.0),
    ("Core 2 Duo E6420", 6.0),
    ("Core 2 Duo E6400", 6.0),
    ("Core 2 Duo E6320", 6.0),
    ("Core 2 Duo E6300", 5.0),
    ("Core 2 Duo E4700", 5.0),
    ("Core 2 Duo E4600", 5.0),
    ("Core 2 Duo E4500", 4.0),
    ("Core 2 Duo E4400", 4.0),
    ("Core 2 Duo E4300", 3.0),
    ("Ryzen 7 2700X", 36.0),
    ("Ryzen 7 2700", 34.0),
    ("Ryzen 7 2700E", 32.0),
    ("Ryzen 5 2600X", 40.0),
    ("Ryzen 5 2600", 38.0),
    ("Ryzen 5 2600E", 36.0),
    ("Ryzen 5 2500X", 37.0),
    ("Ryzen 5 2400G", 33.0),
    ("Ryzen 5 2400GE", 30.0),
    ("Ryzen 3 2300X", 35.0),
    ("Ryzen 3 2200G", 30.0),
    ("Ryzen 3 2200GE", 28.0),
    ("Ryzen 7 1800X", 33.0),
    ("Ryzen 7 1700X", 32.0),
    ("Ryzen 7 1700", 31.0),
    ("Ryzen 7 1700E", 29.0),
    ("Ryzen 5 1600X", 33.0),
    ("Ryzen 5 1600", 31.0),
    ("Ryzen 5 1600E", 29.0),
    ("Ryzen 5 1500X", 30.0),
    ("Ryzen 5 1400", 29.0),
    ("Ryzen 3 1300X", 26.0),
    ("Ryzen 3 1200", 24.0),
    ("FX-9590", 22.0),
    ("FX-9370", 21.0),
    ("FX-8370", 20.0),
    ("FX-8370E", 19.0),
    ("FX-8350", 20.0),
    ("FX-8320", 19.0),
    ("FX-8320E", 18.0),
    ("FX-8310", 18.0),
    ("FX-8300", 17.0),
    ("FX-8150", 18.0),
    ("FX-8120", 17.0),
    ("FX-8100", 16.0),
    ("FX-6350", 18.0),
    ("FX-6300", 17.0),
    ("FX-6200", 16.0),
    ("FX-6130", 16.0),
    ("FX-6120", 15.0),
    ("FX-6100", 15.0),
    ("FX-4350", 16.0),
    ("FX-4320", 15.0),
    ("FX-4300", 15.0),
    ("FX-4200", 14.0),
    ("FX-4170", 15.0),
    ("FX-4150", 14.0),
    ("FX-4130", 14.0),
    ("FX-4100", 13.0),
    ("Phenom II X6 1100T", 18.0),
    ("Phenom II X6 1090T", 17.0),
    ("Phenom II X6 1075T", 16.0),
    ("Phenom II X6 1065T", 16.0),
    ("Phenom II X6 1055T", 15.0),
    ("Phenom II X6 1045T", 15.0),
    ("Phenom II X6 1035T", 14.0),
    ("Phenom II X4 980", 15.0),
    ("Phenom II X4 975", 15.0),
    ("Phenom II X4 970", 14.0),
    ("Phenom II X4 965", 14.0),
    ("Phenom II X4 955", 13.0),
    ("Phenom II X4 945", 13.0),
    ("Phenom II X4 940", 12.0),
    ("Phenom II X4 925", 12.0),
    ("Phenom II X4 910e", 11.0),
    ("Phenom II X3 720", 11.0),
    ("Phenom II X3 710", 10.0),
    ("Phenom II X2 560", 10.0),
    ("Phenom II X2 555", 10.0),
    ("Phenom II X2 550", 9.0),
    ("Phenom II X2 545", 9.0),
    ("Athlon II X4 650", 12.0),
    ("Athlon II X4 645", 12.0),
    ("Athlon II X4 640", 11.0),
    ("Athlon II X4 635", 11.0),
    ("Athlon II X4 630", 11.0),
    ("Athlon II X4 620", 10.0),
    ("Athlon II X3 460", 10.0),
    ("Athlon II X3 455", 10.0),
    ("Athlon II X3 450", 10.0),
    ("Athlon II X3 445", 9.0),
    ("Athlon II X3 440", 9.0),
    ("Athlon II X3 435", 9.0),
    ("Athlon II X2 280", 9.0),
    ("Athlon II X2 270", 9.0),
    ("Athlon II X2 265", 8.0),
    ("Athlon II X2 260", 8.0),
    ("Athlon II X2 255", 8.0),
    ("Athlon II X2 250", 8.0),
    ("Athlon II X2 245", 8.0),
    ("Athlon II X2 240", 7.0),
    ("Athlon II X2 220", 7.0),
    ("Athlon II X2 215", 7.0),
    ("A10-7890K", 18.0),
    ("A10-7870K", 17.0),
    ("A10-7860K", 17.0),
    ("A10-7850K", 16.0),
    ("A10-7800", 15.0),
    ("A10-6800K", 15.0),
    ("A10-6700", 14.0),
    ("A10-6700T", 12.0),
    ("A10-5800K", 14.0),
    ("A10-5700", 13.0),
    ("A8-7600", 14.0),
    ("A8-6600K", 13.0),
    ("A8-5600K", 12.0),
    ("A8-5500", 11.0),
    ("A8-3870K", 11.0),
    ("A8-3850", 10.0),
    ("A8-3820", 10.0),
    ("A8-3800", 9.0),
    ("A6-7400K", 10.0),
    ("A6-6400K", 9.0),
    ("A6-5400K", 8.0),
    ("A6-3670K", 8.0),
    ("A6-3650", 8.0),
    ("A6-3620", 7.0),
    ("A6-3600", 7.0),
    ("A6-3500", 7.0),
    ("A4-7300", 7.0),
    ("A4-6300", 6.0),
    ("A4-5300", 6.0),
    ("A4-4020", 5.0),
    ("A4-4000", 5.0),
    ("A4-3420", 5.0),
    ("A4-3400", 5.0),
    ("A4-3300", 4.0),
    ("Core Ultra 9 185H", 82.0),
    ("Core Ultra 7 165H", 76.0),
    ("Core Ultra 7 155H", 74.0),
    ("Core Ultra 5 135H", 70.0),
    ("Core Ultra 5 125H", 68.0),
    ("Core Ultra 7 145H", 72.0),
    ("Core i9-14900H", 90.0),
    ("Core i9-13900H", 88.0),
    ("Core i9-13905H", 87.0),
    ("Core i9-12900H", 82.0),
    ("Core i9-12900HK", 84.0),
    ("Core i7-14700H", 85.0),
    ("Core i7-13700H", 82.0),
    ("Core i7-13705H", 81.0),
    ("Core i7-12700H", 78.0),
    ("Core i7-12800H", 80.0),
    ("Core i7-12650H", 76.0),
    ("Core i7-11800H", 72.0),
    ("Core i7-1185G7", 70.0),
    ("Core i7-1165G7", 68.0),
    ("Core i7-10875H", 70.0),
    ("Core i7-10870H", 69.0),
    ("Core i7-10750H", 68.0),
    ("Core i7-10850H", 69.0),
    ("Core i5-14500H", 78.0),
    ("Core i5-13500H", 74.0),
    ("Core i5-13450H", 72.0),
    ("Core i5-12500H", 70.0),
    ("Core i5-12450H", 68.0),
    ("Core i5-11400H", 64.0),
    ("Core i5-11300H", 62.0),
    ("Core i5-10500H", 64.0),
    ("Core i5-10400H", 62.0),
    ("Core i5-10300H", 60.0),
    ("Core i5-10210U", 56.0),
    ("Core i3-1315U", 52.0),
    ("Core i3-1305U", 50.0),
    ("Core i3-1215U", 48.0),
    ("Core i3-1115G4", 46.0),
    ("Core i3-10110U", 44.0),
    ("Core i3-1005G1", 42.0),
    ("Core Ultra 9 285HX", 93.0),
    ("Core i9-14900HX", 89.0),
    ("Core i9-13900HX", 88.0),
    ("Core i9-13980HX", 90.0),
    ("Core i9-12900HX", 84.0),
    ("Core i9-12950HX", 86.0),
    ("Core i7-14700HX", 87.0),
    ("Core i7-13700HX", 84.0),
    ("Core i7-12800HX", 82.0),
    ("Core i7-12850HX", 83.0),
    ("Core i5-14500HX", 80.0),
    ("Core i5-13500HX", 78.0),
    ("Core i5-12600HX", 76.0),
    ("Core i5-12500HX", 74.0),
    ("Core i7-1365U", 72.0),
    ("Core i7-1355U", 70.0),
    ("Core i7-1265U", 70.0),
    ("Core i7-1255U", 68.0),
    ("Core i7-1195G7", 68.0),
    ("Core i7-1185G7", 70.0),
    ("Core i7-1165G7", 68.0),
    ("Core i7-1065G7", 62.0),
    ("Core i7-10510U", 60.0),
    ("Core i5-1345U", 68.0),
    ("Core i5-1335U", 66.0),
    ("Core i5-1245U", 66.0),
    ("Core i5-1235U", 64.0),
    ("Core i5-1145G7", 64.0),
    ("Core i5-1135G7", 62.0),
    ("Core i5-1130G7", 60.0),
    ("Core i5-1035G7", 58.0),
    ("Core i5-1035G4", 56.0),
    ("Core i5-10210U", 56.0),
    ("Core i5-8265U", 52.0),
    ("Core i5-8250U", 45.0),
    ("Core i5-7200U", 40.0),
    ("Core i5-6200U", 38.0),
    ("Core i5-5200U", 34.0),
    ("Core i5-4200U", 23.0),
    ("Core i5-3317U", 28.0),
    ("Core i5-2467M", 20.0),
    ("Core i5-2450M", 21.0),
    ("Core i5-2430M", 20.0),
    ("Core i5-2410M", 19.0),
    ("Core i3-1315U", 52.0),
    ("Core i3-1125G4", 48.0),
    ("Core i3-1115G4", 46.0),
    ("Core i3-10110U", 44.0),
    ("Core i3-1005G1", 42.0),
    ("Core i3-8145U", 44.0),
    ("Core i3-8130U", 42.0),
    ("Core i3-7130U", 40.0),
    ("Core i3-7100U", 38.0),
    ("Core i3-6100U", 36.0),
    ("Core i3-6006U", 34.0),
    ("Core i3-5005U", 30.0),
    ("Core i3-4005U", 24.0),
    ("Core i3-3217U", 26.0),
    ("Core i3-3110M", 24.0),
    ("Core i3-2370M", 22.0),
    ("Core i3-2350M", 21.0),
    ("Core i3-2330M", 20.0),
    ("Core i3-2310M", 19.0),
    ("Ryzen 9 8945HS", 88.0),
    ("Ryzen 9 7945HX", 87.0),
    ("Ryzen 9 7940HX", 86.0),
    ("Ryzen 9 7940HS", 88.0),
    ("Ryzen 9 7845HX", 84.0),
    ("Ryzen 9 6900HX", 78.0),
    ("Ryzen 9 6980HX", 80.0),
    ("Ryzen 7 8845HS", 82.0),
    ("Ryzen 7 7840HS", 82.0),
    ("Ryzen 7 7840U", 78.0),
    ("Ryzen 7 7735HS", 76.0),
    ("Ryzen 7 7730U", 70.0),
    ("Ryzen 7 7735U", 70.0),
    ("Ryzen 7 7736U", 70.0),
    ("Ryzen 7 6800U", 72.0),
    ("Ryzen 7 6800H", 74.0),
    ("Ryzen 7 5800H", 68.0),
    ("Ryzen 7 5800U", 64.0),
    ("Ryzen 7 5700U", 60.0),
    ("Ryzen 5 7640U", 72.0),
    ("Ryzen 5 7640HS", 74.0),
    ("Ryzen 5 7535HS", 68.0),
    ("Ryzen 5 7530U", 62.0),
    ("Ryzen 5 7535U", 62.0),
    ("Ryzen 5 7520U", 56.0),
    ("Ryzen 5 6600U", 64.0),
    ("Ryzen 5 6600H", 66.0),
    ("Ryzen 5 5600H", 62.0),
    ("Ryzen 5 5600U", 58.0),
    ("Ryzen 5 5500U", 54.0),
    ("Ryzen 5 5560U", 56.0),
    ("Ryzen 5 4600H", 56.0),
    ("Ryzen 5 4600U", 52.0),
    ("Ryzen 5 4500U", 48.0),
    ("Ryzen 3 7330U", 52.0),
    ("Ryzen 3 7335U", 52.0),
    ("Ryzen 3 7320U", 46.0),
    ("Ryzen 3 5400U", 46.0),
    ("Ryzen 3 5425U", 48.0),
    ("Ryzen 3 5300U", 42.0),
    ("Ryzen 3 4300U", 38.0),
    ("Ryzen 3 3350U", 36.0),
    ("Ryzen 3 3300U", 34.0),
    ("Ryzen 3 3250U", 32.0),
    ("Ryzen 3 3200U", 30.0),
    ("Ryzen 7 8700G", 72.0),
    ("Ryzen 5 8600G", 66.0),
    ("Ryzen 5 8500G", 62.0),
    ("Ryzen 5 8300G", 58.0),
    ("Ryzen 7 9700X", 82.0),
    ("Ryzen 5 9600X", 83.0),
    ("Pentium Gold G7400", 20.0),
    ("Pentium Gold G7400T", 18.0),
    ("Pentium Gold G6605", 26.0),
    ("Pentium Gold G6600", 25.0),
    ("Pentium Gold G6505", 25.0),
    ("Pentium Gold G6500", 24.0),
    ("Pentium Gold G6405", 24.0),
    ("Pentium Gold G6400", 25.0),
    ("Pentium Gold G5620", 24.0),
    ("Pentium Gold G5600", 23.0),
    ("Pentium Gold G5500", 22.0),
    ("Pentium Gold G5420", 22.0),
    ("Pentium Gold G5400", 21.0),
    ("Pentium G4620", 21.0),
    ("Pentium G4600", 20.0),
    ("Pentium G4560", 30.0),
    ("Pentium G4520", 19.0),
    ("Pentium G4500", 18.0),
    ("Pentium G4400", 17.0),
    ("Pentium G3470", 18.0),
    ("Pentium G3460", 17.0),
    ("Pentium G3450", 17.0),
    ("Pentium G3440", 16.0),
    ("Pentium G3430", 16.0),
    ("Pentium G3420", 16.0),
    ("Pentium G3260", 16.0),
    ("Pentium G3258", 16.0),
    ("Pentium G3250", 15.0),
    ("Pentium G3240", 15.0),
    ("Pentium G3220", 15.0),
    ("Pentium G3210", 14.0),
    ("Pentium Gold 4417U", 16.0),
    ("Pentium Gold 4415U", 15.0),
    ("Pentium Silver N6000", 21.0),
    ("Pentium Silver N5100", 19.0),
    ("Pentium Silver N5030", 18.0),
    ("Pentium Silver N5000", 17.0),
    ("Pentium Silver J5040", 18.0),
    ("Pentium Silver J5005", 16.0),
    ("Pentium N4200", 14.0),
    ("Pentium N3710", 13.0),
    ("Pentium N3700", 12.0),
    ("Pentium N3540", 12.0),
    ("Pentium N3520", 11.0),
    ("Pentium J3710", 12.0),
    ("Pentium J2900", 10.0),
    ("Pentium G2130", 15.0),
    ("Pentium G2120", 15.0),
    ("Pentium G2030", 14.0),
    ("Pentium G2020", 14.0),
    ("Pentium G2010", 13.0),
    ("Pentium G870", 14.0),
    ("Pentium G860", 14.0),
    ("Pentium G850", 13.0),
    ("Pentium G840", 13.0),
    ("Pentium G645", 13.0),
    ("Pentium G640", 13.0),
    ("Pentium G632", 12.0),
    ("Pentium G630", 12.0),
    ("Pentium G622", 12.0),
    ("Pentium G620", 12.0),
    ("Pentium G550", 11.0),
    ("Pentium G540", 11.0),
    ("Pentium G530", 10.0),
    ("Celeron G5925", 18.0),
    ("Celeron G5905", 17.0),
    ("Celeron G5900", 16.0),
    ("Celeron G555", 12.0),
    ("Celeron G550", 12.0),
    ("Celeron G540", 11.0),
    ("Celeron G530", 10.0),
    ("Celeron G4950", 16.0),
    ("Celeron G4930", 16.0),
    ("Celeron G4920", 16.0),
    ("Celeron G4900", 16.0),
    ("Celeron G3960", 15.0),
    ("Celeron G3950", 15.0),
    ("Celeron G3930", 15.0),
    ("Celeron G3920", 14.0),
    ("Celeron G3900", 14.0),
    ("Celeron G1850", 13.0),
    ("Celeron G1840", 13.0),
    ("Celeron G1830", 12.0),
    ("Celeron G1820", 12.0),
    ("Celeron G1630", 11.0),
    ("Celeron G1620", 11.0),
    ("Celeron G1610", 10.0),
    ("Celeron G570", 10.0),
    ("Celeron G560", 10.0),
    ("Celeron G550", 9.0),
    ("Celeron G540", 9.0),
    ("Celeron G530", 8.0),
    ("Celeron G470", 8.0),
    ("Celeron G465", 8.0),
    ("Celeron G460", 7.0),
    ("Celeron G440", 6.0),
    ("Celeron G3930T", 13.0),
    ("Celeron G3900T", 12.0),
    ("Celeron G1820T", 10.0),
    ("Celeron G1610T", 9.0),
    ("Celeron N5105", 18.0),
    ("Celeron N5100", 17.0),
    ("Celeron N5095", 17.0),
    ("Celeron N4505", 16.0),
    ("Celeron N4500", 16.0),
    ("Celeron N4120", 14.0),
    ("Celeron N4020", 13.0),
    ("Celeron N4000", 12.0),
    ("Celeron N3450", 12.0),
    ("Celeron N3350", 11.0),
    ("Celeron N3160", 11.0),
    ("Celeron N3150", 10.0),
    ("Celeron N3060", 9.0),
    ("Celeron N3050", 8.0),
    ("Celeron N3010", 8.0),
    ("Celeron N3000", 8.0),
    ("Celeron N2940", 9.0),
    ("Celeron N2930", 8.0),
    ("Celeron N2920", 7.0),
    ("Celeron N2840", 7.0),
    ("Celeron N2830", 6.0),
    ("Celeron N2820", 6.0),
    ("Celeron N2815", 5.0),
    ("Celeron N2810", 5.0),
    ("Celeron N2808", 5.0),
    ("Celeron N2807", 4.0),
    ("Celeron N2806", 4.0),
    ("Celeron J4125", 14.0),
    ("Celeron J4115", 13.0),
    ("Celeron J4105", 12.0),
    ("Celeron J4025", 12.0),
    ("Celeron J4005", 11.0),
    ("Celeron J3455", 10.0),
    ("Celeron J3355", 9.0),
    ("Celeron J1900", 8.0),
    ("Celeron J1850", 7.0),
    ("Celeron J1800", 7.0),
    ("Celeron J1750", 6.0),
    ("Atom x7-Z8750", 11.0),
    ("Atom x7-Z8700", 10.0),
    ("Atom x5-Z8550", 9.0),
    ("Atom x5-Z8500", 9.0),
    ("Atom x5-Z8350", 9.0),
    ("Atom x5-Z8300", 8.0),
    ("Atom x5-Z8100", 7.0),
    ("Atom x3-Z8350", 7.0),
    ("Atom x3-Z8300", 6.0),
    ("Atom Z3795", 7.0),
    ("Atom Z3775", 7.0),
    ("Atom Z3770", 6.0),
    ("Atom Z3760", 6.0),
    ("Atom Z3740", 5.0),
    ("Atom Z3735", 5.0),
    ("Atom Z2760", 4.0),
    ("Atom Z2580", 4.0),
    ("Atom Z2560", 3.0),
    ("Atom Z2520", 3.0),
    ("Atom N2930", 6.0),
    ("Atom N2920", 5.0),
    ("Atom N2820", 5.0),
    ("Atom N2810", 4.0),
    ("Atom N2800", 4.0),
    ("Atom N2600", 3.0),
    ("Atom N570", 3.0),
    ("Atom N550", 2.0),
    ("Atom N470", 2.0),
    ("Atom N450", 1.0),
    ("Atom N280", 1.0),
    ("Atom N270", 1.0),
    ("Atom N2600", 3.0),
    ("Atom N2300", 2.0),
    ("Atom D525", 2.0),
    ("Atom D510", 2.0),
    ("Atom D425", 1.0),
    ("Atom D410", 1.0),
    ("Atom D2700", 2.0),
    ("Atom D2550", 2.0),
    ("Atom D2500", 1.0),
    ("Core i3-N305", 40.0),
    ("Core i3-N300", 38.0),
    ("Core i5-N50", 42.0),
    ("Core i7-N100", 44.0),
    ("Core i7-N150", 46.0),
    ("Core i7-N200", 48.0),
    ("Core i7-N305", 50.0),
    ("Core i7-N300", 48.0),
    ("Core i7-N50", 44.0),
    ("Core i7-N60", 46.0),
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
    ("Apple M5 Max", 82.0),
    ("Apple M5 Pro", 75.0),
    ("Apple M5", 68.0),
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
