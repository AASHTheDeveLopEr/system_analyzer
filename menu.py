#!/usr/bin/env python3
"""The menu of System Analyzer.

Run it:  python menu.py

Detection is done by libraries, not by hand: psutil for RAM and disks,
py-cpuinfo for the CPU, pynvml (and wmi on Windows) for GPUs. Anything a
library cannot read is recorded as ``None`` and printed as "unknown" instead of
being guessed, so a blocked or missing API can never crash the tool.

The detectors return records (bytes and counts), never display strings, and
formatting happens at the edge. The decision layer needs the numbers -- "8 GiB
of dedicated VRAM" and "8 GiB of shared system RAM" are the same digits and two
very different answers to "can I run X?".
"""

from __future__ import annotations

import contextlib
import os
import platform
from dataclasses import dataclass
from typing import Any

import cpuinfo
import psutil
import pynvml

GIB = 1024 ** 3
MIB = 1024 ** 2
GHZ = 1e9

UNKNOWN = "unknown"


def _format_size(num_bytes: int) -> str:
    """Human size that keeps sub-GiB values readable (512 MiB beats 0.5 GiB)."""
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.1f} GiB"
    return f"{num_bytes / MIB:.0f} MiB"


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Memory:
    """Physical RAM. ``None`` means "not readable", never "zero"."""

    total_bytes: int | None = None
    available_bytes: int | None = None

    def display(self) -> str:
        if self.total_bytes is None:
            return UNKNOWN
        total = _format_size(self.total_bytes)
        if self.available_bytes is None:
            return f"{total} total"
        # "available" is what a new process can actually claim, which is not
        # the same as the "free" figure: the OS cache counts as available.
        return f"{total} total, {_format_size(self.available_bytes)} available"


@dataclass(frozen=True)
class Gpu:
    """One display adapter.

    ``vram_bytes`` is *dedicated* video memory and is ``None`` when no source
    reported it honestly. That is not the same as 0: a 2 GiB number and no
    number are different facts, and only the first one can answer a question
    about running something.

    ``source`` says where the name came from -- ``nvml`` (an NVIDIA driver
    answered, so the card is real and its VRAM figure is driver-reported) or
    ``wmi`` (Windows' own adapter list). Prefer ``nvml`` cards when a judgment
    needs the strongest adapter; ``wmi``-only entries are usually integrated
    graphics, but nothing here claims that, because nothing here can see it.
    """

    name: str
    vram_bytes: int | None = None
    source: str = "unknown"
    note: str = ""

    def display(self) -> str:
        if self.vram_bytes is not None:
            return f"{self.name} ({_format_size(self.vram_bytes)} VRAM)"
        return f"{self.name} (VRAM {UNKNOWN}{f': {self.note}' if self.note else ''})"

    def to_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dedicated_vram_gib": _gib(self.vram_bytes),
            "detected_via": self.source,
            "note": self.note or None,
        }


@dataclass(frozen=True)
class Disk:
    """One mounted volume."""

    mount: str
    total_bytes: int
    free_bytes: int
    used_percent: float

    def display(self) -> str:
        return f"{self.mount} {_format_size(self.total_bytes)} ({self.used_percent:.0f}% used)"

    def to_state(self) -> dict[str, Any]:
        # The mount point is a path: it is shown locally but never leaves the
        # machine with the state (FR-010).
        return {
            "total_gib": _gib(self.total_bytes),
            "free_gib": _gib(self.free_bytes),
            "used_percent": round(self.used_percent, 1),
        }


@dataclass(frozen=True)
class Inventory:
    """Everything we detected about this machine, numbers included."""

    os_name: str = UNKNOWN
    cpu_name: str = UNKNOWN
    cpu_cores: int | None = None
    cpu_ghz: float | None = None
    memory: Memory = Memory()
    gpus: tuple[Gpu, ...] = ()
    disks: tuple[Disk, ...] = ()
    python_version: str = UNKNOWN

    def lines(self) -> list[tuple[str, str]]:
        """The "label: value" rows the menu prints."""
        cpu = self.cpu_name
        if self.cpu_ghz and not any(unit in cpu for unit in ("GHz", "MHz")):
            cpu = f"{cpu} @ {self.cpu_ghz:.1f} GHz"
        gpus = ", ".join(gpu.display() for gpu in self.gpus) if self.gpus else UNKNOWN
        disks = ", ".join(disk.display() for disk in self.disks) if self.disks else UNKNOWN
        cores = f"{self.cpu_cores} logical" if self.cpu_cores else UNKNOWN
        return [
            ("OS", self.os_name),
            ("CPU", cpu),
            ("Cores", cores),
            ("RAM", self.memory.display()),
            ("GPU", gpus),
            ("Disks", disks),
            ("Python", self.python_version),
        ]

    def to_state(self) -> dict[str, Any]:
        """The scrubbed payload for Jev: hardware facts only, no paths or names.

        Every field is present even when it is ``None``, so a missing reading
        stays visible as unknown instead of quietly becoming a zero the model
        would reason about (FR-002).
        """
        return {
            "operating_system": self.os_name,
            "cpu": {
                "name": self.cpu_name,
                "logical_cores": self.cpu_cores,
                "clock_ghz": round(self.cpu_ghz, 2) if self.cpu_ghz else None,
            },
            "memory": {
                "total_gib": _gib(self.memory.total_bytes),
                "available_gib": _gib(self.memory.available_bytes),
            },
            "gpus": [gpu.to_state() for gpu in self.gpus],
            "storage": [disk.to_state() for disk in self.disks],
        }


def _gib(num_bytes: int | None) -> float | None:
    return round(num_bytes / GIB, 2) if num_bytes is not None else None


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------
# Each one returns its record and swallows the library's own failures, so a
# blocked API shows up as "unknown" (SC-001) rather than a traceback.


def detect_os() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def detect_cpu() -> tuple[str, int | None, float | None]:
    """(name, logical cores, clock in GHz)."""
    cores = psutil.cpu_count(logical=True) or None
    try:
        info = cpuinfo.get_cpu_info()
    except Exception:  # noqa: BLE001 - py-cpuinfo parses /proc and registry
        return platform.processor() or platform.machine() or UNKNOWN, cores, None

    name = info.get("brand_raw") or platform.processor() or platform.machine() or UNKNOWN
    hz = info.get("hz_actual") or 0
    if isinstance(hz, (list, tuple)):
        # py-cpuinfo reports every frequency it could measure on some
        # platforms. Take the lowest: a rating is about sustained capability,
        # and a turbo spike is not something the machine owes its user.
        hz = min(hz, default=0)
    return name, cores, (hz / GHZ if hz else None)


def detect_memory() -> Memory:
    try:
        memory = psutil.virtual_memory()
    except (psutil.Error, OSError):
        return Memory()
    return Memory(total_bytes=memory.total, available_bytes=memory.available)


def detect_disks() -> tuple[Disk, ...]:
    """Every readable volume. Drives that answer with no media are skipped."""
    disks = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue  # empty card slot, disconnected drive, ...
        if usage.total <= 0:
            continue  # an empty optical drive reports zero, not a size
        disks.append(
            Disk(
                mount=part.mountpoint,
                total_bytes=usage.total,
                free_bytes=usage.free,
                used_percent=usage.percent,
            )
        )
    return tuple(disks)


# Windows reports these for VMs and remote sessions. They are not hardware, and
# the spec wants an unknown field there rather than an invented GPU.
SOFTWARE_ADAPTERS = (
    "basic render driver",
    "remote display adapter",
    "indirect display",
    "microsoft basic",
    "hyper-v",
    "vmware",
    "virtualbox",
    "virtio",
    "parallels",
    "qxl",
    "bochs",
)

# Why a WMI-only adapter has no VRAM number. Windows' AdapterRAM is a 32-bit
# field: it wraps above 4 GiB, so a 24 GB card reads as 0. Reporting a wrong
# number is worse than reporting none.
WMI_NO_VRAM = "Windows exposes only a 32-bit size for this adapter"


def _is_software_adapter(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in SOFTWARE_ADAPTERS)


def detect_gpus() -> tuple[Gpu, ...]:
    """Every real GPU, pynvml first, then Windows' own list for the rest.

    pynvml reads NVIDIA cards on any OS and is the only source here that
    reports VRAM honestly, so it goes first and the Windows pass only adds
    adapters pynvml did not name.
    """
    cards = list(_nvidia_gpus())
    if os.name == "nt":
        known = [_normalize(gpu.name) for gpu in cards]
        for name in _windows_gpu_names():
            if _is_software_adapter(name):
                continue
            if any(_normalize(name) in seen or seen in _normalize(name) for seen in known):
                continue  # pynvml already reported this one, spelled differently
            cards.append(Gpu(name=name, vram_bytes=None, source="wmi", note=WMI_NO_VRAM))
            known.append(_normalize(name))
    return tuple(cards)


def _normalize(name: str) -> str:
    """Fold the vendor prefix and spacing so the same card matches itself."""
    lowered = name.lower().replace("nvidia", "").replace("(r)", "").replace("(tm)", "")
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered).split())


def _nvidia_gpus() -> list[Gpu]:
    """(name, vram) for each NVIDIA card; empty if there is none or no driver."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return []  # no NVIDIA card, or the driver is not loaded

    try:
        cards = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):  # older pynvml versions return bytes
                name = name.decode("utf-8", "replace")
            # One unreadable device must not cost us the others: a display-only
            # or half-initialised card can fail the memory query on its own.
            try:
                vram = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            except pynvml.NVMLError:
                vram = None
            cards.append(
                Gpu(
                    name=name,
                    vram_bytes=vram,
                    source="nvml",
                    note="" if vram is not None else "driver did not report VRAM",
                )
            )
        return cards
    except pynvml.NVMLError:
        return []
    finally:
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()


def _windows_gpu_names() -> list[str]:
    """Names of every display adapter Windows knows, or an empty list."""
    try:
        import wmi
    except ImportError:
        return []  # optional dependency, not installed

    try:
        return [card.Name for card in wmi.WMI().Win32_VideoController() if card.Name]
    except Exception:  # noqa: BLE001 - WMI raises bare exceptions when disabled
        return []


def detect_inventory() -> Inventory:
    """Detect this machine. Nothing here raises; anything missing stays None."""
    cpu_name, cores, ghz = detect_cpu()
    return Inventory(
        os_name=detect_os(),
        cpu_name=cpu_name,
        cpu_cores=cores,
        cpu_ghz=ghz,
        memory=detect_memory(),
        gpus=detect_gpus(),
        disks=detect_disks(),
        python_version=platform.python_version(),
    )


def get_specs() -> dict[str, str]:
    """The detected specs as "label: value" pairs, for the CLI and for tests."""
    return dict(detect_inventory().lines())


# There is deliberately no main() here any more. The menu, the rating and the
# question loop all live in main.py now, so there is exactly one entry point;
# run ``python main.py`` (or ``python main.py --help``) instead of this module.
