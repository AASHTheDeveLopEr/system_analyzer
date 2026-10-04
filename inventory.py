#!/usr/bin/env python3
"""The hardware inventory of System Analyzer.

Detection only: this module answers one question -- what hardware does this
machine have? Menus, commands and output live in ``main.py``.

Detection is done by libraries, not by hand: psutil for RAM and disks,
py-cpuinfo for the CPU, pynvml for NVIDIA GPUs on any OS, and gpu-detector
(which wraps lspci, /sys/class/drm, system_profiler, WMI/WMIC) for AMD/Intel/
Apple GPUs. Anything a library cannot read is recorded as ``None`` and printed
as "unknown" instead of being guessed, so a blocked or missing API can never
crash the tool.

The detectors return records (bytes and counts), never display strings, and
formatting happens at the edge. The decision layer needs the numbers -- "8 GiB
of dedicated VRAM" and "8 GiB of shared system RAM" are the same digits and two
very different answers to "can I run X?".

CPU and RAM usage are live readings taken for the local display: they describe
this moment, not the machine, so they stay out of the state sent to the model.
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

#: How long the CPU-usage sample runs. ``psutil.cpu_percent()`` with no
#: interval returns a placeholder 0.0 on its first call in a process, so a
#: real (short) window is the only honest way to take the reading without
#: making the user wait a full second for it.
CPU_SAMPLE_SECONDS = 0.1


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
    used_percent: float | None = None

    def display(self) -> str:
        if self.total_bytes is None:
            return UNKNOWN
        total = _format_size(self.total_bytes)
        if self.available_bytes is None:
            return f"{total} total"
        # "available" is what a new process can actually claim, which is not
        # the same as the "free" figure: the OS cache counts as available.
        text = f"{total} total, {_format_size(self.available_bytes)} available"
        if self.used_percent is not None:
            text += f" ({self.used_percent:.0f}% used)"
        return text


@dataclass(frozen=True)
class Gpu:
    """One display adapter.

    ``vram_bytes`` is *dedicated* video memory and is ``None`` when no source
    reported it honestly. That is not the same as 0: a 2 GiB number and no
    number are different facts, and only the first one can answer a question
    about running something.

    ``source`` says where the name came from -- ``nvml`` (an NVIDIA driver
    answered, so the card is real and its VRAM figure is driver-reported) or
    ``platform`` (gpu-detector found the card on Windows WMI, Linux /sys/class/
    drm, or macOS system_profiler, but VRAM is not trusted from those paths).
    Prefer ``nvml`` cards when a judgment needs the strongest adapter;
    ``platform``-only entries are usually integrated graphics, but nothing
    here claims that, because nothing here can see it.
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
        # machine with the state.
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
    cpu_usage_percent: float | None = None
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
        # The CPU's live reading rides with its core count: "23% busy" needs
        # the number of cores next to it to mean anything.
        cores = [f"{self.cpu_cores} logical"] if self.cpu_cores else []
        if self.cpu_usage_percent is not None:
            cores.append(f"{self.cpu_usage_percent:.0f}% busy")
        cores_text = ", ".join(cores) if cores else UNKNOWN
        return [
            ("OS", self.os_name),
            ("CPU", cpu),
            ("Cores", cores_text),
            ("RAM", self.memory.display()),
            ("GPU", gpus),
            ("Disks", disks),
            ("Python", self.python_version),
        ]

    def to_state(self) -> dict[str, Any]:
        """The scrubbed payload for Jev: hardware facts only, no paths or names.

        Every field is present even when it is ``None``, so a missing reading
        stays visible as unknown instead of quietly becoming a zero the model
        would reason about. Live usage readings are deliberately
        absent: they describe the moment, not the machine.
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
# blocked API shows up as "unknown" rather than a traceback.


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


def detect_cpu_usage() -> float | None:
    """CPU busy percentage over a short real window; ``None`` if unreadable.

    ``psutil.cpu_percent()`` with no interval returns a placeholder 0.0 on its
    first call in a process -- there is no earlier sample to compare against --
    and printing that as "0% busy" would be an invented reading. A real
    interval, kept short, is the honest way to take one.
    """
    try:
        return psutil.cpu_percent(interval=CPU_SAMPLE_SECONDS)
    except (psutil.Error, OSError):
        return None


def detect_memory() -> Memory:
    try:
        memory = psutil.virtual_memory()
    except (psutil.Error, OSError):
        return Memory()
    return Memory(
        total_bytes=memory.total,
        available_bytes=memory.available,
        used_percent=memory.percent,
    )


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

# Virtual and software-only adapters across platforms. They are not hardware,
# and the field stays unknown there rather than an invented GPU.
SOFTWARE_ADAPTERS = (
    # Windows virtual / remote
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
    # Linux virtual
    "llvmpipe",
    "softpipe",
    "swrast",
    "virtual",
    # macOS software / virtual
    "apple software",
    "displaylink",
)

# Vendors that gpu-detector reports as virtual or software rather than real
# hardware. Cards from these vendors are filtered out.
VIRTUAL_VENDORS = frozenset({
    "VMware", "Red Hat", "QEMU/Bochs", "VirtualBox",
})

# Why a non-NVIDIA platform-detected adapter has no VRAM number. The Windows
# WMI AdapterRAM is a 32-bit field that wraps above 4 GiB; macOS often reports
# shared memory as VRAM. We only trust VRAM from pynvml (NVIDIA) or Linux DRM
# sysfs (AMD dedicated VRAM), so adapters from other paths are unknown.
NO_VRAM_NOTE = (
    "platform API did not report dedicated VRAM for this adapter"
)


def _is_software_adapter(name: str) -> bool:
    """True if *name* looks like a virtual or software-only GPU, not real hardware."""
    lowered = name.lower()
    return any(marker in lowered for marker in SOFTWARE_ADAPTERS)


def detect_gpus() -> tuple[Gpu, ...]:
    """Every real GPU on this machine, across NVIDIA and the host platform.

    pynvml reads NVIDIA cards on any OS and reports VRAM most reliably, so it
    goes first and its entries are preferred. ``gpu-detector`` then fills the
    gap for AMD, Intel, and other vendors on Linux, Windows (WMI), and macOS
    (system_profiler). Software and virtual adapters are filtered out.
    """
    cards = _nvidia_gpus()
    known = [_normalize(gpu.name) for gpu in cards]
    for name, vram_bytes, source, note in _platform_gpu_results():
        if _is_software_adapter(name):
            continue
        normalized = _normalize(name)
        # Substring matching is unsafe here: distinct Intel generations
        # share a name prefix ("Intel UHD Graphics" and "Intel UHD
        # Graphics 630" are different cards), so a substring test would
        # silently drop a real adapter. Only an exact normalized-name
        # match means pynvml already reported this card, spelled differently.
        if normalized in known:
            continue
        cards.append(Gpu(name=name, vram_bytes=vram_bytes, source=source, note=note))
        known.append(normalized)
    return tuple(cards)


def _platform_gpu_results() -> list[tuple[str, int | None, str, str]]:
    """Non-NVIDIA GPUs from gpu-detector, filtered for software adapters.

    Returns a list of ``(name, vram_bytes, source, note)`` tuples. VRAM is only
    populated from Linux DRM sysfs (``mem_info_vram_total`` or PCI BAR sizes)
    where the driver genuinely reports dedicated memory. On macOS and Windows
    the VRAM from the OS API is not trusted (shared memory, 32-bit wraparound),
    so it stays ``None`` with an explanatory note.
    """
    info_list = _detect_platform_gpus()
    results: list[tuple[str, int | None, str, str]] = []
    is_linux = platform.system() == "Linux"
    for info in info_list:
        if info.vendor is not None and info.vendor in VIRTUAL_VENDORS:
            continue
        vram_mb = info.memory_mb
        if vram_mb is not None and is_linux:
            # On Linux, gpu-detector reads mem_info_vram_total (AMD) or the
            # largest PCI BAR, both of which are honest dedicated-VRAM numbers.
            vram_bytes = vram_mb * MIB
            source = "sysfs" if info.source and "sysfs" in info.source else info.source or "platform"
            note = ""
        else:
            # macOS and Windows: VRAM is not trusted (see NO_VRAM_NOTE above).
            vram_bytes = None
            source = "platform"
            note = NO_VRAM_NOTE
        results.append((info.name, vram_bytes, source, note))
    return results


def _detect_platform_gpus() -> list:
    """Call gpu-detector to enumerate all GPUs; return raw GPUInfo list.

    gpu-detector tries ``lspci`` then ``/sys/class/drm`` on Linux,
    ``system_profiler`` on macOS, and PowerShell/WMIC on Windows. Returns an
    empty list if the library is not installed or the platform API fails.
    """
    try:
        from gpu_detector import detect as _detect
    except ImportError:
        return []  # optional dependency; not all installs have it
    try:
        return _detect()
    except Exception:  # noqa: BLE001 - any failure here is "no data"
        return []


def _normalize(name: str) -> str:
    """Fold vendor prefixes, branding words and spacing so the same
    card matches itself across sources.

    Sources disagree about the vendor prefix: WMI reports "Intel UHD
    Graphics 770" where gpu-detector reports "UHD Graphics 770".
    Stripping the brand markers makes those one card, and case is
    folded the same way, so "NVIDIA" and "nvidia" match too. Only
    an exact normalized match counts: distinct generations ("UHD
    Graphics" and "UHD Graphics 630") stay apart.
    """
    lowered = name.lower()
    for marker in ("nvidia", "intel", "amd", "radeon", "geforce", "(r)", "(tm)", "graphics"):
        lowered = lowered.replace(marker, "")
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered).split())


def _nvidia_gpus() -> list[Gpu]:
    """(name, vram) for each NVIDIA card; empty if there is none or no driver.

    pynvml reads NVIDIA cards on any OS and is the only source that reports
    VRAM honestly across platforms, so NVIDIA always goes through here.
    """
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


def detect_inventory() -> Inventory:
    """Detect this machine. Nothing here raises; anything missing stays None."""
    cpu_name, cores, ghz = detect_cpu()
    return Inventory(
        os_name=detect_os(),
        cpu_name=cpu_name,
        cpu_cores=cores,
        cpu_ghz=ghz,
        cpu_usage_percent=detect_cpu_usage(),
        memory=detect_memory(),
        gpus=detect_gpus(),
        disks=detect_disks(),
        python_version=platform.python_version(),
    )


# There is deliberately no main() here. The commands, the menus and the
# question loop all live in main.py, so there is exactly one entry point;
# run ``python main.py`` (or ``python main.py --help``) instead of this module.
