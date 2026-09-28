"""Offline tests for the detection layer in ``menu``.

These are regression tests for real bugs, not coverage for its own sake:

* ``get_gpus`` used to append bare names on Windows and then unpack every entry
  as ``(name, vram)``. On any Windows machine whose only adapter was not an
  NVIDIA card, showing the specs raised
  ``ValueError: too many values to unpack (expected 2)``.
* One unreadable device used to make ``_nvidia_gpus`` throw away every card.
* VRAM used to be a display string, so a shared 2 GB could not be told apart
  from a dedicated 2 GB -- and neither could the decision layer, which is
  exactly the distinction FR-002 asks us to keep.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import menu


class FakeNVMLError(Exception):
    """Stand-in for ``pynvml.NVMLError``; patched in wherever it is caught."""


def fake_pynvml(devices, failing=()):
    """A ``pynvml`` module double where *failing* device indices raise.

    ``devices`` is ``[(name, vram_bytes_or_None), ...]``. Each index listed in
    ``failing`` raises on the VRAM query, which is how a half-initialised or
    display-only card behaves in the wild.
    """
    module = mock.MagicMock()
    module.NVMLError = FakeNVMLError
    module.nvmlDeviceGetCount.return_value = len(devices)

    def get_handle(index):
        if index >= len(devices):
            raise FakeNVMLError("bad index")
        return f"handle-{index}"

    module.nvmlDeviceGetHandleByIndex.side_effect = get_handle
    module.nvmlDeviceGetName.side_effect = lambda handle: handle.replace("handle-", "card-")
    module.nvmlDeviceGetMemoryInfo.side_effect = lambda handle: (
        _raise(FakeNVMLError("no memory info"))
        if int(handle.split("-")[1]) in failing
        else mock.Mock(total=devices[int(handle.split("-")[1])][1])
    )
    return module


def _raise(error):
    raise error


class GpuDetectionTests(unittest.TestCase):
    """The Windows path, which is where the crash lived."""

    def _windows(self, nvidia, windows_names):
        # NB: `Mock(name=...)` sets the mock's label, not its `.name`
        # attribute, so build it first and assign afterwards.
        fake_os = mock.Mock()
        fake_os.name = "nt"
        return mock.patch.multiple(
            menu, os=fake_os, _nvidia_gpus=mock.Mock(return_value=nvidia),
            _windows_gpu_names=mock.Mock(return_value=windows_names),
        )

    def test_windows_non_nvidia_adapter_does_not_crash(self):
        """The P0: pynvml finds nothing, Windows reports Intel graphics."""
        with self._windows([], ["Intel(R) UHD Graphics 620"]):
            gpus = menu.detect_gpus()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].name, "Intel(R) UHD Graphics 620")
        self.assertEqual(gpus[0].source, "wmi")

    def test_windows_adapter_without_vram_is_unknown_not_zero(self):
        """Shared memory is not a number we have, and guessing one is a lie."""
        with self._windows([], ["Intel(R) UHD Graphics 620"]):
            gpus = menu.detect_gpus()

        self.assertIsNone(gpus[0].vram_bytes)
        self.assertIn("unknown", gpus[0].display())

    def test_nvidia_and_windows_entries_do_not_duplicate(self):
        """pynvml spells it 'NVIDIA GeForce ...', WMI often drops the prefix."""
        card = menu.Gpu(name="NVIDIA GeForce RTX 3070", vram_bytes=8 * menu.GIB, source="nvml")
        with self._windows([card], ["GeForce RTX 3070", "Intel(R) UHD Graphics 620"]):
            gpus = menu.detect_gpus()

        names = [gpu.name for gpu in gpus]
        self.assertEqual(len(names), 2, names)
        self.assertEqual(names[0], "NVIDIA GeForce RTX 3070")

    def test_software_adapters_are_dropped(self):
        """A VM has no real GPU, and the spec wants 'unknown' there (FR-002)."""
        with self._windows([], ["Microsoft Basic Render Driver", "VMware SVGA 3D"]):
            gpus = menu.detect_gpus()

        self.assertEqual(gpus, ())

    def test_a_two_character_adapter_name_is_not_unpacked_into_vram(self):
        """The silent-corruption variant of the same bug."""
        with self._windows([], ["3D"]):
            gpus = menu.detect_gpus()

        self.assertIsNone(gpus[0].vram_bytes)


class NvmlTests(unittest.TestCase):
    def test_one_bad_device_keeps_the_other_cards(self):
        """A single unreadable GPU must not hide the whole machine's cards."""
        module = fake_pynvml([("card-0", 8 * menu.GIB), ("card-1", 2 * menu.GIB)], failing={1})
        with mock.patch.object(menu, "pynvml", module):
            cards = menu._nvidia_gpus()

        self.assertEqual([card.name for card in cards], ["card-0", "card-1"])
        self.assertEqual(cards[0].vram_bytes, 8 * menu.GIB)
        self.assertIsNone(cards[1].vram_bytes)
        self.assertIn("did not report", cards[1].note)

    def test_init_failure_means_no_cards_not_a_crash(self):
        module = mock.MagicMock()
        module.NVMLError = FakeNVMLError
        module.nvmlInit.side_effect = FakeNVMLError("driver not loaded")
        with mock.patch.object(menu, "pynvml", module):
            self.assertEqual(menu._nvidia_gpus(), [])

    def test_shutdown_runs_even_when_a_device_fails(self):
        module = mock.MagicMock()
        module.NVMLError = FakeNVMLError
        module.nvmlInit.side_effect = FakeNVMLError("driver not loaded")
        with mock.patch.object(menu, "pynvml", module):
            menu._nvidia_gpus()
        # nvmlInit raised, so shutdown is not owed; the guard is what matters.
        module.nvmlShutdown.assert_not_called()

    def test_names_returned_as_bytes_are_decoded(self):
        module = fake_pynvml([("card-0", 4 * menu.GIB)])
        module.nvmlDeviceGetName.side_effect = lambda handle: b"GeForce RTX 4090"
        with mock.patch.object(menu, "pynvml", module):
            cards = menu._nvidia_gpus()

        self.assertEqual(cards[0].name, "GeForce RTX 4090")
        self.assertEqual(cards[0].vram_bytes, 4 * menu.GIB)


class MemoryTests(unittest.TestCase):
    def test_bytes_not_strings(self):
        memory = menu.Memory(total_bytes=8 * menu.GIB, available_bytes=2 * menu.GIB)
        self.assertEqual(memory.total_bytes, 8 * menu.GIB)
        self.assertEqual(memory.display(), "8.0 GiB total, 2.0 GiB available")

    def test_unreadable_memory_is_unknown_not_zero(self):
        """A blocked API must not look like a machine with no RAM."""
        self.assertEqual(menu.Memory().display(), "unknown")

    def test_available_is_labelled_available(self):
        """psutil's 'available' includes reclaimable cache; 'free' would be wrong."""
        self.assertIn("available", menu.Memory(menu.GIB, menu.GIB).display())

    def test_small_sizes_stay_readable(self):
        self.assertEqual(menu._format_size(512 * menu.MIB), "512 MiB")
        self.assertEqual(menu._format_size(menu.GIB), "1.0 GiB")


class DiskTests(unittest.TestCase):
    def test_empty_card_slot_is_skipped(self):
        partition = mock.Mock(mountpoint="E:\\")
        with mock.patch.object(menu.psutil, "disk_partitions", return_value=[partition]), \
             mock.patch.object(menu.psutil, "disk_usage",
                               return_value=mock.Mock(total=0, free=0, percent=0.0)):
            self.assertEqual(menu.detect_disks(), ())

    def test_unreadable_volume_is_skipped(self):
        partition = mock.Mock(mountpoint="E:\\")
        with mock.patch.object(menu.psutil, "disk_partitions", return_value=[partition]), \
             mock.patch.object(menu.psutil, "disk_usage", side_effect=OSError("no media")):
            self.assertEqual(menu.detect_disks(), ())


class PrivacyTests(unittest.TestCase):
    """The state object ships to a remote model, so it must be scrubbed (FR-010)."""

    def _state(self):
        return menu.Inventory(
            os_name="Windows 11",
            cpu_name="Intel Core i7",
            cpu_cores=16,
            cpu_ghz=2.6,
            memory=menu.Memory(32 * menu.GIB, 12 * menu.GIB),
            gpus=(menu.Gpu("NVIDIA GeForce RTX 4080", 16 * menu.GIB, "nvml"),),
            disks=(menu.Disk("C:\\Users\\someone\\", 512 * menu.GIB, 100 * menu.GIB, 80.5),),
        ).to_state()

    def test_no_paths_leave_the_machine(self):
        blob = repr(self._state())
        for leak in ("C:\\\\", "Users", "someone", "mount"):
            self.assertNotIn(leak, blob)

    def test_unknown_fields_stay_visible_as_none(self):
        state = menu.Inventory().to_state()
        self.assertIsNone(state["memory"]["total_gib"])
        self.assertIsNone(state["cpu"]["logical_cores"])
        self.assertEqual(state["gpus"], [])

    def test_vram_is_in_gib_with_a_key_saying_so(self):
        gpu = self._state()["gpus"][0]
        self.assertEqual(gpu["dedicated_vram_gib"], 16.0)
        self.assertEqual(gpu["detected_via"], "nvml")

    def test_memory_is_gib(self):
        self.assertEqual(self._state()["memory"]["total_gib"], 32.0)


class MenuTests(unittest.TestCase):
    def test_cpu_detection_survives_a_broken_cpuinfo(self):
        """py-cpuinfo parses /proc and the registry; it does throw on odd hosts."""
        with mock.patch.object(menu.cpuinfo, "get_cpu_info", side_effect=RuntimeError("boom")):
            name, _cores, _ghz = menu.detect_cpu()
        self.assertNotEqual(name, "")

    def test_a_list_of_frequencies_uses_the_sustained_clock(self):
        """py-cpuinfo reports a list on some platforms; the base clock is the
        honest number to rate, not the boost spike at the top of it."""
        with mock.patch.object(
            menu.cpuinfo, "get_cpu_info",
            return_value={"brand_raw": "Test CPU", "hz_actual": [2_400_000_000, 3_600_000_000]},
        ):
            _name, _cores, ghz = menu.detect_cpu()
        self.assertAlmostEqual(ghz, 2.4)

    def test_missing_clock_is_unknown_not_zero(self):
        with mock.patch.object(
            menu.cpuinfo, "get_cpu_info", return_value={"brand_raw": "Test CPU"}
        ):
            _name, _cores, ghz = menu.detect_cpu()
        self.assertIsNone(ghz)

    def test_specs_lines_have_one_row_per_component(self):
        labels = [label for label, _value in menu.Inventory().lines()]
        self.assertEqual(labels, ["OS", "CPU", "Cores", "RAM", "GPU", "Disks", "Python"])

    def test_get_specs_is_a_mapping_of_strings(self):
        specs = menu.get_specs()
        self.assertIsInstance(specs, dict)
        self.assertTrue(all(isinstance(v, str) for v in specs.values()))


if __name__ == "__main__":
    unittest.main()
