"""Offline tests for the detection layer in ``inventory``.

These are regression tests for real bugs, not coverage for its own sake:

* ``get_gpus`` used to append bare names on Windows and then unpack every entry
  as ``(name, vram)``. On any Windows machine whose only adapter was not an
  NVIDIA card, showing the specs raised
  ``ValueError: too many values to unpack (expected 2)``.
* One unreadable device used to make ``_nvidia_gpus`` throw away every card.
* VRAM used to be a display string, so a shared 2 GB could not be told apart
  from a dedicated 2 GB -- and neither could the decision layer, which is
  exactly the distinction the decision layer needs kept.

Run:  python -m unittest discover -s tests
"""

import os
import types
import unittest
from unittest import mock
import inventory


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


def _fake_gpuinfo(name, vendor=None, memory_mb=None, source=None):
    """Build a lightweight stand-in for gpu_detector.models.GPUInfo."""
    info = types.SimpleNamespace(name=name, vendor=vendor, memory_mb=memory_mb)
    info.source = source or (f"sysfs:/sys/class/drm" if vendor == "AMD" else "platform")
    return info


class GpuDetectionTests(unittest.TestCase):
    """GPU detection across Windows, Linux, and macOS via gpu-detector."""

    def _with_platform(self, nvidia, platform_infos, system_name="Windows"):
        """Patch _nvidia_gpus and _detect_platform_gpus with controlled inputs."""
        fake_platform = mock.Mock()
        fake_platform.system = mock.Mock(return_value=system_name)
        return mock.patch.multiple(
            inventory,
            _nvidia_gpus=mock.Mock(return_value=nvidia),
            _detect_platform_gpus=mock.Mock(return_value=platform_infos),
            platform=fake_platform,
        )

    def test_windows_non_nvidia_adapter_does_not_crash(self):
        """The P0: pynvml finds nothing, platform reports Intel graphics."""
        intel = _fake_gpuinfo("Intel(R) UHD Graphics 620", vendor="Intel")
        with self._with_platform([], [intel], system_name="Windows"):
            gpus = inventory.detect_gpus()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].name, "Intel(R) UHD Graphics 620")
        self.assertEqual(gpus[0].source, "platform")

    def test_non_nvidia_adapter_without_vram_is_unknown_not_zero(self):
        """Shared memory is not a number we have, and guessing one is a lie."""
        intel = _fake_gpuinfo("Intel(R) UHD Graphics 620", vendor="Intel")
        with self._with_platform([], [intel], system_name="Windows"):
            gpus = inventory.detect_gpus()

        self.assertIsNone(gpus[0].vram_bytes)
        self.assertIn("unknown", gpus[0].display())

    def test_nvidia_and_platform_entries_do_not_duplicate(self):
        """pynvml spells it 'NVIDIA GeForce ...', platform drops the prefix."""
        nvidia_card = inventory.Gpu(name="NVIDIA GeForce RTX 3070", vram_bytes=8 * inventory.GIB, source="nvml")
        intel = _fake_gpuinfo("Intel(R) UHD Graphics 620", vendor="Intel")
        with self._with_platform([nvidia_card], [intel], system_name="Windows"):
            gpus = inventory.detect_gpus()

        names = [gpu.name for gpu in gpus]
        self.assertEqual(len(names), 2, names)
        self.assertEqual(names[0], "NVIDIA GeForce RTX 3070")

    def test_software_adapters_are_dropped(self):
        """A VM has no real GPU, so 'unknown' is recorded rather than invented."""
        software1 = _fake_gpuinfo("Microsoft Basic Render Driver", vendor="Microsoft")
        software2 = _fake_gpuinfo("VMware SVGA 3D", vendor="VMware")
        with self._with_platform([], [software1, software2], system_name="Windows"):
            gpus = inventory.detect_gpus()

        self.assertEqual(gpus, ())

    def test_virtual_vendor_adapters_are_dropped(self):
        """gpu-detector's vendor field marks these as virtual, not hardware."""
        vmware = _fake_gpuinfo("VMware SVGA 3D", vendor="VMware")
        qemu = _fake_gpuinfo("QXL", vendor="QEMU/Bochs")
        with self._with_platform([], [vmware, qemu], system_name="Linux"):
            gpus = inventory.detect_gpus()

        self.assertEqual(gpus, ())

    def test_linux_amd_gpu_with_vram(self):
        """On Linux, AMD VRAM from sysfs is trusted and converted to bytes."""
        amd = _fake_gpuinfo("AMD Radeon RX 7800 XT", vendor="AMD",
                            memory_mb=16384, source="sysfs:/sys/class/drm")
        with self._with_platform([], [amd], system_name="Linux"):
            gpus = inventory.detect_gpus()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].name, "AMD Radeon RX 7800 XT")
        self.assertEqual(gpus[0].vram_bytes, 16384 * inventory.MIB)
        self.assertEqual(gpus[0].source, "sysfs")
        self.assertEqual(gpus[0].note, "")

    def test_macos_intel_gpu_without_dedicated_vram(self):
        """On macOS, shared-memory VRAM from system_profiler is not trusted."""
        intel = _fake_gpuinfo("Intel Iris Xe Graphics", vendor="Intel",
                            memory_mb=1536, source="system_profiler:SPDisplaysDataType -json")
        with self._with_platform([], [intel], system_name="Darwin"):
            gpus = inventory.detect_gpus()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].name, "Intel Iris Xe Graphics")
        # macOS VRAM is not trusted -> None
        self.assertIsNone(gpus[0].vram_bytes)
        self.assertEqual(gpus[0].source, "platform")
        self.assertIn("dedicated", gpus[0].note)


class NvmlTests(unittest.TestCase):
    def test_one_bad_device_keeps_the_other_cards(self):
        """A single unreadable GPU must not hide the whole machine's cards."""
        module = fake_pynvml([("card-0", 8 * inventory.GIB), ("card-1", 2 * inventory.GIB)], failing={1})
        with mock.patch.object(inventory, "pynvml", module):
            cards = inventory._nvidia_gpus()

        self.assertEqual([card.name for card in cards], ["card-0", "card-1"])
        self.assertEqual(cards[0].vram_bytes, 8 * inventory.GIB)
        self.assertIsNone(cards[1].vram_bytes)
        self.assertIn("did not report", cards[1].note)

    def test_init_failure_means_no_cards_not_a_crash(self):
        module = mock.MagicMock()
        module.NVMLError = FakeNVMLError
        module.nvmlInit.side_effect = FakeNVMLError("driver not loaded")
        with mock.patch.object(inventory, "pynvml", module):
            self.assertEqual(inventory._nvidia_gpus(), [])

    def test_shutdown_runs_even_when_a_device_fails(self):
        module = mock.MagicMock()
        module.NVMLError = FakeNVMLError
        module.nvmlInit.side_effect = FakeNVMLError("driver not loaded")
        with mock.patch.object(inventory, "pynvml", module):
            inventory._nvidia_gpus()
        # nvmlInit raised, so shutdown is not owed; the guard is what matters.
        module.nvmlShutdown.assert_not_called()

    def test_names_returned_as_bytes_are_decoded(self):
        module = fake_pynvml([("card-0", 4 * inventory.GIB)])
        module.nvmlDeviceGetName.side_effect = lambda handle: b"GeForce RTX 4090"
        with mock.patch.object(inventory, "pynvml", module):
            cards = inventory._nvidia_gpus()

        self.assertEqual(cards[0].name, "GeForce RTX 4090")
        self.assertEqual(cards[0].vram_bytes, 4 * inventory.GIB)


class MemoryTests(unittest.TestCase):
    def test_bytes_not_strings(self):
        memory = inventory.Memory(total_bytes=8 * inventory.GIB, available_bytes=2 * inventory.GIB)
        self.assertEqual(memory.total_bytes, 8 * inventory.GIB)
        self.assertEqual(memory.display(), "8.0 GiB total, 2.0 GiB available")

    def test_unreadable_memory_is_unknown_not_zero(self):
        """A blocked API must not look like a machine with no RAM."""
        self.assertEqual(inventory.Memory().display(), "unknown")

    def test_available_is_labelled_available(self):
        """psutil's 'available' includes reclaimable cache; 'free' would be wrong."""
        self.assertIn("available", inventory.Memory(inventory.GIB, inventory.GIB).display())

    def test_live_usage_is_shown_with_the_capacity(self):
        memory = inventory.Memory(
            total_bytes=8 * inventory.GIB, available_bytes=2 * inventory.GIB, used_percent=75.0
        )
        self.assertEqual(memory.display(), "8.0 GiB total, 2.0 GiB available (75% used)")

    def test_a_missing_usage_reading_is_not_shown_as_zero(self):
        """No reading means no number; a measured zero is still a number."""
        no_reading = inventory.Memory(total_bytes=8 * inventory.GIB, available_bytes=2 * inventory.GIB)
        measured_zero = inventory.Memory(
            total_bytes=8 * inventory.GIB, available_bytes=8 * inventory.GIB, used_percent=0.0
        )
        self.assertNotIn("used", no_reading.display())
        self.assertIn("(0% used)", measured_zero.display())

    def test_detection_carries_the_usage_reading(self):
        fake = mock.Mock(total=16 * inventory.GIB, available=12 * inventory.GIB, percent=25.0)
        with mock.patch.object(inventory.psutil, "virtual_memory", return_value=fake):
            memory = inventory.detect_memory()
        self.assertEqual(memory.used_percent, 25.0)

    def test_small_sizes_stay_readable(self):
        self.assertEqual(inventory._format_size(512 * inventory.MIB), "512 MiB")
        self.assertEqual(inventory._format_size(inventory.GIB), "1.0 GiB")


class DiskTests(unittest.TestCase):
    def test_empty_card_slot_is_skipped(self):
        partition = mock.Mock(mountpoint="E:\\")
        with mock.patch.object(inventory.psutil, "disk_partitions", return_value=[partition]), \
             mock.patch.object(inventory.psutil, "disk_usage",
                               return_value=mock.Mock(total=0, free=0, percent=0.0)):
            self.assertEqual(inventory.detect_disks(), ())

    def test_unreadable_volume_is_skipped(self):
        partition = mock.Mock(mountpoint="E:\\")
        with mock.patch.object(inventory.psutil, "disk_partitions", return_value=[partition]), \
             mock.patch.object(inventory.psutil, "disk_usage", side_effect=OSError("no media")):
            self.assertEqual(inventory.detect_disks(), ())


class PrivacyTests(unittest.TestCase):
    """The state object ships to a remote model, so it must be scrubbed."""

    def _state(self):
        return inventory.Inventory(
            os_name="Windows 11",
            cpu_name="Intel Core i7",
            cpu_cores=16,
            cpu_ghz=2.6,
            memory=inventory.Memory(32 * inventory.GIB, 12 * inventory.GIB),
            gpus=(inventory.Gpu("NVIDIA GeForce RTX 4080", 16 * inventory.GIB, "nvml"),),
            disks=(inventory.Disk("C:\\Users\\someone\\", 512 * inventory.GIB, 100 * inventory.GIB, 80.5),),
        ).to_state()

    def test_no_paths_leave_the_machine(self):
        blob = repr(self._state())
        for leak in ("C:\\\\", "Users", "someone", "mount"):
            self.assertNotIn(leak, blob)

    def test_unknown_fields_stay_visible_as_none(self):
        state = inventory.Inventory().to_state()
        self.assertIsNone(state["memory"]["total_gib"])
        self.assertIsNone(state["cpu"]["logical_cores"])
        self.assertEqual(state["gpus"], [])

    def test_vram_is_in_gib_with_a_key_saying_so(self):
        gpu = self._state()["gpus"][0]
        self.assertEqual(gpu["dedicated_vram_gib"], 16.0)
        self.assertEqual(gpu["detected_via"], "nvml")

    def test_memory_is_gib(self):
        self.assertEqual(self._state()["memory"]["total_gib"], 32.0)

    def test_live_readings_stay_out_of_the_payload(self):
        """Usage describes the moment, not the machine, so it never ships.

        The capability score is a machine property (which part is
        installed), so it does ship; the usage percentage is a live
        reading, so it does not.
        """
        state = inventory.Inventory(
            cpu_usage_percent=23.4,
            memory=inventory.Memory(8 * inventory.GIB, 2 * inventory.GIB, 75.0),
        ).to_state()
        self.assertEqual(
            set(state["cpu"]), {"name", "logical_cores", "clock_ghz", "capability_score"}
        )
        self.assertIsNone(state["cpu"]["capability_score"])
        self.assertEqual(set(state["memory"]), {"total_gib", "available_gib"})

    def test_the_capability_score_rides_in_the_payload(self):
        """The tier is what every model call receives as grounding."""
        state = inventory.Inventory(
            cpu_name="AMD Ryzen 5 3600",
            cpu_capability=57.0,
            gpus=(inventory.Gpu("NVIDIA GeForce RTX 3060", 12 * inventory.GIB,
                                "nvml", "", 43.0),),
        ).to_state()
        self.assertEqual(state["cpu"]["capability_score"], 57.0)
        self.assertEqual(state["gpus"][0]["capability_score"], 43.0)


class MenuTests(unittest.TestCase):
    def test_cpu_detection_survives_a_broken_cpuinfo(self):
        """py-cpuinfo parses /proc and the registry; it does throw on odd hosts."""
        with mock.patch.object(inventory.cpuinfo, "get_cpu_info", side_effect=RuntimeError("boom")):
            name, _cores, _ghz = inventory.detect_cpu()
        self.assertNotEqual(name, "")

    def test_a_list_of_frequencies_uses_the_sustained_clock(self):
        """py-cpuinfo reports a list on some platforms; the base clock is the
        honest number to rate, not the boost spike at the top of it."""
        with mock.patch.object(
            inventory.cpuinfo, "get_cpu_info",
            return_value={"brand_raw": "Test CPU", "hz_actual": [2_400_000_000, 3_600_000_000]},
        ):
            _name, _cores, ghz = inventory.detect_cpu()
        self.assertAlmostEqual(ghz, 2.4)

    def test_missing_clock_is_unknown_not_zero(self):
        with mock.patch.object(
            inventory.cpuinfo, "get_cpu_info", return_value={"brand_raw": "Test CPU"}
        ):
            _name, _cores, ghz = inventory.detect_cpu()
        self.assertIsNone(ghz)

    def test_cpu_usage_comes_from_a_real_sample_window(self):
        """The no-interval call returns a placeholder 0.0 on its first use, and
        a placeholder printed as "0% busy" would be an invented reading."""
        with mock.patch.object(inventory.psutil, "cpu_percent", return_value=23.4) as percent:
            self.assertEqual(inventory.detect_cpu_usage(), 23.4)
        self.assertIsNotNone(percent.call_args.kwargs.get("interval"))

    def test_unreadable_cpu_usage_is_none_not_zero(self):
        blocked = inventory.psutil.Error("no access")
        with mock.patch.object(inventory.psutil, "cpu_percent", side_effect=blocked):
            self.assertIsNone(inventory.detect_cpu_usage())

    def test_specs_lines_have_one_row_per_component(self):
        labels = [label for label, _value in inventory.Inventory().lines()]
        self.assertEqual(labels, ["OS", "CPU", "Cores", "RAM", "GPU", "Disks", "Python"])

    def test_the_cores_line_carries_the_live_reading(self):
        lines = dict(inventory.Inventory(cpu_cores=16, cpu_usage_percent=23.4).lines())
        self.assertEqual(lines["Cores"], "16 logical, 23% busy")

    def test_the_cores_line_has_no_number_when_nothing_was_read(self):
        lines = dict(inventory.Inventory(cpu_cores=16).lines())
        self.assertEqual(lines["Cores"], "16 logical")


if __name__ == "__main__":
    unittest.main()
