"""Offline tests for the curated capability rankings.

The promises under test:

* name matching is conservative -- exact after normalization, a
  documented laptop penalty, and ``None`` (unknown) for anything
  the tables do not place;
* a named-part requirement whose part and machine are both in the
  tables is settled by arithmetic, with no model call;
* a named part the tables cannot place falls back to the model,
  exactly as before;
* the scores ride in the state payload, show in the display, can
  be overridden through the config, and are dated by a version.
"""

from __future__ import annotations

import json
import types
import unittest
from unittest import mock

import config as config_module
import inventory
import rankings
import requirements
from tests._helpers import FakeClient, TempDir, noul


def _ranks_at_or_above(faster, slower):
    """True when *faster* is known and ranks at or above *slower*.

    A tie is legitimate: the hierarchies agree "at or above".
    """
    return (faster is not None and slower is not None
            and faster >= slower)


class MatchingTests(unittest.TestCase):
    """The lookup: normalization, variants, laptop penalty, unknown."""

    def test_exact_name_matches(self):
        self.assertEqual(rankings.gpu_score("RTX 3060"), 43.0)
        self.assertEqual(rankings.cpu_score("Ryzen 5 3600"), 57.0)

    def test_vendor_prefixes_and_case_do_not_matter(self):
        self.assertEqual(rankings.gpu_score("NVIDIA GeForce RTX 3060"), 43.0)
        self.assertEqual(rankings.gpu_score("nvidia geforce rtx 3060"), 43.0)
        self.assertEqual(rankings.gpu_score("AMD Radeon RX 580"), 27.0)
        self.assertEqual(
            rankings.cpu_score("11th Gen Intel(R) Core(TM) i5-1135G7"), 62.0
        )

    def test_a_suffix_variant_is_a_different_card(self):
        """The Ti is faster than the base card, so it must not match it."""
        self.assertEqual(rankings.gpu_score("RTX 3060 Ti"), 52.0)
        self.assertNotEqual(rankings.gpu_score("RTX 3060 Ti"), 43.0)

    def test_prose_around_the_part_is_absorbed(self):
        self.assertEqual(rankings.gpu_score("GTX 970 or better"), 18.0)
        self.assertEqual(rankings.gpu_score("at least a GTX 970"), 18.0)

    def test_laptop_variants_get_the_documented_penalty(self):
        """A laptop card is its desktop equivalent at x0.8."""
        self.assertEqual(rankings.gpu_score("RTX 3060 Laptop GPU"), 34.4)
        self.assertEqual(rankings.gpu_score("GeForce RTX 3060 Max-Q"), 34.4)
        self.assertEqual(rankings.gpu_score("RTX 3060 Mobile"), 34.4)
        self.assertEqual(rankings.gpu_score("RTX 3060 M"), 34.4)

    def test_an_f_sku_is_the_same_chip(self):
        """An F variant is the same CPU without iGPU: same score."""
        self.assertEqual(rankings.cpu_score("Core i5-12400F"), 69.0)
        self.assertEqual(rankings.cpu_score("Core i7-13700KF"), 85.0)
        self.assertEqual(rankings.cpu_score("14th Gen Intel Core i5-14400F"), 73.0)
        # An F SKU scores the same as its base chip, not the K.
        self.assertEqual(rankings.cpu_score("Core i5-12400F"),
                         rankings.cpu_score("Core i5-12400"))
        self.assertNotEqual(rankings.cpu_score("Core i7-13700KF"),
                            rankings.cpu_score("Core i7-13700"))

    def test_a_laptop_only_part_has_an_entry_of_its_own(self):
        """The GeForce 610M has no desktop equivalent to penalize."""
        self.assertEqual(rankings.gpu_score("GeForce 610M"), 6.0)
        self.assertEqual(rankings.gpu_score("NVIDIA GeForce 610M"), 6.0)

    def test_an_unknown_part_is_none_never_a_guess(self):
        self.assertIsNone(rankings.gpu_score("GTX Mystery 9000"))
        self.assertIsNone(rankings.gpu_score(""))
        self.assertIsNone(rankings.cpu_score("A CPU That Does Not Exist"))
        self.assertIsNone(rankings.cpu_score("Intel Core i5"))  # which one?

    def test_a_family_word_is_not_required(self):
        """The family word folds away, so "i7-13700KF" matches."""
        self.assertEqual(rankings.cpu_score("i7-13700KF"), 85.0)
        self.assertEqual(rankings.cpu_score("ryzen 5 5600X"), 65.0)
        self.assertEqual(rankings.cpu_score("pentium gold g7400"), 20.0)
        self.assertEqual(rankings.cpu_score("atom x5-z8350"), 9.0)

    def test_a_bare_model_number_matches_one_part(self):
        """A model number that names exactly one part is enough."""
        self.assertEqual(rankings.cpu_score("13700K"), 85.0)
        self.assertEqual(rankings.cpu_score("5600X"), 65.0)
        # "4060" names two cards (the RTX 4060 and the RTX
        # 4060 Ti), so it stays unknown rather than picking
        # one; "7900" names three.
        self.assertIsNone(rankings.gpu_score("4060"))
        self.assertIsNone(rankings.gpu_score("7900"))

    def test_a_d_variant_is_the_same_card(self):
        """A China "D" variant is the same capability, no penalty."""
        self.assertEqual(rankings.gpu_score("RTX 4090D"), 93.0)
        self.assertEqual(rankings.gpu_score("RTX 4090 D"), 93.0)

    def test_a_glued_mobile_marker_is_a_laptop_variant(self):
        """A glued M ("RTX 4080M") is the desktop card at x0.8."""
        self.assertEqual(rankings.gpu_score("RTX 4080M"), 71.2)
        self.assertEqual(rankings.gpu_score("GTX 980M"), 16.0)
        self.assertEqual(rankings.gpu_score("Radeon RX 6800M"), 55.2)
        # Laptop-only parts have entries of their own: the glued
        # M stays glued, the first pass matches, no penalty.
        self.assertEqual(rankings.gpu_score("GeForce 610M"), 6.0)
        self.assertEqual(rankings.gpu_score("GT 740M"), 9.0)
        self.assertEqual(rankings.gpu_score("GTX 960M"), 12.0)

    def test_the_extended_cpu_entries_are_placed(self):
        """The HX and older chips the table was extended with.

        These are estimates calibrated against the nearest
        anchors, not measured scores.
        """
        self.assertEqual(rankings.cpu_score("Core Ultra 9 285HX"), 93.0)
        self.assertEqual(rankings.cpu_score("Core i9-14900HX"), 89.0)
        self.assertEqual(rankings.cpu_score("Core i7-14700HX"), 87.0)
        self.assertEqual(rankings.cpu_score("Core i9-13900HX"), 88.0)
        self.assertEqual(rankings.cpu_score("Core i7-13700HX"), 84.0)
        self.assertEqual(rankings.cpu_score("Ryzen 9 7945HX"), 87.0)
        self.assertEqual(rankings.cpu_score("Ryzen 9 7940HX"), 86.0)
        self.assertEqual(rankings.cpu_score("Core i7-10750H"), 68.0)
        self.assertEqual(rankings.cpu_score("Core i5-8250U"), 45.0)
        self.assertEqual(rankings.cpu_score("Core i5-7200U"), 40.0)
        self.assertEqual(rankings.cpu_score("Core i5-4200U"), 23.0)
        self.assertEqual(rankings.cpu_score("Pentium G4560"), 30.0)
        self.assertEqual(rankings.cpu_score("AMD FX-8350"), 20.0)
        self.assertEqual(rankings.cpu_score("Core 2 Duo E8400"), 12.0)

    def test_describe_names_the_part_itself(self):
        self.assertEqual(
            rankings.describe("RTX 3060"),
            "mid-range, ~43/100 (comparable to a RTX 3060)",
        )
        self.assertEqual(
            rankings.describe("GTX 1060"),
            "entry-level, ~25/100 (comparable to a GTX 1060)",
        )
        self.assertEqual(
            rankings.describe("Ryzen 5 3600"),
            "mid-range, ~57/100 (comparable to a Ryzen 5 3600)",
        )
        # Both the GTX 970 and the GTX 1050 Ti sit at
        # 18.0: the card anchors to itself, not to the
        # first entry at its score.
        self.assertEqual(
            rankings.describe("GTX 970"),
            "entry-level, ~18/100 (comparable to a GTX 970)",
        )
        self.assertIsNone(rankings.describe("GTX Mystery 9000"))

    def test_describe_uses_the_nearest_anchor_for_prose(self):
        # "GTX 970 or better" is not itself a table
        # entry, so the label anchors to the nearest
        # entry at its score.
        self.assertEqual(
            rankings.describe("GTX 970 or better"),
            "entry-level, ~18/100 (comparable to a GTX 1050 Ti)",
        )

    def test_a_mobile_variant_names_its_desktop_twin(self):
        self.assertEqual(
            rankings.describe("GeForce GTX 1080 Max-Q"),
            "entry-level, ~28/100 (a GTX 1080 at laptop clocks)",
        )
        self.assertEqual(
            rankings.describe("RTX 4080M"),
            "high-end, ~71.2/100 (a RTX 4080 at laptop clocks)",
        )

    def test_describe_honors_a_table_override(self):
        config = config_module.Config(
            {"gpu_rankings": [["My GPU", 50.0]]}
        )
        self.assertEqual(
            rankings.describe("My GPU", config),
            "mid-range, ~50/100 (comparable to a My GPU)",
        )
        # Without the config the built-in table does not know it.
        self.assertIsNone(rankings.describe("My GPU"))

    def test_tier_text_and_words(self):
        self.assertEqual(rankings.tier_text(43.0), "mid-range 43/100")
        self.assertEqual(rankings.tier_text(100.0), "flagship 100/100")
        self.assertEqual(rankings.tier_text(None), "unknown")
        self.assertEqual(rankings.tier_word(95.0), "flagship")
        self.assertEqual(rankings.tier_word(75.0), "high-end")
        self.assertEqual(rankings.tier_word(43.0), "mid-range")
        self.assertEqual(rankings.tier_word(25.0), "entry-level")
        self.assertEqual(rankings.tier_word(15.0), "entry-level")


class AnchorTests(unittest.TestCase):
    """The scale is only meaningful if its anchors hold."""

    def test_the_flagship_is_a_hundred(self):
        self.assertEqual(rankings.gpu_score("RTX 5090"), 100.0)
        self.assertEqual(rankings.cpu_score("Ryzen 9 9950X3D"), 100.0)

    def test_the_documented_anchors(self):
        self.assertEqual(rankings.gpu_score("GTX 1060"), 25.0)
        self.assertEqual(rankings.gpu_score("GTX 970"), 18.0)
        self.assertEqual(rankings.gpu_score("GeForce 610M"), 6.0)
        self.assertEqual(rankings.cpu_score("Ryzen 5 3600"), 57.0)
        self.assertEqual(rankings.cpu_score("Core i5-2450M"), 21.0)

    def test_the_rtx_4060_class_sits_mid_table(self):
        score = rankings.gpu_score("RTX 4060")
        self.assertIsNotNone(score)
        self.assertTrue(48.0 <= score <= 52.0, score)

    def test_the_ordering_follows_the_published_hierarchies(self):
        """Where a source measured both parts, the table must agree.

        Every pair is checked and every inversion reported, so
        one bad entry cannot hide the ones after it.
        """
        gpu = rankings.gpu_score
        # Tom's 2026 GPU hierarchy, measured percentages of an RTX 4090.
        hierarchy = (
            ("RTX 5090", "RTX 4090"),
            ("RTX 4090", "RX 7900 XTX"),
            ("RX 7900 XTX", "RTX 4080 Super"),
            ("RTX 4080 Super", "RTX 4080"),
            ("RTX 4080", "RX 7900 XT"),
            ("RX 7900 XT", "RTX 4070 Ti Super"),
            ("RTX 4070 Ti Super", "RTX 4070 Super"),
            ("RTX 4070 Super", "RTX 4070 Ti"),
            ("RTX 4070 Ti", "RTX 3090 Ti"),
            ("RTX 3090 Ti", "RTX 3090"),
            ("RTX 3090", "RX 7800 XT"),
            ("RX 7800 XT", "RTX 3080 Ti"),
            ("RTX 3080 Ti", "RTX 3080"),
            ("RTX 3080", "RX 7700 XT"),
            ("RX 7700 XT", "RTX 3070 Ti"),
            ("RTX 3070 Ti", "RX 6750 XT"),
            ("RX 6750 XT", "RTX 4060 Ti"),
            ("RTX 4060 Ti", "RTX 2080 Ti"),
            ("RTX 2080 Ti", "RX 6700 XT"),
            ("RX 6700 XT", "RTX 3070"),
            ("RTX 3070", "RTX 2080"),
            ("RTX 2080", "RX 6650 XT"),
            ("RX 6650 XT", "RTX 3060 Ti"),
            ("RTX 3060 Ti", "RX 6600 XT"),
            ("RX 6600 XT", "RTX 2070 Super"),
            ("RTX 2070 Super", "RTX 4060"),
            ("RTX 4060", "RTX 2070"),
            ("RTX 2070", "RX 7600"),
            ("RX 7600", "RTX 2060 Super"),
            ("RTX 2060 Super", "Arc A770"),
            ("Arc A770", "RX 5700 XT"),
            ("RX 5700 XT", "RTX 3060"),
            ("RTX 3060", "Arc A750"),
            ("Arc A750", "RTX 2060"),
            ("RTX 2060", "RX Vega 64"),
            ("RX Vega 64", "GTX 1080 Ti"),
            ("GTX 1080 Ti", "RX 6600"),
        )
        inversions = [
            f"{faster} ({gpu(faster)}) should rank at or "
            f"above {slower} ({gpu(slower)})"
            for faster, slower in hierarchy
            if not _ranks_at_or_above(gpu(faster), gpu(slower))
        ]
        self.assertEqual(inversions, [])
        # The classic cross-generation lines, still true after the
        # recalibration: a 2060 beats a 1080 Ti, a 1060 beats a 970.
        self.assertGreater(gpu("RTX 2060"), gpu("GTX 1080 Ti"))
        self.assertGreater(gpu("GTX 1060"), gpu("GTX 970"))
        self.assertGreaterEqual(gpu("GTX 1050 Ti"), gpu("GTX 970"))

    def test_the_cpu_ordering_follows_the_hierarchies(self):
        """Every pair is checked; all inversions are reported."""
        cpu = rankings.cpu_score
        hierarchy = (
            ("Ryzen 9 9950X3D", "Ryzen 7 9800X3D"),
            ("Ryzen 7 9800X3D", "Ryzen 9 9950X"),
            ("Ryzen 9 9950X", "Core Ultra 9 285K"),
            ("Core Ultra 9 285K", "Core i9-14900K"),
            ("Core i9-14900K", "Core i9-13900K"),
            ("Core i9-13900K", "Core i7-14700K"),
            ("Core i7-14700K", "Ryzen 9 7950X"),
            ("Ryzen 9 7950X", "Core i7-13700K"),
            ("Core i7-13700K", "Ryzen 7 7800X3D"),
            ("Ryzen 7 7800X3D", "Core i5-14600K"),
            ("Core i5-14600K", "Core i7-12700K"),
            ("Core i7-12700K", "Core i5-12600K"),
            ("Core i5-12600K", "Ryzen 7 5800X3D"),
            ("Ryzen 7 5800X3D", "Ryzen 7 5800X"),
            ("Ryzen 7 5800X", "Core i5-12400"),
            ("Core i5-12400", "Ryzen 5 5600X"),
            ("Ryzen 5 5600X", "Core i5-10400"),
            ("Core i5-10400", "Ryzen 5 3600"),
            ("Ryzen 5 3600", "Core i7-7700K"),
            ("Core i7-7700K", "Core i7-6700K"),
            ("Core i7-6700K", "Core i7-4790K"),
            ("Core i7-4790K", "Core i7-3770"),
            ("Core i7-3770", "Core i5-2500"),
            ("Core i5-2500", "Core i5-2450M"),
            ("Core i5-2450M", "Pentium Gold G7400"),
            ("Pentium Gold G7400", "Celeron G5905"),
            ("Celeron G5905", "Atom x5-Z8350"),
        )
        inversions = [
            f"{faster} ({cpu(faster)}) should rank at or "
            f"above {slower} ({cpu(slower)})"
            for faster, slower in hierarchy
            if not _ranks_at_or_above(cpu(faster), cpu(slower))
        ]
        self.assertEqual(inversions, [])

    def test_every_score_sits_on_the_scale(self):
        for _pattern, score in rankings.GPU_RANKINGS + rankings.CPU_RANKINGS:
            self.assertTrue(0.0 <= score <= 100.0, (score, _pattern))


class RequirementCheckTests(TempDir):
    """The deterministic tier comparison for named parts."""

    MACHINE = {
        "cpu": {"name": "AMD Ryzen 5 3600", "capability_score": 57.0},
        "gpus": [
            {"name": "NVIDIA GeForce RTX 3060", "capability_score": 43.0},
        ],
        "memory": {"total_gib": 16.0},
        "storage": [{"total_gib": 512.0, "free_gib": 200.0}],
    }

    def test_a_known_machine_beats_a_weaker_requirement_without_the_model(self):
        client = FakeClient(answers={"req_0": noul(0.99)})
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX 970", None, client
        )
        item = result.items[0]
        self.assertEqual(item.state, "passed")
        self.assertEqual(item.path, "tier-table")
        self.assertIn("GTX 970 ~ 18/100", item.detail)
        self.assertIn("RTX 3060 ~ 43/100", item.detail)
        self.assertIn("curated tier table v", item.detail)
        self.assertEqual(client.calls, [])  # no model call was spent

    def test_a_known_machine_fails_a_stronger_requirement(self):
        client = FakeClient(answers={"req_0": noul(0.99)})
        result = requirements.check_requirements(
            self.MACHINE, "task", "RTX 4080", None, client
        )
        item = result.items[0]
        self.assertEqual(item.state, "failed")
        self.assertEqual(item.path, "tier-table")
        self.assertIn("RTX 4080 ~ 89/100", item.detail)
        self.assertIn("RTX 3060 ~ 43/100", item.detail)
        self.assertEqual(result.verdict, "Below the minimum")
        self.assertEqual(client.calls, [])

    def test_an_equal_score_passes_the_at_least_test(self):
        result = requirements.check_requirements(
            self.MACHINE, "task", "RTX 3060", None, None
        )
        self.assertEqual(result.items[0].state, "passed")

    def test_cpu_named_parts_compare_on_the_cpu_table(self):
        result = requirements.check_requirements(
            self.MACHINE, "task", "Core i5-2450M", None, None
        )
        self.assertEqual(result.items[0].component, "cpu_model")
        self.assertEqual(result.items[0].state, "passed")
        self.assertEqual(result.items[0].path, "tier-table")
        result = requirements.check_requirements(
            self.MACHINE, "task", "Core i9-14900K", None, None
        )
        self.assertEqual(result.items[0].state, "failed")

    def test_requirement_prose_is_absorbed(self):
        result = requirements.check_requirements(
            self.MACHINE, "task", "NVIDIA GTX 970 or better", None, None
        )
        self.assertEqual(result.items[0].state, "passed")
        self.assertEqual(result.items[0].path, "tier-table")

    def test_recommended_named_parts_are_settled_too(self):
        result = requirements.check_requirements(
            self.MACHINE, "task", "4 GB RAM", "GTX 970", None
        )
        by_raw = {item.raw: item for item in result.items}
        self.assertEqual(by_raw["GTX 970"].state, "passed")
        self.assertEqual(by_raw["GTX 970"].path, "tier-table")

    def test_a_machine_without_scores_falls_back_to_the_model(self):
        """The state carries no capability scores: the model decides."""
        state = {
            "cpu": {"name": "AMD Ryzen 5 3600"},
            "gpus": [{"name": "NVIDIA GeForce RTX 3060"}],
        }
        client = FakeClient(answers={"req_0": noul(0.93)})
        result = requirements.check_requirements(state, "task", "GTX 970", None, client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["questions"]["req_0"]["type"], "noul")
        self.assertEqual(result.items[0].state, "passed")
        self.assertEqual(result.items[0].path, "model")

    def test_an_unknown_required_part_falls_back_to_the_model(self):
        client = FakeClient(answers={"req_0": noul(0.04)})
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX Mystery 9000", None, client
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.items[0].state, "failed")
        self.assertEqual(result.items[0].path, "model")

    def test_known_parts_never_spend_a_model_call(self):
        client = FakeClient(answers={"req_0": noul(0.5)})
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX 970; Core i5-2450M", None, client
        )
        self.assertEqual(client.calls, [])
        self.assertTrue(all(i.state == "passed" for i in result.items))
        self.assertEqual(result.confidence, 100)

    def test_an_outage_reports_zero_confidence(self):
        """A model that never answered judged nothing: 0, not 100."""
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX Mystery 9000", None, None
        )
        self.assertTrue(result.model_unavailable)
        self.assertEqual(result.items[0].state, "unknown")
        self.assertEqual(result.confidence, 0)

    def test_a_coin_flip_is_at_most_fifty_confidence(self):
        """A coin flip is 50% at best, whatever its state."""
        client = FakeClient(answers={"req_0": noul(0.5)})
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX Mystery 9000", None, client
        )
        self.assertEqual(result.items[0].state, "unknown")
        self.assertEqual(result.confidence, 50)

    def test_the_detail_names_the_configured_version(self):
        config = config_module.Config({"rankings_version": "2026-11"})
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX 970", None, None, config
        )
        self.assertIn("curated tier table v2026-11", result.items[0].detail)

    def test_unresolved_parts_share_one_model_call(self):
        """The fallback still batches: one call, one question per part."""
        client = FakeClient(answers={"req_0": noul(0.9), "req_1": noul(0.1)})
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX Mystery 9000; Ryzen Mystery 9000",
            None, client,
        )
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(set(client.calls[0]["questions"]), {"req_0", "req_1"})
        self.assertEqual(
            [item.state for item in result.items], ["passed", "failed"]
        )

    def test_judge_named_batches_by_questions_per_call(self):
        """QUESTIONS_PER_CALL bounds the fallback's batch size."""
        items = [
            requirements.CheckItem(
                tier="minimum", component="gpu_model", raw=f"GTX Mystery {index}",
                state="unknown",
            )
            for index in range(requirements.QUESTIONS_PER_CALL + 1)
        ]
        client = FakeClient(
            answers={f"req_{i}": noul(0.9) for i in range(len(items))}
        )
        notes: list[str] = []
        requirements._judge_named(
            items, list(range(len(items))), self.MACHINE, client, None, notes
        )
        self.assertEqual(
            [len(call["questions"]) for call in client.calls],
            [requirements.QUESTIONS_PER_CALL, 1],
        )
        self.assertTrue(all(item.state == "passed" for item in items))

    def test_a_coin_flip_from_the_model_is_still_unknown(self):
        client = FakeClient(answers={"req_0": noul(0.52)})
        result = requirements.check_requirements(
            self.MACHINE, "task", "GTX Mystery 9000", None, client
        )
        self.assertEqual(result.items[0].state, "unknown")
        self.assertEqual(result.verdict, "Cannot tell")

    def test_numeric_checks_keep_their_path_blank(self):
        result = requirements.check_requirements(
            self.MACHINE, "task", "8 GB RAM", None, None
        )
        self.assertEqual(result.items[0].path, "")


class DetectionTests(unittest.TestCase):
    """Detection fills in the capability scores."""

    def _with_platform(self, platform_infos, system_name="Windows"):
        fake_platform = mock.Mock()
        fake_platform.system = mock.Mock(return_value=system_name)
        return mock.patch.multiple(
            inventory,
            _nvidia_gpus=mock.Mock(return_value=[]),
            _detect_platform_gpus=mock.Mock(return_value=platform_infos),
            platform=fake_platform,
        )

    @staticmethod
    def _gpu_info(name, vendor):
        info = types.SimpleNamespace(name=name, vendor=vendor, memory_mb=None)
        info.source = "platform"
        return info

    def test_detect_gpus_looks_up_each_card(self):
        cards = [self._gpu_info("NVIDIA GeForce RTX 3060", "NVIDIA"),
                 self._gpu_info("Intel(R) UHD Graphics 620", "Intel")]
        with self._with_platform(cards):
            gpus = inventory.detect_gpus()

        self.assertEqual([gpu.capability for gpu in gpus], [43.0, 8.0])

    def test_an_unknown_card_keeps_capability_none(self):
        with self._with_platform([self._gpu_info("Mystery GPU", "NVIDIA")]):
            gpus = inventory.detect_gpus()

        self.assertIsNone(gpus[0].capability)

    def test_detect_inventory_scores_the_cpu(self):
        with mock.patch.object(
            inventory, "detect_cpu",
            return_value=("AMD Ryzen 5 3600", 6, 3.6),
        ), mock.patch.object(inventory, "detect_gpus", return_value=()):
            machine = inventory.detect_inventory()

        self.assertEqual(machine.cpu_capability, 57.0)

    def test_detect_inventory_honors_a_table_override(self):
        config = config_module.Config(
            {"cpu_rankings": [["AMD Ryzen 5 3600", 99.0]]}
        )
        with mock.patch.object(
            inventory, "detect_cpu",
            return_value=("AMD Ryzen 5 3600", 6, 3.6),
        ), mock.patch.object(inventory, "detect_gpus", return_value=()):
            machine = inventory.detect_inventory(config)

        self.assertEqual(machine.cpu_capability, 99.0)


class DisplayTests(unittest.TestCase):
    """The tier shows in the specs output."""

    def test_gpu_display_carries_the_tier(self):
        gpu = inventory.Gpu("GeForce RTX 3060", 12 * inventory.GIB, "nvml",
                            "", 43.0)
        self.assertEqual(
            gpu.display(), "GeForce RTX 3060 (12.0 GiB VRAM, mid-range 43/100)"
        )

    def test_gpu_display_says_unknown_when_unranked(self):
        gpu = inventory.Gpu("Mystery", 8 * inventory.GIB, "wmi")
        self.assertEqual(gpu.display(), "Mystery (8.0 GiB VRAM, unknown)")

    def test_cpu_line_carries_the_tier(self):
        machine = inventory.Inventory(
            cpu_name="AMD Ryzen 5 3600", cpu_ghz=3.6, cpu_capability=57.0
        )
        lines = dict(machine.lines())
        self.assertEqual(lines["CPU"], "AMD Ryzen 5 3600 @ 3.6 GHz (mid-range 57/100)")

    def test_an_unknown_cpu_stays_unknown(self):
        lines = dict(inventory.Inventory().lines())
        self.assertEqual(lines["CPU"], "unknown")


class ConfigTests(TempDir):
    """The tables can be overridden through the config file."""

    def test_the_builtin_tables_are_the_default(self):
        config = config_module.Config()
        self.assertEqual(config.gpu_rankings, rankings.GPU_RANKINGS)
        self.assertEqual(config.cpu_rankings, rankings.CPU_RANKINGS)

    def test_an_override_replaces_the_table(self):
        config = config_module.Config({
            "gpu_rankings": [["My GPU", 50.0], ["Better GPU", 80.0]],
        })
        self.assertEqual(rankings.gpu_score("My GPU", config), 50.0)
        self.assertEqual(rankings.gpu_score("Better GPU", config), 80.0)
        # A part outside the override is unknown, not silently
        # scored by the built-in table.
        self.assertIsNone(rankings.gpu_score("RTX 3060", config))

    def test_the_override_loads_from_a_file(self):
        path = self.write("config.json", json.dumps({
            "gpu_rankings": [["Custom Card", 42.0]],
        }))
        config = config_module.load(path)
        self.assertEqual(rankings.gpu_score("Custom Card", config), 42.0)

    def test_an_out_of_range_score_is_rejected(self):
        with self.assertRaises(config_module.ConfigError):
            config_module.load(self.write(
                "config.json", json.dumps({"gpu_rankings": [["Bad", 150.0]]})
            ))

    def test_an_empty_pattern_is_rejected(self):
        with self.assertRaises(config_module.ConfigError):
            config_module.load(self.write(
                "config.json", json.dumps({"gpu_rankings": [["", 10.0]]})
            ))

    def test_a_malformed_entry_is_rejected(self):
        with self.assertRaises(config_module.ConfigError):
            config_module.load(self.write(
                "config.json", json.dumps({"gpu_rankings": ["not-a-pair"]})
            ))
        with self.assertRaises(config_module.ConfigError):
            config_module.load(self.write(
                "config.json", json.dumps({"gpu_rankings": []})
            ))

    def test_an_unknown_key_is_still_rejected(self):
        with self.assertRaises(config_module.ConfigError):
            config_module.load(self.write(
                "config.json", json.dumps({"gpu_rankngs": []})
            ))

    def test_the_version_is_disclosed(self):
        document = config_module.Config().to_dict()
        self.assertEqual(document["rankings_version"], rankings.RANKINGS_VERSION)
        self.assertTrue(document["rankings_version"])

    def test_an_override_may_declare_its_own_version(self):
        path = self.write("config.json", json.dumps({
            "rankings_version": "2026-11",
            "gpu_rankings": [["Custom Card", 42.0]],
        }))
        config = config_module.load(path)
        self.assertEqual(config.rankings_version, "2026-11")
        self.assertEqual(config.to_dict()["rankings_version"], "2026-11")


if __name__ == "__main__":
    unittest.main()
