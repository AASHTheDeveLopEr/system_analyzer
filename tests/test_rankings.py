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


class MatchingTests(unittest.TestCase):
    """The lookup: normalization, variants, laptop penalty, unknown."""

    def test_exact_name_matches(self):
        self.assertEqual(rankings.gpu_score("RTX 3060"), 55.0)
        self.assertEqual(rankings.cpu_score("Ryzen 5 3600"), 45.0)

    def test_vendor_prefixes_and_case_do_not_matter(self):
        self.assertEqual(rankings.gpu_score("NVIDIA GeForce RTX 3060"), 55.0)
        self.assertEqual(rankings.gpu_score("nvidia geforce rtx 3060"), 55.0)
        self.assertEqual(rankings.gpu_score("AMD Radeon RX 580"), 29.0)
        self.assertEqual(
            rankings.cpu_score("11th Gen Intel(R) Core(TM) i5-1135G7"), 62.0
        )

    def test_a_suffix_variant_is_a_different_card(self):
        """The Ti is faster than the base card, so it must not match it."""
        self.assertEqual(rankings.gpu_score("RTX 3060 Ti"), 60.0)
        self.assertNotEqual(rankings.gpu_score("RTX 3060 Ti"), 55.0)

    def test_prose_around_the_part_is_absorbed(self):
        self.assertEqual(rankings.gpu_score("GTX 970 or better"), 28.0)
        self.assertEqual(rankings.gpu_score("at least a GTX 970"), 28.0)

    def test_laptop_variants_get_the_documented_penalty(self):
        """A laptop card is its desktop equivalent at x0.8, rounded."""
        self.assertEqual(rankings.gpu_score("RTX 3060 Laptop GPU"), 44.0)
        self.assertEqual(rankings.gpu_score("GeForce RTX 3060 Max-Q"), 44.0)
        self.assertEqual(rankings.gpu_score("RTX 3060 Mobile"), 44.0)
        self.assertEqual(rankings.gpu_score("RTX 3060 M"), 44.0)

    def test_a_laptop_only_part_has_an_entry_of_its_own(self):
        """The GeForce 610M has no desktop equivalent to penalize."""
        self.assertEqual(rankings.gpu_score("GeForce 610M"), 8.0)
        self.assertEqual(rankings.gpu_score("NVIDIA GeForce 610M"), 8.0)

    def test_an_unknown_part_is_none_never_a_guess(self):
        self.assertIsNone(rankings.gpu_score("GTX Mystery 9000"))
        self.assertIsNone(rankings.gpu_score(""))
        self.assertIsNone(rankings.cpu_score("A CPU That Does Not Exist"))
        self.assertIsNone(rankings.cpu_score("Intel Core i5"))  # which one?

    def test_describe_names_the_nearest_anchor(self):
        self.assertEqual(
            rankings.describe("RTX 3060"),
            "mid-range, ~55/100 (comparable to a RTX 3060)",
        )
        self.assertEqual(
            rankings.describe("GTX 1060"),
            "entry-level, ~35/100 (comparable to a GTX 1060)",
        )
        self.assertEqual(
            rankings.describe("Ryzen 5 3600"),
            "entry-level, ~45/100 (comparable to a Ryzen 5 3600)",
        )
        self.assertIsNone(rankings.describe("GTX Mystery 9000"))

    def test_tier_text_and_words(self):
        self.assertEqual(rankings.tier_text(55.0), "mid-range 55/100")
        self.assertEqual(rankings.tier_text(100.0), "flagship 100/100")
        self.assertEqual(rankings.tier_text(None), "unknown")
        self.assertEqual(rankings.tier_word(95.0), "flagship")
        self.assertEqual(rankings.tier_word(75.0), "high-end")
        self.assertEqual(rankings.tier_word(55.0), "mid-range")
        self.assertEqual(rankings.tier_word(35.0), "entry-level")
        self.assertEqual(rankings.tier_word(8.0), "low-end")


class AnchorTests(unittest.TestCase):
    """The scale is only meaningful if its anchors hold."""

    def test_the_flagship_is_a_hundred(self):
        self.assertEqual(rankings.gpu_score("RTX 5090"), 100.0)
        self.assertEqual(rankings.cpu_score("Ryzen 9 9950X"), 100.0)

    def test_the_documented_anchors(self):
        self.assertEqual(rankings.gpu_score("GTX 1060"), 35.0)
        self.assertEqual(rankings.gpu_score("GTX 970"), 28.0)
        self.assertEqual(rankings.gpu_score("GeForce 610M"), 8.0)
        self.assertEqual(rankings.cpu_score("Ryzen 5 3600"), 45.0)
        self.assertEqual(rankings.cpu_score("Core i5-2450M"), 10.0)

    def test_the_rtx_4060_class_sits_mid_table(self):
        score = rankings.gpu_score("RTX 4060")
        self.assertIsNotNone(score)
        self.assertTrue(55.0 <= score <= 60.0, score)

    def test_every_score_sits_on_the_scale(self):
        for _pattern, score in rankings.GPU_RANKINGS + rankings.CPU_RANKINGS:
            self.assertTrue(0.0 <= score <= 100.0, (score, _pattern))


class RequirementCheckTests(TempDir):
    """The deterministic tier comparison for named parts."""

    MACHINE = {
        "cpu": {"name": "AMD Ryzen 5 3600", "capability_score": 45.0},
        "gpus": [
            {"name": "NVIDIA GeForce RTX 3060", "capability_score": 55.0},
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
        self.assertIn("GTX 970 ~ 28/100", item.detail)
        self.assertIn("RTX 3060 ~ 55/100", item.detail)
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
        self.assertIn("RTX 4080 ~ 88/100", item.detail)
        self.assertIn("RTX 3060 ~ 55/100", item.detail)
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

        self.assertEqual([gpu.capability for gpu in gpus], [55.0, 13.0])

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

        self.assertEqual(machine.cpu_capability, 45.0)

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
                            "", 55.0)
        self.assertEqual(
            gpu.display(), "GeForce RTX 3060 (12.0 GiB VRAM, mid-range 55/100)"
        )

    def test_gpu_display_says_unknown_when_unranked(self):
        gpu = inventory.Gpu("Mystery", 8 * inventory.GIB, "wmi")
        self.assertEqual(gpu.display(), "Mystery (8.0 GiB VRAM, unknown)")

    def test_cpu_line_carries_the_tier(self):
        machine = inventory.Inventory(
            cpu_name="AMD Ryzen 5 3600", cpu_ghz=3.6, cpu_capability=45.0
        )
        lines = dict(machine.lines())
        self.assertEqual(lines["CPU"], "AMD Ryzen 5 3600 @ 3.6 GHz (entry-level 45/100)")

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
