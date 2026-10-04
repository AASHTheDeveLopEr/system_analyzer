"""Offline tests for the requirements checklist.

The promises under test:

* numeric requirements are checked by direct comparison against the measured
  machine -- the model is never consulted for arithmetic;
* named parts ("GTX 970") go to the provider as atomic questions, and a coin
  flip counts as unknown, never as a pass or a fail;
* anything unparsable is reported as unchecked, never guessed.
"""

from __future__ import annotations

import unittest

import requirements
import system_one
from system_one import Decision
from config import Config
from tests._helpers import TempDir

STATE = {
    "cpu": {"logical_cores": 4, "clock_ghz": None},
    "memory": {"total_gib": 7.78},
    "gpus": [
        {"name": "GeForce 610M", "dedicated_vram_gib": 2.0},
        {"name": "Intel HD Graphics 3000", "dedicated_vram_gib": None},
    ],
    "storage": [
        {"total_gib": 183.69, "free_gib": 12.18},
        {"total_gib": 38.85, "free_gib": 2.94},
    ],
}


class StubClient:
    """Answers every question with a fixed noul (per-id overrides win)."""

    def __init__(self, default: float, per_id: dict[str, float] | None = None):
        self.default = default
        self.per_id = per_id or {}
        self.calls: list[dict] = []

    def ask(self, state, questions):
        self.calls.append(dict(questions))
        answers = {
            qid: {"type": "noul", "noul": self.per_id.get(qid, self.default)}
            for qid in questions
        }
        return Decision(model="stub", answers=answers, provider="stub", source="local")


class ParsingTests(TempDir):
    def test_ram_is_read_from_either_word_order(self):
        parsed, _unchecked, _ignored = requirements.parse_requirements("8 GB RAM")
        self.assertEqual((parsed[0].component, parsed[0].value), ("ram", 8.0))
        parsed, *_ = requirements.parse_requirements("Memory: 16 GB")
        self.assertEqual((parsed[0].component, parsed[0].value), ("ram", 16.0))

    def test_vram_in_gb_and_mb(self):
        parsed, *_ = requirements.parse_requirements("NVIDIA GTX 970 4GB")
        self.assertEqual((parsed[0].component, parsed[0].value), ("vram", 4.0))
        parsed, *_ = requirements.parse_requirements("512 MB graphics")
        self.assertEqual(parsed[0].value, 0.5)

    def test_storage_means_free_space(self):
        parsed, *_ = requirements.parse_requirements("60 GB storage")
        self.assertEqual((parsed[0].component, parsed[0].value), ("storage_free", 60.0))

    def test_processor_speed_and_core_count(self):
        parsed, *_ = requirements.parse_requirements("2.4 GHz processor")
        self.assertEqual((parsed[0].component, parsed[0].value), ("cpu_ghz", 2.4))
        parsed, *_ = requirements.parse_requirements("4 core CPU")
        self.assertEqual((parsed[0].component, parsed[0].value), ("cpu_cores", 4.0))

    def test_named_parts_stay_named(self):
        parsed, *_ = requirements.parse_requirements("GTX 970")
        self.assertEqual(parsed[0].component, "gpu_model")
        parsed, *_ = requirements.parse_requirements("Intel Core i5-4460")
        self.assertEqual(parsed[0].component, "cpu_model")

    def test_a_speed_beats_a_named_processor(self):
        parsed, *_ = requirements.parse_requirements("Intel Core i5 at 3.2 GHz")
        self.assertEqual(parsed[0].component, "cpu_ghz")

    def test_platform_notes_are_ignored_not_unchecked(self):
        _p, unchecked, ignored = requirements.parse_requirements("DirectX 12; 8 GB RAM")
        self.assertEqual(ignored, ["DirectX 12"])
        self.assertEqual(unchecked, [])

    def test_unknown_clauses_are_reported_unchecked(self):
        _p, unchecked, _ignored = requirements.parse_requirements("8 GB RAM; VR Ready headset")
        self.assertEqual(unchecked, ["VR Ready headset"])

    def test_duplicate_requirements_collapse(self):
        parsed, *_ = requirements.parse_requirements("8 GB RAM, 8 GB RAM")
        self.assertEqual(len(parsed), 1)

    def test_one_line_can_hold_several_requirements(self):
        parsed, *_ = requirements.parse_requirements("8 GB RAM and 2 GB VRAM and 30 GB storage")
        self.assertEqual([r.component for r in parsed], ["ram", "vram", "storage_free"])


class NumericCheckTests(TempDir):
    def test_a_met_requirement_passes_with_the_numbers_shown(self):
        result = requirements.check_requirements(STATE, "task", "4 GB RAM", None, None)
        item = result.items[0]
        self.assertEqual(item.state, "passed")
        self.assertIn("7.78", item.detail)
        self.assertEqual(result.verdict, "Meets the minimum")
        self.assertEqual(result.confidence, 100)

    def test_a_failed_requirement_fails_with_both_numbers(self):
        result = requirements.check_requirements(STATE, "task", "16 GB RAM", None, None)
        self.assertEqual(result.items[0].state, "failed")
        self.assertEqual(result.verdict, "Below the minimum")

    def test_the_largest_free_volume_is_what_storage_is_checked_against(self):
        result = requirements.check_requirements(STATE, "task", "10 GB storage", None, None)
        self.assertEqual(result.items[0].state, "passed")
        result = requirements.check_requirements(STATE, "task", "600 GB storage", None, None)
        self.assertEqual(result.items[0].state, "failed")

    def test_an_unreadable_machine_value_is_unknown_not_zero(self):
        result = requirements.check_requirements(STATE, "task", "2 GHz CPU", None, None)
        self.assertEqual(result.items[0].state, "unknown")
        self.assertEqual(result.verdict, "Cannot tell")

    def test_recommended_specs_separate_runs_from_runs_well(self):
        result = requirements.check_requirements(STATE, "task", "4 GB RAM", "16 GB RAM", None)
        self.assertEqual(result.verdict, "Meets the minimum -- it will run, but tightly")
        result = requirements.check_requirements(STATE, "task", "4 GB RAM", "4 GB RAM", None)
        self.assertEqual(result.verdict, "Meets the recommended specs -- it should run well")

    def test_unparsable_text_is_rejected_loudly(self):
        with self.assertRaises(requirements.RequirementError):
            requirements.check_requirements(STATE, "task", "hello world", None, None)


class NamedPartTests(TempDir):
    def test_named_parts_are_asked_as_atomic_questions(self):
        client = StubClient(0.93)
        result = requirements.check_requirements(STATE, "task", "GTX 970", None, client)
        self.assertEqual(client.calls[0]["req_0"]["type"], "noul")
        self.assertEqual(result.items[0].state, "passed")
        self.assertEqual(result.confidence, 93)

    def test_a_coin_flip_is_unknown_never_a_fail(self):
        client = StubClient(0.52)
        result = requirements.check_requirements(STATE, "task", "GTX 970", None, client)
        self.assertEqual(result.items[0].state, "unknown")
        self.assertEqual(result.verdict, "Cannot tell")

    def test_a_low_probability_is_an_honest_fail(self):
        client = StubClient(0.04)
        result = requirements.check_requirements(STATE, "task", "GTX 970", None, client)
        self.assertEqual(result.items[0].state, "failed")
        self.assertEqual(result.verdict, "Below the minimum")

    def test_without_a_client_named_parts_are_unknown_and_flagged(self):
        result = requirements.check_requirements(STATE, "task", "GTX 970", None, None)
        self.assertTrue(result.model_unavailable)
        self.assertTrue(any("no System One endpoint" in n for n in result.notes))
        self.assertEqual(result.verdict, "Cannot tell")

    def test_numeric_requirements_pass_even_when_the_model_judges_the_rest(self):
        client = StubClient(0.9)
        result = requirements.check_requirements(STATE, "task", "4 GB RAM; GTX 970", None, client)
        states = {item.raw: item.state for item in result.items}
        self.assertEqual(states["4 GB RAM"], "passed")
        self.assertEqual(states["GTX 970"], "passed")


class ReportableFloorTests(TempDir):
    """The checklist gets the same reportable floor as the verdict.

    ``ask`` and ``check`` must treat the same confidence the same
    way: a checklist under the floor keeps its verdict and its
    number, and gains the flag, the note, and the label.
    """

    def test_a_checklist_below_the_floor_is_called_a_guess(self):
        # One named part judged at P=0.9 earns confidence 90;
        # with the floor raised to 95 the checklist is under it.
        result = requirements.check_requirements(
            STATE, "task", "GTX 970", None, StubClient(0.9),
            config=Config({"min_reportable_confidence": 95}),
        )
        self.assertEqual(result.confidence, 90)
        self.assertTrue(result.low_confidence)
        self.assertTrue(result.to_dict()["low_confidence"])
        self.assertTrue(any("reportable floor" in n for n in result.notes))
        self.assertTrue(
            any("treat this verdict as a guess" in n for n in result.notes)
        )

    def test_a_confident_checklist_is_not_flagged(self):
        result = requirements.check_requirements(
            STATE, "task", "GTX 970", None, StubClient(0.9),
        )
        self.assertGreaterEqual(result.confidence, Config().min_reportable_confidence)
        self.assertFalse(result.low_confidence)
        self.assertFalse(result.to_dict()["low_confidence"])
        self.assertFalse(any("reportable floor" in n for n in result.notes))

    def test_the_floor_labels_without_changing_the_checklist(self):
        # Same answers, same confidence; only the floor differs.
        flagged = requirements.check_requirements(
            STATE, "task", "GTX 970", None, StubClient(0.9),
            config=Config({"min_reportable_confidence": 95}),
        )
        plain = requirements.check_requirements(
            STATE, "task", "GTX 970", None, StubClient(0.9),
        )
        self.assertEqual(flagged.verdict, plain.verdict)
        self.assertEqual(flagged.confidence, plain.confidence)
        self.assertTrue(flagged.low_confidence)
        self.assertFalse(plain.low_confidence)

    def test_a_config_without_the_setting_still_composes(self):
        # getattr, so config objects that predate the setting
        # fall back to the declared default of 20.
        class LegacyConfig:
            uncertain_margin = 0.10

        result = requirements.check_requirements(
            STATE, "task", "GTX 970", None, StubClient(0.9),
            config=LegacyConfig(),
        )
        self.assertEqual(result.confidence, 90)
        self.assertFalse(result.low_confidence)

    def test_the_floor_is_read_from_any_config_object(self):
        class LegacyConfig:
            uncertain_margin = 0.10
            min_reportable_confidence = 95

        result = requirements.check_requirements(
            STATE, "task", "GTX 970", None, StubClient(0.9),
            config=LegacyConfig(),
        )
        self.assertTrue(result.low_confidence)
        self.assertTrue(any("reportable floor (95%)" in n for n in result.notes))


if __name__ == "__main__":
    unittest.main()
