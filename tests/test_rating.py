"""Tests for the rating: component scores, the rule total, and rate_system.

Nothing here touches the network. The model is a FakeClient with hand-written
answers, so these tests pin down the arithmetic and the honesty rules rather
than whatever a live model happens to say today.
"""

from __future__ import annotations

import unittest

import menu
import system_one
from config import Config
from system_one import ComponentScore, SystemOneError, rate_system, rules_rating
from tests._helpers import FakeClient, sample_inventory, score


def rating_answers(score_value: float, legend: dict | None = None) -> dict:
    return {
        "overall_tier": score(
            score_value,
            legend or {"0": "Entry level", "4": "Workstation"},
        )
    }


class CurveTests(unittest.TestCase):
    def test_a_value_below_the_first_anchor_is_the_first_score(self) -> None:
        curve = Config().curve("cpu")
        self.assertEqual(system_one._curve_score(curve, 0), curve[0][1])
        self.assertEqual(system_one._curve_score(curve, -99), curve[0][1])

    def test_a_value_above_the_last_anchor_is_the_last_score(self) -> None:
        curve = Config().curve("memory")
        self.assertEqual(system_one._curve_score(curve, 10_000), curve[-1][1])

    def test_interpolation_is_monotonic(self) -> None:
        curve = Config().curve("gpu")
        scores = [system_one._curve_score(curve, v / 2) for v in range(0, 60)]
        self.assertEqual(scores, sorted(scores))

    def test_an_unknown_value_scores_nothing_rather_than_zero(self) -> None:
        # Zero is a real, terrible score. Unknown is not the same thing, and
        # conflating them would quietly punish machines we failed to read.
        self.assertIsNone(system_one._curve_score(Config().curve("cpu"), None))


class ComponentScoreTests(unittest.TestCase):
    def test_a_good_machine_scores_worse_than_a_bad_one(self) -> None:
        good = {c.name: c for c in system_one.component_scores(sample_inventory())}
        bad = {
            c.name: c
            for c in system_one.component_scores(
                sample_inventory(
                    cpu_cores=2,
                    memory=menu.Memory(4 * 1024**3, 2 * 1024**3),
                    gpus=(menu.Gpu("NVIDIA GTX 750 Ti", 2 * 1024**3, "nvml"),),
                    disks=(menu.Disk("/", 64 * 1024**3, 256 * 1024**3, 75.0),),
                )
            )
        }
        for name, component in good.items():
            self.assertGreater(component.score, bad[name].score, name)

    def test_a_software_adapter_is_unknown_not_zero(self) -> None:
        scores = {
            c.name: c
            for c in system_one.component_scores(
                sample_inventory(gpus=(menu.Gpu("Intel UHD 770", None, "software"),))
            )
        }
        self.assertFalse(scores["GPU"].known)
        self.assertIn("no dedicated VRAM", scores["GPU"].detail)

    def test_no_adapter_at_all_says_so_plainly(self) -> None:
        scores = {
            c.name: c for c in system_one.component_scores(sample_inventory(gpus=()))
        }
        self.assertFalse(scores["GPU"].known)
        self.assertIn("no display adapter", scores["GPU"].detail)

    def test_shared_memory_is_not_counted_as_dedicated_vram(self) -> None:
        # The whole point of a separate vram field: an iGPU that borrows system
        # RAM has not got VRAM, and rating it as if it had is the bug that made
        # integrated machines look like they had a graphics card.
        scores = {
            c.name: c
            for c in system_one.component_scores(
                sample_inventory(
                    memory=menu.Memory(32 * 1024**3, 16 * 1024**3),
                    gpus=(menu.Gpu("Intel Arc", None, "wmi"),),
                )
            )
        }
        self.assertIsNone(scores["GPU"].score)

    def test_the_state_mapping_gives_the_same_answer_as_the_inventory(self) -> None:
        from_object = {c.name: c.score for c in system_one.component_scores(sample_inventory())}
        from_state = {c.name: c.score for c in system_one.component_scores(sample_inventory().to_state())}
        self.assertEqual(from_object, from_state)

    def test_one_bad_adapter_does_not_hide_the_others(self) -> None:
        scores = {
            c.name: c
            for c in system_one.component_scores(
                sample_inventory(
                    gpus=(
                        menu.Gpu("NVIDIA RTX 3060", 12 * 1024**3, "nvml"),
                        menu.Gpu("Mystery", None, "wmi", "driver did not report VRAM"),
                    )
                )
            )
        }
        self.assertTrue(scores["GPU"].known)
        self.assertEqual(scores["GPU"].score, system_one._curve_score(Config().curve("gpu"), 12))


class RulesRatingTests(unittest.TestCase):
    def test_unknown_components_are_dropped_and_the_rest_renormalise(self) -> None:
        """A machine we could not read about is not a machine we can score low."""
        strong = [ComponentScore("CPU", 100.0, "12 cores"), ComponentScore("GPU", 100.0, "16 GiB")]
        full, _ = rules_rating(strong + [ComponentScore("Memory", None, "unknown")])
        partial, known = rules_rating([ComponentScore("CPU", 100.0, "12 cores")])
        self.assertAlmostEqual(full, 100.0, places=6)
        self.assertAlmostEqual(partial, 100.0, places=6)
        self.assertEqual([c.name for c in known], ["CPU"])

    def test_the_whole_rule_total_is_exactly_one_hundred_when_everything_is_maxed(self) -> None:
        perfect = [ComponentScore("CPU", 100.0, ""), ComponentScore("GPU", 100.0, "")]
        total, _known = rules_rating(perfect)
        self.assertAlmostEqual(total, 100.0, places=6)
    def test_nothing_known_scores_zero_rather_than_crashing(self) -> None:
        total, known = rules_rating([ComponentScore("CPU", None, "unknown")])
        self.assertEqual(total, 0.0)
        self.assertEqual(known, [])


class RateSystemTests(unittest.TestCase):
    def test_the_model_and_the_rules_are_both_visible_in_the_result(self) -> None:
        client = FakeClient(rating_answers(1.0))
        rating = rate_system(sample_inventory(), client, Config({"model_blend": 0.5}))

        self.assertEqual(rating.method, "model+rules")
        self.assertEqual(rating.model_total, 100.0)
        self.assertAlmostEqual(rating.total, (100.0 + rating.rules_total) / 2, delta=1.0)
        self.assertEqual(rating.model, "fake-model")
        self.assertEqual(rating.provider, "fake")
        self.assertEqual(rating.source, "free")

    def test_a_blend_of_zero_ignores_the_model(self) -> None:
        client = FakeClient(rating_answers(0.0))
        rating = rate_system(sample_inventory(), client, Config({"model_blend": 0.0}))
        self.assertAlmostEqual(rating.total, rating.rules_total, delta=1.0)

    def test_the_model_is_asked_to_place_the_machine_in_ordered_tiers(self) -> None:
        client = FakeClient(rating_answers(0.5))
        rate_system(sample_inventory(), client, Config())
        asked = client.calls[0]["questions"]["overall_tier"]
        self.assertEqual(asked["type"], "score")
        self.assertIsInstance(asked["criteria"], list)
        self.assertGreaterEqual(len(asked["criteria"]), 2)

    def test_the_tier_name_comes_from_the_legend_the_model_returned(self) -> None:
        client = FakeClient(rating_answers(0.5, {"2": "Performance tier"}))
        rating = rate_system(sample_inventory(), client, Config())
        self.assertIn("Performance", rating.tier)

    def test_no_endpoint_means_a_rules_rating_not_no_rating(self) -> None:
        client = FakeClient(error=SystemOneError("network unreachable"))
        rating = rate_system(sample_inventory(), client, Config())

        self.assertEqual(rating.method, "rules")
        self.assertEqual(rating.source, "rules")
        self.assertIsNone(rating.model)
        self.assertGreater(rating.total, 0)
        self.assertTrue(any("rule-only" in note for note in rating.notes))

    def test_the_degraded_rating_says_plainly_that_no_model_answered(self) -> None:
        rating = rate_system(
            sample_inventory(), FakeClient(error=SystemOneError("boom")), Config()
        )
        self.assertTrue(any("No model was available" in f.text for f in rating.factors))

    def test_at_least_three_factors_always(self) -> None:
        # SC-002: a single reason is not an explanation.
        rating = rate_system(sample_inventory(), FakeClient(rating_answers(0.6)), Config())
        self.assertGreaterEqual(len(rating.factors), 3)

    def test_the_component_factors_run_biggest_pull_first(self) -> None:
        rating = rate_system(sample_inventory(), FakeClient(rating_answers(0.0)), Config())
        pulls = [
            f.text
            for f in rating.factors
            if f.effect in ("up", "down") and not f.text.startswith("Model")
        ]
        self.assertTrue(pulls)
        biggest = max(
            (c for c in rating.subscores if c.known),
            key=lambda c: abs(c.score - rating.total),
        )
        self.assertIn(biggest.name, pulls[0])

    def test_a_factor_says_the_same_thing_twice(self) -> None:
        # The text says "pulls the score up/down" and ``effect`` repeats it for
        # anything reading the JSON. When these disagree the reader gets two
        # opposite answers from one factor, which is worse than no direction.
        for model_value in (0.0, 0.5, 0.95):
            rating = rate_system(sample_inventory(), FakeClient(rating_answers(model_value)), Config())
            for factor in rating.factors:
                if "pulls the score" not in factor.text:
                    continue
                self.assertEqual(
                    "up" if "pulls the score up" in factor.text else "down",
                    factor.effect,
                    factor.text,
                )

    def test_the_tier_is_always_shown_when_the_model_gave_one(self) -> None:
        rating = rate_system(sample_inventory(), FakeClient(rating_answers(0.5)), Config())
        self.assertTrue(rating.factors[0].text.startswith("Model placed this machine in the"))

    def test_a_missing_reading_is_called_out_both_ways(self) -> None:
        rating = rate_system(
            sample_inventory(gpus=()), FakeClient(rating_answers(0.5)), Config()
        )
        self.assertTrue(rating.estimated)
        self.assertTrue(any("not readable" in f.text for f in rating.factors))
        self.assertTrue(any("estimated" in note for note in rating.notes))

    def test_the_whole_thing_is_stays_on_the_scale(self) -> None:
        for value in (-5.0, 0.0, 0.5, 1.0, 7.0):
            rating = rate_system(sample_inventory(), FakeClient(rating_answers(value)), Config())
            self.assertGreaterEqual(rating.total, 0)
            self.assertLessEqual(rating.total, 100)

    def test_the_json_view_carries_the_provenance_and_the_working(self) -> None:
        rating = rate_system(sample_inventory(), FakeClient(rating_answers(0.6)), Config())
        payload = rating.to_dict()
        for key in ("total", "method", "source", "model", "subscores", "factors", "tier"):
            self.assertIn(key, payload)
        self.assertEqual(len(payload["subscores"]), 4)


if __name__ == "__main__":
    unittest.main()
