"""Tests for answer_capability: the bands, and the honesty rules.

The band thresholds are configuration, and the rules that hold confidence down
are the reason to trust the confidence. These tests drive a FakeClient so each
rule can be triggered on its own: a confident answer, a coin flip, two questions
that disagree, and a request the specs simply do not decide.
"""

from __future__ import annotations

import unittest

import inventory
import system_one
from config import Config
from system_one import SystemOneError
from judgment import answer_capability
from tests._helpers import FakeClient, capability_answers, sample_inventory


def verdict_for(answers: dict, question: str = "Can I run Cyberpunk 2077?", config=None):
    return answer_capability(sample_inventory(), question, FakeClient(answers), config or Config())


class QuestionShapeTests(unittest.TestCase):
    def test_an_empty_question_is_refused_before_any_request(self) -> None:
        client = FakeClient(capability_answers(0.9))
        with self.assertRaises(ValueError):
            answer_capability(sample_inventory(), "   ", client, Config())
        self.assertEqual(client.calls, [])

    def test_all_three_questions_go_out_in_one_request(self) -> None:
        client = FakeClient(capability_answers(0.9))
        answer_capability(sample_inventory(), "can I game?", client, Config())
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            set(client.calls[0]["questions"]), {"verdict", "grounding", "second_opinion"}
        )

    def test_the_wording_of_the_question_is_carried_inside_each_instruction(self) -> None:
        # Question ids are not sent to the model, so the instructions have to
        # carry the whole question themselves.
        client = FakeClient(capability_answers(0.9))
        answer_capability(sample_inventory(), "Can I run Blender?", client, Config())
        for question in client.calls[0]["questions"].values():
            self.assertIn("Can I run Blender?", question["instructions"])

    def test_choice_criteria_are_a_map_not_a_list(self) -> None:
        client = FakeClient(capability_answers(0.9))
        answer_capability(sample_inventory(), "can I game?", client, Config())
        for qid in ("grounding", "second_opinion"):
            self.assertIsInstance(client.calls[0]["questions"][qid]["criteria"], dict)


class BandTests(unittest.TestCase):
    def test_a_near_certain_yes_reads_as_yes(self) -> None:
        verdict = verdict_for(capability_answers(0.95))
        self.assertEqual(verdict.verdict, "Yes")
        self.assertTrue(verdict.grounded)
        self.assertFalse(verdict.uncertain)

    def test_the_bands_come_from_configuration(self) -> None:
        # With Yes pushed up to 0.99, a 0.95 is no longer a Yes. If the verdict
        # ignored the config this test would still pass on the defaults.
        strict = verdict_for(capability_answers(0.95), config=Config({"verdict_bands": {"yes_at": 0.99}}))
        self.assertEqual(strict.verdict, "Likely")

    def test_a_near_certain_no_reads_as_no(self) -> None:
        verdict = verdict_for(capability_answers(0.04, second="no"))
        self.assertEqual(verdict.verdict, "No")

    def test_a_fuzzy_answer_does_not_claim_to_be_yes(self) -> None:
        verdict = verdict_for(capability_answers(0.6, second="yes"))
        self.assertNotEqual(verdict.verdict, "Yes")
        self.assertEqual(verdict.verdict, "Likely")

    def test_the_explanation_always_names_the_question_asked(self) -> None:
        verdict = verdict_for(capability_answers(0.9), question="Can I run Blender?")
        self.assertIn("Blender", verdict.explanation)


class HonestyTests(unittest.TestCase):
    def test_a_coin_flip_never_comes_out_confident(self) -> None:
        config = Config()
        verdict = verdict_for(capability_answers(0.5, second_confidence=100.0), config=config)
        self.assertTrue(verdict.uncertain)
        self.assertLessEqual(verdict.confidence, config.uncertain_confidence_cap)
        self.assertIn("coin flip", verdict.explanation)

    def test_a_decisive_answer_can_be_confident(self) -> None:
        verdict = verdict_for(capability_answers(0.97, second_confidence=95.0))
        self.assertGreater(verdict.confidence, 70)

    def test_two_questions_that_disagree_reduce_the_confidence(self) -> None:
        agree = verdict_for(capability_answers(0.95, second="yes", second_confidence=95.0))
        disagree = verdict_for(capability_answers(0.95, second="no", second_confidence=95.0))
        self.assertLess(disagree.confidence, agree.confidence)
        self.assertTrue(any("disagreed" in note for note in disagree.notes))

    def test_the_confidence_never_exceeds_the_models_own(self) -> None:
        verdict = verdict_for(capability_answers(0.99, second_confidence=40.0))
        self.assertLessEqual(verdict.confidence, 40)

    def test_a_question_the_specs_cannot_answer_gets_no_verdict(self) -> None:
        # "can I stream 4K on Twitch" depends on bandwidth and
        # upload speed, which are not in a hardware inventory.
        verdict = verdict_for(
            capability_answers(
                0.9, grounding="specs_do_not_decide_it", grounding_confidence=85.0
            ),
            question="Can I stream on Twitch?",
        )
        self.assertEqual(verdict.verdict, "Cannot tell")
        self.assertFalse(verdict.grounded)

    def test_an_ungrounded_answer_says_how_to_make_it_answerable(self) -> None:
        verdict = verdict_for(
            capability_answers(0.9, grounding="specs_do_not_decide_it"),
            question="Can I get a refund for this laptop?",
        )
        self.assertIn("requirements", verdict.explanation)
        self.assertLessEqual(verdict.confidence, Config().ungrounded_confidence_cap)

    def test_an_ungrounded_answer_never_looks_confident_even_at_full_marks(self) -> None:
        verdict = verdict_for(
            capability_answers(
                0.99,
                grounding="specs_do_not_decide_it",
                second_confidence=100.0,
                grounding_confidence=100.0,
            )
        )
        self.assertLessEqual(verdict.confidence, Config().ungrounded_confidence_cap)

    def test_a_strong_yes_has_a_high_probability(self) -> None:
        self.assertGreater(verdict_for(capability_answers(0.93)).probability, 0.9)

    def test_the_confidence_is_always_a_plain_percent(self) -> None:
        for value in (0.0, 0.3, 0.5, 0.77, 1.0):
            verdict = verdict_for(capability_answers(value))
            self.assertIsInstance(verdict.confidence, int)
            self.assertGreaterEqual(verdict.confidence, 0)
            self.assertLessEqual(verdict.confidence, 100)


class BottleneckTests(unittest.TestCase):
    def test_a_weak_component_is_named(self) -> None:
        verdict = answer_capability(
            sample_inventory(
                cpu_cores=2,
                memory=inventory.Memory(2 * 1024**3, 1 * 1024**3),
            ),
            "Can I run Cyberpunk 2077?",
            FakeClient(capability_answers(0.4, second="no")),
            Config(),
        )
        self.assertTrue(verdict.bottlenecks)
        self.assertIn("CPU", " ".join(verdict.bottlenecks))
        self.assertIn("CPU", verdict.explanation)

    def test_a_strong_machine_still_says_something_about_its_limits(self) -> None:
        verdict = verdict_for(capability_answers(0.9))
        self.assertTrue(verdict.explanation.strip())
        self.assertTrue(verdict.sub_scores)

    def test_the_sub_scores_travel_with_the_verdict(self) -> None:
        payload = verdict_for(capability_answers(0.9)).to_dict()
        self.assertEqual(len(payload["sub_scores"]), 4)
        self.assertIn("source", payload)
        self.assertIn("model", payload)


class ConfidenceScaleTests(unittest.TestCase):
    """The live endpoint sends ``confidence`` as a 0..1 fraction, not a percent.

    Reading it as a percent is the kind of bug that only shows up in front of a
    real model: every answer would report 0-1% confidence and still look
    plausible in a test written with 90.0.
    """

    @staticmethod
    def _decision(answer: dict) -> system_one.Decision:
        return system_one.Decision(model="m", answers={"q": answer}, provider="p", source="free", cost=0.0)

    def test_a_fraction_becomes_a_percent(self) -> None:
        decision = self._decision(
            {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9}, "confidence": 0.86}
        )
        self.assertEqual(round(decision.choice("q")[2]), 86)

    def test_a_value_that_is_already_a_percent_is_left_alone(self) -> None:
        decision = self._decision(
            {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9}, "confidence": 86}
        )
        self.assertEqual(decision.choice("q")[2], 86.0)

    def test_a_score_answer_uses_the_same_scale(self) -> None:
        decision = self._decision({"type": "score", "score": 2.0, "legend": {}, "confidence": 0.21})
        self.assertEqual(round(decision.score("q")[3]), 21)

    def test_a_missing_confidence_is_zero_not_an_invented_number(self) -> None:
        decision = self._decision({"type": "choice", "choice": "yes", "probabilities": {}})
        self.assertEqual(decision.choice("q")[2], 0.0)

    def test_a_fraction_confidence_reaches_the_shown_percent(self) -> None:
        verdict = verdict_for(capability_answers(0.9, second_confidence=0.8))
        self.assertGreater(verdict.confidence, 60)


class FailureTests(unittest.TestCase):
    def test_a_dead_endpoint_raises_rather_than_inventing_a_verdict(self) -> None:
        # The CLI turns this into exit code 3 and a plain message. Returning a
        # confident verdict here would be the worst possible failure mode.
        client = FakeClient(error=SystemOneError("no route to host"))
        with self.assertRaises(SystemOneError):
            answer_capability(sample_inventory(), "can I game?", client, Config())


if __name__ == "__main__":
    unittest.main()
