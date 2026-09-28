"""Live contract tests against a real Jev endpoint.

**These are skipped unless you ask for them**, so ``python -m unittest discover
-s tests`` stays offline and fast::

    SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_jev

They exist to pin down the parts of the contract that the local validator
implements but cannot prove on its own: whether a body the validator accepts is
one the endpoint actually answers, and vice versa. Several of these cases were
found by running the shapes by hand and watching the real responses.

They also cost money on the paid tier and are rate limited on the free one, so
each test asks as little as possible: one question, one request.
"""

from __future__ import annotations

import os
import unittest

import system_one
from system_one import SystemOneClient, SystemOneError

LIVE = os.environ.get("SYSTEM_ANALYZER_LIVE_TESTS") == "1"

STATE = "A desktop PC with 6 CPU cores, 16 GB of RAM and an 8 GB NVIDIA GPU."


@unittest.skipUnless(LIVE, "set SYSTEM_ANALYZER_LIVE_TESTS=1 to run the live contract tests")
class LiveContractTests(unittest.TestCase):
    client: SystemOneClient

    @classmethod
    def setUpClass(cls):
        cls.client = SystemOneClient()
        if not cls.client.providers:  # pragma: no cover - defensive
            raise unittest.SkipTest("no System One endpoint is configured")

    def ask(self, questions, state=STATE):
        return self.client.ask(state, questions)

    # -- what the validator promises the endpoint will accept --------------

    def test_noul(self):
        decision = self.ask({"q": {"type": "noul", "instructions": "Is this a modern gaming PC?"}})
        probability = decision.noul("q")
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertIn("input_tokens", decision.usage)

    def test_choice_with_map_criteria(self):
        decision = self.ask({
            "q": {
                "type": "choice",
                "instructions": "Which best describes this machine?",
                "criteria": {
                    "High": "built for demanding games and 3D work",
                    "Medium": "everyday computing and light gaming",
                    "Low": "office work and web browsing only",
                },
            }
        })
        chosen, probabilities, _confidence = decision.choice("q")
        self.assertIn(chosen, ("High", "Medium", "Low"))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    def test_score_with_list_criteria(self):
        decision = self.ask({
            "q": {
                "type": "score",
                "instructions": "How demanding a workload can this machine handle?",
                "criteria": [
                    "Entry: office work and web browsing",
                    "Mainstream: modern games and video editing",
                    "Enthusiast: 3D rendering and local AI models",
                ],
            }
        })
        score, legend, _probabilities, _confidence = decision.score("q")
        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(len(legend), 3)

    def test_mixed_types_in_one_request(self):
        """All three primitives answer from a single state and a single call."""
        decision = self.ask({
            "n": {"type": "noul", "instructions": "Does this machine have discrete graphics?"},
            "c": {"type": "choice", "instructions": "How would you buy this machine?",
                  "criteria": {"Yes": "now", "No": "not yet"}},
            "s": {"type": "score", "instructions": "How fast is this machine?",
                  "criteria": ["Slow", "Medium", "Fast"]},
        })
        self.assertEqual(set(decision.answers), {"n", "c", "s"})

    def test_object_state_is_accepted(self):
        """An object is the shape the app actually sends, so it must work."""
        decision = self.ask(
            {"q": {"type": "noul", "instructions": "Is this machine good for local AI?"}},
            state={
                "cpu": {"name": "Intel Core i7", "logical_cores": 16},
                "memory": {"total_gib": 32.0},
                "gpus": [{"name": "NVIDIA RTX 4080", "dedicated_vram_gib": 16.0}],
            },
        )
        self.assertGreaterEqual(decision.noul("q"), 0.0)

    # -- what the endpoint rejects, which the validator must catch first --

    def _ask_raw(self, questions, state=STATE):
        """Bypass ``validate_request`` and post exactly this body.

        These cases are here to document the *endpoint's* behaviour. The
        normal path never sends them -- ``validate_request`` refuses first,
        which is the whole point -- so the only honest way to check that the
        local rule is still needed is to go around it.
        """
        return system_one._post_json(self.client.providers[0], state, questions)

    def test_choice_criteria_as_a_list_is_rejected_by_the_endpoint(self):
        """Documents the 422 that made this a local validation rule."""
        with self.assertRaises(SystemOneError) as caught:
            self._ask_raw({"q": {"type": "choice", "instructions": "Pick one",
                                 "criteria": ["High", "Low"]}})
        self.assertIn("422", str(caught.exception))

    def test_score_criteria_as_a_map_is_rejected_by_the_endpoint(self):
        with self.assertRaises(SystemOneError) as caught:
            self._ask_raw({"q": {"type": "score", "instructions": "How fast?",
                                 "criteria": {"0": "Slow", "1": "Fast"}}})
        self.assertIn("422", str(caught.exception))

    def test_the_validator_catches_what_the_endpoint_would_reject(self):
        """And the cheap way: the same two bodies, refused without a request."""
        for questions in (
            {"q": {"type": "choice", "instructions": "Pick one", "criteria": ["High", "Low"]}},
            {"q": {"type": "score", "instructions": "How fast?",
                   "criteria": {"0": "Slow", "1": "Fast"}}},
        ):
            error = system_one.validate_request(STATE, questions)
            self.assertIsNotNone(error, f"{questions} should be rejected locally")
            self.assertIn("Invalid question", error)

    def test_an_unknown_model_is_rejected(self):
        """A bad model is a 401 on this gateway; it must not look like success."""
        provider = dict(self.client.providers[0])
        provider["model"] = "jev-does-not-exist"
        client = SystemOneClient([provider])
        with self.assertRaises(SystemOneError) as caught:
            client.ask(STATE, {"q": {"type": "noul", "instructions": "Is it working?"}})
        self.assertIn("not supported", str(caught.exception).lower())


@unittest.skipUnless(LIVE, "set SYSTEM_ANALYZER_LIVE_TESTS=1 to run the live contract tests")
class LiveStabilityTests(unittest.TestCase):
    """FR-007 wants a confidence band, and a band needs evidence that the
    model is at least repeatable. This measures it rather than assuming it."""

    def test_repeated_answers_stay_close(self):
        client = SystemOneClient()
        question = {
            "q": {
                "type": "noul",
                "instructions": "Can this machine run a 7B-parameter model locally?",
            }
        }
        answers = [client.ask(STATE, question).noul("q") for _ in range(3)]
        spread = max(answers) - min(answers)
        self.assertLessEqual(
            spread, 0.5,
            f"noul moved by {spread:.2f} across {answers}; a confidence band "
            f"based on these would be meaningless",
        )


if __name__ == "__main__":
    unittest.main()
