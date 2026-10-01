"""Live contract tests against a real local Ollama. Opt-in, like the Zen ones.

    SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_ollama

Everything here is skipped unless a local Ollama actually answers and has a
decision-capable model installed, so an offline machine loses nothing.
"""

from __future__ import annotations

import os
import unittest

import providers
from system_one import SystemOneClient

LIVE = os.environ.get("SYSTEM_ANALYZER_LIVE_TESTS", "").strip().lower() in ("1", "true", "yes", "on")


@unittest.skipUnless(LIVE, "opt-in: set SYSTEM_ANALYZER_LIVE_TESTS=1")
class LiveOllamaTests(unittest.TestCase):
    def setUp(self):
        self.base = providers.resolve_ollama_url()
        status = providers.discover_ollama(self.base)
        if not status.reachable:
            self.skipTest(f"no Ollama answered at {self.base}")
        if not status.models:
            self.skipTest(
                f"no decision-capable model installed; try: ollama pull {providers.RECOMMENDED_PULL_MODEL}"
            )
        self.model = status.models[0]

    def _client(self) -> SystemOneClient:
        provider = {
            "name": "ollama",
            "url": self.base.rstrip("/") + "/v1/systemone",
            "model": self.model,
            "source": "local",
            "auth": "",
        }
        return SystemOneClient([provider])

    def test_a_real_local_decision_model_answers_a_noul_question(self):
        decision = self._client().ask(
            "A desktop with 8 CPU cores, 16 GB of RAM and a discrete GPU with 8 GB of VRAM.",
            {
                "is_desktop": {
                    "type": "noul",
                    "instructions": (
                        "Is the machine described here a desktop computer, "
                        "rather than a laptop or a phone?"
                    ),
                }
            },
        )
        probability = decision.noul("is_desktop")
        self.assertTrue(0.0 <= probability <= 1.0)
        self.assertEqual(decision.source, "local")

    def test_a_real_local_decision_model_answers_a_choice_question(self):
        decision = self._client().ask(
            "A desktop with 8 CPU cores, 16 GB of RAM and a discrete GPU with 8 GB of VRAM.",
            {
                "bigger": {
                    "type": "choice",
                    "instructions": "Which of these numbers is the larger one?",
                    "criteria": {"two": "the number two", "seven": "the number seven"},
                }
            },
        )
        chosen, probabilities, confidence = decision.choice("bigger")
        self.assertIn(chosen, ("two", "seven"))
        self.assertTrue(0.0 <= confidence <= 100.0)


if __name__ == "__main__":
    unittest.main()
