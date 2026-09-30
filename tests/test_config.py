"""Tests for config.py -- the declared thresholds, weights and curves.

FR-006 says the verdict bands live in configuration, not in the code that
decides a verdict. These tests pin that down: the defaults are documented, an
override actually takes effect, and a nonsense override is refused loudly
rather than silently ignored.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import config as config_module
from config import Config, ConfigError
from tests._helpers import TempDir


class DefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def test_the_built_in_weights_add_up_to_one(self) -> None:
        weights = self.config.weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_every_component_has_a_weight_and_a_curve(self) -> None:
        for component in ("cpu", "memory", "gpu", "storage"):
            self.assertIn(component, self.config.weights())
            curve = self.config.curve(component)
            self.assertGreaterEqual(len(curve), 2)
            self.assertEqual(curve[0][1], 0.0)
            self.assertEqual(curve[-1][1], 100.0)

    def test_the_curves_only_ever_climb(self) -> None:
        for name in ("cpu", "memory", "gpu", "storage"):
            scores = [y for _x, y in self.config.curve(name)]
            self.assertEqual(scores, sorted(scores), f"{name} curve goes downhill")

    def test_bands_are_ordered_from_certain_to_hopeless(self) -> None:
        bands = self.config.bands()
        self.assertEqual([label for _t, label in bands], ["Yes", "Likely", "Probably not", "No"])
        thresholds = [t for t, _label in bands]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))
        self.assertEqual(thresholds[-1], 0.0)

    def test_the_unsure_margin_is_symmetric_around_a_coin_flip(self) -> None:
        self.assertAlmostEqual(self.config.uncertain_margin, 0.10, places=6)
        self.assertLess(self.config.uncertain_confidence_cap, 100)

    def test_an_ungrounded_answer_can_never_look_confident(self) -> None:
        self.assertLess(self.config.ungrounded_confidence_cap, 60)

    def test_the_model_blend_is_a_fraction(self) -> None:
        self.assertGreaterEqual(self.config.model_blend, 0.0)
        self.assertLessEqual(self.config.model_blend, 1.0)

    def test_tiers_are_five_self_contained_labels(self) -> None:
        tiers = self.config.rating_tiers
        self.assertEqual(len(tiers), 5)
        for tier in tiers:
            self.assertGreater(len(tier.split()), 3, f"{tier!r} is too thin to stand alone")

    def test_an_unknown_curve_name_is_a_loud_error(self) -> None:
        with self.assertRaises(KeyError):
            self.config.curve("network")


class ValidationTests(TempDir):
    def _load(self, payload: dict) -> Config:
        path = self.write("config.json", json.dumps(payload))
        return config_module.load(path)

    def test_an_override_wins_over_the_default(self) -> None:
        config = self._load({"verdict_bands": {"yes_at": 0.95, "likely_at": 0.7}})
        self.assertEqual(config.bands()[0], (0.95, "Yes"))

    def test_weights_that_do_not_add_up_are_refused(self) -> None:
        with self.assertRaises(ConfigError):
            self._load({"rating_weights": {"cpu": 0.5, "memory": 0.5, "gpu": 0.5, "storage": 0.1}})

    def test_a_typo_in_a_key_is_refused_not_ignored(self) -> None:
        # Silently dropping an unknown key is how "I changed the config" turns
        # into "I changed nothing and did not find out".
        with self.assertRaises(ConfigError):
            self._load({"verdict_band": {"yes_at": 0.9}})

    def test_a_bands_value_outside_zero_to_one_is_refused(self) -> None:
        with self.assertRaises(ConfigError):
            self._load({"verdict_bands": {"yes_at": 1.4}})

    def test_a_missing_file_is_a_loud_error(self) -> None:
        with self.assertRaises(ConfigError):
            config_module.load(Path("nope-does-not-exist.json"))

    def test_broken_json_is_a_loud_error(self) -> None:
        with self.assertRaises(ConfigError):
            config_module.load(self.write("config.json", "{not json"))

    def test_to_dict_round_trips_through_load(self) -> None:
        original = Config()
        path = self.write("config.json", json.dumps(original.to_dict()))
        self.assertEqual(config_module.load(path).to_dict(), original.to_dict())


if __name__ == "__main__":
    unittest.main()
