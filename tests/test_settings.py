"""Offline tests for the settings layer: precedence, files, and the chain.

Nothing here touches the network. The layer's promises under test:

* a real environment variable beats .env, which beats config.json, which
  beats the built-in default -- so a script can always override the menus;
* a corrupt config.json fails loudly (SettingsError), while one bad .env
  line is skipped with a warning instead of taking the keys down with it;
* secrets are masked anywhere they are printed.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import providers
import settings
from tests._helpers import TempDir


class QuietEnv(TempDir):
    """TempDir with the machine's real key variables removed.

    A developer machine may carry a live TYPESAFE_API_KEY globally; the
    environment layer must win over the fixtures these tests write, so the
    variables are cleared rather than trusted to be absent.
    """

    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in (
            settings.TYPESAFE_KEY_ENV,
            settings.ZEN_KEY_ENV,
            settings.PROVIDERS_ENV,
            settings.OLLAMA_URL_ENV,
            settings.OLLAMA_MODEL_ENV,
            settings.CLOUD_FALLBACK_ENV,
        ):
            os.environ.pop(name, None)


class PrecedenceTests(QuietEnv):
    def test_the_default_value_is_used_when_nothing_else_says(self):
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "")
        self.assertEqual(loaded.origin("ollama_model"), "default")

    def test_config_json_beats_the_default(self):
        self.write("config.json", json.dumps({"ollama_model": "nimble"}))
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "nimble")
        self.assertEqual(loaded.origin("ollama_model"), "config.json")

    def test_the_env_file_beats_config_json(self):
        self.write("config.json", json.dumps({"ollama_model": "nimble"}))
        self.write(".env", "SYSTEM_ANALYZER_OLLAMA_MODEL=tev1:0.8b\n")
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "tev1:0.8b")
        self.assertEqual(loaded.origin("ollama_model"), ".env")

    def test_a_real_environment_variable_beats_everything(self):
        self.write(".env", "SYSTEM_ANALYZER_OLLAMA_MODEL=tev1:0.8b\n")
        with mock.patch.dict(os.environ, {"SYSTEM_ANALYZER_OLLAMA_MODEL": "nimble"}):
            loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "nimble")
        self.assertEqual(loaded.origin("ollama_model"), "environment")


class EnvFileTests(QuietEnv):
    def test_comments_blanks_and_export_prefixes_are_tolerated(self):
        self.write(
            ".env",
            "# a comment\n\nexport SYSTEM_ANALYZER_OLLAMA_MODEL=nimble\n"
            "TYPESAFE_API_KEY='sk quoted'\n",
        )
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "nimble")
        self.assertEqual(loaded.get("typesafe_api_key"), "sk quoted")

    def test_a_malformed_line_is_skipped_with_a_warning(self):
        self.write(".env", "this line has no equals sign\nSYSTEM_ANALYZER_OLLAMA_MODEL=nimble\n")
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "nimble")
        self.assertTrue(any("line 1" in w for w in loaded.warnings))

    def test_a_value_that_is_not_a_variable_name_is_skipped(self):
        self.write(".env", "not a name=1\n")
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "")
        self.assertTrue(any("not a variable name" in w for w in loaded.warnings))

    def test_an_unreadable_env_file_degrades_to_a_warning(self):
        (self.tmp / ".env").mkdir()
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("ollama_model"), "")
        self.assertTrue(any("could not read" in w for w in loaded.warnings))


class ConfigFileTests(QuietEnv):
    def test_a_missing_file_is_just_defaults(self):
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("provider_chain"), "zen")

    def test_corrupt_json_fails_loudly(self):
        self.write("config.json", "{not json")
        with self.assertRaises(settings.SettingsError) as caught:
            settings.load_settings(self.tmp)
        self.assertIn("not valid JSON", str(caught.exception))

    def test_a_non_object_fails_loudly(self):
        self.write("config.json", "[1, 2]")
        with self.assertRaises(settings.SettingsError):
            settings.load_settings(self.tmp)

    def test_an_unknown_key_is_rejected_not_ignored(self):
        self.write("config.json", json.dumps({"providor_chain": "zen"}))
        with self.assertRaises(settings.SettingsError) as caught:
            settings.load_settings(self.tmp)
        self.assertIn("providor_chain", str(caught.exception))

    def test_an_unknown_provider_id_is_rejected(self):
        self.write("config.json", json.dumps({"provider_chain": "zen,hal9000"}))
        with self.assertRaises(settings.SettingsError) as caught:
            settings.load_settings(self.tmp)
        self.assertIn("hal9000", str(caught.exception))

    def test_an_empty_chain_is_rejected(self):
        self.write("config.json", json.dumps({"provider_chain": " , "}))
        with self.assertRaises(settings.SettingsError):
            settings.load_settings(self.tmp)


class SaveTests(QuietEnv):
    def test_save_config_merges_and_preserves_other_keys(self):
        self.write("config.json", json.dumps({"provider_chain": "zen"}))
        path = settings.save_config({"ollama_model": "nimble"}, self.tmp)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["provider_chain"], "zen")
        self.assertEqual(saved["ollama_model"], "nimble")

    def test_save_config_refuses_to_stomp_a_corrupt_file(self):
        self.write("config.json", "{broken")
        with self.assertRaises(settings.SettingsError):
            settings.save_config({"ollama_model": "nimble"}, self.tmp)
        self.assertEqual((self.tmp / "config.json").read_text(encoding="utf-8"), "{broken")

    def test_save_config_rejects_unknown_keys(self):
        with self.assertRaises(settings.SettingsError):
            settings.save_config({"not_a_setting": "1"}, self.tmp)

    def test_save_env_replaces_in_place_and_keeps_the_rest(self):
        self.write(".env", "# my keys\nOPENCODE_ZEN_API_KEY=old\nSYSTEM_ANALYZER_OLLAMA_MODEL=nimble\n")
        settings.save_env("OPENCODE_ZEN_API_KEY", "new", self.tmp)
        lines = (self.tmp / ".env").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, [
            "# my keys", "OPENCODE_ZEN_API_KEY=new", "SYSTEM_ANALYZER_OLLAMA_MODEL=nimble",
        ])

    def test_save_env_appends_when_the_key_is_new(self):
        settings.save_env("TYPESAFE_API_KEY", "sk-1", self.tmp)
        self.assertIn("TYPESAFE_API_KEY=sk-1", (self.tmp / ".env").read_text(encoding="utf-8"))

    def test_save_env_refuses_a_non_secret_key(self):
        with self.assertRaises(settings.SettingsError):
            settings.save_env("SYSTEM_ANALYZER_OLLAMA_MODEL", "nimble", self.tmp)

    def test_the_value_saved_to_the_env_file_is_read_back(self):
        settings.save_env("TYPESAFE_API_KEY", "sk-live", self.tmp)
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.get("typesafe_api_key"), "sk-live")
        self.assertEqual(loaded.origin("typesafe_api_key"), ".env")

    def test_the_legacy_backup_flag_is_not_a_setting_anymore(self):
        with self.assertRaises(settings.SettingsError):
            settings.save_config({"cloud_fallback": "1"}, self.tmp)


class DisplayTests(QuietEnv):
    def test_secrets_are_masked_in_printed_output(self):
        settings.save_env("TYPESAFE_API_KEY", "sk-secret-value", self.tmp)
        loaded = settings.load_settings(self.tmp)
        document = json.dumps(loaded.to_dict())
        self.assertNotIn("sk-secret-value", document)
        self.assertIn("set", document)

    def test_secrets_are_visible_when_explicitly_asked(self):
        settings.save_env("TYPESAFE_API_KEY", "sk-secret-value", self.tmp)
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.to_dict(include_secrets=True)["typesafe_api_key"]["value"], "sk-secret-value")

    def test_the_chain_is_rendered_as_a_list(self):
        self.write("config.json", json.dumps({"provider_chain": "ollama, zen"}))
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.to_dict()["provider_chain"]["value"], ["ollama", "zen"])


class ChainValidationTests(TempDir):
    def test_order_is_kept_and_repeats_removed(self):
        self.assertEqual(settings.validate_chain("zen, ollama, zen"), ["zen", "ollama"])

    def test_semicolons_are_accepted_as_separators(self):
        self.assertEqual(settings.validate_chain("zen;typesafe"), ["zen", "typesafe"])

    def test_an_unknown_id_names_the_valid_ones(self):
        with self.assertRaises(settings.SettingsError) as caught:
            settings.validate_chain("zen,openai")
        self.assertIn("known providers", str(caught.exception))

    def test_openrouter_is_a_known_provider(self):
        self.assertEqual(settings.validate_chain("zen,openrouter"), ["zen", "openrouter"])

    def test_the_openrouter_key_is_declared_as_a_secret(self):
        """save_env writes to .env, so an unmarked secret would leak on disk."""
        self.assertTrue(settings._BY_ENV[settings.OPENROUTER_KEY_ENV].secret)
        with self.assertRaises(settings.SettingsError):
            settings.save_env(settings.PROVIDERS_ENV, "zen", self.tmp)

    def test_an_openrouter_chain_with_no_key_loads_cleanly(self):
        """A missing key is a skip later, not a settings error."""
        self.write("config.json", json.dumps({"provider_chain": "openrouter"}))
        loaded = settings.load_settings(self.tmp)
        self.assertEqual(loaded.chain(), ["openrouter"])
        self.assertEqual(loaded.get("openrouter_api_key"), "")

    def test_the_openrouter_model_defaults_to_the_known_good_one(self):
        self.assertEqual(
            settings.load_settings(self.tmp).get("openrouter_model"),
            settings.OPENROUTER_DEFAULT_MODEL,
        )


class ConfiguredTests(TempDir):
    def test_fresh_and_unconfigured(self):
        self.assertFalse(settings.configured(self.tmp))

    def test_a_saved_config_file_counts(self):
        settings.save_config({"provider_chain": "zen"}, self.tmp)
        self.assertTrue(settings.configured(self.tmp))

    def test_a_deliberate_environment_choice_counts(self):
        with mock.patch.dict(os.environ, {settings.PROVIDERS_ENV: "zen,ollama"}):
            self.assertTrue(settings.configured(self.tmp))


class ChainBuildingTests(QuietEnv):
    """providers.chain() must follow the chain the user configured."""

    def test_the_chain_order_is_the_try_order(self):
        self.write("config.json", json.dumps({"provider_chain": "ollama,zen", "ollama_model": "nimble"}))
        endpoints, _skipped = providers.chain()
        self.assertEqual([p["name"] for p in endpoints], ["ollama", "jev-free"])
        self.assertEqual(endpoints[0]["source"], "local")
        self.assertEqual(endpoints[0]["auth"], "")

    def test_the_local_endpoint_is_the_ollama_base_plus_the_wire_path(self):
        self.write("config.json", json.dumps({"provider_chain": "ollama", "ollama_model": "nimble"}))
        endpoints, _skipped = providers.chain()
        self.assertEqual(endpoints[0]["url"], "http://localhost:11434/v1/systemone")

    def test_ollama_without_a_model_is_skipped_with_a_reason(self):
        self.write("config.json", json.dumps({"provider_chain": "ollama,zen"}))
        endpoints, skipped = providers.chain()
        self.assertEqual([p["name"] for p in endpoints], ["jev-free"])
        self.assertTrue(any("setup" in reason for reason in skipped))

    def test_typesafe_in_the_chain_needs_a_key_but_not_the_flag(self):
        self.write("config.json", json.dumps({"provider_chain": "zen,typesafe"}))
        endpoints, skipped = providers.chain()
        self.assertEqual(len(endpoints), 1)
        self.assertTrue(any(settings.TYPESAFE_KEY_ENV in reason for reason in skipped))

        with mock.patch.dict(os.environ, {settings.TYPESAFE_KEY_ENV: "sk-test"}):
            endpoints, _skipped = providers.chain()
        self.assertEqual([p["name"] for p in endpoints], ["jev-free", "typesafe"])

    def test_the_legacy_backup_flag_joins_when_no_chain_is_configured(self):
        """The pre-chain contract, kept for the scripts that rely on it."""
        with mock.patch.dict(
            os.environ, {settings.TYPESAFE_KEY_ENV: "sk-test", settings.CLOUD_FALLBACK_ENV: "1"}
        ):
            endpoints, _skipped = providers.chain()
        self.assertEqual([p["name"] for p in endpoints], ["jev-free", "typesafe"])

    def test_once_a_chain_is_configured_the_flag_is_ignored(self):
        """A written chain is the entire truth; the flag may not extend it."""
        self.write("config.json", json.dumps({"provider_chain": "zen"}))
        with mock.patch.dict(
            os.environ, {settings.TYPESAFE_KEY_ENV: "sk-test", settings.CLOUD_FALLBACK_ENV: "1"}
        ):
            endpoints, _skipped = providers.chain()
        self.assertEqual([p["name"] for p in endpoints], ["jev-free"])


if __name__ == "__main__":
    unittest.main()
