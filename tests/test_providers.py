"""Offline tests for the provider catalog: discovery, guidance, and the chain.

Every Ollama probe failure mode gets its own case, because the guidance is
only as good as the diagnosis behind it: nothing listening, something else
listening, Ollama without decision models, and Ollama with the configured
model missing are four different messages to the user.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import providers
import settings
from tests._helpers import IsolatedSettings, TempDir


def _response(status: int = 200, body: object = None, json_error: Exception | None = None):
    response = mock.Mock()
    response.status_code = status
    if json_error is not None:
        response.json.side_effect = json_error
    else:
        response.json.return_value = body
    response.text = str(body)
    return response


OLLAMA_BODY = {
    "models": [
        {"name": "tev1:0.8b", "capabilities": ["decision", "tools", "completion"]},
        {"name": "Qwen3.8-2B", "capabilities": ["tools", "thinking", "completion"]},
        {"name": "nameless", "capabilities": ["decision"]},  # odd name, but decision-capable
        {"name": 42, "capabilities": ["decision"]},          # not a name
        "not even an object",
    ]
}


class UrlResolutionTests(TempDir):
    def test_an_explicit_setting_wins(self):
        self.write("config.json", json.dumps({"ollama_url": "https://ai.box.example:4000"}))
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop(providers.OLLAMA_HOST_ENV, None)
            self.assertEqual(
                providers.resolve_ollama_url(settings.load_settings(self.tmp)),
                "https://ai.box.example:4000",
            )

    def test_ollama_host_with_a_port_is_used_as_the_client_target(self):
        with mock.patch.dict("os.environ", {providers.OLLAMA_HOST_ENV: "box:1234"}):
            self.assertEqual(providers.resolve_ollama_url(settings.load_settings(self.tmp)), "http://box:1234")

    def test_ollama_host_without_a_port_gets_ollama_default_port(self):
        with mock.patch.dict("os.environ", {providers.OLLAMA_HOST_ENV: "box"}):
            self.assertEqual(providers.resolve_ollama_url(settings.load_settings(self.tmp)), "http://box:11434")

    def test_a_bind_style_address_is_read_as_this_machine(self):
        with mock.patch.dict("os.environ", {providers.OLLAMA_HOST_ENV: "0.0.0.0:11434"}):
            self.assertEqual(providers.resolve_ollama_url(settings.load_settings(self.tmp)), "http://127.0.0.1:11434")
        with mock.patch.dict("os.environ", {providers.OLLAMA_HOST_ENV: ":11434"}):
            self.assertEqual(providers.resolve_ollama_url(settings.load_settings(self.tmp)), "http://127.0.0.1:11434")

    def test_the_default_is_the_local_default_port(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop(providers.OLLAMA_HOST_ENV, None)
            self.assertEqual(providers.resolve_ollama_url(settings.load_settings(self.tmp)), providers.DEFAULT_OLLAMA_URL)


class OllamaDiscoveryTests(TempDir):
    def test_only_decision_capable_models_are_offered(self):
        with mock.patch.object(providers.requests, "get", return_value=_response(200, OLLAMA_BODY)):
            status = providers.discover_ollama("http://localhost:11434")
        self.assertTrue(status.reachable)
        self.assertEqual(status.models, ["tev1:0.8b", "nameless"])

    def test_a_dead_port_reports_a_connection_failure(self):
        error = providers.requests.ConnectionError("refused")
        with mock.patch.object(providers.requests, "get", side_effect=error):
            status = providers.discover_ollama("http://localhost:11434")
        self.assertFalse(status.reachable)
        self.assertIn("connection failed", status.detail)

    def test_something_else_on_the_port_is_reported_as_such(self):
        with mock.patch.object(providers.requests, "get", return_value=_response(500, {})):
            status = providers.discover_ollama("http://localhost:11434")
        self.assertFalse(status.reachable)
        self.assertIn("HTTP 500", status.detail)

    def test_non_json_is_reported(self):
        with mock.patch.object(providers.requests, "get", return_value=_response(200, None, json_error=ValueError())):
            status = providers.discover_ollama("http://localhost:11434")
        self.assertFalse(status.reachable)
        self.assertIn("not JSON", status.detail)

    def test_a_wrong_shape_is_reported(self):
        with mock.patch.object(providers.requests, "get", return_value=_response(200, ["models"])):
            status = providers.discover_ollama("http://localhost:11434")
        self.assertFalse(status.reachable)
        self.assertIn("not an Ollama model list", status.detail)


class OllamaGuidanceTests(TempDir):
    def setUp(self):
        super().setUp()
        self.config = settings.load_settings(self.tmp)

    def test_an_unreachable_default_address_says_to_start_ollama(self):
        status = providers.OllamaStatus(False, [], "connection failed: refused")
        lines = providers.ollama_guidance(status, providers.DEFAULT_OLLAMA_URL, self.config)
        self.assertTrue(any("start it" in line for line in lines))

    def test_an_unreachable_custom_address_points_at_the_setting(self):
        self.write("config.json", json.dumps({"ollama_url": "http://box:1234"}))
        config = settings.load_settings(self.tmp)
        status = providers.OllamaStatus(False, [], "connection failed: refused")
        lines = providers.ollama_guidance(status, "http://box:1234", config)
        self.assertTrue(any("Settings" in line for line in lines))

    def test_no_decision_models_offers_the_pull_command(self):
        status = providers.OllamaStatus(True, [], "")
        lines = providers.ollama_guidance(status, providers.DEFAULT_OLLAMA_URL, self.config)
        self.assertTrue(any("ollama pull" in line for line in lines))
        self.assertTrue(any(providers.RECOMMENDED_PULL_MODEL in line for line in lines))

    def test_a_configured_model_that_is_not_installed_is_named(self):
        self.write("config.json", json.dumps({"ollama_model": "nimble"}))
        config = settings.load_settings(self.tmp)
        status = providers.OllamaStatus(True, ["tev1:0.8b"], "")
        lines = providers.ollama_guidance(status, providers.DEFAULT_OLLAMA_URL, config)
        self.assertTrue(any("nimble" in line and "tev1:0.8b" in line for line in lines))

    def test_a_ready_setup_needs_no_guidance(self):
        status = providers.OllamaStatus(True, ["tev1:0.8b"], "")
        self.assertEqual(providers.ollama_guidance(status, providers.DEFAULT_OLLAMA_URL, self.config), [])


class TypeSafeDiscoveryTests(TempDir):
    def test_without_a_key_the_models_cannot_be_listed(self):
        models, reason = providers.discover_typesafe("")
        self.assertIsNone(models)
        self.assertIn("key", reason)

    def test_a_rejected_key_is_reported_as_rejected(self):
        with mock.patch.object(providers.requests, "get", return_value=_response(403, {})):
            models, reason = providers.discover_typesafe("sk-bad")
        self.assertIsNone(models)
        self.assertIn("rejected", reason)

    def test_the_model_names_are_read_out_of_the_envelope(self):
        body = {"models": [{"name": "jev-latest"}, {"name": "jev-preview"}]}
        with mock.patch.object(providers.requests, "get", return_value=_response(200, body)):
            models, reason = providers.discover_typesafe("sk-test")
        self.assertEqual(models, ["jev-latest", "jev-preview"])
        self.assertEqual(reason, "")

    def test_a_network_failure_is_reported(self):
        with mock.patch.object(providers.requests, "get", side_effect=providers.requests.ConnectionError("down")):
            models, reason = providers.discover_typesafe("sk-test")
        self.assertIsNone(models)
        self.assertIn("failed", reason)


class OllamaProviderTests(TempDir):
    def test_without_a_model_the_provider_is_skipped_with_a_reason(self):
        config = settings.load_settings(self.tmp)
        provider, reason = providers.ollama_provider(config)
        self.assertIsNone(provider)
        self.assertIn("setup", reason)

    def test_with_a_model_the_endpoint_is_the_wire_path_on_the_base(self):
        self.write("config.json", json.dumps({"ollama_model": "tev1:0.8b"}))
        provider, reason = providers.ollama_provider(settings.load_settings(self.tmp))
        self.assertEqual(reason, "")
        self.assertEqual(provider["url"], "http://localhost:11434/v1/systemone")
        self.assertEqual(provider["source"], "local")
        self.assertEqual(provider["auth"], "")


class ChainOptInTests(IsolatedSettings):
    """Cloud escalation must stay off unless the user asked for it."""

    def setUp(self):
        super().setUp()
        self.environ = mock.patch.dict(os.environ, {}, clear=False)
        self.environ.start()
        for name in (settings.TYPESAFE_KEY_ENV, settings.CLOUD_FALLBACK_ENV, settings.ZEN_KEY_ENV):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environ.stop()

    def test_paid_is_off_without_a_key(self):
        provider, reason = providers.paid_provider()
        self.assertIsNone(provider)
        self.assertIn(settings.TYPESAFE_KEY_ENV, reason)

    def test_paid_is_off_without_the_opt_in_even_with_a_key(self):
        os.environ[settings.TYPESAFE_KEY_ENV] = "sk-test"
        provider, reason = providers.paid_provider()
        self.assertIsNone(provider)
        self.assertIn(settings.CLOUD_FALLBACK_ENV, reason)

    def test_paid_joins_when_both_are_set(self):
        os.environ[settings.TYPESAFE_KEY_ENV] = "sk-test"
        os.environ[settings.CLOUD_FALLBACK_ENV] = "1"
        provider, reason = providers.paid_provider()
        self.assertIsNotNone(provider)
        self.assertEqual(provider["url"], providers.PAID_URL)
        self.assertEqual(reason, "")

    def test_free_works_without_a_key(self):
        endpoints, _ = providers.chain()
        self.assertEqual(endpoints[0]["name"], "jev-free")
        # No key means no auth header: "Bearer public" would be a
        # credential Zen never issued.
        self.assertEqual(endpoints[0]["auth"], "")

    def test_a_free_key_is_used_when_present(self):
        """So the tool survives the day the open tier closes."""
        os.environ[settings.ZEN_KEY_ENV] = "zen-test"
        endpoints, _ = providers.chain()
        self.assertEqual(endpoints[0]["auth"], "Bearer zen-test")

    def test_the_key_is_never_part_of_the_printed_provider_summary(self):
        """The probe prints provider fields; a key must not ride along."""
        os.environ[settings.ZEN_KEY_ENV] = "zen-secret"
        os.environ[settings.TYPESAFE_KEY_ENV] = "sk-secret"
        os.environ[settings.CLOUD_FALLBACK_ENV] = "1"
        endpoints, _ = providers.chain()
        shown = " ".join(
            f"{p['name']} {p['url']} {p['model']} {p['source']}" for p in endpoints
        )
        self.assertNotIn("zen-secret", shown)
        self.assertNotIn("sk-secret", shown)


class DefaultChainTests(IsolatedSettings):
    """Pins the standing decision: the default chain is one remote free endpoint.

    A local provider appearing in the *default* chain is not a bug, so this
    test failing means the decision changed on purpose -- update it and this
    file together rather than deleting the test.
    """

    def setUp(self):
        super().setUp()
        self.environ = mock.patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        # The clear above wiped the settings pointer; put it back so a config
        # file left in the repo cannot change the defaults under test here.
        os.environ["SYSTEM_ANALYZER_SETTINGS_DIR"] = str(self._settings_tmp)

    def tearDown(self):
        self.environ.stop()

    def test_the_default_chain_is_one_remote_free_endpoint(self):
        endpoints, _reasons = providers.chain()
        self.assertEqual(len(endpoints), 1)
        self.assertEqual(endpoints[0]["name"], "jev-free")
        self.assertEqual(endpoints[0]["url"], providers.FREE_URL)
        self.assertEqual(endpoints[0]["model"], "jev-1.13-free")
        self.assertTrue(endpoints[0]["url"].startswith("https://"))

    def test_no_provider_talks_to_a_local_address(self):
        """The default chain stays remote; 'local' would mean a socket here."""
        endpoints, _reasons = providers.chain()
        for provider in endpoints:
            self.assertNotIn("localhost", provider["url"])
            self.assertNotIn("127.0.0.1", provider["url"])
            self.assertNotIn("0.0.0.0", provider["url"])


if __name__ == "__main__":
    unittest.main()
