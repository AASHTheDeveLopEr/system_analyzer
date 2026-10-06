"""Offline tests for the decision layer in ``system_one``.

Nothing here touches the network. The live contract tests live in
``test_live_jev.py`` and only run when asked for.

The validation cases are the interesting part. Each one exists because a
malformed body was sent to the real endpoint during development and the server
answered something less useful than a local check would have:

* ``choice`` criteria as a **list** is accepted by a naive check and rejected
  by the endpoint with ``422 Endpoint is unavailable`` -- which points at the
  service instead of at the body that caused it.
* ``score`` criteria as a **dict** does the same, for the same reason.
* ``cost`` arrives as the string ``"0"``.

Run:  python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest
from unittest import mock

import system_one
from system_one import Decision, SystemOneClient, SystemOneError, validate_request

STATE = "A desktop with 6 CPU cores, 16 GB RAM and an 8 GB discrete GPU."


def question(**kwargs):
    return {"instructions": "Is this a good machine?", **kwargs}


class ValidationTests(unittest.TestCase):
    def assertRejected(self, questions, because):
        error = validate_request(STATE, questions)
        self.assertIsNotNone(error, "expected this body to be rejected")
        self.assertIn(because, error)

    def assertAccepted(self, questions):
        self.assertIsNone(validate_request(STATE, questions))

    # -- state ------------------------------------------------------------

    def test_accepts_string_object_and_array_state(self):
        for state in ("some text", {"cpu": "fast"}, ["a", "b"]):
            self.assertIsNone(validate_request(state, {"q": question(type="noul")}))

    def test_rejects_missing_or_blank_state(self):
        for state in (None, "", "   "):
            error = validate_request(state, {"q": question(type="noul")})
            self.assertIsNotNone(error, f"state {state!r} should be rejected")
            self.assertIn("Missing state", error)

    def test_rejects_a_state_the_endpoint_cannot_take(self):
        """Jev takes text: a string, an object or an array -- not a number."""
        error = validate_request(42, {"q": question(type="noul")})
        self.assertIsNotNone(error)
        self.assertIn("Invalid state", error)

    # -- questions --------------------------------------------------------

    def test_rejects_an_empty_question_map(self):
        self.assertRejected({}, "Missing questions")

    def test_rejects_an_unknown_type(self):
        self.assertRejected({"q": question(type="chat")}, "type must be one of")

    def test_rejects_missing_instructions(self):
        self.assertRejected({"q": {"type": "noul"}}, "instructions are required")

    def test_rejects_a_non_object_question(self):
        self.assertRejected({"q": "noul please"}, "must be an object")

    # -- choice: a map of options ----------------------------------------

    def test_accepts_choice_criteria_as_a_map(self):
        self.assertAccepted(
            {"q": question(type="choice", criteria={"Yes": "it can", "No": "it cannot"})}
        )

    def test_rejects_choice_criteria_as_a_list(self):
        """The endpoint answers 422 for this, naming the wrong problem."""
        self.assertRejected(
            {"q": question(type="choice", criteria=["Yes", "No"])},
            "must be an object",
        )

    def test_rejects_choice_with_no_criteria(self):
        self.assertRejected({"q": question(type="choice")}, "choice needs criteria")

    def test_rejects_choice_with_empty_criteria(self):
        self.assertRejected({"q": question(type="choice", criteria={})}, "choice needs criteria")

    # -- score: an ordered list of levels -------------------------------

    def test_accepts_score_criteria_as_a_list(self):
        self.assertAccepted(
            {"q": question(type="score", criteria=["Entry", "Mainstream", "Enthusiast"])}
        )

    def test_rejects_score_criteria_as_a_map(self):
        """Also a 422 upstream, for the same reason as choice."""
        self.assertRejected(
            {"q": question(type="score", criteria={"0": "Entry", "1": "Mainstream"})},
            "must be a list",
        )

    def test_rejects_a_single_level_score(self):
        """The endpoint tolerates one level; a question that cannot discriminate
        is not worth a round trip, so we refuse it locally."""
        self.assertRejected(
            {"q": question(type="score", criteria=["Entry"])}, "at least 2 ordered levels"
        )

    # -- noul -------------------------------------------------------------

    def test_noul_needs_no_criteria(self):
        self.assertAccepted({"q": question(type="noul")})


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.decision = Decision(
            model="jev-1.13-free",
            answers={
                "can_run": {"type": "noul", "noul": 0.93},
                "verdict": {
                    "type": "choice",
                    "choice": "Likely",
                    "probabilities": {"Yes": 0.10, "Likely": 0.82, "No": 0.08},
                    "confidence": 0.73,
                },
                "tier": {
                    "type": "score",
                    "score": 1.05,
                    "legend": {"0": "Entry", "1": "Mainstream", "2": "Enthusiast"},
                    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                    "confidence": 0.92,
                },
            },
            provider="jev-free",
            source="free",
        )

    def test_noul(self):
        self.assertAlmostEqual(self.decision.noul("can_run"), 0.93)

    def test_noul_has_no_confidence_field(self):
        """A noul answer carries only the probability; nothing to read."""
        with self.assertRaises(SystemOneError):
            self.decision._answer("can_run", "choice")

    def test_choice(self):
        chosen, probabilities, confidence = self.decision.choice("verdict")
        self.assertEqual(chosen, "Likely")
        # Confidence is read as a percent even though the wire sends 0.73.
        self.assertAlmostEqual(confidence, 73.0)
        self.assertAlmostEqual(probabilities["Likely"], 0.82)

    def test_score_is_a_weighted_position_not_a_level_index(self):
        score, legend, _probabilities, confidence = self.decision.score("tier")
        self.assertAlmostEqual(score, 1.05)
        self.assertEqual(legend["2"], "Enthusiast")
        self.assertAlmostEqual(confidence, 92.0)

    def test_missing_answer_names_the_question(self):
        with self.assertRaises(SystemOneError) as caught:
            self.decision.noul("nope")
        self.assertIn("nope", str(caught.exception))

    def test_wrong_type_is_reported(self):
        with self.assertRaises(SystemOneError):
            self.decision.score("can_run")


class CostTests(unittest.TestCase):
    def test_string_cost_becomes_a_number(self):
        """The free endpoint sends ``"0"`` as a string, not a number."""
        self.assertEqual(system_one._as_float("0"), 0.0)
        self.assertEqual(system_one._as_float("1.5"), 1.5)

    def test_junk_cost_is_none_not_a_crash(self):
        for value in (None, "", "free", [], {}, True):
            self.assertIsNone(system_one._as_float(value))


class CostPlacementTests(unittest.TestCase):
    """Endpoints disagree about where cost lives, so both places are read.

    OpenRouter nests it under ``usage``; the free endpoint and TypeSafe send
    it top-level, as a string. A user with billing switched on needs the real
    number from whichever endpoint answered, and needs silence when neither
    place carried one.
    """

    def _cost_from(self, body):
        provider = {"name": "p", "url": "https://x.example", "model": "m",
                    "source": "s", "auth": "Bearer k"}
        client = SystemOneClient([provider])
        with mock.patch.object(system_one, "_post_json", return_value=body):
            return client.ask(STATE, {"q": question(type="noul")}).cost

    def test_cost_nested_under_usage_is_found(self):
        """OpenRouter's shape."""
        body = {"answers": {}, "usage": {"input_tokens": 476, "cost": 0.000019992}}
        self.assertAlmostEqual(self._cost_from(body), 0.000019992)

    def test_a_string_zero_cost_survives_the_fallback(self):
        """A real free-tier cost is 0.0, which is falsy.

        Reading the fallback with ``or`` would swallow it and report no cost
        at all -- the answer would be wrong in the direction of hiding a real
        number.
        """
        body = {"answers": {}, "cost": "0"}
        self.assertEqual(self._cost_from(body), 0.0)

    def test_a_top_level_cost_wins_over_usage(self):
        body = {"answers": {}, "cost": 1.5, "usage": {"cost": 2.5}}
        self.assertEqual(self._cost_from(body), 1.5)

    def test_no_cost_anywhere_is_reported_as_absent(self):
        """Not zero: absent. A guessed 0.0 would be a claim about someone's bill."""
        self.assertIsNone(self._cost_from({"answers": {}}))

    def test_a_non_dict_usage_does_not_crash_the_decode(self):
        self.assertIsNone(self._cost_from({"answers": {}, "usage": "nope"}))


class ErrorMessageTests(unittest.TestCase):
    """The endpoints do not share an error envelope; both shapes must be read."""

    def _response(self, status, body, headers=None):
        response = mock.Mock(status_code=status, headers=headers or {})
        response.json.return_value = body
        response.text = str(body)
        return response

    def test_fastapi_detail_string(self):
        response = self._response(
            400, {"detail": "Choice question must have at least one choice: q"}
        )
        self.assertIn("at least one choice", system_one._error_message(response))

    def test_opencode_error_envelope(self):
        response = self._response(
            401, {"type": "error", "error": {"type": "ModelError", "message": "Model x is not supported"}}
        )
        message = system_one._error_message(response)
        self.assertIn("not supported", message)
        self.assertIn("ModelError", message)

    def test_console_error_envelope(self):
        response = self._response(
            422, {"error": {"type": "server_error", "message": "Error from provider (Console)"}}
        )
        self.assertIn("Console", system_one._error_message(response))

    def test_non_json_body_is_still_reported(self):
        response = mock.Mock(status_code=502, headers={}, text="<html>bad gateway</html>")
        response.json.side_effect = ValueError("not json")
        self.assertIn("bad gateway", system_one._error_message(response))

    def test_retry_after_in_seconds(self):
        response = self._response(429, {"detail": "slow down"}, headers={"Retry-After": "2"})
        self.assertEqual(system_one._retry_after(response), 2.0)

    def test_retry_after_absent_means_no_wait(self):
        self.assertEqual(system_one._retry_after(self._response(429, {})), 0.0)


class ProviderChainTests(unittest.TestCase):
    def _client(self, responses):
        """A client whose endpoints fail or succeed in the given order."""
        providers = [
            {"name": "first", "url": "https://a.example", "model": "m1",
             "source": "free", "auth": ""},
            {"name": "second", "url": "https://b.example", "model": "m2",
             "source": "paid", "auth": "Bearer secret"},
        ]
        client = SystemOneClient(providers)
        return client, mock.Mock(side_effect=responses)

    def test_a_failure_falls_through_to_the_next_endpoint(self):
        client, post = self._client(
            [SystemOneError("HTTP 401: nope"),
             {"model": "m2", "answers": {"q": {"type": "noul", "noul": 0.5}}, "cost": "0"}]
        )
        with mock.patch.object(system_one, "_post_json", post):
            decision = client.ask(STATE, {"q": question(type="noul")})

        self.assertEqual(decision.provider, "second")
        self.assertEqual(decision.cost, 0.0)
        self.assertIsInstance(decision.cost, float)

    def test_every_failure_is_reported_together(self):
        client, post = self._client(
            [SystemOneError("HTTP 500: first"), SystemOneError("HTTP 429: second")]
        )
        with mock.patch.object(system_one, "_post_json", post):
            with self.assertRaises(SystemOneError) as caught:
                client.ask(STATE, {"q": question(type="noul")})

        self.assertIn("first", str(caught.exception))
        self.assertIn("second", str(caught.exception))

    def test_an_invalid_body_fails_before_any_request(self):
        client, post = self._client([{"model": "m1", "answers": {}}])
        with mock.patch.object(system_one, "_post_json", post):
            with self.assertRaises(SystemOneError):
                client.ask(STATE, {"q": question(type="choice", criteria=["Yes"])})
        post.assert_not_called()


class AuthorizationHeaderTests(unittest.TestCase):
    """The local Ollama endpoint takes no auth: an empty auth
    means the header is omitted entirely, never filled with a
    fabricated token."""

    def _posted_headers(self, auth):
        provider = {"name": "fake", "url": "https://x.example/v1/systemone",
                    "model": "m", "source": "free", "auth": auth}
        response = mock.Mock(status_code=200)
        response.json.return_value = {"model": "m", "answers": {}}
        with mock.patch.object(system_one.requests, "post", return_value=response) as post:
            client = SystemOneClient([provider])
            client.ask(STATE, {"q": question(type="noul")})
        return post.call_args.kwargs["headers"]

    def test_no_authorization_header_when_auth_is_empty(self):
        headers = self._posted_headers("")
        self.assertNotIn("Authorization", headers)

    def test_the_token_rides_in_the_authorization_header(self):
        headers = self._posted_headers("Bearer secret")
        self.assertEqual(headers["Authorization"], "Bearer secret")


if __name__ == "__main__":
    unittest.main()
