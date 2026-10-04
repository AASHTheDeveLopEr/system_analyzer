"""Tests for the CLI in main.py.

The point of these is the contract a script depends on: ``--json`` stdout is
nothing but JSON, hints go to stderr, and the exit code says what happened.
Detection and the model are both stubbed, so none of this needs a network or a
particular machine.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import main as cli
import inventory
import system_one
from system_one import SystemOneError
from tests._helpers import (
    FakeClient,
    TempDir,
    capability_answers,
    sample_inventory,
    score,
)


def rating_answers(value: float = 0.5) -> dict:
    return {"overall_tier": score(value, {"0": "Entry level", "4": "Workstation"})}


class CliCase(TempDir):
    """Runs the CLI with detection and the model replaced."""

    def run_cli(self, *argv: str, answers: dict | None = None, error: Exception | None = None) -> tuple[int, str, str]:
        client = FakeClient(answers if answers is not None else rating_answers(), error=error)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with redirect_stdout(out), redirect_stderr(err):
                    code = cli.main(["--history-file", str(self.tmp / "history.json"), *argv])
        return code, out.getvalue(), err.getvalue()

    def history_path(self):
        return self.tmp / "history.json"


class SpecsTests(CliCase):
    def test_specs_prints_what_was_detected(self) -> None:
        code, out, _err = self.run_cli("specs")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Ryzen 5 3600", out)

    def test_specs_json_is_the_scrubbed_state(self) -> None:
        code, out, _err = self.run_cli("specs", "--json")
        payload = json.loads(out)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(payload["inventory"]["cpu"]["logical_cores"], 6)
        self.assertEqual(payload["inventory"]["memory"]["total_gib"], 16.0)

    def test_an_unreadable_component_is_shown_as_unreadable(self) -> None:
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory(gpus=())):
            client = FakeClient(rating_answers())
            out = io.StringIO()
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    cli.main(["--history-file", str(self.history_path()), "specs"])
        self.assertIn("Not readable", out.getvalue())


class RateTests(CliCase):
    def test_rate_prints_a_number_and_its_reasons(self) -> None:
        code, out, _err = self.run_cli("rate")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("/100", out)
        self.assertIn("What moved it", out)
        self.assertIn("Source", out)

    def test_rate_json_is_valid_json_with_the_working_attached(self) -> None:
        code, out, _err = self.run_cli("rate", "--json")
        payload = json.loads(out)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIsInstance(payload["total"], int)
        self.assertEqual(len(payload["subscores"]), 4)
        self.assertTrue(payload["factors"])

    def test_a_dead_endpoint_still_rates_and_says_it_degraded(self) -> None:
        code, out, _err = self.run_cli("rate", error=SystemOneError("offline"))
        self.assertEqual(code, cli.EXIT_NO_MODEL)
        self.assertIn("/100", out)
        self.assertIn("local rules", out)

    def test_a_dead_endpoint_in_json_mode_still_emits_valid_json(self) -> None:
        code, out, _err = self.run_cli("rate", "--json", error=SystemOneError("offline"))
        payload = json.loads(out)
        self.assertEqual(code, cli.EXIT_NO_MODEL)
        self.assertEqual(payload["method"], "rules")
        self.assertIsNone(payload["model"])

    def test_json_mode_keeps_stdout_free_of_prose(self) -> None:
        # The whole contract of --json: stdout parses, everything else does not.
        _code, out, _err = self.run_cli("rate", "--json")
        json.loads(out)

    def test_json_mode_puts_the_human_hints_on_stderr(self) -> None:
        _code, out, err = self.run_cli("rate", "--json", error=SystemOneError("offline"))
        json.loads(out)
        self.assertIn("note:", err)

    def test_repeat_reports_the_spread_and_says_whether_it_was_stable(self) -> None:
        code, out, _err = self.run_cli("rate", "--repeat", "3", "--json")
        payload = json.loads(out)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(len(payload["totals"]), 3)
        self.assertEqual(payload["spread"], 0)
        self.assertTrue(payload["stable"])


class AskTests(CliCase):
    def _run_ask_tty(
        self,
        *argv: str,
        answers: dict | None = None,
        replies: list[str] | None = None,
    ) -> tuple[int, str, str, FakeClient]:
        """Run one-shot ``ask`` as if both ends of the console were a TTY.

        ``_interactive_terminal`` is patched directly: a test's redirected
        stdout is never a real terminal, and the offer only fires when
        both sides agree. ``replies`` feeds the paste blocks; an empty
        first reply ends the block immediately, so no checklist runs.
        """
        client = FakeClient(answers)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch.object(cli, "_interactive_terminal", return_value=True):
                    with mock.patch("builtins.input", side_effect=replies or [""]):
                        with redirect_stdout(out), redirect_stderr(err):
                            code = cli.main(["--history-file", str(self.history_path()), *argv])
        return code, out.getvalue(), err.getvalue(), client

    def test_ask_prints_a_verdict_and_a_confidence(self) -> None:
        code, out, _err = self.run_cli("ask", "Can I run Cyberpunk 2077?", answers=capability_answers(0.95))
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Yes", out)
        self.assertIn("Confidence", out)
        self.assertIn("Cyberpunk", out)

    def test_an_ungrounded_question_exits_four_and_says_cannot_tell(self) -> None:
        answers = capability_answers(0.9, grounding="specs_do_not_decide_it")
        code, out, _err = self.run_cli("ask", "Can I stream on Twitch?", answers=answers)
        self.assertEqual(code, cli.EXIT_UNGROUNDED)
        self.assertIn("Cannot tell", out)
        self.assertIn("requirements", out)

    def test_an_ungrounded_verdict_is_offered_the_requirements_checklist(self) -> None:
        # The offer exists for verdicts the specs cannot settle, and
        # one-shot ask is exactly where a human meets them: it must
        # offer the checklist the same way the interactive session does.
        answers = capability_answers(0.9, grounding="specs_do_not_decide_it")
        code, out, _err, client = self._run_ask_tty(
            "ask", "Can I stream on Twitch?", answers=answers, replies=[""]
        )
        self.assertEqual(code, cli.EXIT_UNGROUNDED)
        self.assertIn("Cannot tell", out)
        self.assertIn("MINIMUM requirements", out)
        # Entering nothing ends the paste block, so no checklist runs.
        self.assertEqual(len(client.calls), 1)

    def test_a_confident_grounded_verdict_is_not_offered_the_checklist(self) -> None:
        answers = capability_answers(0.95)
        code, out, _err, _client = self._run_ask_tty(
            "ask", "Can I game?", answers=answers, replies=[""]
        )
        self.assertEqual(code, cli.EXIT_OK)
        self.assertNotIn("The model could not judge", out)
        self.assertNotIn("MINIMUM requirements", out)

    def test_json_mode_never_stops_for_the_offer(self) -> None:
        # A script cannot answer an interactive ask-back: it gets a
        # machine-readable pointer in the document instead, and its
        # stdout stays pure JSON.
        answers = capability_answers(0.9, grounding="specs_do_not_decide_it")
        code, out, _err, client = self._run_ask_tty(
            "ask", "Can I stream on Twitch?", "--json", answers=answers, replies=[""]
        )
        self.assertEqual(code, cli.EXIT_UNGROUNDED)
        self.assertNotIn("MINIMUM requirements", out)
        self.assertEqual(len(client.calls), 1)
        payload = json.loads(out)
        self.assertTrue(payload["requirements_needed"])
        self.assertIn("hint", payload)

    def test_ask_with_no_question_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_cli("ask")
        self.assertEqual(raised.exception.code, cli.EXIT_USAGE)

    def test_an_unknown_command_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_cli("frobnicate")
        self.assertEqual(raised.exception.code, cli.EXIT_USAGE)

    def test_a_question_on_the_wrong_command_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.run_cli("rate", "why?")
        self.assertEqual(raised.exception.code, cli.EXIT_USAGE)

    def test_a_dead_endpoint_exits_three_with_no_verdict_invented(self) -> None:
        code, out, _err = self.run_cli("ask", "can I game?", error=SystemOneError("offline"))
        self.assertEqual(code, cli.EXIT_NO_MODEL)
        self.assertIn("no verdict", out.lower())

    def test_a_dead_endpoint_in_json_mode_says_so_in_the_document(self) -> None:
        code, out, _err = self.run_cli("ask", "can I game?", "--json", error=SystemOneError("offline"))
        payload = json.loads(out)
        self.assertEqual(code, cli.EXIT_NO_MODEL)
        self.assertEqual(payload["verdict"], None)
        self.assertEqual(payload["error"], "no_endpoint_answered")

    def test_repeat_reports_whether_the_verdict_moves(self) -> None:
        code, out, _err = self.run_cli(
            "ask", "can I game?", "--repeat", "3", "--json", answers=capability_answers(0.95)
        )
        payload = json.loads(out)
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(payload["verdicts"], ["Yes", "Yes", "Yes"])
        self.assertTrue(payload["stable"])


class HistoryTests(CliCase):
    def test_questions_are_logged_with_their_verdict_and_source(self) -> None:
        self.run_cli("ask", "can I game?", answers=capability_answers(0.95))
        entries = json.loads(self.history_path().read_text(encoding="utf-8"))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["verdict"], "Yes")
        self.assertEqual(entries[0]["model"], "fake-model")
        self.assertIn("at", entries[0])

    def test_no_history_leaves_the_file_alone(self) -> None:
        self.run_cli("--no-history", "ask", "can I game?", answers=capability_answers(0.95))
        self.assertFalse(self.history_path().exists())

    def test_history_shows_what_was_asked(self) -> None:
        self.run_cli("ask", "can I game?", answers=capability_answers(0.95))
        code, out, _err = self.run_cli("history")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("can I game?", out)

    def test_history_is_json_on_request(self) -> None:
        self.run_cli("ask", "can I game?", answers=capability_answers(0.95))
        _code, out, _err = self.run_cli("history", "--json")
        self.assertEqual(len(json.loads(out)), 1)

    def test_an_empty_history_is_not_an_error(self) -> None:
        code, out, _err = self.run_cli("history")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("No questions", out)

    def test_clear_removes_the_log(self) -> None:
        self.run_cli("ask", "can I game?", answers=capability_answers(0.95))
        code, _out, _err = self.run_cli("clear")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertFalse(self.history_path().exists())

    def test_a_corrupt_history_file_does_not_stop_the_app(self) -> None:
        self.write("history.json", "{not json")
        code, out, _err = self.run_cli("ask", "can I game?", answers=capability_answers(0.95))
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("can I game?", out)
        self.assertIn("could not read the history", _err)


class PayloadAndConfigTests(CliCase):
    def test_payload_shows_exactly_what_would_be_sent(self) -> None:
        _code, out, _err = self.run_cli("payload")
        payload = json.loads(out)
        self.assertIn("state", payload)
        self.assertEqual(payload["state"]["cpu"]["logical_cores"], 6)
        self.assertTrue(payload["providers"])

    def test_config_prints_the_thresholds_in_force(self) -> None:
        _code, out, _err = self.run_cli("config")
        payload = json.loads(out)
        self.assertIn("verdict_bands", payload)
        self.assertIn("rating_weights", payload)

    def test_a_config_file_changes_the_answer(self) -> None:
        path = self.write("strict.json", json.dumps({"verdict_bands": {"yes_at": 0.99}}))
        code, out, _err = self.run_cli("--config", str(path), "ask", "can I game?", answers=capability_answers(0.95))
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Likely", out)

    def test_a_broken_config_is_a_usage_error_not_a_traceback(self) -> None:
        path = self.write("bad.json", "{nope")
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=FakeClient(rating_answers())):
                with redirect_stdout(out), redirect_stderr(err):
                    code = cli.main(["--config", str(path), "rate"])
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("config error", err.getvalue())
        self.assertEqual(out.getvalue(), "")


class SessionTests(CliCase):
    def _run_session(self, replies: list[str], tty: bool = True) -> tuple[int, str, str, FakeClient]:
        client = FakeClient(capability_answers(0.95))
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch("builtins.input", side_effect=replies):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=tty):
                        with redirect_stdout(out), redirect_stderr(err):
                            code = cli.main(["--history-file", str(self.history_path()), "session"])
        return code, out.getvalue(), err.getvalue(), client

    def test_a_session_answers_each_question_in_turn(self) -> None:
        code, out, _err, client = self._run_session(["can I game?", "can I edit 4K video?", "quit"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("game", client.calls[0]["questions"]["verdict"]["instructions"])
        self.assertIn("4K video", client.calls[1]["questions"]["verdict"]["instructions"])
        self.assertIn("Yes", out)

    def test_quit_ends_the_session_without_asking_anything(self) -> None:
        _code, _out, _err, client = self._run_session(["quit"])
        self.assertEqual(client.calls, [])

    def test_a_question_is_answered_even_when_the_endpoint_is_dead(self) -> None:
        client = FakeClient(error=SystemOneError("offline"))
        out = io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch("builtins.input", side_effect=["can I game?", "quit"]):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with redirect_stdout(out), redirect_stderr(io.StringIO()):
                            code = cli.main(["--history-file", str(self.history_path()), "session"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("No verdict this time", out.getvalue())

    def test_a_piped_question_is_answered_and_the_pipe_ends_it(self) -> None:
        client = FakeClient(capability_answers(0.95))
        out = io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch.object(cli.sys, "stdin", io.StringIO("can I game?\n")):
                    with redirect_stdout(out), redirect_stderr(io.StringIO()):
                        cli.main(["--history-file", str(self.history_path()), "session"])
        self.assertEqual(len(client.calls), 1)
        self.assertIn("Yes", out.getvalue())

    def _run_menu(self, choices: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=FakeClient(capability_answers(0.95))):
                with mock.patch("builtins.input", side_effect=choices):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with redirect_stdout(out), redirect_stderr(err):
                            code = cli.main(["--history-file", str(self.history_path()), "menu"])
        return code, out.getvalue(), err.getvalue()

    def test_the_menu_offers_the_features_the_app_actually_has(self) -> None:
        out = io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=FakeClient(rating_answers())):
                with mock.patch("builtins.input", side_effect=["0"]):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with redirect_stdout(out), redirect_stderr(io.StringIO()):
                            cli.main([])
        text = out.getvalue()
        for label in ("Show my specs", "Rate my PC", "Ask a question", "History"):
            self.assertIn(label, text)

    def test_every_numbered_menu_option_actually_runs(self) -> None:
        # A menu that prints a label is not a menu that works. Each option gets
        # driven in turn, so a wrong call signature or a stub raises here
        # instead of on the one evening someone picks that number.
        code, out, _err = self._run_menu(["1", "3", "can I game?", "4", "0"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Ryzen 5 3600", out)          # 1. specs
        self.assertIn("Verdict", out)                # 3. ask, and the typed question
        self.assertIn("90% confident", out)          # 4. history, showing the ask above

    def test_typing_a_question_at_the_menu_prompt_asks_it(self) -> None:
        # The menu is a convenience, not a gate: the question itself is a valid
        # answer to "Ask or choose >".
        code, out, _err = self._run_menu(["can I game?", "0"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Verdict", out)

    def test_a_wrong_number_gets_a_hint_not_a_model_question(self) -> None:
        # Typing "7" at the menu is a keypress mistake; it must never reach the
        # model as a question about what this machine can do.
        client = FakeClient(capability_answers(0.95))
        out = io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch("builtins.input", side_effect=["9", "0"]):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with redirect_stdout(out), redirect_stderr(io.StringIO()):
                            code = cli.main(["--history-file", str(self.history_path()), "menu"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("not a menu choice", out.getvalue())
        self.assertEqual(client.calls, [])

    def test_typing_check_at_the_prompt_runs_the_checklist_without_a_model_call(self) -> None:
        # "check <task>" is the fast path for requirements already in hand; it
        # must not spend an open-ended model call first.
        client = FakeClient(capability_answers(0.95))
        out = io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch("builtins.input", side_effect=["check Office", "4 GB RAM", "", "", "0"]):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with redirect_stdout(out), redirect_stderr(io.StringIO()):
                            code = cli.main(["--history-file", str(self.history_path()), "menu"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Checklist  : Office", out.getvalue())
        self.assertIn("Meets the minimum", out.getvalue())
        self.assertEqual(client.calls, [])

    def test_option_3_accepts_check_to_go_straight_to_the_checklist(self) -> None:
        # Option 3 answers questions; typing "check <task>" into it reaches the
        # same checklist, so the feature needs no menu slot of its own.
        client = FakeClient(capability_answers(0.95))
        out = io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch(
                    "builtins.input",
                    side_effect=["3", "check Office", "4 GB RAM", "", "", "0"],
                ):
                    with mock.patch.object(cli.sys.stdin, "isatty", return_value=True):
                        with redirect_stdout(out), redirect_stderr(io.StringIO()):
                            code = cli.main(["--history-file", str(self.history_path()), "menu"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Checklist  : Office", out.getvalue())
        self.assertIn("Meets the minimum", out.getvalue())
        self.assertEqual(client.calls, [])


class CheckTests(CliCase):
    def test_a_checklist_compares_numbers_against_this_machine(self) -> None:
        code, out, _err = self.run_cli("check", "Office", "--min-reqs", "4 GB RAM; 2 GB storage")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Office", out)
        self.assertIn("pass", out)

    def test_below_the_minimum_names_the_requirement_and_fails(self) -> None:
        code, out, _err = self.run_cli("check", "Game", "--min-reqs", "64 GB RAM")
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("FAIL", out)
        self.assertIn("Below the minimum", out)

    def test_check_without_requirements_is_a_usage_error(self) -> None:
        code, _out, err = self.run_cli("check", "Office")
        self.assertEqual(code, cli.EXIT_USAGE)
        self.assertIn("no requirements given", err)


class ProgressLineTests(CliCase):
    """The wait for a model answer is announced, not silent."""

    PROVIDERS = [{"name": "fake", "url": "https://x.example", "model": "m",
                  "source": "free", "timeout": 7}]

    def test_the_wait_is_explained_before_the_model_is_asked(self) -> None:
        client = FakeClient(rating_answers(), providers=self.PROVIDERS)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with redirect_stdout(out), redirect_stderr(err):
                    code = cli.main(["--history-file", str(self.history_path()), "rate"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("Asking fake", out.getvalue())
        self.assertIn("up to 7 s", out.getvalue())

    def test_json_mode_keeps_the_progress_line_off_stdout(self) -> None:
        client = FakeClient(rating_answers(), providers=self.PROVIDERS)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with redirect_stdout(out), redirect_stderr(err):
                    cli.main(["--history-file", str(self.history_path()), "rate", "--json"])
        json.loads(out.getvalue())  # stdout is still pure JSON
        self.assertIn("Asking fake", err.getvalue())

    def test_a_client_without_endpoints_stays_silent(self) -> None:
        client = FakeClient(rating_answers(), providers=[])
        out, _err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    cli.main(["--history-file", str(self.history_path()), "rate"])
        self.assertNotIn("Asking", out.getvalue())

    def test_the_source_line_reports_how_long_the_model_took(self) -> None:
        client = FakeClient(rating_answers(), providers=self.PROVIDERS)
        out, _err = io.StringIO(), io.StringIO()
        with mock.patch.object(inventory, "detect_inventory", return_value=sample_inventory()):
            with mock.patch.object(cli, "SystemOneClient", return_value=client):
                with mock.patch.object(cli.time, "monotonic", side_effect=[0.0, 5.0]):
                    with redirect_stdout(out), redirect_stderr(io.StringIO()):
                        code = cli.main(["--history-file", str(self.history_path()), "rate"])
        self.assertEqual(code, cli.EXIT_OK)
        self.assertIn("(5.0 s)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
