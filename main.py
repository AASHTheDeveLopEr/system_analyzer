#!/usr/bin/env python3
"""System Analyzer -- one entry point for the whole app.

One-shot commands, or an interactive session::

    python main.py specs                  # what was detected, and what is unknown
    python main.py rate                   # 0-100 with the factors behind it
    python main.py ask "Can I run Cyberpunk 2077?"
    python main.py ask "stream on Twitch" --repeat 5   # how stable is the answer?
    python main.py check "Office" --min-reqs "4 GB RAM"
    python main.py session                # interactive: menu, or just type questions
    python main.py history                # what has been asked so far
    python main.py payload                # the exact payload that leaves the machine
    python main.py config                 # the thresholds in force right now

Machine-readable mode puts JSON on stdout and every hint on stderr, with exit
codes that mean something::

    python main.py rate --json | jq .total
    python main.py ask "..." --json; echo $?

    0  worked
    1  something went wrong
    2  bad command line, or a config.json that cannot be used
    3  no System One endpoint answered -- degraded, output is still valid
    4  the question could not be grounded in your hardware -- no verdict given

Everything a rating or verdict is built from is in ``--json`` output as well:
the per-component sub-scores, the model, and whether the number is estimated
from partial readings. Nothing is hidden behind a formatted string.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config as config_module
import history as history_module
import inventory
import judgment
import providers as providers_module
import requirements as requirements_module
import settings as settings_module
import system_one
from history import History
from system_one import SystemOneClient, SystemOneError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2  # what argparse itself uses
EXIT_NO_MODEL = 3
EXIT_UNGROUNDED = 4


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


class Reporter:
    """Where text goes, which depends on whether stdout is data or prose.

    In ``--json`` mode stdout belongs to the JSON document alone, so every
    human-facing line is pushed to stderr. A script that pipes this into jq
    must not have to strip banners out of its input.
    """

    def __init__(self, as_json: bool, stream: Any = None, errors: Any = None) -> None:
        self.as_json = as_json
        self.stream = stream or sys.stdout
        self.errors = errors or sys.stderr

    def say(self, text: str = "") -> None:
        print(text, file=self.errors if self.as_json else self.stream)

    def note(self, text: str) -> None:
        """A hint for the human. Never on stdout, in either mode."""
        print(text, file=self.errors)

    def data(self, payload: Any) -> None:
        print(json.dumps(payload, indent=2, sort_keys=False), file=self.stream)


def _provenance(decision_or_rating: Any) -> str:
    """Every judgment says where it came from."""
    source = getattr(decision_or_rating, "source", "rules")
    model = getattr(decision_or_rating, "model", None)
    provider = getattr(decision_or_rating, "provider", None)
    if source == "rules" or not model:
        return "local rules (no model involved)"
    return f"{source} via {provider or 'jev'} / {model}"


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def _detect() -> Any:
    """This machine's hardware inventory."""
    return inventory.detect_inventory()


def _client() -> SystemOneClient:
    return SystemOneClient()


def _report_asking(reporter: Reporter, client: Any) -> None:
    """One line before the model call, so the wait has an explanation.

    A local model can legitimately think for two minutes; silence that long
    reads as a hang.
    """
    endpoints = getattr(client, "providers", None) or []
    if not endpoints:
        return
    ceiling = max(int(p.get("timeout", system_one.TIMEOUT_SECONDS)) for p in endpoints)
    names = ", ".join(p["name"] for p in endpoints)
    reporter.say(f"  Asking {names} (up to {ceiling} s) ...")


def _elapsed_text(seconds: float) -> str:
    """The model's answering time, appended to the source line.

    A local model's answer time is worth knowing; a rules-only path answers in
    microseconds, so nothing is shown for it.
    """
    if seconds < 0.05:
        return ""
    return f" ({seconds:.1f} s)"


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_specs(args: argparse.Namespace, reporter: Reporter, detected: Any = None) -> int:
    machine = detected or _detect()

    if args.json:
        reporter.data({"inventory": machine.to_state(), "labels": dict(machine.lines())})
        return EXIT_OK

    reporter.say("")
    for label, value in machine.lines():
        reporter.say(f"  {label:<8}: {value}")
    unknown = [
        c for c in judgment.component_scores(machine, args.config_object)
        if not c.known
    ]
    if unknown:
        reporter.say("")
        reporter.say("  Not readable, and not guessed:")
        for component in unknown:
            reporter.say(f"    - {component.name}: {component.detail}")
    reporter.say("")
    return EXIT_OK


def cmd_payload(args: argparse.Namespace, reporter: Reporter) -> int:
    """Show exactly what would be sent to a System One endpoint."""
    state = _detect().to_state()
    endpoints, skipped = providers_module.chain()
    reporter.data(
        {
            "state": state,
            "providers": [
                {"name": p["name"], "url": p["url"], "model": p["model"], "source": p["source"]}
                for p in endpoints
            ],
            "skipped": skipped,
        }
    )
    return EXIT_OK


def cmd_config(args: argparse.Namespace, reporter: Reporter) -> int:
    document = args.config_object.to_dict()
    document["providers"] = _providers_document()
    reporter.data(document)
    return EXIT_OK


def _providers_document() -> dict[str, Any]:
    """The provider settings in force, with where each value came from.

    Secrets are masked: this document is printed, and a key must not ride
    along in it.
    """
    settings = settings_module.load_settings()
    document = settings.to_dict()
    endpoints, skipped = providers_module.chain()
    document["chain"] = {
        "endpoints": [
            {"name": p["name"], "url": p["url"], "model": p["model"], "source": p["source"]}
            for p in endpoints
        ],
        "skipped": skipped,
    }
    return document


def _print_rating(rating: Any, reporter: Reporter) -> None:
    reporter.say("")
    reporter.say(f"  Rating    : {rating.total}/100" + ("   (estimated)" if rating.estimated else ""))
    if rating.tier:
        reporter.say(f"  Tier      : {rating.tier}")
    reporter.say(f"  Method    : {rating.method}")
    reporter.say("")
    reporter.say("  What it is based on:")
    for component in rating.subscores:
        score = f"{component.score:5.1f}/100" if component.known else "  unknown"
        reporter.say(f"    {component.name:<9}{score}  {component.detail}")
    reporter.say("")
    reporter.say("  What moved it:")
    for factor in rating.factors:
        reporter.say(f"    [{factor.effect:>7}] {factor.text}")
    for note in rating.notes:
        reporter.note(f"  note: {note}")


def cmd_rate(
    args: argparse.Namespace, reporter: Reporter, detected: Any = None, client: Any = None
) -> int:
    machine = detected or _detect()
    client = client or _client()
    repeat = getattr(args, "repeat", 1)

    if repeat > 1:
        return _rate_repeated(machine, client, repeat, args, reporter)

    _report_asking(reporter, client)
    started = time.monotonic()
    rating = judgment.rate_system(machine, client, args.config_object)
    elapsed = time.monotonic() - started
    if args.json:
        reporter.data(rating.to_dict())
        # The notes are in the document, but a degraded rating is also something
        # a human watching the run should be told about, so it goes to stderr.
        for note in rating.notes:
            reporter.note(f"  note: {note}")
    else:
        _print_rating(rating, reporter)
        reporter.say("")
        reporter.say(f"  Source    : {_provenance(rating)}{_elapsed_text(elapsed)}")
        reporter.say("")
    return EXIT_NO_MODEL if rating.method == "rules" else EXIT_OK


def _rate_repeated(
    machine: Any, client: Any, repeat: int, args: argparse.Namespace, reporter: Reporter
) -> int:
    """Ask the same thing N times and report the spread."""
    results = []
    for attempt in range(1, repeat + 1):
        reporter.say(f"  run {attempt}/{repeat} ...")
        results.append(judgment.rate_system(machine, client, args.config_object))
    totals = [r.total for r in results]
    bands = {r.tier for r in results}
    tolerance = args.config_object.stability_tolerance
    spread = max(totals) - min(totals)
    stable = spread <= tolerance and len(bands) == 1

    if args.json:
        reporter.data(
            {
                "repeat": repeat,
                "totals": totals,
                "spread": spread,
                "tiers": sorted(bands),
                "tolerance": tolerance,
                "stable": stable,
                "rating": results[0].to_dict(),
            }
        )
    else:
        _print_rating(results[0], reporter)
        reporter.say("")
        reporter.say(f"  {repeat} runs: {min(totals)}-{max(totals)} (spread {spread})")
        reporter.say(f"  tiers seen : {', '.join(sorted(b for b in bands if b)) or 'unknown'}")
        reporter.say(f"  stable     : {'yes' if stable else 'no'} (tolerance {tolerance})")
        reporter.say("")
    return EXIT_OK


def _print_verdict(verdict: Any, reporter: Reporter) -> None:
    reporter.say("")
    reporter.say(f"  {verdict.question}")
    reporter.say("")
    reporter.say(f"  Verdict    : {verdict.verdict}")
    reporter.say(f"  Confidence : {verdict.confidence}%")
    reporter.say("")
    reporter.say(f"  {verdict.explanation}")
    if verdict.bottlenecks:
        reporter.say("")
        reporter.say("  Weakest link(s):")
        for name in verdict.bottlenecks:
            reporter.say(f"    - {name}")
    for note in verdict.notes:
        reporter.note(f"  note: {note}")


def _ask_once(
    machine: Any,
    question: str,
    client: Any,
    args: argparse.Namespace,
    history: History,
    reporter: Reporter,
) -> tuple[Any, int]:
    _report_asking(reporter, client)
    started = time.monotonic()
    verdict = judgment.answer_capability(machine, question, client, args.config_object)
    elapsed = time.monotonic() - started

    if not args.no_history:
        history.add(
            {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "question": verdict.question,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "probability": round(verdict.probability, 3),
                "source": verdict.source,
                "provider": verdict.provider,
                "model": verdict.model,
                "grounded": verdict.grounded,
            }
        )

    if args.json:
        document = verdict.to_dict()
        if verdict.uncertain or not verdict.grounded:
            # A script cannot answer the interactive ask-back, so it gets a
            # machine-readable pointer to the checklist instead.
            document["requirements_needed"] = True
            document["hint"] = "rerun with --check and --min-reqs to check stated requirements"
        reporter.data(document)
    else:
        _print_verdict(verdict, reporter)
        reporter.say("")
        reporter.say(f"  Source    : {_provenance(verdict)}{_elapsed_text(elapsed)}")
        reporter.say("")
    return verdict, EXIT_UNGROUNDED if not verdict.grounded else EXIT_OK


def cmd_ask(
    args: argparse.Namespace, reporter: Reporter, detected: Any = None, client: Any = None
) -> int:
    if getattr(args, "check", False):
        return cmd_check(args, reporter, detected=detected, client=client)

    machine = detected or _detect()
    client = client or _client()
    history = History(args.history_file, reporter)

    try:
        repeat = getattr(args, "repeat", 1)
        if repeat > 1:
            return _ask_repeated(machine, args.question, client, repeat, args, history, reporter)

        _verdict, code = _ask_once(machine, args.question, client, args, history, reporter)
        # Offered unconditionally: the checklist is for the verdicts that
        # most need it (coin-flip or ungrounded), and _offer_requirements
        # no-ops for grounded verdicts, --json, and non-interactive runs.
        checklist = _offer_requirements(machine, _verdict, args, reporter, client)
        if _settled_by_checklist(checklist):
            # The pasted requirements answered the question, so the
            # checklist's grounded verdict decides the exit code.
            code = EXIT_OK
        return code
    except SystemOneError as error:
        reporter.note(f"no System One endpoint answered: {error}")
        if args.json:
            reporter.data({"error": "no_endpoint_answered", "detail": str(error), "verdict": None})
        else:
            reporter.say("")
            reporter.say("  I could not reach a Jev endpoint, so I have no verdict.")
            reporter.say("  Everything stays on this machine; try again in a moment.")
            reporter.say("")
        return EXIT_NO_MODEL


def _ask_repeated(
    machine: Any,
    question: str,
    client: Any,
    repeat: int,
    args: argparse.Namespace,
    history: History,
    reporter: Reporter,
) -> int:
    """The same question must land in the same band, every time."""
    verdicts = []
    for attempt in range(1, repeat + 1):
        reporter.say(f"  run {attempt}/{repeat} ...")
        verdicts.append(judgment.answer_capability(machine, question, client, args.config_object))

    bands = [v.verdict for v in verdicts]
    confidences = [v.confidence for v in verdicts]
    spread = max(confidences) - min(confidences)
    tolerance = args.config_object.stability_tolerance
    stable_band = len(set(bands)) == 1
    stable = stable_band and spread <= tolerance

    if not args.no_history and verdicts:
        last = verdicts[-1]
        history.add(
            {
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "question": last.question,
                "verdict": last.verdict,
                "confidence": last.confidence,
                "probability": round(last.probability, 3),
                "source": last.source,
                "provider": last.provider,
                "model": last.model,
                "grounded": last.grounded,
                "repeated": repeat,
            }
        )

    if args.json:
        reporter.data(
            {
                "repeat": repeat,
                "verdicts": bands,
                "confidences": confidences,
                "band_spread": spread,
                "tolerance": tolerance,
                "stable_band": stable_band,
                "stable": stable,
                "verdict": verdicts[0].to_dict(),
            }
        )
    else:
        _print_verdict(verdicts[0], reporter)
        reporter.say("")
        reporter.say(f"  {repeat} runs: {', '.join(bands)}")
        reporter.say(f"  confidence : {min(confidences)}-{max(confidences)}% (spread {spread}, tolerance {tolerance})")
        reporter.say(f"  stable     : {'yes' if stable else 'no'}")
        reporter.say("")
    return EXIT_OK


# --------------------------------------------------------------------------
# Interactive session
# --------------------------------------------------------------------------

_MENU_CHOICES = frozenset({"1", "2", "3", "4", "5"})


def cmd_interactive(args: argparse.Namespace, reporter: Reporter) -> int:
    """The interactive app: there is exactly one.

    In a terminal it shows a menu, and any line that is not a number is treated
    as a question -- the menu is a convenience, not a gate. With piped stdin
    the same loop reads one question per line and answers until the pipe ends.
    ``session`` and ``menu`` run this; there is no second interactive mode.
    """
    history = History(args.history_file, reporter)
    if _onboarding_due(args):
        code = _run_onboarding(reporter, first_run=True)
        if code != EXIT_OK:
            return code

    menu_mode = sys.stdin.isatty()
    machine: Any = None
    client: Any = None

    if menu_mode:
        _print_banner(reporter)

    while True:
        if menu_mode:
            _print_menu(reporter)
            try:
                line = input("Ask or choose > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        else:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

        lowered = line.lower()
        if lowered in ("quit", "exit", "q", ":q") or (menu_mode and line == "0"):
            break

        if menu_mode and line in _MENU_CHOICES:
            if line == "1":
                machine = machine or _detect()
                cmd_specs(args, reporter, detected=machine)
            elif line == "2":
                machine = machine or _detect()
                client = client or _client()
                cmd_rate(args, reporter, detected=machine, client=client)
            elif line == "3":
                try:
                    question = input("What do you want to do? > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question:
                    machine = machine or _detect()
                    client = client or _client()
                    if question.lower() == "check" or question.lower().startswith("check "):
                        # Requirements already in hand: straight to the checklist,
                        # no open-ended model call spent first.
                        _interactive_check(
                            args, reporter, task=question[6:].strip() or None,
                            detected=machine, client=client,
                        )
                    else:
                        _ask_in_session(args, reporter, history, question, machine, client)
            elif line == "4":
                history.show(args.json)
            elif line == "5":
                _settings_menu(reporter)
            continue

        if menu_mode and line.isdigit():
            # A wrong number is a keypress mistake, not a question about what
            # this machine can do; say so instead of asking the model "7".
            reporter.say(
                f"  {line} is not a menu choice; pick 1-5 (or 0 to quit), or just type a question."
            )
            continue

        if lowered == "check" or lowered.startswith("check "):
            machine = machine or _detect()
            client = client or _client()
            _interactive_check(
                args, reporter, task=line[6:].strip() or None, detected=machine, client=client
            )
            continue

        # Anything else is the question itself, so "can I game?" works without
        # going through the menu.
        machine = machine or _detect()
        client = client or _client()
        _ask_in_session(args, reporter, history, line, machine, client)

    if menu_mode:
        reporter.say("Bye.")
    return EXIT_OK


def _ask_in_session(
    args: argparse.Namespace,
    reporter: Reporter,
    history: History,
    question: str,
    machine: Any,
    client: Any,
) -> int:
    """One question inside the interactive loop: answer, then offer the checklist.

    Returns the question's exit code: a grounded checklist that
    settles the question counts as answered, everything else
    keeps the verdict's own code.
    """
    try:
        _verdict, code = _ask_once(machine, question, client, args, history, reporter)
        if not args.json:
            checklist = _offer_requirements(machine, _verdict, args, reporter, client)
            if _settled_by_checklist(checklist):
                code = EXIT_OK
        return code
    except SystemOneError as error:
        reporter.note(f"no System One endpoint answered: {error}")
        reporter.say("  No verdict this time -- the endpoint did not answer.")
        return EXIT_NO_MODEL
    except ValueError as error:
        # A bad answer from the model layer (an empty question, for
        # one) is soft: the session continues. SettingsError is
        # deliberately not a ValueError (see settings.py), so a
        # broken config.json escapes this catch and reaches
        # main()'s exit-2 wrapper instead of being printed here.
        reporter.say(f"  {error}")
        return EXIT_ERROR


def _print_banner(reporter: Reporter) -> None:
    reporter.say()
    reporter.say("=" * 46)
    reporter.say("            SYSTEM ANALYZER")
    reporter.say("=" * 46)
    reporter.say("  Ask anything about what this machine can do, e.g.")
    reporter.say("    can I run Cyberpunk 2077?")
    reporter.say("    is 8 GB of RAM enough for a local 7B model?")
    reporter.say("  Know the requirements already? Skip the wait:")
    reporter.say("    check can I run Cyberpunk 2077")
    reporter.say("  Type 'quit' to leave.")
    reporter.say("-" * 46)


def _print_menu(reporter: Reporter) -> None:
    reporter.say()
    reporter.say("  1. Show my specs")
    reporter.say("  2. Rate my PC out of 100")
    reporter.say("  3. Ask a question")
    reporter.say("  4. History")
    reporter.say("  5. Settings")
    reporter.say("  0. Quit")
    reporter.say("-" * 46)
    reporter.say('  ...or just type a question, e.g. "can I game?"')
    reporter.say('  ...or "check <task>" to paste a program\'s requirements')
    reporter.say("      and check them against this machine")


# --------------------------------------------------------------------------
# Requirement checklists (the ask-back)
# --------------------------------------------------------------------------


def cmd_check(
    args: argparse.Namespace, reporter: Reporter, detected: Any = None, client: Any = None
) -> int:
    """Check pasted requirements ("8 GB RAM; GTX 970") against this machine.

    Numeric requirements are compared directly against the measured specs;
    named parts ("GTX 970") go to the provider as atomic questions.
    """
    machine = detected or _detect()
    task = args.question or "your task"
    minimum_text = getattr(args, "min_reqs", None)
    recommended_text = getattr(args, "rec_reqs", None)
    if not minimum_text and not args.json and _interactive_terminal():
        minimum_text = _paste_block(reporter, "  Paste the MINIMUM requirements, one per line")
        recommended_text = _paste_block(reporter, "  Recommended specs, one per line (optional)")
    if not minimum_text:
        reporter.note(
            'no requirements given; use --min-reqs "8 GB RAM; 2 GHz CPU" or run interactively'
        )
        return EXIT_USAGE
    code, _result = _run_checklist(
        machine, task, minimum_text, recommended_text, args, reporter, client
    )
    return code


def _run_checklist(
    machine: Any,
    task: str,
    minimum_text: str,
    recommended_text: str | None,
    args: argparse.Namespace,
    reporter: Reporter,
    client: Any = None,
) -> tuple[int, Any]:
    """Run the checklist, report it, and hand back what it decided.

    Returns the exit code plus the ``CheckResult`` the checklist
    produced, so a caller that offered it can treat a grounded
    verdict as the answer. The result is ``None`` when no
    checklist could run at all: unparsable requirements, or no
    endpoint to judge the named parts.
    """
    client = client or _client()
    try:
        result = requirements_module.check_requirements(
            machine.to_state(), task, minimum_text, recommended_text,
            client, args.config_object,
            progress=lambda: _report_asking(reporter, client),
        )
    except requirements_module.RequirementError as error:
        reporter.note(f"could not check the requirements: {error}")
        return EXIT_USAGE, None
    except SystemOneError as error:
        reporter.note(f"no System One endpoint answered: {error}")
        if args.json:
            reporter.data({"error": "no_endpoint_answered", "detail": str(error)})
        return EXIT_NO_MODEL, None

    if args.json:
        reporter.data(result.to_dict())
    else:
        _print_checklist(result, reporter)
    if result.verdict == "Cannot tell":
        return EXIT_UNGROUNDED, result
    if result.model_unavailable:
        return EXIT_NO_MODEL, result
    return EXIT_OK, result


def _print_checklist(result: Any, reporter: Reporter) -> None:
    reporter.say("")
    reporter.say(f"  Checklist  : {result.task}")
    reporter.say(f"  Verdict    : {result.verdict}")
    confidence_line = f"  Confidence : {result.confidence}%"
    if result.low_confidence:
        confidence_line += "  (below the reportable floor -- a guess)"
    reporter.say(confidence_line)
    reporter.say("")
    for item in result.items:
        if item.state in ("passed", "failed"):
            mark = "pass" if item.state == "passed" else "FAIL"
            tier = f" [{item.tier}]" if item.tier == "recommended" else ""
            reporter.say(f"    [{mark:>4}]{tier} {item.raw}")
            reporter.say(f"           {item.detail}")
    for item in result.items:
        if item.state == "unknown":
            reporter.say(f"    [ ??? ] {item.raw}")
            reporter.say(f"           {item.detail}")
    for note in result.notes:
        reporter.note(f"  note: {note}")
    reporter.say("")


def _interactive_check(
    args: argparse.Namespace,
    reporter: Reporter,
    task: str | None = None,
    detected: Any = None,
    client: Any = None,
) -> None:
    """The requirements checklist without the open-ended question first.

    The user already knows the specs they need, so no model call is spent on
    a judgment the checklist will not use: paste requirements, get numbers.
    """
    if not task:
        task = _prompt("What is it for? > ") or "your task"
    machine = detected or _detect()
    minimum = _paste_block(reporter, "  Paste the MINIMUM requirements, one per line")
    if not minimum:
        reporter.say("  Nothing to check.")
        return
    recommended = _paste_block(reporter, "  Recommended specs, one per line (optional)")
    _run_checklist(machine, task, minimum, recommended, args, reporter, client)


def _paste_block(reporter: Reporter, prompt: str) -> str:
    """Read requirement lines until an empty line; Ctrl-C keeps what is held."""
    reporter.say(prompt)
    reporter.say("  (Enter on an empty line to finish)")
    lines: list[str] = []
    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


def _offer_requirements(
    machine: Any, verdict: Any, args: argparse.Namespace,
    reporter: Reporter, client: Any,
) -> Any:
    """After a coin-flip or ungrounded verdict, offer the requirements checklist.

    Interactive only. A script that cannot answer questions is never stopped
    by one: it gets ``requirements_needed`` in its JSON instead.

    Returns the ``CheckResult`` the checklist produced, so the caller
    can let a grounded checklist settle the exit code, or ``None``
    whenever no checklist ran: a grounded verdict, ``--json``, a
    non-interactive run, or an empty paste.
    """
    if not (verdict.uncertain or not verdict.grounded):
        return None
    if args.json or not _interactive_terminal():
        return None
    reporter.say("")
    reporter.say("  The model could not judge that with confidence.")
    minimum = _paste_block(
        reporter, "  Paste the program's MINIMUM requirements, one per line"
    )
    if not minimum:
        reporter.say("")
        return None
    recommended = _paste_block(reporter, "  Recommended specs, one per line (optional)")
    _code, checklist = _run_checklist(
        machine, verdict.question, minimum, recommended, args, reporter, client
    )
    return checklist


def _settled_by_checklist(checklist: Any) -> bool:
    """True when a checklist ran and grounded its verdict.

    A checklist that ran but landed on "Cannot tell" settled
    nothing: the question stays ungrounded and the ask's own
    exit code stands.
    """
    return checklist is not None and checklist.verdict != "Cannot tell"


# --------------------------------------------------------------------------
# Setup and settings (provider chain, onboarding)
# --------------------------------------------------------------------------


class _SetupAborted(Exception):
    """The user ended setup early; nothing is saved."""


def _prompt(prompt: str) -> str:
    """Read one line, or raise _SetupAborted on EOF/Ctrl-C."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise _SetupAborted() from error


def _secret_prompt(prompt: str) -> str:
    """Read a secret without echoing it."""
    try:
        return getpass.getpass(prompt).strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise _SetupAborted() from error


def _interactive_terminal() -> bool:
    """True when both sides of the console are a real terminal.

    Both ends have to agree: a piped stdout belongs to redirected output even
    when stdin is a terminal, and asking setup questions into a script's
    captured stdin would only hang it.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def cmd_setup(args: argparse.Namespace, reporter: Reporter) -> int:
    """Set up the provider chain: probe Ollama, ask, save.

    Deliberately interactive-only. A script that cannot answer questions must
    never be stopped by one; non-interactive runs configure providers through
    environment variables instead.
    """
    if args.json or not _interactive_terminal():
        reporter.note(
            "setup is interactive only; for scripts, set the provider settings "
            f"as environment variables (e.g. {settings_module.PROVIDERS_ENV})"
        )
        return EXIT_USAGE
    return _run_onboarding(reporter, first_run=not settings_module.configured())


def _onboarding_due(args: argparse.Namespace) -> bool:
    """First interactive run, and no provider configuration exists yet."""
    if args.json or not _interactive_terminal():
        return False
    return not settings_module.configured()


def _run_onboarding(reporter: Reporter, first_run: bool) -> int:
    """Probe what is reachable, ask only what the probe could not settle, save."""
    try:
        return _onboarding_flow(reporter, first_run)
    except _SetupAborted:
        reporter.say("")
        reporter.say("  Setup ended early; nothing was saved.")
        reporter.say("")
        return EXIT_OK


def _onboarding_flow(reporter: Reporter, first_run: bool) -> int:
    settings = settings_module.load_settings()

    reporter.say("")
    reporter.say("=" * 46)
    reporter.say("  WELCOME TO SYSTEM ANALYZER" if first_run else "            SETUP")
    reporter.say("=" * 46)
    if first_run:
        reporter.say("  A few choices, saved next to the app; all of it can be")
        reporter.say("  changed later with: python main.py setup")

    # One probe decides what is worth asking: a user whose Ollama is running
    # with a model installed should not be asked whether they have Ollama.
    base_url = providers_module.resolve_ollama_url(settings)
    reporter.say("")
    reporter.say(f"  Looking for Ollama at {base_url} ...")
    status = providers_module.discover_ollama(base_url)
    for line in providers_module.ollama_guidance(status, base_url, settings):
        reporter.say(f"  - {line}")
    if status.reachable and status.models:
        reporter.say(f"  - decision models installed: {', '.join(status.models)}")

    chain = _ask_chain(reporter)

    ollama_url = ""
    ollama_model = settings.get("ollama_model")
    if "ollama" in chain:
        if not status.reachable:
            answer = _prompt(
                "  Ollama's address, if it runs elsewhere (host:port, "
                "Enter to skip): "
            )
            if answer:
                ollama_url = answer
                settings_module.save_config({"ollama_url": answer})
                settings = settings_module.load_settings()
                base_url = providers_module.resolve_ollama_url(settings)
                status = providers_module.discover_ollama(base_url)
                for line in providers_module.ollama_guidance(status, base_url, settings):
                    reporter.say(f"  - {line}")
        ollama_model = _ask_ollama_model(reporter, status, settings)

    if "typesafe" in chain and not settings.get("typesafe_api_key"):
        _ask_typesafe_key(reporter)

    values: dict[str, Any] = {"provider_chain": ",".join(chain)}
    if "ollama" in chain:
        values["ollama_model"] = ollama_model
    if ollama_url:
        values["ollama_url"] = ollama_url
    try:
        path = settings_module.save_config(values)
    except settings_module.SettingsError as error:
        reporter.say(f"  Could not save the settings: {error}")
        return EXIT_ERROR

    reporter.say("")
    reporter.say(f"  Saved {path}")
    reporter.say("  Providers, in order:")
    for pid in chain:
        reporter.say(f"    - {providers_module.PROVIDER_LABELS[pid]}")
    reporter.say("")
    return EXIT_OK


def _ask_chain(reporter: Reporter) -> list[str]:
    """Which providers, in which order. Excluded means never contacted.

    The free default is listed first, so "just press Enter" picks it and the
    numbering always agrees with what Enter does.
    """
    ids = providers_module.PROVIDER_IDS
    order = [pid for pid in ("zen", "ollama", "typesafe") if pid in ids]
    reporter.say("")
    reporter.say("  Who should answer your questions, in order of preference?")
    for number, pid in enumerate(order, 1):
        reporter.say(f"    {number}. {providers_module.PROVIDER_LABELS[pid]}")
    reporter.say("  A provider left out is never contacted -- not even as a fallback.")

    default = order[0]
    while True:
        answer = _prompt(f"  Choices, e.g. 1 or 2,1 (Enter for {default}): ")
        if not answer:
            return [default]
        picked: list[str] = []
        for part in answer.replace(";", ",").split(","):
            part = part.strip().lower()
            if part in ids:
                pid = part
            elif part.isdigit() and 1 <= int(part) <= len(order):
                pid = order[int(part) - 1]
            else:
                reporter.say(f"  {part!r} is not one of the choices; try again.")
                picked = []
                break
            if pid not in picked:
                picked.append(pid)
        if picked:
            return picked


def _ask_ollama_model(reporter: Reporter, status: Any, settings: Any) -> str:
    """Pick an installed decision model; an empty result leaves it unset."""
    if status.reachable and status.models:
        reporter.say("")
        reporter.say("  Which model should Ollama answer with?")
        for number, name in enumerate(status.models, 1):
            reporter.say(f"    {number}. {name}")
        while True:
            answer = _prompt(f"  1-{len(status.models)}, by number or name (Enter for 1): ")
            if not answer:
                return status.models[0]
            if answer in status.models:
                return answer
            if answer.isdigit() and 1 <= int(answer) <= len(status.models):
                return status.models[int(answer) - 1]
            reporter.say("  Pick one of the numbers or names above.")

    if status.reachable:  # running, but nothing decision-capable installed
        while True:
            answer = _prompt(
                f"  Install one with:  ollama pull {providers_module.RECOMMENDED_PULL_MODEL}"
                "  -- then press Enter to re-check (or type skip): "
            )
            if answer.lower() == "skip":
                reporter.say("  Ollama stays in the chain but is skipped until a model is configured.")
                return ""
            status = providers_module.discover_ollama(providers_module.resolve_ollama_url(settings))
            if status.reachable and status.models:
                return _ask_ollama_model(reporter, status, settings)
            reporter.say("  Still no decision-capable model installed.")

    reporter.say("  Ollama stays in the chain but is skipped until it answers.")
    return ""


def _ask_typesafe_key(reporter: Reporter) -> None:
    """Collect the paid key into .env; a stored key alone enables nothing."""
    reporter.say("")
    reporter.say("  TypeSafe is paid. The key is stored in .env, never displayed,")
    reporter.say("  and used only because TypeSafe is in your provider list.")
    key = _secret_prompt("  TypeSafe API key (Enter to leave unset): ")
    if not key:
        reporter.say("  No key set: TypeSafe will be skipped until one is saved.")
        return
    try:
        settings_module.save_env("TYPESAFE_API_KEY", key)
    except settings_module.SettingsError as error:
        reporter.say(f"  Could not save the key: {error}")
        return
    reporter.say("  Key saved to .env.")


def _settings_menu(reporter: Reporter) -> int:
    """Edit any setting; secrets are written to .env, the rest to config.json."""
    while True:
        settings = settings_module.load_settings()
        for warning in settings.warnings:
            reporter.say(f"  note: {warning}")
        reporter.say("")
        reporter.say("  Settings (stored next to the app):")
        entries = list(settings_module.SETTINGS)
        for number, setting in enumerate(entries, 1):
            value = settings.get(setting.key)
            if setting.secret:
                shown = "set" if value else "not set"
            elif setting.kind == "chain":
                shown = ", ".join(p for p in value.split(",") if p.strip()) or "(empty)"
            else:
                shown = value or "(empty)"
            reporter.say(
                f"    {number}. {setting.label}: {shown}   [{settings.origin(setting.key)}]"
            )
        reporter.say("    0. Back")
        choice = _prompt("  Number to change (0 for back): ")
        if choice in ("", "0", "q", "quit"):
            return EXIT_OK
        if not choice.isdigit() or not 1 <= int(choice) <= len(entries):
            continue
        _edit_setting(entries[int(choice) - 1], reporter)


def _edit_setting(setting: Any, reporter: Reporter) -> None:
    """One setting, edited and saved; the current value is kept on a bare Enter."""
    reporter.say("")
    reporter.say(f"  {setting.label} -- {setting.help}")
    if setting.kind == "bool":
        answer = _prompt("  on or off (Enter keeps the current value): ").lower()
        if answer in ("on", "yes", "y", "1", "true"):
            value = "1"
        elif answer in ("off", "no", "n", "0", "false"):
            value = "0"
        else:
            reporter.say("  Unchanged.")
            return
        save = lambda: settings_module.save_config({setting.key: value})
    elif setting.kind == "chain":
        reporter.say(
            "  Providers: "
            + "; ".join(f"{pid} = {providers_module.PROVIDER_LABELS[pid]}" for pid in providers_module.PROVIDER_IDS)
        )
        answer = _prompt("  Comma-separated, in order (Enter keeps the current value): ")
        if not answer:
            reporter.say("  Unchanged.")
            return
        try:
            settings_module.validate_chain(answer)
        except settings_module.SettingsError as error:
            reporter.say(f"  Not saved: {error}")
            return
        value = ",".join(settings_module.validate_chain(answer))
        save = lambda: settings_module.save_config({setting.key: value})
    elif setting.secret:
        value = _secret_prompt("  New value (Enter keeps the current one): ")
        if not value:
            reporter.say("  Unchanged.")
            return
        save = lambda: settings_module.save_env(setting.env, value)
    else:
        value = _prompt("  New value (Enter keeps the current one): ")
        if not value:
            reporter.say("  Unchanged.")
            return
        save = lambda: settings_module.save_config({setting.key: value})

    try:
        path = save()
    except settings_module.SettingsError as error:
        reporter.say(f"  Not saved: {error}")
        return
    reporter.say(f"  Saved to {path.name}.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _cmd_history(args: argparse.Namespace, reporter: Reporter) -> int:
    return History(args.history_file, reporter).show(args.json)


def _cmd_clear(args: argparse.Namespace, reporter: Reporter) -> int:
    return History(args.history_file, reporter).clear()


def build_parser() -> argparse.ArgumentParser:
    """The command line: one shared parent, one subparser per command.

    Subparsers give every command its own ``--help`` and only the flags that
    command actually uses. The shared flags are ``default=SUPPRESS`` so the
    same option works in either position -- before or after the command --
    without one parser's default clobbering the other's parsed value.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="JSON on stdout, hints on stderr")
    shared.add_argument("--config", metavar="PATH", default=argparse.SUPPRESS,
                        help="JSON file of thresholds and weights")
    shared.add_argument("--history-file", metavar="PATH", default=argparse.SUPPRESS,
                        help="where to keep the question log")
    shared.add_argument("--no-history", action="store_true", default=argparse.SUPPRESS,
                        help="do not write to the question log")

    parser = argparse.ArgumentParser(
        prog="system-analyzer",
        parents=[shared],
        description="Detect this machine's specs, rate it, and answer what it can do.",
        epilog=textwrap.dedent(
            """\
            examples:
              python main.py specs                                 what was detected
              python main.py rate                                  the 0-100 rating
              python main.py ask "can I run Cyberpunk 2077?"
              python main.py check "Office" --min-reqs "4 GB RAM; 2 GHz CPU"
              python main.py                                       interactive (terminal)
              echo "can I game?" | python main.py                  one answer per line
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", metavar="command")

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, description=help_text, parents=[shared])

    command("specs", "what was detected, and what is unreadable")

    rate = command("rate", "rate this machine 0-100, with the factors behind it")
    _add_repeat(rate)

    ask = command("ask", 'ask what this machine can do, e.g. "can I run Cyberpunk 2077?"')
    ask.add_argument("question", nargs="*", metavar="QUESTION", help="the question, in words")
    _add_repeat(ask)
    ask.add_argument("--check", action="store_true",
                     help="skip the open-ended judgment and check pasted requirements")
    _add_requirements(ask)

    check = command("check", "check pasted requirements against this machine")
    check.add_argument("question", nargs="*", metavar="TASK", help="what the requirements are for")
    _add_requirements(check)

    command("session", "the interactive app (menu in a terminal, one answer per line when piped)")
    command("menu", "same as session")
    command("history", "the questions asked so far")
    command("clear", "forget the question log")
    command("config", "thresholds, weights and the provider chain, with origins")
    command("payload", "the scrubbed state that would be sent to the model")
    command("setup", "choose providers and models interactively (terminal only)")

    return parser


def _add_repeat(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-n", "--repeat", type=int, default=1, metavar="N",
        help="ask N times and report how much the answer moves",
    )


def _add_requirements(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-reqs", metavar="TEXT",
                        help='minimum requirements, e.g. "8 GB RAM; GTX 970; 60 GB storage"')
    parser.add_argument("--rec-reqs", metavar="TEXT", help="recommended requirements to check")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Shared flags are SUPPRESS-defaulted so either position parses; settle
    # them here so the rest of the app can read attributes without guards.
    args.json = getattr(args, "json", False)
    args.no_history = getattr(args, "no_history", False)
    args.config = getattr(args, "config", None)
    args.history_file = getattr(args, "history_file", None)

    if getattr(args, "repeat", 1) < 1:
        parser.error("--repeat must be at least 1")

    try:
        args.config_object = config_module.load(args.config)
    except config_module.ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return EXIT_USAGE

    # Settings are validated here, at launch: a config.json that
    # cannot be used fails fast with exit 2, before any command
    # runs -- even one like specs that never touches a provider.
    # The provider chain itself stays lazy; it is built when a
    # command first asks for it, so a run that never consults a
    # model never probes an endpoint.
    try:
        settings_module.load_settings()
    except settings_module.SettingsError as error:
        print(f"settings error: {error}", file=sys.stderr)
        return EXIT_USAGE

    if args.history_file is None:
        args.history_file = Path(os.environ.get(history_module.HISTORY_ENV)
                                 or history_module.DEFAULT_HISTORY)
    else:
        args.history_file = Path(args.history_file)

    reporter = Reporter(args.json)

    # argparse leaves the question as a list of words; the rest of the app
    # wants it as one string, and shell quoting should not leak through.
    if hasattr(args, "question"):
        args.question = " ".join(args.question).strip()

    if args.command == "ask" and not args.question and not getattr(args, "check", False):
        parser.error('ask needs a question, e.g. ask "Can I run Cyberpunk 2077?"')

    handlers: dict[str, Callable[[argparse.Namespace, Reporter], int]] = {
        "specs": cmd_specs,
        "rate": cmd_rate,
        "ask": cmd_ask,
        "check": cmd_check,
        "session": cmd_interactive,
        "menu": cmd_interactive,
        "history": _cmd_history,
        "clear": _cmd_clear,
        "config": cmd_config,
        "payload": cmd_payload,
        "setup": cmd_setup,
    }

    try:
        if args.command is None:
            # No command: a terminal gets the interactive app, a pipe gets a
            # question loop -- one answer per line.
            return cmd_interactive(args, reporter)
        return handlers[args.command](args, reporter)
    except settings_module.SettingsError as error:
        # Backstop: settings are validated at launch (above), but
        # commands still load and save settings themselves -- the
        # settings menu, onboarding, setup. A failure there is
        # reported the same way: the one line that says which
        # file to fix, and exit 2. In the interactive session
        # this also means the session ends; a broken config.json
        # is not something to keep running on.
        print(f"settings error: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(EXIT_OK) from None
