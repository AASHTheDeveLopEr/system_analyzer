#!/usr/bin/env python3
"""System Analyzer -- one entry point for the whole app (spec FR-000).

One-shot commands, or an interactive session::

    python main.py specs                  # what was detected, and what is unknown
    python main.py rate                   # 0-100 with the factors behind it
    python main.py ask "Can I run Cyberpunk 2077?"
    python main.py ask "stream on Twitch" --repeat 5   # FR-007 stability check
    python main.py session                # interactive loop
    python main.py history                # what has been asked this session
    python main.py payload                # the exact payload that leaves the machine
    python main.py config                 # the thresholds in force right now

Machine-readable mode puts JSON on stdout and every hint on stderr, with exit
codes that mean something (FR-015)::

    python main.py rate --json | jq .total
    python main.py ask "..." --json; echo $?

    0  worked
    1  something went wrong
    2  bad command line
    3  no System One endpoint answered -- degraded, output is still valid
    4  the question could not be grounded in your hardware -- no verdict given

Everything a rating or verdict is built from is in ``--json`` output as well:
the per-component sub-scores, the model, and whether the number is estimated
from partial readings (FR-011). Nothing is hidden behind a formatted string.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config as config_module
import menu
import system_one
from system_one import SystemOneClient, SystemOneError

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2  # what argparse itself uses
EXIT_NO_MODEL = 3
EXIT_UNGROUNDED = 4

HISTORY_ENV = "SYSTEM_ANALYZER_HISTORY"
DEFAULT_HISTORY = Path(__file__).resolve().parent / "history.json"
HISTORY_LIMIT = 200


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
    """FR-011: every judgment says where it came from."""
    source = getattr(decision_or_rating, "source", "rules")
    model = getattr(decision_or_rating, "model", None)
    provider = getattr(decision_or_rating, "provider", None)
    if source == "rules" or not model:
        return "local rules (no model involved)"
    return f"{source} via {provider or 'jev'} / {model}"


# --------------------------------------------------------------------------
# History (FR-012)
# --------------------------------------------------------------------------


class History:
    """Local question log with verdicts. Never leaves the machine."""

    def __init__(self, path: Path, reporter: Reporter) -> None:
        self.path = path
        self.reporter = reporter

    def load(self) -> list[dict[str, Any]]:
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, OSError) as error:
            # A corrupt log must not stop the tool working; say so and start over.
            self.reporter.note(f"could not read the history file ({error}); starting a new one")
            return []
        return entries if isinstance(entries, list) else []

    def add(self, entry: dict[str, Any]) -> None:
        entries = [e for e in self.load() if isinstance(e, dict)]
        entries.append(entry)
        entries = entries[-HISTORY_LIMIT:]
        try:
            self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        except OSError as error:
            self.reporter.note(f"could not save history ({error}) -- the answer below is still valid")

    def clear(self) -> int:
        existed = self.path.exists()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            self.reporter.note(f"could not clear history: {error}")
            return EXIT_ERROR
        return 0 if existed else EXIT_OK

    def show(self, as_json: bool) -> int:
        entries = self.load()
        if as_json:
            self.reporter.data(entries)
            return EXIT_OK
        if not entries:
            self.reporter.say("No questions yet.")
            return EXIT_OK
        self.reporter.say("")
        for index, entry in enumerate(entries, 1):
            when = entry.get("at", "?")
            self.reporter.say(
                f"  {index:>3}. [{when}] {entry.get('question', '')}\n"
                f"       {entry.get('verdict', '?')} -- {entry.get('confidence', '?')}% confident "
                f"({entry.get('source', '?')})"
            )
        self.reporter.say("")
        return EXIT_OK


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def _inventory_and_state() -> tuple[Any, dict[str, Any]]:
    inventory = menu.detect_inventory()
    return inventory, inventory.to_state()


def _client() -> SystemOneClient:
    return SystemOneClient()


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_specs(args: argparse.Namespace, reporter: Reporter) -> int:
    inventory, state = _inventory_and_state()

    if args.json:
        reporter.data({"inventory": state, "labels": dict(inventory.lines())})
        return EXIT_OK

    reporter.say("")
    for label, value in inventory.lines():
        reporter.say(f"  {label:<8}: {value}")
    unknown = [
        c for c in system_one.component_scores(inventory, args.config_object)
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
    """Show exactly what would be sent to a System One endpoint (FR-010)."""
    _inventory, state = _inventory_and_state()
    providers, skipped = system_one._providers()
    reporter.data(
        {
            "state": state,
            "providers": [
                {"name": p["name"], "url": p["url"], "model": p["model"], "source": p["source"]}
                for p in providers
            ],
            "skipped": skipped,
        }
    )
    return EXIT_OK


def cmd_config(args: argparse.Namespace, reporter: Reporter) -> int:
    reporter.data(args.config_object.to_dict())
    return EXIT_OK


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


def cmd_rate(args: argparse.Namespace, reporter: Reporter) -> int:
    inventory, _state = _inventory_and_state()
    client = _client()

    if args.repeat > 1:
        return _rate_repeated(inventory, client, args, reporter)

    rating = system_one.rate_system(inventory, client, args.config_object)
    if args.json:
        reporter.data(rating.to_dict())
        # The notes are in the document, but a degraded rating is also something
        # a human watching the run should be told about, so it goes to stderr.
        for note in rating.notes:
            reporter.note(f"  note: {note}")
    else:
        _print_rating(rating, reporter)
        reporter.say("")
        reporter.say(f"  Source    : {_provenance(rating)}")
        reporter.say("")
    return EXIT_NO_MODEL if rating.method == "rules" else EXIT_OK


def _rate_repeated(
    inventory: Any, client: SystemOneClient, args: argparse.Namespace, reporter: Reporter
) -> int:
    """Ask the same thing N times and report the spread (FR-007, SC-007)."""
    results = [system_one.rate_system(inventory, client, args.config_object) for _ in range(args.repeat)]
    totals = [r.total for r in results]
    bands = {r.tier for r in results}
    tolerance = args.config_object.stability_tolerance

    if args.json:
        reporter.data(
            {
                "repeat": args.repeat,
                "totals": totals,
                "spread": max(totals) - min(totals),
                "tiers": sorted(bands),
                "tolerance": tolerance,
                "stable": (max(totals) - min(totals)) <= tolerance and len(bands) == 1,
                "rating": results[0].to_dict(),
            }
        )
    else:
        _print_rating(results[0], reporter)
        reporter.say("")
        reporter.say(f"  {args.repeat} runs: {min(totals)}-{max(totals)} (spread {max(totals) - min(totals)})")
        reporter.say(f"  tiers seen : {', '.join(sorted(b for b in bands if b)) or 'unknown'}")
        reporter.say(f"  stable     : {'yes' if (max(totals) - min(totals)) <= tolerance and len(bands) == 1 else 'no'} (tolerance {tolerance})")
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
    inventory: Any,
    question: str,
    client: SystemOneClient,
    args: argparse.Namespace,
    history: History,
    reporter: Reporter,
) -> tuple[Any, int]:
    verdict = system_one.answer_capability(inventory, question, client, args.config_object)

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
        reporter.data(verdict.to_dict())
    else:
        _print_verdict(verdict, reporter)
        reporter.say("")
        reporter.say(f"  Source    : {_provenance(verdict)}")
        reporter.say("")
    return verdict, EXIT_UNGROUNDED if not verdict.grounded else EXIT_OK


def cmd_ask(args: argparse.Namespace, reporter: Reporter) -> int:
    inventory, _state = _inventory_and_state()
    client = _client()
    history = History(args.history_file, reporter)

    try:
        if args.repeat > 1:
            return _ask_repeated(inventory, args, reporter, history)

        _verdict, code = _ask_once(inventory, args.question, client, args, history, reporter)
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
    inventory: Any, args: argparse.Namespace, reporter: Reporter, history: History
) -> int:
    """FR-007: the same question must land in the same band, every time."""
    client = _client()
    verdicts = [
        system_one.answer_capability(inventory, args.question, client, args.config_object)
        for _ in range(args.repeat)
    ]
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
                "source": last.source,
                "model": last.model,
                "grounded": last.grounded,
                "repeated": args.repeat,
            }
        )

    if args.json:
        reporter.data(
            {
                "repeat": args.repeat,
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
        reporter.say(f"  {args.repeat} runs: {', '.join(bands)}")
        reporter.say(f"  confidence : {min(confidences)}-{max(confidences)}% (spread {spread}, tolerance {tolerance})")
        reporter.say(f"  stable     : {'yes' if stable else 'no'}")
        reporter.say("")
    return EXIT_OK


# --------------------------------------------------------------------------
# Interactive session (FR-000, US2)
# --------------------------------------------------------------------------


def cmd_session(args: argparse.Namespace, reporter: Reporter) -> int:
    inventory, _state = _inventory_and_state()
    client = _client()
    history = History(args.history_file, reporter)

    if not args.json:
        _print_banner(reporter)

    while True:
        try:
            if args.json or not sys.stdin.isatty():
                # No menu for a pipe: read one question, answer, stop.
                line = sys.stdin.readline()
                if not line:
                    break
                question = line.strip()
                if not question:
                    continue
            else:
                _print_menu(reporter)
                try:
                    question = input("Ask > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
        except KeyboardInterrupt:
            break

        if question.lower() in ("quit", "exit", "q", ":q"):
            break
        if not question:
            continue

        try:
            _verdict, _code = _ask_once(inventory, question, client, args, history, reporter)
        except SystemOneError as error:
            reporter.note(f"no System One endpoint answered: {error}")
            reporter.say("  No verdict this time -- the endpoint did not answer.")
        except ValueError as error:
            reporter.say(f"  {error}")

    if not args.json:
        reporter.say("Bye.")
    return EXIT_OK


def _print_banner(reporter: Reporter) -> None:
    reporter.say()
    reporter.say("=" * 46)
    reporter.say("            SYSTEM ANALYZER")
    reporter.say("=" * 46)
    reporter.say("  Ask anything about what this machine can do, e.g.")
    reporter.say("    can I run Cyberpunk 2077?")
    reporter.say("    is 8 GB of RAM enough for a local 7B model?")
    reporter.say("  Type 'quit' to leave.")
    reporter.say("-" * 46)


def _print_menu(reporter: Reporter) -> None:
    reporter.say()
    reporter.say("  1. Show my specs")
    reporter.say("  2. Rate my PC out of 100")
    reporter.say("  3. Ask a question")
    reporter.say("  4. History")
    reporter.say("  0. Quit")
    reporter.say("-" * 46)


def cmd_interactive(args: argparse.Namespace, reporter: Reporter) -> int:
    """The numbered menu, wired to the real features (FR-000)."""
    history = History(args.history_file, reporter)
    _print_banner(reporter)

    while True:
        _print_menu(reporter)
        try:
            choice = input("Choose > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice in ("0", "q", "quit", "exit"):
            break
        if choice == "1":
            cmd_specs(args, reporter)
        elif choice == "2":
            cmd_rate(args, reporter)
        elif choice == "3":
            try:
                question = input("What do you want to do? > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question:
                _dispatch_ask(args, reporter, history, question)
        elif choice == "4":
            history.show(args.json)
        elif choice:
            # Anything else is treated as the question itself, so "can I game?"
            # works without going through the menu.
            _dispatch_ask(args, reporter, history, choice)
    reporter.say("Bye.")
    return EXIT_OK


def _dispatch_ask(
    args: argparse.Namespace, reporter: Reporter, history: History, question: str
) -> None:
    inventory, _state = _inventory_and_state()
    client = _client()
    parsed = argparse.Namespace(**{**vars(args), "question": question, "repeat": 1})
    try:
        _ask_once(inventory, question, client, parsed, history, reporter)
    except SystemOneError as error:
        reporter.note(f"no System One endpoint answered: {error}")
        reporter.say("  No verdict this time -- the endpoint did not answer.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="system-analyzer",
        description="Detect this machine's specs, rate it, and answer what it can do.",
    )
    parser.add_argument("--json", action="store_true", help="JSON on stdout, hints on stderr")
    parser.add_argument("--config", metavar="PATH", help="JSON file of thresholds and weights")
    parser.add_argument("--history-file", metavar="PATH", help="where to keep the question log")
    parser.add_argument("--no-history", action="store_true", help="do not write to the log")
    parser.add_argument(
        "-n", "--repeat", type=int, default=1, metavar="N",
        help="ask N times and report how much the answer moves (FR-007)",
    )
    parser.add_argument(
        "command", nargs="?", default=None,
        help="specs | rate | ask | session | history | clear | config | payload | menu",
    )
    parser.add_argument("question", nargs="*", help="the question, for 'ask'")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    try:
        args.config_object = config_module.load(args.config)
    except config_module.ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return EXIT_USAGE

    if args.history_file is None:
        args.history_file = Path(os.environ.get(HISTORY_ENV) or DEFAULT_HISTORY)
    else:
        args.history_file = Path(args.history_file)

    reporter = Reporter(args.json)

    command = args.command
    # argparse leaves the question as a list of words; the rest of the app
    # wants it as one string, and shell quoting should not leak through.
    args.question = " ".join(args.question).strip()
    question = args.question

    if command == "ask" and not question:
        parser.error("ask needs a question, e.g. ask \"Can I run Cyberpunk 2077?\"")
    if command not in (None, "ask") and question:
        parser.error(f"'{command}' does not take a question")

    handlers: dict[str, Callable[[argparse.Namespace, Reporter], int]] = {
        "specs": cmd_specs,
        "rate": cmd_rate,
        "ask": cmd_ask,
        "session": cmd_session,
        "history": lambda a, r: History(a.history_file, r).show(a.json),
        "clear": lambda a, r: History(a.history_file, r).clear(),
        "config": cmd_config,
        "payload": cmd_payload,
        "menu": cmd_interactive,
    }

    if command is None:
        # No command: a terminal gets the menu, a pipe gets a question loop.
        return cmd_interactive(args, reporter) if sys.stdin.isatty() else cmd_session(args, reporter)
    if command not in handlers:
        parser.error(f"unknown command {command!r}; try: {', '.join(sorted(handlers))}")
    return handlers[command](args, reporter)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(EXIT_OK) from None
