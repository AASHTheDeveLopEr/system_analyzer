"""The local question log: what was asked, and what was answered.

The log lives next to the app (or wherever ``--history-file`` or
``SYSTEM_ANALYZER_HISTORY`` points) and never leaves the machine: it records
the questions the user asked and the verdicts given, nothing else. A log that
cannot be read must not stop the tool from answering, so every failure here
is reported and survived, never raised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Where the log location comes from; tests point this at a scratch file.
HISTORY_ENV = "SYSTEM_ANALYZER_HISTORY"
DEFAULT_HISTORY = Path(__file__).resolve().parent / "history.json"

#: Entries beyond this are dropped: the log is a convenience, not an archive.
HISTORY_LIMIT = 200


class History:
    """Local question log with verdicts. Never leaves the machine."""

    def __init__(self, path: Path, reporter: Any) -> None:
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
            return 0
        except OSError as error:
            self.reporter.note(f"could not clear history: {error}")
            return 1
        return 0

    def show(self, as_json: bool) -> int:
        entries = self.load()
        if as_json:
            self.reporter.data(entries)
            return 0
        if not entries:
            self.reporter.say("No questions yet.")
            return 0
        self.reporter.say("")
        for index, entry in enumerate(entries, 1):
            when = entry.get("at", "?")
            self.reporter.say(
                f"  {index:>3}. [{when}] {entry.get('question', '')}\n"
                f"       {entry.get('verdict', '?')} -- {entry.get('confidence', '?')}% confident "
                f"({entry.get('source', '?')})"
            )
        self.reporter.say("")
        return 0
