"""Shared test helpers.

Two things live here so no test has to repeat them:

* ``TempDir`` -- a scratch directory that works everywhere. ``tempfile`` picks
  the platform temp dir, which on a locked-down machine can be unwritable; this
  keeps scratch files inside the repo and cleans up after itself.
* ``FakeClient`` -- a stand-in for SystemOneClient that returns hand-written
  answers. The composition layer can then be tested against exactly the model
  behaviour we care about (a coin-flip answer, a contradictory answer, a
  question the specs do not determine) without a network call and without
  hoping a live model behaves.
"""

from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest import mock

import settings

REPO = Path(__file__).resolve().parent.parent


class TempDir(unittest.TestCase):
    """A TestCase with a writable ``self.tmp`` directory.

    ``tempfile`` is deliberately not used: its temp dir is not writable on every
    machine this runs on, and a test suite that cannot write its own fixtures
    is a suite that quietly stops testing.

    The settings layer is pointed at the scratch directory too, so a
    ``config.json`` or ``.env`` left in the repo by a real setup run cannot
    leak into what a test asserts.
    """

    def setUp(self) -> None:
        super().setUp()
        self._tmp = REPO / f".test-tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._tmp.mkdir(parents=True, exist_ok=True)
        self.tmp = self._tmp
        self._settings_env = mock.patch.dict(
            os.environ, {settings.SETTINGS_DIR_ENV: str(self._tmp)}
        )
        self._settings_env.start()

    def tearDown(self) -> None:
        self._settings_env.stop()
        shutil.rmtree(self._tmp, ignore_errors=True)
        super().tearDown()

    def write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path


class IsolatedSettings(unittest.TestCase):
    """A scratch settings directory for tests that do not want a TempDir."""

    def setUp(self) -> None:
        super().setUp()
        self._settings_tmp = REPO / f".test-tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._settings_tmp.mkdir(parents=True, exist_ok=True)
        patcher = mock.patch.dict(
            os.environ, {settings.SETTINGS_DIR_ENV: str(self._settings_tmp)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self._settings_tmp, True)


class FakeClient:
    """Returns the answers it is handed, and records what it was asked.

    ``answers`` maps a question id to the answer body the real endpoint would
    return. ``error`` makes every call fail, which is how the rule-only
    fallback is exercised. ``providers`` mirrors the real client's endpoint
    list, so the CLI's "asking ..." progress line can be tested.
    """

    def __init__(
        self,
        answers: dict[str, dict[str, Any]] | None = None,
        error: Exception | None = None,
        providers: list[dict[str, Any]] | None = None,
    ) -> None:
        self.answers = answers or {}
        self.error = error
        self.providers = providers or []
        self.calls: list[dict[str, Any]] = []

    def ask(self, state: Any, questions: dict[str, Any]) -> Any:
        from system_one import Decision, SystemOneError

        if self.error is not None:
            raise self.error
        self.calls.append({"state": state, "questions": questions})
        answers = {qid: dict(body) for qid, body in self.answers.items()}
        return Decision(
            model="fake-model",
            answers=answers,
            provider="fake",
            source="free",
            cost=0.0,
        )


def noul(value: float) -> dict[str, Any]:
    return {"type": "noul", "noul": value}


def choice(chosen: str, probabilities: dict[str, float], confidence: float) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": chosen,
        "probabilities": probabilities,
        "confidence": confidence,
    }


def score(value: float, legend: dict[str, str], confidence: float = 0.9) -> dict[str, Any]:
    return {
        "type": "score",
        "score": value,
        "legend": legend,
        "probabilities": {},
        "confidence": confidence,
    }


def capability_answers(
    probability: float,
    second: str = "yes",
    grounding: str = "specs_decide_it",
    second_confidence: float = 90.0,
    grounding_confidence: float = 90.0,
) -> dict[str, dict[str, Any]]:
    """A well-formed three-question answer set for ``answer_capability``."""
    return {
        "verdict": noul(probability),
        "grounding": choice(
            grounding,
            {grounding: 0.8, "specs_do_not_decide_it" if grounding == "specs_decide_it" else "specs_decide_it": 0.2},
            grounding_confidence,
        ),
        "second_opinion": choice(
            second,
            {"yes": 0.9 if second == "yes" else 0.1, "no": 0.1 if second == "yes" else 0.9},
            second_confidence,
        ),
    }


def sample_inventory(**overrides: Any) -> Any:
    """A small but complete machine, built without touching the real hardware."""
    import inventory

    machine = inventory.Inventory(
        os_name="Linux 6.1 (x86_64)",
        cpu_name="AMD Ryzen 5 3600",
        cpu_cores=6,
        cpu_ghz=3.6,
        cpu_usage_percent=14.0,
        memory=inventory.Memory(
            total_bytes=16 * 1024 ** 3, available_bytes=9 * 1024 ** 3, used_percent=43.8
        ),
        gpus=(inventory.Gpu("NVIDIA RTX 3060", 12 * 1024 ** 3, "nvml"),),
        disks=(inventory.Disk("/home", 1024 * 1024 ** 3, 400 * 1024 ** 3, 60.0),),
        python_version="3.14.6",
    )
    if not overrides:
        return machine
    fields = {**{f: getattr(machine, f) for f in machine.__dataclass_fields__}, **overrides}
    return inventory.Inventory(**fields)
