"""The app's own settings: config.json, .env and the environment.

Two files sit next to the code, both git-ignored:

* ``config.json`` -- choices: which providers to use, which Ollama model,
  feature flags.
* ``.env``        -- secrets only: API keys. Masked everywhere they are shown.

The same setting can be given in three places, and the winner is always the
one a script can control:

1. a real environment variable (CI, containers, one-off runs)
2. the same name in ``.env``
3. the value in ``config.json``
4. the built-in default

Every setting is declared exactly once, in :data:`SETTINGS`, with its
environment name, a plain-language label, help text and default. The settings
menu, the ``config`` command and the validation are all driven by that table,
so a setting cannot be editable in one place and invisible in another.

A corrupt ``config.json`` raises :class:`SettingsError` -- the app exits with
code 2 rather than quietly running on defaults the user believed they had
changed. A malformed ``.env`` line is skipped with a warning instead: a bad
comment must not take the API keys down with it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Where the settings directory comes from; tests point this at a scratch dir.
SETTINGS_DIR_ENV = "SYSTEM_ANALYZER_SETTINGS_DIR"

CONFIG_FILE = "config.json"
ENV_FILE = ".env"

#: Env names live here so system_one and the menus quote the same strings.
PROVIDERS_ENV = "SYSTEM_ANALYZER_PROVIDERS"
OLLAMA_URL_ENV = "SYSTEM_ANALYZER_OLLAMA_URL"
OLLAMA_MODEL_ENV = "SYSTEM_ANALYZER_OLLAMA_MODEL"
#: Legacy, and not a setting anymore: honored only when no provider chain is
#: configured anywhere (see providers.chain). Once the user writes a chain,
#: the chain is the entire truth and this flag is ignored.
CLOUD_FALLBACK_ENV = "SYSTEM_ANALYZER_CLOUD_FALLBACK"
ZEN_KEY_ENV = "OPENCODE_ZEN_API_KEY"
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"


@dataclass(frozen=True)
class Setting:
    """One editable setting, declared once and rendered everywhere from this."""

    key: str
    env: str
    label: str
    help: str
    default: str = ""
    kind: str = "str"  # "str" | "bool" | "chain"
    secret: bool = False


#: The whole editable surface, in the order the settings menu shows it.
SETTINGS: tuple[Setting, ...] = (
    Setting(
        "provider_chain",
        PROVIDERS_ENV,
        "Providers, in order",
        "Which decision providers to try, and in what order. "
        "Comma-separated: ollama, zen, typesafe. A provider left out is "
        "never contacted.",
        default="zen",
        kind="chain",
    ),
    Setting(
        "ollama_url",
        OLLAMA_URL_ENV,
        "Ollama address",
        "Where Ollama runs, e.g. localhost:11434. Left empty, the address "
        "from OLLAMA_HOST is used, then the default port.",
    ),
    Setting(
        "ollama_model",
        OLLAMA_MODEL_ENV,
        "Ollama model",
        "Which installed decision model to ask. Leave empty and the Ollama "
        "provider is skipped with a note.",
    ),
    Setting(
        "zen_api_key",
        ZEN_KEY_ENV,
        "OpenCode Zen API key",
        "Optional today: the free tier answers without a key. Needed only if "
        "the open tier ever closes.",
        secret=True,
    ),
    Setting(
        "typesafe_api_key",
        TYPESAFE_KEY_ENV,
        "TypeSafe API key",
        "The paid TypeSafe endpoint. A key alone changes nothing: TypeSafe "
        "only joins the chain when listed in Providers (or the backup flag "
        "is on).",
        secret=True,
    ),
)

_BY_KEY = {setting.key: setting for setting in SETTINGS}
_BY_ENV = {setting.env: setting for setting in SETTINGS}

_CHAIN_IDS = ("ollama", "zen", "typesafe")
_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SettingsError(Exception):
    """A settings file says something that cannot be used.

    Deliberately not a ``ValueError``: the interactive ask block
    catches ``ValueError`` to keep a bad answer from ending the
    session, and that catch must never be able to swallow a settings
    failure. The app's promise for a broken config.json is exit 2,
    not a soft printed line, so the class hierarchy has to keep the
    two apart structurally.
    """


class Settings:
    """Effective settings with their origins, loaded from all three layers."""

    def __init__(self, values: dict[str, tuple[str, str]], warnings: list[str]) -> None:
        # key -> (value, origin); origin is "environment", ".env",
        # "config.json" or "default".
        self._values = values
        self.warnings = warnings

    def get(self, key: str) -> str:
        return self._values[key][0]

    def origin(self, key: str) -> str:
        return self._values[key][1]

    def chain(self) -> list[str]:
        """The provider chain as ids, in the user's order, without repeats."""
        return validate_chain(self.get("provider_chain"))

    def to_dict(self, include_secrets: bool = False) -> dict[str, Any]:
        """Every setting with its value and origin.

        Secrets are reported as ``"set"``/``"not set"`` unless
        ``include_secrets`` is true, because this document is printed.
        """
        document: dict[str, Any] = {}
        for setting in SETTINGS:
            value = self.get(setting.key)
            if setting.secret and not include_secrets:
                value = "set" if value else "not set"
            elif setting.kind == "chain":
                value = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
            document[setting.key] = {"value": value, "origin": self.origin(setting.key)}
        return document


def validate_chain(raw: str) -> list[str]:
    """Parse a provider chain string; unknown ids and an empty chain fail loudly.

    An unknown provider id is a typo in the chain and must be reported, not
    silently dropped: half a chain is a privacy decision the user did not make.
    """
    chain: list[str] = []
    for part in (raw or "").replace(";", ",").split(","):
        pid = part.strip().lower()
        if not pid:
            continue
        if pid not in _CHAIN_IDS:
            raise SettingsError(
                f"provider_chain has unknown provider {pid!r}; "
                f"known providers: {', '.join(_CHAIN_IDS)}"
            )
        if pid not in chain:
            chain.append(pid)
    if not chain:
        raise SettingsError("provider_chain is empty; name at least one provider")
    return chain


def settings_dir(base: str | os.PathLike[str] | None = None) -> Path:
    """The directory holding config.json and .env."""
    if base is not None:
        return Path(base)
    from_env = os.environ.get(SETTINGS_DIR_ENV, "").strip()
    if from_env:
        return Path(from_env)
    return Path(__file__).resolve().parent


def load_settings(base: str | os.PathLike[str] | None = None) -> Settings:
    """Defaults, overlaid with .env and config.json, under the environment."""
    directory = settings_dir(base)
    file_values = _read_config_file(directory / CONFIG_FILE)
    env_values, warnings = _read_env_file(directory / ENV_FILE)

    values: dict[str, tuple[str, str]] = {}
    for setting in SETTINGS:
        if setting.env in os.environ:
            values[setting.key] = (os.environ[setting.env], "environment")
        elif setting.env in env_values:
            values[setting.key] = (env_values[setting.env], ENV_FILE)
        elif setting.key in file_values:
            values[setting.key] = (str(file_values[setting.key]), CONFIG_FILE)
        else:
            values[setting.key] = (setting.default, "default")

    settings = Settings(values, warnings)
    try:
        settings.chain()  # fail fast on an unknown provider id
    except SettingsError as error:
        # The chain can come from the environment, .env or
        # config.json; name which, so the one failing line says
        # what to fix.
        raise SettingsError(
            f"{settings.origin('provider_chain')}: {error}"
        ) from error
    return settings


def configured(base: str | os.PathLike[str] | None = None) -> bool:
    """Has this install been set up, by file or by deliberate environment?"""
    if (settings_dir(base) / CONFIG_FILE).exists():
        return True
    return bool(os.environ.get(PROVIDERS_ENV, "").strip())


def save_config(values: dict[str, Any], base: str | os.PathLike[str] | None = None) -> Path:
    """Merge ``values`` into config.json and return its path.

    Refuses to write over a file it cannot parse: silently replacing a broken
    settings file would delete evidence of what the user actually configured.
    """
    path = settings_dir(base) / CONFIG_FILE
    current: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise SettingsError(
                f"{path} cannot be parsed ({error}); fix or delete it before saving"
            ) from error
        if not isinstance(loaded, dict):
            raise SettingsError(f"{path} must contain a JSON object")
        current = loaded

    unknown = sorted(set(values) - set(_BY_KEY))
    if unknown:
        raise SettingsError(
            f"unknown setting(s): {', '.join(unknown)}; "
            f"known settings: {', '.join(sorted(_BY_KEY))}"
        )
    current.update(values)
    try:
        path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise SettingsError(f"could not write {path}: {error}") from error
    return path


def save_env(key: str, value: str, base: str | os.PathLike[str] | None = None) -> Path:
    """Set one variable in .env, keeping every other line as it is."""
    if key not in _BY_ENV or not _BY_ENV[key].secret:
        raise SettingsError(f"{key} is not a known secret; refusing to write it")
    path = settings_dir(base) / ENV_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    except (OSError, UnicodeDecodeError) as error:
        raise SettingsError(f"could not read {path}: {error}") from error

    entry = f"{key}={value}"
    replaced = False
    for index, line in enumerate(lines):
        if _env_key(line) == key:
            lines[index] = entry
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(entry)
    try:
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except OSError as error:
        raise SettingsError(f"could not write {path}: {error}") from error
    return path


def _env_key(line: str) -> str:
    """The KEY out of a ``KEY=VALUE`` line, tolerating ``export`` and blanks."""
    text = line.strip()
    if text.startswith("export "):
        text = text[len("export "):].strip()
    return text.partition("=")[0].strip()


def _read_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    """The KEY=VALUE pairs in .env; bad lines are skipped with a warning."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, []
    except (OSError, UnicodeDecodeError) as error:
        return {}, [f"could not read {path.name} ({error}); its values are ignored"]

    values: dict[str, str] = {}
    warnings: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            warnings.append(f"{path.name} line {number}: not KEY=VALUE, ignored")
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not _KEY_PATTERN.fullmatch(key):
            warnings.append(f"{path.name} line {number}: {key!r} is not a variable name, ignored")
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values, warnings


def _read_config_file(path: Path) -> dict[str, Any]:
    """config.json as a dict; unreadable or unknown content fails loudly."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise SettingsError(f"could not read {path}: {error}") from error

    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SettingsError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise SettingsError(f"{path} must contain a JSON object")

    unknown = sorted(set(loaded) - set(_BY_KEY))
    if unknown:
        raise SettingsError(
            f"{path} has unknown setting(s): {', '.join(unknown)}. "
            f"Known settings: {', '.join(sorted(_BY_KEY))}"
        )
    return loaded
