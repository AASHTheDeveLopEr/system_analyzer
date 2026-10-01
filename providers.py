"""The provider catalog: who can answer, where they live, and how to find out.

Three providers speak the same wire format (see ``system_one``):

    ollama     decision models served by Ollama on this machine -- local
    zen        OpenCode Zen's free Jev tier -- remote, the default
    typesafe   TypeSafe's paid Jev -- remote, joins only when asked for

This module owns everything about providers: their endpoints and keys, the
provider chain that decides who is asked and in what order, Ollama discovery,
and the per-provider guidance. A provider left out of the chain is never
contacted -- not even as a fallback -- so the chain itself is the privacy
decision.

The app never assumes which Ollama models exist. Whatever ``GET /api/tags``
reports with the ``decision`` capability is the whole menu; when that list is
empty the advice is to pull the smallest known decision model, and the app
never installs anything itself.

Chain building reads files and the environment, never the network, so asking
a question cannot block on discovery. The one exception is discovery itself,
which is short-timeout by design: a wrong port on localhost should cost three
seconds, not thirty.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import requests

import settings as settings_module

PROVIDER_IDS = ("ollama", "zen", "typesafe")

OLLAMA_HOST_ENV = "OLLAMA_HOST"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: The advice when Ollama answers but no decision model is installed: the
#: smallest download that can do the job. Advice, never an action -- the app
#: does not install anything on its own.
RECOMMENDED_PULL_MODEL = "tev1:0.8b"

DISCOVERY_TIMEOUT = 3.0

#: Local decision models are small and often run on CPU, so a rating question
#: can legitimately take a minute. The cloud endpoints share the 30s default;
#: the local one gets its own ceiling.
OLLAMA_TIMEOUT_SECONDS = 120.0

PAID_MODELS_URL = "https://api.typesafe.ai/v1/models"

PROVIDER_LABELS = {
    "ollama": "Ollama -- runs on this machine, private, works offline",
    "zen": "OpenCode Zen -- free, runs online, no key needed",
    "typesafe": "TypeSafe -- paid, runs online, needs a key",
}


# --------------------------------------------------------------------------
# The chain
# --------------------------------------------------------------------------
# The default chain is OpenCode Zen's free `jev-1.13-free`, remote. A local
# Ollama provider and the paid TypeSafe backup join only where the user's
# provider chain puts them. The free tier needs no key today, but its key is
# read too (OPENCODE_ZEN_API_KEY) so the tool keeps working if the open tier
# ever closes to anonymous callers.

FREE_URL = "https://opencode.ai/zen/v1/systemone"
FREE_MODEL = "jev-1.13-free"

PAID_URL = "https://api.typesafe.ai/v1/systemone"
PAID_MODEL = "jev-latest"


def zen_provider(config: settings_module.Settings) -> dict:
    """The free Zen endpoint, anonymous unless a key is set."""
    free_key = config.get("zen_api_key").strip()
    # Anonymous callers are accepted, so a missing key is not an error.
    # Note that the probe prints these dicts: never put the key in one.
    return {
        "name": "jev-free",
        "url": FREE_URL,
        "model": FREE_MODEL,
        "source": "free",
        "auth": f"Bearer {free_key}" if free_key else "Bearer public",
    }


def paid_provider(
    config: settings_module.Settings | None = None, explicit: bool = False
) -> tuple[dict | None, str]:
    """The paid TypeSafe endpoint, plus a reason when it is out of the chain.

    Listing TypeSafe in the provider chain is itself the opt-in; without
    that, the backup flag and the key must both be set. The reason is
    returned so it can be shown to the user instead of silently dropping
    the backup.
    """
    config = config or settings_module.load_settings()
    api_key = config.get("typesafe_api_key").strip()
    if not api_key:
        return None, f"{settings_module.TYPESAFE_KEY_ENV} is not set"
    if not explicit:
        # The legacy backup path: the pre-chain contract, where flag and key
        # together let the paid endpoint join as a last resort. Only reached
        # when the user has written no provider chain of their own.
        flag = settings_module.CLOUD_FALLBACK_ENV
        if os.environ.get(flag, "").strip().lower() not in ("1", "true", "yes", "on"):
            return None, f"cloud escalation is off (set {flag}=1 to enable it)"

    return {
        "name": "typesafe",
        "url": PAID_URL,
        "model": PAID_MODEL,
        "source": "paid",
        "auth": f"Bearer {api_key}",
    }, ""


def chain() -> tuple[list[dict], list[str]]:
    """Endpoints to try in order, and the reasons the rest are out.

    The chain itself is a setting: provider ids in the user's order, and a
    provider left out is never contacted. The paid backup joins after the
    configured chain only when the cloud flag is on with a key -- never
    before.
    """
    config = settings_module.load_settings()
    ordered = config.chain()
    # The paid backup is a legacy path: once the user has written a chain of
    # their own, the chain is the entire truth and the flag is ignored.
    legacy_backup = config.origin("provider_chain") == "default"

    endpoints: list[dict] = []
    skipped: list[str] = []
    for pid in ordered:
        if pid == "ollama":
            provider, reason = ollama_provider(config)
            if reason:
                reason = f"ollama skipped: {reason}"
        elif pid == "zen":
            provider, reason = zen_provider(config), ""
        else:  # config.chain() validates ids, so "typesafe" is the only case left
            provider, reason = paid_provider(config, explicit=True)
            if reason:
                reason = f"typesafe skipped: {reason}"
        if provider:
            endpoints.append(provider)
        elif reason:
            skipped.append(reason)

    if legacy_backup:
        paid, reason = paid_provider(config)
        if paid:
            endpoints.append(paid)
        elif reason:
            skipped.append(f"typesafe skipped: {reason}")
    return endpoints, skipped


@dataclass(frozen=True)
class OllamaStatus:
    """What one discovery probe against Ollama found."""

    reachable: bool
    models: list[str] = field(default_factory=list)
    detail: str = ""


def resolve_ollama_url(config: settings_module.Settings | None = None) -> str:
    """The base URL to talk to Ollama on, from the setting, OLLAMA_HOST or default.

    A bare ``host:port`` (the shape OLLAMA_HOST uses) gains an ``http://``
    scheme and, when no port was given, Ollama's default port. A full URL is
    taken as written. A bind-style address such as ``0.0.0.0`` is a client
    target here, so it is read as this machine.
    """
    config = config or settings_module.load_settings()
    configured = config.get("ollama_url").strip()
    if configured:
        return _normalize_base_url(configured, default_port=None)
    host = os.environ.get(OLLAMA_HOST_ENV, "").strip()
    if host:
        return _normalize_base_url(host, default_port=11434)
    return DEFAULT_OLLAMA_URL


def _normalize_base_url(raw: str, default_port: int | None) -> str:
    text = raw.strip().rstrip("/")
    if "://" not in text:
        text = f"http://{text}"
        # OLLAMA_HOST-style input means Ollama's port when none was written.
        parts = urlsplit(text)
        if parts.port is None and default_port is not None:
            text = f"{parts.scheme}://{parts.hostname or ''}:{default_port}"
    parts = urlsplit(text)
    hostname = parts.hostname
    if not hostname:
        hostname = "127.0.0.1"
    elif hostname in ("0.0.0.0", "::", "*"):
        hostname = "127.0.0.1"
    netloc = hostname if parts.port is None else f"{hostname}:{parts.port}"
    return f"{parts.scheme}://{netloc}"


def discover_ollama(base_url: str, timeout: float = DISCOVERY_TIMEOUT) -> OllamaStatus:
    """Ask Ollama what it serves, and keep only decision-capable models.

    Every way this can fail is reported as a status with a readable detail,
    because the whole point of probing is to tell the user *which* thing is
    wrong: nothing there, something else on that port, or Ollama with no
    usable model installed.
    """
    url = base_url.rstrip("/") + "/api/tags"
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as error:
        return OllamaStatus(False, [], f"connection failed: {error}")

    if response.status_code != 200:
        return OllamaStatus(
            False, [], f"HTTP {response.status_code} from {base_url} -- "
            "something answered, but it did not look like Ollama"
        )

    try:
        body = response.json()
    except ValueError:
        return OllamaStatus(False, [], "the answer was not JSON")

    if not isinstance(body, dict) or not isinstance(body.get("models"), list):
        return OllamaStatus(False, [], "the answer was not an Ollama model list")

    models = [
        entry["name"]
        for entry in body["models"]
        if isinstance(entry, dict)
        and isinstance(entry.get("name"), str)
        and isinstance(entry.get("capabilities"), list)
        and "decision" in entry["capabilities"]
    ]
    return OllamaStatus(True, models, "")


def ollama_guidance(status: OllamaStatus, base_url: str, config: settings_module.Settings) -> list[str]:
    """What the user should do next, for exactly the situation the probe found.

    Four situations have four different fixes, and conflating them is how an
    app ends up telling someone to set an environment variable when their
    daemon is simply not running.
    """
    address_is_custom = bool(
        config.get("ollama_url").strip() or os.environ.get(OLLAMA_HOST_ENV, "").strip()
    )
    configured_model = config.get("ollama_model").strip()

    if not status.reachable:
        if address_is_custom:
            return [
                f"Ollama did not answer at {base_url}: is it running, and is the "
                "address right? Check the Ollama address in Settings."
            ]
        return [
            f"Ollama did not answer at the default address ({base_url}).",
            "If Ollama is installed, start it and try again; if it runs "
            "somewhere else, set its address in Settings.",
        ]
    if not status.models:
        return [
            "Ollama is running, but no decision-capable model is installed.",
            f"To install the smallest one:  ollama pull {RECOMMENDED_PULL_MODEL}",
        ]
    if configured_model and configured_model not in status.models:
        return [
            f"The configured model {configured_model!r} is not installed; "
            f"installed decision models: {', '.join(status.models)}",
        ]
    return []


def discover_typesafe(api_key: str, timeout: float = DISCOVERY_TIMEOUT) -> tuple[list[str] | None, str]:
    """TypeSafe's model list, or ``None`` and the reason it could not be listed."""
    if not api_key.strip():
        return None, "a TypeSafe key is required before its models can be listed"
    try:
        response = requests.get(
            PAID_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
        )
    except requests.RequestException as error:
        return None, f"request failed: {error}"

    if response.status_code in (401, 403):
        return None, f"the key was rejected (HTTP {response.status_code})"
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}"

    try:
        body = response.json()
    except ValueError:
        return None, "the answer was not JSON"
    if not isinstance(body, dict) or not isinstance(body.get("models"), list):
        return None, "unexpected response shape"
    models = [
        entry["name"]
        for entry in body["models"]
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]
    return models, ""


def ollama_provider(config: settings_module.Settings) -> tuple[dict | None, str]:
    """The local provider entry for the chain, or the reason it is out.

    The model comes from settings alone: chain building never touches the
    network, so asking a question stays deterministic even when Ollama is
    mid-upgrade. An unconfigured model skips the provider with a note.
    """
    base = resolve_ollama_url(config)
    model = config.get("ollama_model").strip()
    if not model:
        return None, "no Ollama model is configured (run: python main.py setup)"
    return {
        "name": "ollama",
        "url": base.rstrip("/") + "/v1/systemone",
        "model": model,
        "source": "local",
        "auth": "",
        "timeout": OLLAMA_TIMEOUT_SECONDS,
    }, ""
