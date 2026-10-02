# AGENTS.md — System Analyzer

## Overview

A terminal-only Python application that detects this machine's hardware specs (CPU, RAM, GPU/VRAM, disks, OS) and uses an AI decision model (Jev via the System One endpoint) to rate the system out of 100 and answer questions like "can I run X?".

**Key design principle:** The model supplies judgment; local code supplies arithmetic and honesty. Unreadable values are reported as `unknown` (never guessed), and when no endpoint answers the rating falls back to rule-only scores — clearly flagged.

## Project Structure

| File | Responsibility |
|------|----------------|
| `main.py` | The whole user surface: argument parsing (one subparser per command), the commands, the single interactive loop, onboarding/settings menus, JSON output, exit codes |
| `inventory.py` | Hardware detection only: CPU, RAM, GPU/VRAM, disks, OS → frozen records (`Inventory`, `Memory`, `Gpu`, `Disk`) |
| `system_one.py` | The wire client: request validation, HTTP transport with failover, answer decoding (`Decision`) |
| `judgment.py` | The judgment layer: measured facts + model answers → 0–100 `Rating` and `CapabilityVerdict`, with the honesty rules |
| `providers.py` | The provider catalog: endpoint builders, the provider chain (`providers.chain()`), Ollama discovery (`/api/tags`, `decision` capability), per-provider guidance |
| `requirements.py` | Requirement checklists: parse pasted requirement lines, check numerics against measured specs locally, judge named parts via atomic nouls, compose tiered verdicts |
| `settings.py` | The settings registry: config.json + .env storage, precedence (env > .env > config.json > default), origins, masking |
| `history.py` | The local question log: load/add/clear/show, corrupt-file survival |
| `config.py` | All thresholds, weights, and curves (declared in config, not hardcoded) |
| `history.json` | Persisted question log (local only, never sent remotely) |
| `config.json`, `.env` | Provider settings and secrets, git-ignored, written by setup/the settings menu |
| `tests/` | Offline test suite (stdlib `unittest`, no test dependencies) |

## Architecture

```
inventory.py ──detect──▶ main.py ──ask──▶ system_one.py (wire protocol, failover)
 (hardware facts)         │                    ▲
                          │                    │ chain: who answers, in order
                          │              providers.py (catalog + discovery)
                          ├──compose──▶ judgment.py (facts + answers → rating/verdict)
                          ├──check────▶ requirements.py (pasted requirements → checklist)
                          ├──settings─▶ settings.py (config.json/.env) + config.py (tunables)
                          └──log──────▶ history.py (local question log)
```

**Detection flow:**
1. `inventory.detect_inventory()` → `Inventory` (frozen dataclass with `Memory`, `Gpu`, `Disk` sub-records)
2. `Inventory.to_state()` → scrubbed JSON (hardware facts only, no paths/names)
3. `system_one.SystemOneClient` POSTs state + questions to the first endpoint in `providers.chain()` that answers
4. `judgment.rate_system` / `judgment.answer_capability` turn answers into a 0–100 `Rating` or `CapabilityVerdict`

## Conventions

### Code Style
- **Python 3.10+** required (uses `X | Y` union syntax throughout)
- All modules begin with `from __future__ import annotations`
- Dataclasses are `frozen=True` for all records (`Gpu`, `Memory`, `Disk`, `Inventory`, `ComponentScore`, `Factor`, `Rating`, `CapabilityVerdict`)
- `None` means "unknown/unreadable", never "zero". A missing VRAM reading is `None`, not `0`.
- Every public function/class has a docstring; internal helpers start with `_`
- Error handling: detectors swallow library exceptions and return `unknown`, never raise
- Use `contextlib.suppress(...)` for cleanup paths (e.g., `pynvml.nvmlShutdown()`)
- Comments and docstrings state constraints as facts (what the code must do and why), never as references to external documents or requirement IDs

### Testing
- **Offline-first:** `python -m unittest discover -s tests` runs 279 tests with no network (12 opt-in live tests skipped)
- **Live tests opt-in:** `SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_jev` (and `tests.test_live_ollama`)
- Tests use `unittest.mock` to patch hardware — never test against real hardware in CI
- Follow existing test structure: class-per-concern, descriptive method names
- **Never** write tests for incidental behavior, wiring, or mock echoes — test observable contracts

### Key Constants
```python
# In inventory.py
GIB = 1024 ** 3
MIB = 1024 ** 2
GHZ = 1e9
UNKNOWN = "unknown"

# Exit codes (main.py)
EXIT_OK = 0          # answered successfully
EXIT_ERROR = 1       # app-level failure
EXIT_USAGE = 2       # bad arguments or config
EXIT_NO_MODEL = 3    # no endpoint answered
EXIT_UNGROUNDED = 4  # specs cannot determine the question
```

### Data Flow Conventions
- GPU detection: **pynvml first** (cross-platform, reports VRAM honestly), **gpu-detector as fallback** for non-NVIDIA adapters (Windows WMI, Linux lspci/sysfs, macOS system_profiler). Integrated GPUs that share system memory are listed as `unknown` VRAM, not the shared amount. Linux DRM sysfs VRAM is trusted; macOS/Windows platform VRAM is not.
- Software adapters (VM, remote, basic render drivers) are dropped via `SOFTWARE_ADAPTERS` filter
- Deduplicated by normalized name (`_normalize()`) to avoid double-counting cards pynvml and gpu-detector both see
- The `Gpu.source` field says where the name came from: `"nvml"` (strongest — driver-reported VRAM) or `"platform"`/`"sysfs"` (name only, except Linux DRM sysfs VRAM which is trusted)
- The interactive loop detects hardware once per session and reuses it; one-shot commands detect fresh

### CLI Conventions
- One subparser per command (`main.build_parser()`): every command has its own `--help` and only the flags it actually uses
- Shared flags (`--json`, `--config`, `--history-file`, `--no-history`) work in either position — before or after the command — because they are `default=SUPPRESS` on a parent parser
- There is exactly one interactive mode (`cmd_interactive`); `session` and `menu` are aliases. In a terminal it shows a menu where any non-number line is treated as a question; with piped stdin it answers one question per line
- The requirements checklist has no menu slot: it is offered automatically after an unsure answer, and `check <task>` typed at the prompt (or as option 3's answer) goes straight to it without an open-ended model call. The `check` subcommand remains for one-shot/script use
- Before every model call, the CLI prints who is being asked and the timeout (`Asking zen (up to 30 s) ...`); repeats print `run 2/5 ...`; the answer's source line ends with how long the model took

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Answer produced (or rule-only rating when model unavailable) |
| 1 | App-level error (exception, broken config) |
| 2 | Usage error or unreadable config file |
| 3 | No System One endpoint answered |
| 4 | Specs cannot determine the question — no verdict invented |

## Commands

```bash
python main.py                      # interactive (menu on a terminal; question loop if piped)
python main.py specs                # what was detected
python main.py rate                 # 0–100 rating + tier + factors
python main.py ask "can I run X?"   # capability verdict
python main.py rate -n 5            # stability: ask 5×, report spread
python main.py history              # question log
python main.py clear                # forget history
python main.py config               # thresholds/weights + provider chain, with origins
python main.py payload              # scrubbed state sent to the model
python main.py setup                # interactive provider setup (TTY only)
python main.py check "Office" --min-reqs "4 GB RAM" [--rec-reqs "..."]
python main.py ask "run Office" --check          # skip the open-ended judgment
python main.py session              # interactive mode (--json for machine output)
python main.py menu                 # alias of session
```

## Development

### Setup
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests    # verify offline suite
```

### Key Files for Common Tasks
- **Add a config tunable:** `config.py` → `Config` class + `curve()`/`weights()` accessors
- **Change verdict bands:** `config.py` `VERDICT_BANDS`
- **Add a provider setting:** `settings.py` → `SETTINGS` table (one entry drives menus, display and validation)
- **Extend requirement checking:** `requirements.py` → `_parse_clause` (clause → component) + `_QUESTION` (atomic noul wording)
- **Add a detection field:** `inventory.py` → `Inventory` dataclass → `to_state()` (check `judgment._measured()` can read it)
- **Add a subcommand:** `main.py` → `build_parser()` (add a subparser) + `cmd_*()` function + `handlers` dict in `main()`
- **Change provider behavior:** `providers.py` (catalog, `chain()`, discovery, guidance)

### Before Editing
1. Run `python -m unittest discover -s tests` — know the baseline
2. Check if your change affects the state payload (`Inventory.to_state()`) — that's the contract with the remote model
3. If modifying an exported symbol, check all callers (`inventory.detect_inventory` is called from `main.py` and `system_one._probe_state`)

### Verification Checklist
- [ ] `python -m unittest discover -s tests` passes (offline, no regressions)
- [ ] `python main.py specs` runs and detects real hardware
- [ ] `python main.py payload` shows scrubbed state (no paths/filenames)
- [ ] `python main.py config --json` validates with defaults
- [ ] New code path tested with mocks (no real hardware dependency in tests)

## Honesty Rules (Non-Negotiable)

Violate these at your peril:

1. **No guessing**: `None` means unreadable, never `0`. An integrated GPU with no dedicated VRAM → `vram_bytes=None`, not "shared memory amount".
2. **No invented verdicts**: If specs can't determine the question → "Cannot tell" (exit 4), not a fabricated answer.
3. **Confidence honesty**: Coin-flip answers are capped and named uncertain. Two independent questions about the same request must agree; disagreement drops confidence.
4. **Fallback is flagged**: When no endpoint answers, rules-only rating is produced but `notes` explain it and exit code is 3 (not 0).
5. **Privacy**: State payload contains hardware facts only — no usernames, machine names, IPs, mount points, or file names. See `Inventory.to_state()`.

**Linux/macOS GPU detection**: handled via `gpu-detector` (AMD/Intel on
Linux via lspci + /sys/class/drm, Intel/Apple GPUs on macOS via system_profiler,
all vendors on Windows via WMI/PowerShell). NVIDIA VRAM always comes from pynvml.
VRAM from non-NVIDIA platform sources is trusted only on Linux (DRM sysfs),
not on macOS or Windows (shared memory / 32-bit wraparound).

**The provider chain is the privacy decision**: providers are an ordered user-edited list (`SYSTEM_ANALYZER_PROVIDERS` / config.json / setup). A provider left out of the chain is never contacted, not even as fallback. Default chain: `zen` alone. TypeSafe joins when listed in the chain. The `SYSTEM_ANALYZER_CLOUD_FALLBACK` + key combination is a legacy path honored only when no chain is configured anywhere — a written chain cannot be extended by the flag. A fallback that fires must be flagged in `source`/`provider`/`notes`; exit 3 only when nothing answered. The chain lives in `providers.py` (`providers.chain()`), not in `system_one`.

**Settings precedence**: real environment variable > `.env` > `config.json` > built-in default. Every setting is declared once in `settings.SETTINGS` (env name, label, help, kind, secret flag) — the menus, `config` display and validation are generated from that table. Secrets are masked in anything printed. `config.json` corruption raises `SettingsError` (exit 2); a malformed `.env` line is skipped with a warning. Tests isolate via `SYSTEM_ANALYZER_SETTINGS_DIR` (see `tests/_helpers.py`).

**Ollama is discovery-driven**: the app never hardcodes which Ollama models exist — installed decision models come from `GET /api/tags` filtered on the `decision` capability. The app never installs anything; when nothing is installed it recommends `ollama pull tev1:0.8b`. Local models get a 120s timeout (cloud: 30s).

**Rating weights are not calibrated**: CPU 30 / memory 25 / GPU 35 / storage 10 are a starting point, not a measured truth.
