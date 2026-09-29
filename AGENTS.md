# AGENTS.md — System Analyzer

## Overview

A terminal-only Python application that detects this machine's hardware specs (CPU, RAM, GPU/VRAM, disks, OS) and uses an AI decision model (Jev via the System One endpoint) to rate the system out of 100 and answer questions like "can I run X?".

**Key design principle:** The model supplies judgment; local code supplies arithmetic and honesty. Unreadable values are reported as `unknown` (never guessed), and when no endpoint answers the rating falls back to rule-only scores — clearly flagged.

## Project Structure

| File | Responsibility |
|------|----------------|
| `main.py` | Application entry point: argument parsing, subcommands, numbered menu, question log, JSON output, exit codes |
| `menu.py` | Hardware detection only: CPU, RAM, GPU/VRAM, disks, OS → structured records |
| `system_one.py` | Decision layer: typed questions against Jev, composition into ratings/verdicts |
| `config.py` | All thresholds, weights, and curves (declared in config, not hardcoded) |
| `history.json` | Persisted question log (local only, never sent remotely) |
| `tests/` | Offline test suite (stdlib `unittest`, no test dependencies) |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│   menu.py   │───▶│  main.py     │───▶│ system_one.py    │
│  (detect)   │    │  (command   │    │  (System One     │
│  → Inventory │    │   dispatch)  │    │   endpoint)      │
│  → state)   │    │             │    │                  │
└─────────────┘    └──────────────┘    └──────────────────┘
                        │                    ▲
                        │                    │
                        ▼                    │
                  ┌──────────────┐    ┌──────────────┐
                  │   config.py  │    │  (HTTP POST) │
                  │  (weights,   │    │  jev-1.13-free│
                  │   curves,    │    │  on Zen)      │
                  │   bands)     │    │               │
                  └──────────────┘    └──────────────┘
```

**Detection flow:**
1. `menu.detect_inventory()` → `Inventory` (frozen dataclass with `Memory`, `Gpu`, `Disk` sub-records)
2. `Inventory.to_state()` → scrubbed JSON (hardware facts only, no paths/names — FR-010)
3. `system_one.SystemOneClient` POSTs state + questions to the endpoint
4. Local composition layer turns answers into a 0–100 `Rating` or `CapabilityVerdict`

## Conventions

### Code Style
- **Python 3.10+** required (uses `X | Y` union syntax throughout)
- All modules begin with `from __future__ import annotations`
- Dataclasses are `frozen=True` for all records (`Gpu`, `Memory`, `Disk`, `Inventory`, `ComponentScore`, `Factor`, `Rating`, `CapabilityVerdict`)
- `None` means "unknown/unreadable", never "zero". A missing VRAM reading is `None`, not `0`.
- Every public function/class has a docstring; internal helpers start with `_`
- Error handling: detectors swallow library exceptions and return `unknown`, never raise (SC-001)
- Use `contextlib.suppress(...)` for cleanup paths (e.g., `pynvml.nvmlShutdown()`)

### Testing
- **Offline-first:** `python -m unittest discover -s tests` runs 169 tests with no network
- **Live tests opt-in:** `SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_jev`
- Tests use `unittest.mock` to patch hardware — never test against real hardware in CI
- Follow existing test structure: class-per-concern, descriptive method names
- **Never** write tests for incidental behavior, wiring, or mock echoes — test observable contracts

### Key Constants
```python
# In menu.py
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
- GPU detection: **pynvml first** (cross-platform, reports VRAM honestly), **Windows WMI as fallback** for non-NVIDIA adapters. Integrated GPUs that share system memory are listed as `unknown` VRAM, not the shared amount.
- Software adapters (VM, remote, basic render drivers) are dropped via `SOFTWARE_ADAPTERS` filter
- Deduplicated by normalized name (`_normalize()`) to avoid double-counting cards pynvml and WMI both see
- The `Gpu.source` field says where the name came from: `"nvml"` (strongest — driver-reported VRAM) or `"wmi"` (name only)

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
python main.py                      # menu (terminal) or one-shot (piped)
python main.py specs                # what was detected
python main.py rate                 # 0–100 rating + tier + factors
python main.py ask "can I run X?"   # capability verdict
python main.py rate -n 5            # stability: ask 5×, report spread
python main.py history              # question log
python main.py clear                # forget history
python main.py config               # thresholds/weights in force
python main.py payload              # scrubbed state sent to the model
python main.py session              # interactive mode (--json for machine output)
python main.py menu                 # force numbered menu even when piped
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
- **Add a detection field:** `menu.py` → `Inventory` dataclass → `to_state()` (check `system_one._measured()` can read it)
- **Add a subcommand:** `main.py` → `build_parser()` (add argument group) + `cmd_*()` function
- **Change model endpoint:** `system_one.py` → `_providers()` (free → paid fallback ordering)

### Before Editing
1. Run `python -m unittest discover -s tests` — know the baseline
2. Check if your change affects the state payload (`Inventory.to_state()`) — that's the contract with the remote model
3. If modifying an exported symbol, check all callers (`menu.detect_inventory` is called from both `main.py` and `system_one.py._probe_state`)

### Verification Checklist
- [ ] `python -m unittest discover -s tests` passes (offline, no regressions)
- [ ] `python main.py specs` runs and detects real hardware
- [ ] `python main.py payload` shows scrubbed state (no paths/filenames)
- [ ] `python main.py config --json` validates with defaults
- [ ] New code path tested with mocks (no real hardware dependency in tests)

## Honesty Rules (Non-Negotiable)

These are the spec's FR-002/FR-005/FR-006 requirements — violate them at your peril:

1. **No guessing**: `None` means unreadable, never `0`. An integrated GPU with no dedicated VRAM → `vram_bytes=None`, not "shared memory amount".
2. **No invented verdicts**: If specs can't determine the question → "Cannot tell" (exit 4), not a fabricated answer.
3. **Confidence honesty**: Coin-flip answers are capped and named uncertain. Two independent questions about the same request must agree; disagreement drops confidence.
4. **Fallback is flagged**: When no endpoint answers, rules-only rating is produced but `notes` explain it and exit code is 3 (not 0).
5. **Privacy**: State payload contains hardware facts only — no usernames, machine names, IPs, mount points, or file names. See `Inventory.to_state()`.

**Linux/macOS GPU detection**: Now handled via `gpu-detector` (AMD/Intel on
Linux via lspci + /sys/class/drm, Intel/Apple GPUs on macOS via system_profiler,
all vendors on Windows via WMI/PowerShell). NVIDIA VRAM always comes from pynvml.
VRAM from non-NVIDIA platform sources is trusted only on Linux (DRM sysfs),
not on macOS or Windows (shared memory / 32-bit wraparound).
**No local model**: The default is the hosted Jev remote endpoint. There is no offline mode. The local provider (Laya) is deliberately deferred.
**Rating weights are not calibrated**: CPU 30 / memory 25 / GPU 35 / storage 10 are a starting point, not a measured truth.
