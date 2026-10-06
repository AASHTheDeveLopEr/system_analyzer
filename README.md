<div align="center">

# System Analyzer

### A terminal tool that reads this machine and asks if it can run a task

Detects CPU, RAM, GPU, VRAM, disks, and OS, then uses a System One decision model to rate the machine out of 100 and answer questions such as “can I run this?”

<br>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![psutil](https://img.shields.io/badge/psutil-Hardware-2C3E50?style=for-the-badge)
![Local](https://img.shields.io/badge/Runs-Locally-1ABC9C?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-unittest-4A90A4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

</div>

---

# About

**System Analyzer** is a terminal-only Python app. It reads the computer it is started on, sends a scrubbed inventory to a decision model, and turns the answer into a score and a verdict.

Numeric requirement checks stay on the machine. Named parts can be judged by the model. Anything that cannot be read is reported as `unknown`. It is not guessed.

> A local system analyzer that rates this machine out of 100 and checks whether it can run a task.

---

# Modes

| Command | What it does |
|---|---|
| `python main.py` | Interactive menu. A line that is not a menu number is treated as a question. |
| `specs` | Show what was detected. |
| `rate` | Score out of 100, the model tier, and what moved the score. |
| `rate -n 5` | Ask five times and report how much the answer moves. |
| `ask "..."` | Ask if this machine can run a named task. |
| `check "..."` | Compare pasted requirements with this machine. |
| `history` | Show questions asked so far. |
| `clear` | Forget the question log. |
| `config` | Show thresholds, weights, and the provider chain. |
| `payload` | Show the scrubbed state sent to the model. |
| `setup` | Choose providers and models and save the choice. |
| `session` / `menu` | Aliases for the interactive app. |

Shared flags: `--json`, `--config`, `--history-file`, `--no-history`. Each command has its own `--help`.

---

# Features

- Hardware detection for CPU, RAM, GPU, VRAM, disks, and OS
- Total and available RAM, not only “free” memory
- Dedicated VRAM only when a driver exposes it; shared memory stays `unknown`
- Virtual adapters are dropped so a VM does not get a fictional GPU
- Interactive menu and one-question-per-line pipe mode
- 0–100 rating with a rule-only fallback when no endpoint answers
- Requirement checklists: numbers checked locally, named parts judged by the model
- Provider chain: Ollama, Zen, TypeSafe, and OpenRouter
- JSON output on stdout and human hints on stderr
- Local history and git-ignored settings

---

# Files

| File | Role |
|---|---|
| `main.py` | Commands, interactive loop, settings menu, JSON output, exit codes |
| `inventory.py` | Hardware detection only |
| `system_one.py` | Request validation, HTTP transport, failover, typed answers |
| `judgment.py` | Measured facts plus model answers, as a 0–100 rating |
| `providers.py` | Provider catalog, chain, Ollama and OpenRouter discovery |
| `requirements.py` | Pasted requirement checklists |
| `settings.py` | `config.json`, `.env`, menus, and precedence |
| `history.py` | Local question log |
| `config.py` | Thresholds, weights, and curves |
| `tests/` | `unittest` suite, no extra test dependency |
| `requirements.txt` | Runtime libraries |

---

# Install

```bash
python -m pip install -r requirements.txt
```

Detection uses libraries instead of hand-rolled probes:

- `psutil` for RAM, cores, and disks
- `py-cpuinfo` for CPU name and speed
- `pynvml` for NVIDIA GPUs and VRAM
- `gpu-detector` for AMD, Intel, and Apple GPUs
- `requests` for API calls

On Windows, VRAM comes from `pynvml`. `wmi` `AdapterRAM` is not used, because that 32-bit field wraps above 4 GB. On Linux, AMD VRAM comes from the driver file in `/sys/class/drm`. Intel integrated graphics stay `unknown`. On macOS, dedicated VRAM is counted only when the OS exposes it as dedicated.

---

# Run

```bash
python main.py
python main.py specs
python main.py rate
python main.py ask "can I run Cyberpunk 2077?"
python main.py check "Office" --min-reqs "4 GB RAM; 10 GB storage"
```

Interactive menu:

```text
  1. Show my specs
  2. Rate my PC out of 100
  3. Ask a question
  4. History
  5. Settings
  0. Quit
  ...or just type a question
  Ask or choose >
```

Piped input reads one question per line:

```bash
echo "can I game?" | python main.py
```

---

# Requirement checks

A small model cannot reliably answer an open “can this machine run X?” question. When an answer is unsure, the app can check a pasted list instead.

- Numeric lines such as `8 GB RAM` or `60 GB storage` are compared with measured specs. No model call.
- Named parts such as `GTX 970` become one yes/no question. An unsure model counts as unknown.
- Unparsed text is reported as not checked.

Verdicts name the requirement and both numbers: meets recommended, meets minimum, or below minimum.

```bash
python main.py ask "a game" --check
python main.py check "a game" --min-reqs "8 GB RAM; 50 GB storage"
```

In `--json` mode, an unsure ask can return `"requirements_needed": true` instead of blocking on a prompt.

---

# Exit codes

| Exit | Meaning |
|---|---|
| 0 | Answered, or a rules-only rating when rating degraded |
| 1 | The app itself failed |
| 2 | Wrong usage, or an unreadable config file |
| 3 | No System One endpoint answered |
| 4 | Specs cannot settle the question; no verdict was invented |

`ask` exits 3 with `no_endpoint_answered` when nothing responds. `rate` can fall back to the rule-only score, say so in `notes`, and still exit 3.

---

# Providers

The provider chain is an ordered list in `config.json`. A provider left out of the chain is never contacted.

| Provider | Where it runs | Key |
|---|---|---|
| `ollama` | This machine, offline | none |
| `zen` | OpenCode Zen, online | optional |
| `typesafe` | Paid, online | `TYPESAFE_API_KEY` |
| `openrouter` | Paid gateway, online | `OPENROUTER_API_KEY` |

The default chain is `zen` alone. Ollama is local at `http://localhost:11434` unless `OLLAMA_HOST` or `SYSTEM_ANALYZER_OLLAMA_URL` says otherwise. The app does not install models. A local decision model can be added with:

```bash
ollama pull tev1:0.8b
```

OpenRouter is a gateway rather than a model vendor: one key reaches decision models from several vendors, so the model is a setting (`SYSTEM_ANALYZER_OPENROUTER_MODEL`, default `typesafe/jev-1.13`). Requests go to `POST https://openrouter.ai/api/v1/systemone`, which speaks the same wire as TypeSafe, so no extra adapter is involved. Two things to know:

- **The payload leaves this machine** and is forwarded on to whichever vendor serves the model. True of every online provider here, but OpenRouter adds a second party.
- **Listing the models sends no key.** OpenRouter's catalog returns text-output models only unless asked otherwise, so decision models are missing from the default listing. Setup asks for `output_modalities=decisions` and then checks each model's own `decisions` output modality, so the list cannot quietly fill up with chat models that cannot answer a typed question.

Settings precedence:

1. Environment variable
2. `.env`
3. `config.json`
4. Built-in default

`config.json` and `.env` are meant to stay git-ignored. `.env` holds secrets and is not displayed.

---

# Tests

```bash
python -m unittest discover -s tests
python system_one.py --selftest
```

Live tests are opt-in:

```bash
SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_jev
SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_ollama
```

The Ollama live test is skipped unless a local Ollama is running and has a decision-capable model.

```bash
python system_one.py --probe
python system_one.py --probe -n 5
```

---

# Notes

- Rating weights in `config.py` are a starting point, not a measured truth.
- Unread components are excluded from the score and listed in `notes`.
- Do not commit API keys, `.env`, or `config.json` if it contains a secret.

---

# License

Released under the MIT License. See [`LICENSE`](LICENSE).
