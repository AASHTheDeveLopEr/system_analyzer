<div align="center">

# System Analyzer

### A terminal tool that reads this machine and asks if it can run a task

Detects CPU, RAM, GPU, VRAM, disks, and OS, then uses a System One decision model to rate the machine out of 100 and answer questions such as “can I run this?”

<br>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![psutil](https://img.shields.io/badge/psutil-Hardware-2C3E50?style=for-the-badge)
![Local](https://img.shields.io/badge/Runs-Locally-1ABC9C?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-unittest-4A90A4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

</div>

---

# About

**System Analyzer** is a terminal-only Python app. It reads the computer it is started on, sends a scrubbed hardware inventory to a decision model, and turns the answer into a score and a verdict.

The model supplies judgment. Local code supplies arithmetic and honesty. Unreadable values are `unknown`, never guessed. If no endpoint answers, the rating falls back to rule-only scores and says so.

> A local system analyzer that rates this machine out of 100 and checks whether it can run a task.

---

# Modes

| Command | What it does |
|---|---|
| `python main.py` | Interactive menu. A line that is not a number is treated as a question. |
| `specs` | Show what was detected. |
| `rate` | Score out of 100, tier, and the factors that moved it. |
| `rate -n 5` | Ask five times and report the spread. |
| `ask "..."` | Capability verdict for a named task. |
| `ask "..." --check` | Skip the open question and go straight to the checklist. |
| `check "..."` | Compare pasted requirements with this machine. |
| `history` | Show the local question log. |
| `clear` | Forget the log. |
| `config` | Thresholds, weights, and the provider chain, with origins. |
| `payload` | Scrubbed state sent to the model. |
| `setup` | Choose providers and models. Terminal only. |
| `session` / `menu` | Aliases for the interactive app. |

Shared flags work before or after the command: `--json`, `--config`, `--history-file`, `--no-history`. Each command has its own `--help`.

---

# How it is split

```text
inventory.py ──detect──▶ main.py ──ask──▶ system_one.py
                              │                 ▲
                              │                 │ provider chain
                              │           providers.py
                              ├──compose──▶ judgment.py
                              ├──check────▶ requirements.py
                              ├──settings─▶ settings.py + config.py
                              └──log──────▶ history.py
```

| File | Role |
|---|---|
| `main.py` | Commands, interactive loop, JSON output, exit codes |
| `inventory.py` | Hardware only: CPU, RAM, GPU/VRAM, disks, OS |
| `rankings.py` | Curated GPU/CPU capability tables, matching, tier labels |
| `system_one.py` | Request validation, HTTP failover, typed answers |
| `judgment.py` | Facts plus model answers, as a 0–100 rating |
| `providers.py` | Endpoint catalog, chain, Ollama discovery |
| `requirements.py` | Pasted checklists and tiered verdicts |
| `settings.py` | `config.json`, `.env`, precedence, masking |
| `history.py` | Local question log |
| `config.py` | Thresholds, weights, and curves |
| `tests/` | Offline `unittest` suite |

`history.json` stays local. `config.json` and `.env` are git-ignored.

---

# Install

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

Detection uses libraries, not hand-rolled probes:

- `psutil` for RAM, cores, and disks
- `py-cpuinfo` for CPU name and speed
- `pynvml` for NVIDIA GPUs and VRAM
- `gpu-detector` for AMD, Intel, and Apple GPUs
- `requests` for API calls

RAM is reported as total and available. Dedicated VRAM is reported only when a driver exposes it. Integrated adapters that share system memory stay `unknown`. Virtual adapters are dropped.

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

Before a model call the app prints who is being asked and the timeout. Repeats print `run 2/5`. Hardware is detected once per interactive session.

---

# Requirement checks

Numeric lines such as `8 GB RAM` are compared with measured specs. No model call. Named parts such as `GTX 970` are settled by the curated capability tables when both sides are known (`rankings.py`); only an unknown part or unknown machine becomes one yes/no question. An unsure model counts as unknown. Unparsed text is not checked.

Verdicts name the requirement and both numbers: meets recommended, meets minimum, or below minimum.

```bash
python main.py ask "a game" --check
python main.py check "a game" --min-reqs "8 GB RAM" --rec-reqs "16 GB RAM"
```

In `--json` mode, an unsure ask can return `"requirements_needed": true` instead of blocking on a prompt.

---

# Exit codes

| Exit | Meaning |
|---|---|
| 0 | Answered, or a rules-only rating when the model is unavailable |
| 1 | The app itself failed |
| 2 | Wrong usage, or an unreadable config file |
| 3 | No System One endpoint answered |
| 4 | Specs cannot settle the question; no verdict was invented |

---

# Providers

The chain is an ordered list. A provider left out of it is never contacted.

| Provider | Where it runs | Key |
|---|---|---|
| `ollama` | This machine, offline | none |
| `zen` | OpenCode Zen, online | optional |
| `typesafe` | Paid, online | `TYPESAFE_API_KEY` |

The default chain is `zen` alone. Ollama is discovery-driven: installed decision models come from `/api/tags`. The app does not install models. If none are installed, the advice is:

```bash
ollama pull tev1:0.8b
```

Local models get 120 seconds per question. Cloud endpoints get 30.

Settings precedence: environment variable, then `.env`, then `config.json`, then the built-in default. Secrets are masked in anything printed.

---

# Tests

```bash
python -m unittest discover -s tests
python system_one.py --selftest
```

The offline suite does not use the network. Live tests are opt-in:

```bash
SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_jev
SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_ollama
```

```bash
python system_one.py --probe
python system_one.py --probe -n 5
```

---

# Notes

- The payload has hardware facts only: no username, machine name, IP, mount point, or file name.
- Rating weights are a starting point: CPU 30, memory 25, GPU 35, storage 10.
- Do not commit API keys, `.env`, or a `config.json` that contains a secret.

---

# License

Released under the MIT License. See [`LICENSE`](LICENSE).
