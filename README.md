# system_analyzer

Terminal-only Python app that reads this machine's specs and uses a System One
decision model (Jev) to rate the system out of 100 and answer "can I run X?".

## Files

- `main.py` — the whole user surface: argument parsing (one subparser per
  command), the commands, the single interactive loop, the settings menu,
  setup, JSON output and exit codes.
- `inventory.py` — hardware detection only: CPU, RAM, GPU/VRAM, disks, OS.
- `system_one.py` — the wire client: request validation, HTTP transport with
  failover, and decoding the typed `noul` / `choice` / `score` answers.
- `judgment.py` — the judgment layer: measured facts + model answers composed
  into the 0–100 rating and the capability verdicts.
- `providers.py` — the provider catalog: the endpoint builders, the provider
  chain, Ollama discovery and address resolution, TypeSafe model listing,
  and the per-provider guidance.
- `requirements.py` — requirement checklists: pasted requirements parsed,
  numerics checked locally, named parts judged by the model.
- `settings.py` — the settings: one registry driving config.json, .env, the
  menus and the `config` display, with a defined precedence between them.
- `history.py` — the local question log.
- `config.py` — the thresholds, weights and curves, all declared in
  configuration rather than hidden in code.
- `tests/` — the test suite (standard library `unittest`, no test dependency).
- `requirements.txt` — the handful of libraries the app uses.

## Install

```bash
python -m pip install -r requirements.txt
```

Detection is delegated to libraries rather than hand-rolled: `psutil` (RAM,
cores, disks), `py-cpuinfo` (CPU name/speed), `pynvml` (NVIDIA GPUs on any OS,
reports VRAM most reliably), `gpu-detector` (AMD/Intel/Apple GPUs on Linux,
Windows, and macOS), `requests` for the API calls. Anything a library cannot
read is reported as `unknown`, never guessed.

On RAM, GPU and VRAM specifically:

- **RAM** is reported as *total* and *available* (not "free", which would
  exclude reclaimable cache and badly understate what is actually usable).
- **VRAM** is only reported when a driver genuinely exposes dedicated memory.
  Integrated adapters that share system memory are listed as `unknown` rather
  than as the shared amount, because shared memory is not capacity the machine
  owns.
- On Windows, `wmi`'s `AdapterRAM` is **not** used for VRAM: it is a 32-bit
  field that wraps above 4 GB, which reports a 8 GB card as 0. VRAM on Windows
  comes from pynvml (NVIDIA only); other adapters show names but VRAM unknown.
- On Linux, VRAM for AMD GPUs comes from the driver's `mem_info_vram_total` in
  `/sys/class/drm`. Intel integrated graphics have no dedicated VRAM file, so
  VRAM stays `unknown`.
- On macOS, `system_profiler` reports GPU names; dedicated VRAM is only counted
  when the OS exposes it as such (not shared-system-memory).
- Virtual adapters (VMware, VirtualBox, Hyper-V, basic render drivers, llvmpipe,
  swrast, etc.) are dropped, so a VM does not get a fictional GPU.

## Running

```bash
python main.py                      # interactive: menu on a terminal,
                                    # one answer per line if piped
python main.py specs                # what was detected
python main.py rate                 # 0-100, the model's tier, and what moved it
python main.py ask "can I run Cyberpunk 2077?"
python main.py rate -n 5            # ask five times, report how much it moves
python main.py history              # the questions asked so far
python main.py clear                # forget them
python main.py config               # thresholds, weights, and the provider chain
python main.py payload              # the scrubbed state sent to the model
python main.py setup                # choose providers and models, save the choice
python main.py check "Office" --min-reqs "4 GB RAM; 10 GB storage"
```

Every command has its own `--help` (`python main.py ask --help`) listing only
the flags that command actually uses. The shared flags (`--json`, `--config`,
`--history-file`, `--no-history`) work in either position, before or after the
command.

### The interactive app

`python main.py` (or `session` / `menu` — they are aliases) opens one
interactive mode. In a terminal it shows a menu, and *any line that is not a
menu number is treated as a question*, so you never have to enter a submenu
just to ask something:

```
  1. Show my specs
  2. Rate my PC out of 100
  3. Ask a question
  4. History
  5. Settings
  0. Quit
  ...or just type a question, e.g. "can I game?"
  ...or "check <task>" to paste a program's requirements
      and check them against this machine
  Ask or choose > can I run Cyberpunk 2077?
```

With piped stdin the same loop reads one question per line and answers until
the pipe ends: `echo "can I game?" | python main.py`.

So you always know what is happening, the app says what it is doing: before a
model call it prints who is being asked and the timeout (`Asking zen (up to
30 s) ...`), repeated runs count themselves (`run 2/5 ...`), and the answer's
source line ends with how long the model took. Hardware is detected once per
session, not re-probed for every question.

### Requirement checklists (the ask-back)

A decision model cannot know what a program requires, and a small local
model cannot even judge "can this machine run X" open-ended — it answers
coin flips. So when an answer comes back unsure, the app asks *you* for the
requirements (game stores list them), and checks each one in the role that
fits:

- **Numeric requirements** ("8 GB RAM", "2.4 GHz", "60 GB storage") are
  compared directly against the measured specs — deterministic, offline, no
  model call, exact numbers shown.
- **Named parts** ("GTX 970") become one atomic yes/no question for the
  provider, gated by the usual coin-flip rule: an unsure model counts as
  *unknown*, never as a pass or a fail.
- Anything unparsable is reported as **not checked**. Nothing is guessed.

The same requirements text can carry a recommended tier; the gap between
minimum and recommended is the verdict: *Meets the recommended — runs well*,
*Meets the minimum — runs, but tightly*, or *Below the minimum*, always
naming which requirement and both numbers. Scripts get the same checklist
via `--min-reqs` / `--rec-reqs`; an unsure `ask` in `--json` mode carries
`"requirements_needed": true` and a hint instead of blocking on a question.

You never have to wait for an unsure answer to be told you can help, and the
checklist needs no menu of its own: it is offered automatically after an
unsure answer, and typing `check <task>` in the interactive app (at the main
prompt, or as option 3's answer) goes straight to it — no model call is spent
on a judgment the checklist will not use. From the command line,
`ask "<task>" --check` or `check "<task>" --min-reqs "..."` does the same in
one shot.

### Machine-readable mode

`--json` puts the document on stdout and every human hint on stderr, so
`python main.py ask "can I run Minecraft?" --json | jq` works. Failures carry
a reason rather than an empty answer:

| Exit | Meaning |
| ---- | ------- |
| 0 | answered (or the rules-only rating, when the rating degraded) |
| 1 | something in the app itself broke |
| 2 | wrong usage, or an unreadable config file |
| 3 | no System One endpoint answered |
| 4 | the specs cannot determine the question — no verdict was invented |

A dead endpoint never silently produces a number: `ask` exits 3 with
`{"error": "no_endpoint_answered", ...}`, and `rate` falls back to the
rule-only score, says so in `notes`, and still exits 3 so a script can tell.

### Honesty rules

There is no table of "what you need to run game X". Every question goes to the
model, and the answer is then made to earn its confidence:

- The verdict band (`yes` / `likely` / `probably not`) comes from the model's
  probability, cut at thresholds declared in `config.py`.
- An answer sitting near a coin flip is capped and named as uncertain.
- If the model says the specs do not determine the question, the verdict becomes
  "Cannot tell" and the answer explains what requirements would settle it.
- Two of the three questions are asked twice in different shapes, because
  noul answers carry no confidence field. If the two disagree, confidence drops.
- When a component could not be read it is excluded from the rating and listed
  in `notes`, rather than being scored as zero.

`confidence` is read as 0–100 wherever it appears, even though the endpoint
sends it as a 0–1 fraction.

## Tests

```bash
python -m unittest discover -s tests          # 279 offline tests, no network (12 opt-in live skipped)
python system_one.py --selftest               # the decision layer, verbose
```

The live contract tests are opt-in, because they call a real endpoint.
Verified passing (10/10, 2026-09-30):

```bash
SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_jev
SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_ollama
```

The Ollama half is skipped unless a local Ollama answers and has a
decision-capable model installed.

Check the decision layer against the real API:

```bash
python system_one.py --probe            # one live Jev call on this machine
python system_one.py --probe -n 5       # five calls, reporting answer spread
```

`--probe` sends this machine's real inventory (scrubbed of paths and file
names) and re-asks the same questions, which is how you check whether a
confidence band would mean anything before building the rating on top of it.

## Providers and settings

Which endpoint answers is the **provider chain**: an ordered list, stored in
`config.json` next to the code and edited with `python main.py setup` or the
Settings entry in the menu. A provider left out of the chain is never
contacted — not even as a fallback — so the chain itself is the privacy
decision. When a chain falls through to a later provider, the answer's
`source` and a note say so; exit code 3 means nothing answered.

| Provider   | Runs                        | Key                 | Default? |
| ---------- | --------------------------- | ------------------- | -------- |
| `ollama`   | on this machine, offline    | none                | no       |
| `zen`      | online (OpenCode Zen)       | optional            | **yes**  |
| `typesafe` | online (paid)               | `TYPESAFE_API_KEY`  | no       |

The default chain is `zen` alone. TypeSafe joins when listed in the chain.
As a legacy path, when **no** chain is configured anywhere, the paid endpoint
can still join as a last resort if **both** `TYPESAFE_API_KEY` and
`SYSTEM_ANALYZER_CLOUD_FALLBACK` are set — but once you write a chain of your
own, the chain is the entire truth and the flag is ignored.

### Ollama (local decision models)

Ollama serves the same wire format from `http://localhost:11434`. The app
never assumes which models exist: whatever its discovery reports with the
`decision` capability is the whole menu, and the app never installs anything
itself. When Ollama is running but nothing decision-capable is installed, the
advice is:

```bash
ollama pull tev1:0.8b
```

Ollama on another port or machine is picked up from `OLLAMA_HOST`, or fixed
with `SYSTEM_ANALYZER_OLLAMA_URL`. Local models get a 120-second ceiling per
question rather than the cloud endpoints' 30, because a 0.8B model on a CPU
is slow by nature.

### Where settings live

Two git-ignored files sit next to the code: `config.json` (choices) and
`.env` (secrets only — written by the app, never displayed). The same setting
can be given in three places, and the winner is the one a script controls:

1. a real environment variable
2. the same name in `.env`
3. `config.json`
4. the built-in default

`python main.py config` shows every value with where it came from. The
settings: `SYSTEM_ANALYZER_PROVIDERS` (the chain, e.g. `ollama,zen`),
`SYSTEM_ANALYZER_OLLAMA_URL`, `SYSTEM_ANALYZER_OLLAMA_MODEL`,
`OPENCODE_ZEN_API_KEY`, `TYPESAFE_API_KEY`.

## Not settled

The rating's weights and curves (CPU 30 / memory 25 / GPU 35 / storage 10, and
the shape of each curve) are a defensible starting point, not a measured truth.
They are configuration, so changing them is a one-file edit, but nobody has
checked them against how people actually rate machines.
