# system_analyzer

Terminal-only Python app that reads this machine's specs and uses a System One
decision model (Jev) to rate the system out of 100 and answer "can I run X?".

## Files

- `main.py` — the application: argument parsing, the commands, the numbered
  menu, the question log, JSON output and exit codes.
- `menu.py` — hardware detection only: CPU, RAM, GPU/VRAM, disks, OS.
- `system_one.py` — the decision layer: typed `noul` / `choice` / `score`
  questions against Jev, and the composition that turns the answers into a
  rating or a verdict.
- `config.py` — the thresholds, weights and curves, all declared in
  configuration rather than hidden in code.
- `tests/` — the test suite (standard library `unittest`, no test dependency).
- `requirements.txt` — the handful of libraries the app uses.

The spec is not in this repository: `specs/` is git-ignored, so the feature
requirements live with the project brief rather than in the source tree.

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
python main.py                      # menu, if a terminal; one-shot, if piped
python main.py specs                # what was detected
python main.py rate                 # 0-100, the model's tier, and what moved it
python main.py ask "can I run Cyberpunk 2077?"
python main.py rate -n 5            # ask five times, report how much it moves
python main.py history              # the questions asked so far
python main.py clear                # forget them
python main.py config               # the thresholds and weights in force
python main.py payload              # the scrubbed state sent to the model
```

`session` keeps asking questions until you type `quit`; `menu` forces the
numbered menu even when input is piped.

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
python -m unittest discover -s tests          # 169 offline tests, no network
python system_one.py --selftest               # the decision layer, verbose
```

The live contract tests are opt-in, because they call a real endpoint:

```bash
SYSTEM_ANALYZER_LIVE_TESTS=1 python -m unittest tests.test_live_jev
```

Check the decision layer against the real API:

```bash
python system_one.py --probe            # one live Jev call on this machine
python system_one.py --probe -n 5       # five calls, reporting answer spread
```

`--probe` sends this machine's real inventory (scrubbed of paths and file
names) and re-asks the same questions, which is how you check whether a
confidence band would mean anything before building the rating on top of it.

## Keys

No key is required. The default endpoint is **OpenCode Zen's free
`jev-1.13-free` model**, at `https://opencode.ai/zen/v1/systemone`; set
`OPENCODE_ZEN_API_KEY` to use your own Zen key instead of the public one.

**It is a hosted model, not a local one.** This machine's specs are sent to
OpenCode on every rating and every question, and there is no offline mode — a
local Jev-compatible provider (Laya) was on the original plan and is
deliberately deferred. "Free" here means *no key and no cost*, not *on this
machine*.

The paid TypeSafe endpoint stays off unless **both** `TYPESAFE_API_KEY` and
`SYSTEM_ANALYZER_CLOUD_FALLBACK` are set.

## Not settled

The rating's weights and curves (CPU 30 / memory 25 / GPU 35 / storage 10, and
the shape of each curve) are a defensible starting point, not a measured truth.
They are configuration, so changing them is a one-file edit, but nobody has
checked them against how people actually rate machines.
