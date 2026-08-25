# statorscope

**Motor Current Signature Analysis that knows when to refuse.**

[![CI](https://github.com/ali-kin4/statorscope/actions/workflows/ci.yml/badge.svg)](https://github.com/ali-kin4/statorscope/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/statorscope/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Detect broken rotor bars, stator inter-turn faults and air-gap eccentricity from
three-phase motor current — and get told, up front, when your acquisition
**cannot support the answer**.

```python
from statorscope import Motor, Recording, diagnose

motor = Motor(pole_pairs=2, rotor_bars=28, line_hz=50.0)  # pole PAIRS, not poles
rec = Recording.from_text("capture.txt", time_column=0, time_unit="ms")

print(diagnose(rec, motor))
```

---

## Why this exists

Every MCSA tool will happily tell you a motor has a broken rotor bar. Most of them
will tell you that about a *healthy* motor too, and never mention it.

Broken-bar sidebands sit at `(1 ± 2ks)·f` — within a couple of Hz of a fundamental
that is 40–60 dB larger. Two things routinely counterfeit them:

| Counterfeit | What it does | What most tools do |
|---|---|---|
| **Sample-clock jitter** | Smears the fundamental into a phase-noise skirt, right where the sidebands live | Threshold on amplitude → fires every time |
| **Window leakage** | A Hann window has −31 dB sidelobes; the fault you want is at −45 dBc | Use the default window → fires every time |

This library was built after auditing a real 2018 MCSA rig whose detector lit every
fault lamp on every run. Two independent causes: an unsatisfiable comparison in the
code, *and* an acquisition chain that manufactured the evidence. The second one is
the interesting one, because no amount of signal processing fixes it — and nothing
in the tooling told the operator.

**statorscope makes that check a first-class, non-optional part of the pipeline.**

### See it for yourself

`examples/false_positive.py` runs one **healthy** motor through two acquisition
chains and compares a naive amplitude detector with statorscope:

```
A HEALTHY motor. No broken bars. Two acquisition chains.

--- hardware clock ---
  naive amplitude detector : clear (-86.0 dBc)
  statorscope              : clear
  clock verdict            : GOOD
  measured floor           : -86.9 dBc

--- millis() logging ---
  naive amplitude detector : BROKEN BAR (-12.8 dBc)     <-- false positive
  statorscope              : clear
  clock verdict            : UNRELIABLE
  measured floor           : -16.5 dBc
```

Same healthy motor. Change nothing but the clock, and the naive detector
diagnoses a broken rotor bar 73 dB louder than the truth. `statorscope demo
--healthy --jitter` shows the full report, which says why:

```
Clock audit: UNRELIABLE
  samples             : 32,240
  sample rate         : 602.83 Hz
  instantaneous range : 250-1000 Hz
  fast jitter (rms)   : 3513.4 us (211.8% of sample interval)
  slow drift (rms)    : 16.7 ms
  timestamp resolution: 1000 us
  MEASURED floor      : -16.5 dBc (offset 0.5-6.0 Hz)
  incipient fault at  : -50.0 dBc
  headroom            : -33.5 dB
  ! The measured floor (-16.5 dBc) sits ABOVE the level a real fault would
    produce (-50.0 dBc). Sidebands found here cannot be distinguished from
    acquisition noise. Fix the acquisition, not the maths.

VERDICT: UNSUPPORTED. The clock audit failed, so the findings above cannot be
distinguished from acquisition noise. Do not act on them.
```

---

## Install

```console
uv add statorscope             # or: pip install statorscope
uv add "statorscope[cli]"      # + the rich CLI
uv add "statorscope[all]"      # + plotting
```

Python 3.11+. Runtime dependencies are `numpy` and `scipy`, nothing else.

---

## What it does

### 1. Audits the acquisition before it analyses anything

It **measures** the noise floor in the band where sidebands live and compares it
to the level a real fault would produce:

```python
from statorscope import assess_clock

q = assess_clock(rec, line_hz=50.0)
print(q.verdict)  # TrustLevel.UNRELIABLE
print(q.measured_floor_dbc)  # -16.5
print(q.headroom_db)  # -33.5
print(q.explain())
```

Measured, not predicted — and that distinction was earned the hard way. The first
version of this gate *predicted* the floor from the standard aperture-jitter
relation. On real 2018 hardware it was wrong by 30 dB: the model assumes white
jitter, but a software-timed loop **drifts**, and correlated drift concentrates
near the carrier instead of raising the broadband floor. The model is still
shipped (`jitter_noise_floor_dbc`) for designing an acquisition chain up front,
but the verdict comes from the data.

Timing is still characterised when timestamps are present, and split into the two
components that need different fixes:

- **fast jitter** — lands in the sideband band, raises the floor → needs a better clock
- **slow drift** — shifts the frequency axis → needs `grid_lock`, not new hardware

It also flags when apparent jitter is really **timestamp quantisation** (a
1 ms-resolution `millis()` log makes clean sampling *look* jittery). Most tools
conflate all three; this one tells you which you have.

### 2. Calibrates your frequency axis against the grid — free

If your sample rate came from `millis()`, a USB packet counter, or anything other
than a real crystal, your frequency axis has an unknown scale error. Every fault
frequency you search is in the wrong place.

The grid is a calibration tone. Utility frequency is regulated to ~±0.05 Hz, so
any larger deviation in your *measured* fundamental is your own time base:

```python
from statorscope import grid_lock

lock = grid_lock(rec, nominal_hz=50.0)
print(lock.explain())
rec = lock.recording  # corrected fs, use this downstream
```

```
Grid-lock calibration
  measured fundamental : 50.3979 Hz (nominal 50.0 Hz)
  sample rate          : 620.000 -> 615.105 Hz
  time-base error      : +7,958 ppm (0.796% fast)
  ! Uncalibrated, every fault frequency would have been offset by 0.796%.
```

### 3. Measures how wide the carrier actually is

`suppress_fundamental` fits and removes a *single sinusoid*. A carrier smeared by
time-base drift is not one — so a large residue survives right beside it, and that
residue is locally prominent, which makes it look exactly like a low-slip
broken-bar sideband.

statorscope measures the carrier's occupied bandwidth and refuses to look inside it:

```python
q = assess_clock(rec)
q.carrier_halfwidth_hz    # 0.05 Hz on a clean capture, 3.9 Hz on a drifting one
```

A carrier smeared across ±3.9 Hz blinds every slip below 3.9%. Induction motors run
at 1–5% slip, so such a recording cannot see a normally loaded machine at all — and
the audit says so rather than reporting the residue as a fault.

This was found by running the library against the real 2018 recordings that
motivated it: it reported all three fault types on every file. The synthetic suite
had missed it because a clean synthetic carrier suppresses perfectly, leaving no
residue to trip over. `TestSmearedCarrier` covers it now.

### 4. Rejects skirts, keeps sidebands

Detection scores **prominence over a local median floor**, not absolute amplitude.
A discrete sideband stands above its neighbourhood. A phase-noise skirt has level
everywhere and prominence nowhere — so it scores low and is discarded.

The threshold is measured, not chosen by taste: white noise alone reaches ~9 dB
prominence over a few-bin search window, so anything below that flags healthy
machines. `tests/test_detect.py::TestThresholdCalibration` pins the measurement so
a future change has to re-justify itself.

The fundamental is removed by least-squares fit and subtraction in the *time*
domain, which eliminates its leakage entirely rather than attenuating it. The
default window is Blackman-Harris (−92 dB sidelobes).

### 5. Estimates slip without a tachometer

Slip is the crux of MCSA: the sidebands move with it, and a typed-in nameplate RPM
is usually wrong. statorscope searches the sideband geometry directly, then refines
from the interpolated peak separation:

```python
from statorscope import estimate_slip

est = estimate_slip(spectrum, motor)
print(est.slip, est.rpm, est.confident)  # 0.0301  1454 rpm  True
```

### 6. Ships the fault models, correctly

| Mechanism | Signature |
|---|---|
| Broken rotor bar | `(1 ± 2ks)·f` |
| Stator inter-turn | `f·[n(1−s)/p ± k]` |
| Air-gap eccentricity | `f ± k·f_r` |
| Rotor slot harmonics | `f·[(R ± n_d)(1−s)/p ± n_ws]` |
| Bearing races | `abs(f ± m·BPFO/BPFI)` |

`Motor` takes **`pole_pairs`**, spelled out, because poles-vs-pole-pairs is the
single most common unit error in MCSA code.

---

## CLI

```console
statorscope audit capture.txt --line-hz 50          # can I trust this data?
statorscope analyse capture.txt -p 2 -b 28          # full diagnosis
statorscope demo --fault -42                        # synthetic, known answer
statorscope demo --healthy --jitter                 # the false positive, live
```

`audit` and `analyse` exit non-zero when the data cannot support a claim, so they
drop straight into CI or a monitoring cron.

---

## Testing against known truth

The synthesiser generates three-phase current with faults injected at *specified*
levels, plus a jitter model, so the test suite asserts in both directions:

```python
from statorscope import Motor, synthesize, diagnose

motor = Motor(pole_pairs=2, rotor_bars=28)
rec, truth = synthesize(motor, slip=0.03, broken_bar_dbc=-42)

report = diagnose(rec, motor)
assert report.faults[0].detected
assert abs(report.slip.slip - 0.03) < 0.002
```

The headline regression test is the one that matters: **a healthy machine with a
bad clock must never come back `supported`.**

---

## Roadmap

- [ ] Loaders and benchmarks for the public labelled datasets (IEEE DataPort broken-rotor-bar database, Mendeley current-signature set)
- [ ] Startup/transient analysis — track the fault component through the run-up, which sidesteps slip estimation entirely
- [ ] Park's Vector / EPVA, using all three phases (needs simultaneous sampling)
- [ ] Reference firmware for a Pico/ESP32 + ADS131M04 front end (24-bit, simultaneous-sampling, ~$15)
- [ ] MUSIC/ESPRIT high-resolution estimators for short records

---

## Scope and honesty

- These are **screening** tools. A dBc level is not a work order.
- Severity bands are field heuristics, not standards; they are constants you can
  change, and they are documented as such in `detect.SEVERITY_BANDS`.
- The jitter floor assumes white jitter. Structured jitter concentrates into
  discrete spurs and can be worse at specific offsets. It is a floor, not a promise.
- Current-based bearing diagnosis is weak compared to vibration. It is included
  because it is cheap, not because it is equivalent.

---

## Contributing

```console
git clone https://github.com/ali-kin4/statorscope
cd statorscope
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy src
```

Issues and PRs welcome — particularly validation against real labelled data, and
fault models for machine types not covered here.

## License

Apache-2.0.
