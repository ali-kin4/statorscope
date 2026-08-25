"""Synthetic motor current with known ground truth, including a clock-jitter model.

Every detector needs a signal where the answer is known before you look. This
module builds three-phase current with faults injected at *specified* levels, so
tests can assert both directions:

* a fault at ``-45 dBc`` **is** found, at the right slip;
* a healthy machine sampled with a bad clock is **not** reported as faulty.

The jitter model is the important half. It reproduces the failure mode of cheap
acquisition: samples taken at irregular instants, then logged with a coarse
timestamp (``millis()``), which smears the fundamental into a skirt sitting
exactly where broken-bar sidebands live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.signal import get_window

from .faults import broken_rotor_bar, eccentricity
from .signals import FloatArray, Motor, Recording
from .spectrum import DEFAULT_WINDOW

DEFAULT_HARMONICS: dict[int, float] = {3: -38.0, 5: -31.0, 7: -42.0}
"""Typical supply/machine harmonic content in dBc. Odd orders; 5th dominant is
normal for three-phase systems because triplens largely cancel."""


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """What was actually injected, for tests and benchmarks to assert against."""

    motor: Motor
    slip: float
    fs_true: float
    broken_bar_dbc: float | None
    eccentricity_dbc: float | None
    harmonics_dbc: dict[int, float]
    noise_floor_dbc: float
    jitter_rms_s: float
    jitter_model: str
    timestamp_resolution_s: float | None
    healthy: bool
    expected_frequencies: dict[str, FloatArray] = field(default_factory=dict)

    @property
    def rotor_rpm(self) -> float:
        return self.motor.synchronous_rpm * (1.0 - self.slip)


def synthesize(
    motor: Motor,
    *,
    slip: float = 0.03,
    fs: float = 5000.0,
    duration_s: float = 20.0,
    broken_bar_dbc: float | None = None,
    n_sidebands: int = 2,
    eccentricity_dbc: float | None = None,
    harmonics_dbc: dict[int, float] | None = None,
    noise_floor_dbc: float = -85.0,
    jitter_rms_s: float = 0.0,
    jitter_model: Literal["aperture", "interval"] = "aperture",
    timestamp_resolution_s: float | None = None,
    fs_error_ppm: float = 0.0,
    seed: int | None = 0,
    window: str = DEFAULT_WINDOW,
) -> tuple[Recording, GroundTruth]:
    """Generate a three-phase current recording with known faults.

    Args:
        motor: Machine to simulate.
        slip: Per-unit slip. Sets where the sidebands land.
        fs: Nominal sample rate in Hz.
        duration_s: Record length. Longer records lower the per-bin noise floor
            and are the cheapest way to reach a weak sideband.
        broken_bar_dbc: Level of the ``(1 +/- 2ks)f`` sidebands relative to the
            fundamental. ``None`` for a healthy rotor.
        n_sidebands: How many sideband orders ``k`` to inject.
        eccentricity_dbc: Level of ``f +/- k*f_r`` sidebands. ``None`` to omit.
        harmonics_dbc: Supply harmonic content. Defaults to
            :data:`DEFAULT_HARMONICS`.
        noise_floor_dbc: Target *per-bin* white-noise floor in dBc.
        jitter_rms_s: RMS timing error applied to the true sample instants. Set
            this to reproduce a cheap acquisition chain.
        jitter_model: ``"aperture"`` perturbs each sample around a fixed grid
            (bounded, hardware-clock-like). ``"interval"`` perturbs each interval
            so phase error accumulates as a random walk -- the software-timed loop
            case, and far more destructive.
        timestamp_resolution_s: If given, the *reported* timestamps are quantised
            to this step (e.g. ``1e-3`` for ``millis()``) while the true sample
            instants keep their jitter -- exactly the Arduino failure mode.
        fs_error_ppm: Scale error deliberately introduced into the reported ``fs``,
            for exercising :func:`statorscope.calibrate.grid_lock`.
        seed: RNG seed; ``None`` for nondeterministic.
        window: Window assumed when converting ``noise_floor_dbc`` into a
            time-domain noise amplitude.

    Returns:
        ``(recording, ground_truth)``.

    Example:
        >>> from statorscope import Motor, synthesize
        >>> rec, truth = synthesize(Motor(pole_pairs=2), slip=0.03, broken_bar_dbc=-45)
        >>> truth.healthy
        False
    """
    if duration_s <= 0 or fs <= 0:
        raise ValueError("fs and duration_s must be positive")
    if not 0.0 <= slip < 1.0:
        raise ValueError(f"slip must be in [0, 1), got {slip}")

    rng = np.random.default_rng(seed)
    n = round(fs * duration_s)
    harmonics = dict(DEFAULT_HARMONICS if harmonics_dbc is None else harmonics_dbc)

    # True sample instants.
    #
    # Two regimes, and they are not equivalent:
    #   "aperture"  - each sample deviates from a fixed grid by sigma. Bounded.
    #                 This is what a hardware-timed ADC with a noisy clock does.
    #   "interval"  - each *interval* is perturbed, so the phase error accumulates
    #                 as a random walk. This is what a software loop calling
    #                 delay()/millis() does, and it is far more destructive.
    t_true = np.arange(n, dtype=np.float64) / fs
    if jitter_rms_s > 0:
        if jitter_model == "aperture":
            t_true = t_true + rng.normal(0.0, jitter_rms_s, size=n)
        elif jitter_model == "interval":
            steps = 1.0 / fs + rng.normal(0.0, jitter_rms_s, size=n)
            t_true = np.cumsum(np.maximum(steps, 1e-9))
            t_true = t_true - t_true[0]
        else:
            raise ValueError(f"jitter_model must be 'aperture' or 'interval', got {jitter_model!r}")
        t_true = np.maximum.accumulate(t_true)  # keep monotonic

    f0 = motor.line_hz
    expected: dict[str, FloatArray] = {}

    def build_phase(phase_offset: float) -> FloatArray:
        sig = np.cos(2.0 * np.pi * f0 * t_true + phase_offset)
        for order, dbc in harmonics.items():
            sig += _amp(dbc) * np.cos(2.0 * np.pi * f0 * order * t_true + order * phase_offset)
        if broken_bar_dbc is not None:
            sig_brb = broken_rotor_bar(motor, slip, n_sidebands=n_sidebands)
            expected["broken_rotor_bar"] = sig_brb.frequencies
            for f_hz in sig_brb.frequencies:
                ph = rng.uniform(0, 2 * np.pi)
                sig += _amp(broken_bar_dbc) * np.cos(2.0 * np.pi * f_hz * t_true + ph)
        if eccentricity_dbc is not None:
            sig_ecc = eccentricity(motor, slip)
            expected["eccentricity"] = sig_ecc.frequencies
            for f_hz in sig_ecc.frequencies:
                ph = rng.uniform(0, 2 * np.pi)
                sig += _amp(eccentricity_dbc) * np.cos(2.0 * np.pi * f_hz * t_true + ph)
        return sig

    phases = np.column_stack([build_phase(off) for off in (0.0, -2 * np.pi / 3, 2 * np.pi / 3)])

    sigma = _noise_sigma_for_floor(noise_floor_dbc, n, window)
    if sigma > 0:
        phases = phases + rng.normal(0.0, sigma, size=phases.shape)

    # Reported time base: optionally quantised, optionally scale-shifted.
    if timestamp_resolution_s is not None and timestamp_resolution_s > 0:
        t_reported = np.round(t_true / timestamp_resolution_s) * timestamp_resolution_s
        t_reported = np.maximum.accumulate(t_reported)
    else:
        t_reported = t_true

    fs_reported = fs * (1.0 + fs_error_ppm / 1e6)

    recording = Recording(
        current=phases,
        fs=fs_reported,
        timestamps=t_reported,
        name="synthetic",
    )
    truth = GroundTruth(
        motor=motor,
        slip=slip,
        fs_true=fs,
        broken_bar_dbc=broken_bar_dbc,
        eccentricity_dbc=eccentricity_dbc,
        harmonics_dbc=harmonics,
        noise_floor_dbc=noise_floor_dbc,
        jitter_rms_s=jitter_rms_s,
        jitter_model=jitter_model,
        timestamp_resolution_s=timestamp_resolution_s,
        healthy=broken_bar_dbc is None and eccentricity_dbc is None,
        expected_frequencies=expected,
    )
    return recording, truth


def millis_jitter_recording(
    motor: Motor,
    *,
    slip: float = 0.03,
    fs: float = 620.0,
    duration_s: float = 52.0,
    broken_bar_dbc: float | None = None,
    seed: int | None = 0,
) -> tuple[Recording, GroundTruth]:
    """Reproduce a classic ``millis()``-logged Arduino acquisition.

    Convenience preset matching the failure mode this library was built around:
    a low sample rate derived from 1 ms-resolution timestamps, with sub-interval
    jitter. Used in the test suite to prove the detector refuses rather than
    hallucinating sidebands.
    """
    return synthesize(
        motor,
        slip=slip,
        fs=fs,
        duration_s=duration_s,
        broken_bar_dbc=broken_bar_dbc,
        jitter_rms_s=0.5e-3,
        jitter_model="interval",
        timestamp_resolution_s=1e-3,
        noise_floor_dbc=-85.0,
        seed=seed,
    )


def _amp(dbc: float) -> float:
    """Convert a dBc level into a linear amplitude relative to a unit fundamental."""
    return float(10.0 ** (dbc / 20.0))


def _noise_sigma_for_floor(floor_dbc: float, n: int, window: str) -> float:
    """Time-domain noise sigma that lands a given per-bin floor, for a unit carrier.

    The library reports amplitude as ``2|X[k]| / sum(w)``. For white noise of
    standard deviation ``sigma`` this has RMS ``2 * sigma * sqrt(sum(w^2)) / sum(w)``,
    which inverts directly.
    """
    if not np.isfinite(floor_dbc):
        return 0.0
    w = get_window(window, n, fftbins=True)
    return _amp(floor_dbc) * float(np.sum(w)) / (2.0 * float(np.sqrt(np.sum(w**2))))
