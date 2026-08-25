"""Clock auditing: decide whether a recording can support a fault claim at all.

Most MCSA tooling will happily report a broken rotor bar from a recording whose
acquisition cannot physically resolve one. Timing jitter spreads the (very large)
supply fundamental into a phase-noise skirt sitting exactly where the fault
sidebands live, at ``(1 +/- 2ks)f``. A detector that only checks amplitude reads
that skirt as a fault, every time, on a perfectly healthy machine.

**The gate is measured, not predicted.** The noise floor in the sideband
neighbourhood is read directly off the spectrum and compared with the level a real
fault would produce. Predicting it from a jitter model was tried first and was
wrong by 30 dB on real hardware: the standard aperture-jitter relation assumes
white jitter, while a software-timed acquisition drifts, and slow drift smears the
carrier narrowly instead of raising the broadband floor.

The jitter and drift figures are still computed, because they explain *why* a floor
is high and *which* fix applies -- but they inform the notes, not the verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .signals import Recording
from .spectrum import DEFAULT_WINDOW, compute_spectrum, suppress_fundamental

#: Sideband level (dBc, relative to the fundamental) of an *incipient* broken
#: rotor bar. Widely used field heuristic; faults weaker than this are generally
#: not claimed from current data alone.
INCIPIENT_FAULT_DBC = -50.0

#: Offsets from the carrier, in Hz, where broken-bar sidebands live for typical
#: slip. The noise floor is measured across this band.
SIDEBAND_OFFSET_BAND_HZ = (0.5, 6.0)

#: Timing-error components slower than this are counted as drift, not jitter.
DRIFT_CUTOFF_HZ = 1.0

#: Carrier smearing that blinds slip at or above this level makes a recording
#: unusable: induction motors run at roughly 1-5% slip, so a carrier that covers
#: 2% has swallowed the sidebands of a normally loaded machine.
BLIND_SLIP_UNRELIABLE = 0.02

#: Below this, smearing only hides very lightly loaded machines -- worth a warning
#: and a downgrade, not a refusal.
BLIND_SLIP_MARGINAL = 0.005

#: Headroom (dB) of fault level over the measured floor required for each verdict.
_GOOD_HEADROOM_DB = 10.0
_MARGINAL_HEADROOM_DB = 0.0


class TrustLevel(StrEnum):
    """How much weight a detection result from this recording deserves."""

    GOOD = "good"
    """The floor is low enough to resolve an incipient fault with margin."""

    MARGINAL = "marginal"
    """Only gross faults clear the floor. Treat as a screen, not a diagnosis."""

    UNRELIABLE = "unreliable"
    """The floor buries the fault signature. Any detection is unsupported."""

    UNKNOWN = "unknown"
    """The floor could not be measured (no usable carrier)."""


@dataclass(frozen=True, slots=True)
class ClockQuality:
    """Verdict on whether a recording supports fault detection, and why."""

    verdict: TrustLevel
    n_samples: int
    fs_mean: float
    fs_span: tuple[float, float]

    measured_floor_dbc: float
    """Median noise floor in the sideband band. This drives the verdict."""

    reference_fault_dbc: float
    headroom_db: float

    predicted_jitter_floor_dbc: float
    """What the aperture-jitter model expects. Diagnostic only."""

    carrier_halfwidth_hz: float
    """How far the carrier's own energy spreads. Fault frequencies inside this
    offset carry no information -- the carrier is sitting on them."""

    jitter_rms_s: float
    """Fast timing jitter, the component that lands in the sideband band."""

    drift_rms_s: float
    """Slow wander. Shifts the frequency axis; does not raise the floor."""

    relative_jitter: float
    timestamp_resolution_s: float
    quantization_limited: bool
    timestamps_available: bool
    notes: tuple[str, ...]

    @property
    def trustworthy(self) -> bool:
        """True when the recording supports at least a screening-grade claim."""
        return self.verdict in (TrustLevel.GOOD, TrustLevel.MARGINAL)

    def explain(self) -> str:
        """Human-readable audit, suitable for printing straight into a report."""
        lines = [
            f"Clock audit: {self.verdict.upper()}",
            f"  samples             : {self.n_samples:,}",
            f"  sample rate         : {self.fs_mean:.2f} Hz",
        ]
        if self.timestamps_available:
            lines.append(f"  instantaneous range : {self.fs_span[0]:.0f}-{self.fs_span[1]:.0f} Hz")
            lines.append(
                f"  fast jitter (rms)   : {self.jitter_rms_s * 1e6:.1f} us "
                f"({self.relative_jitter * 100:.1f}% of sample interval)"
            )
            lines.append(f"  slow drift (rms)    : {self.drift_rms_s * 1e3:.1f} ms")
            lines.append(f"  timestamp resolution: {self.timestamp_resolution_s * 1e6:.0f} us")
        lines += [
            f"  carrier halfwidth   : {self.carrier_halfwidth_hz:.3f} Hz "
            f"(blinds slip below {self.carrier_halfwidth_hz / 100.0:.4f})",
            f"  MEASURED floor      : {self.measured_floor_dbc:+.1f} dBc "
            f"(offset {SIDEBAND_OFFSET_BAND_HZ[0]}-{SIDEBAND_OFFSET_BAND_HZ[1]} Hz)",
            f"  incipient fault at  : {self.reference_fault_dbc:+.1f} dBc",
            f"  headroom            : {self.headroom_db:+.1f} dB",
        ]
        lines.extend(f"  ! {note}" for note in self.notes)
        return "\n".join(lines)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average via cumulative sums, with edge padding."""
    pad = window // 2
    padded = np.pad(x, (pad, window - 1 - pad), mode="edge")
    cumulative = np.cumsum(np.insert(padded, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / float(window)


@dataclass(frozen=True, slots=True)
class _Timing:
    """Timing characterisation extracted from a recording's timestamps."""

    fs_mean: float
    fs_span: tuple[float, float]
    jitter_rms: float
    drift_rms: float
    relative: float
    resolution: float
    quantization_limited: bool
    predicted_floor_dbc: float
    notes: tuple[str, ...]


def _analyse_timing(
    recording: Recording,
    *,
    line_hz: float,
    drift_cutoff_hz: float,
) -> _Timing:
    """Split the timing error into fast jitter and slow drift.

    They do different damage: fast jitter lands in the sideband band and raises the
    floor there, while slow drift shifts the whole frequency axis. Separating them
    is what makes the resulting advice actionable -- one needs a better clock, the
    other needs :func:`statorscope.calibrate.grid_lock`.
    """

    def blank(notes: tuple[str, ...] = ()) -> _Timing:
        return _Timing(
            fs_mean=recording.fs,
            fs_span=(recording.fs, recording.fs),
            jitter_rms=0.0,
            drift_rms=0.0,
            relative=0.0,
            resolution=0.0,
            quantization_limited=False,
            predicted_floor_dbc=-math.inf,
            notes=notes,
        )

    ts = recording.timestamps
    if ts is None:
        return blank()

    notes: list[str] = []
    dt = np.diff(ts)
    if np.any(dt <= 0):
        notes.append("Non-monotonic timestamps detected - samples may be reordered or duplicated.")
        dt = dt[dt > 0]
    if not dt.size:
        return blank(tuple(notes))

    dt_mean = float(np.mean(dt))
    fs_mean = 1.0 / dt_mean if dt_mean > 0 else recording.fs

    index = np.arange(len(ts), dtype=np.float64)
    slope, intercept = np.polyfit(index, ts, 1)
    error = ts - (slope * index + intercept)
    win = max(3, round(fs_mean / max(drift_cutoff_hz, 1e-6)))
    if win >= len(error):
        fast = error - float(np.mean(error))
        slow = np.zeros_like(error)
    else:
        slow = _moving_average(error, win)
        fast = error - slow

    jitter_rms = float(np.std(fast))
    resolution_source = np.unique(ts)
    resolution = float(np.min(np.diff(resolution_source))) if resolution_source.size > 1 else 0.0

    return _Timing(
        fs_mean=fs_mean,
        fs_span=(float(1.0 / dt.max()), float(1.0 / dt.min())),
        jitter_rms=jitter_rms,
        drift_rms=float(np.std(slow)),
        relative=jitter_rms / dt_mean if dt_mean > 0 else 0.0,
        resolution=resolution,
        quantization_limited=resolution > 0 and resolution > 0.1 * dt_mean,
        predicted_floor_dbc=jitter_noise_floor_dbc(jitter_rms, line_hz, len(recording)),
        notes=tuple(notes),
    )


def jitter_noise_floor_dbc(
    jitter_rms_s: float,
    frequency_hz: float,
    n_samples: int,
) -> float:
    """Per-bin phase-noise floor predicted by white sample-clock jitter, in dBc.

    Args:
        jitter_rms_s: RMS timing error per sample, in seconds.
        frequency_hz: Frequency of the dominant tone (the supply fundamental).
        n_samples: Record length, which sets how the noise spreads across bins.

    Returns:
        Noise floor relative to the carrier, in dBc (negative). ``-inf`` for a
        perfect clock.

    Notes:
        This is a **prediction**, and it assumes white jitter. Real acquisitions
        drift, and drift concentrates near the carrier rather than spreading
        uniformly, so this can be pessimistic by tens of dB. The verdict in
        :func:`assess_clock` uses the *measured* floor instead; this function is
        exposed for diagnosis and for designing an acquisition chain up front.
    """
    if jitter_rms_s <= 0:
        return -math.inf
    if frequency_hz <= 0 or n_samples < 2:
        raise ValueError("frequency_hz must be > 0 and n_samples >= 2")
    snr_db = -20.0 * math.log10(2.0 * math.pi * frequency_hz * jitter_rms_s)
    return -(snr_db + 10.0 * math.log10(n_samples / 2.0))


def measure_noise_floor_dbc(
    recording: Recording,
    *,
    line_hz: float = 50.0,
    phase: int | str = 0,
    offset_band_hz: tuple[float, float] = SIDEBAND_OFFSET_BAND_HZ,
    window: str = DEFAULT_WINDOW,
) -> float:
    """Median spectral level in the sideband neighbourhood, in dBc.

    This is the floor a real fault has to clear, read off the data rather than
    inferred from a model. The fundamental is fitted out first so its main lobe
    does not contaminate the measurement, and the band is taken on both sides of
    the carrier.

    Args:
        recording: Recording to measure.
        line_hz: Supply frequency.
        phase: Which phase to measure.
        offset_band_hz: ``(low, high)`` offsets from the carrier, in Hz.
        window: Analysis window.

    Returns:
        Median level in dBc, or ``-inf`` if no carrier could be located.
    """
    floor, _ = _floor_and_carrier(
        recording, line_hz=line_hz, phase=phase, offset_band_hz=offset_band_hz, window=window
    )
    return floor


def _floor_and_carrier(
    recording: Recording,
    *,
    line_hz: float,
    phase: int | str,
    offset_band_hz: tuple[float, float],
    window: str,
) -> tuple[float, float]:
    """Return ``(measured_floor_dbc, carrier_halfwidth_hz)``.

    The floor is measured *outside* the carrier's occupied bandwidth. Measuring
    through it reports the carrier's own skirt as though it were noise, which is
    both wrong and, worse, flattering: a badly smeared carrier produces a high
    median that still sits below the fault reference, so the recording passes.
    """
    x = recording.phase(phase)
    carrier = compute_spectrum(x, recording.fs, window=window, reference_hz=line_hz)
    if carrier.reference_amplitude <= 0:
        return -math.inf, 0.0

    resid = suppress_fundamental(x, recording.fs, line_hz)
    spec = compute_spectrum(resid, recording.fs, window=window, reference_hz=line_hz)
    spec = spec.rereferenced(carrier.reference_hz, carrier.reference_amplitude)

    halfwidth = spec.carrier_halfwidth_hz()
    f0 = carrier.reference_hz
    lo, hi = offset_band_hz
    lo = max(lo, halfwidth)
    if lo >= hi:
        # The carrier fills the entire sideband band; there is no clean region left.
        return math.inf, halfwidth

    offset = np.abs(spec.freq - f0)
    band = (offset >= lo) & (offset <= hi)
    if not np.any(band):
        return -math.inf, halfwidth
    return float(np.median(spec.dbc[band])), halfwidth


def assess_clock(
    recording: Recording,
    *,
    line_hz: float = 50.0,
    phase: int | str = 0,
    reference_fault_dbc: float = INCIPIENT_FAULT_DBC,
    offset_band_hz: tuple[float, float] = SIDEBAND_OFFSET_BAND_HZ,
    drift_cutoff_hz: float = DRIFT_CUTOFF_HZ,
    window: str = DEFAULT_WINDOW,
) -> ClockQuality:
    """Audit a recording and return a trust verdict.

    The verdict comes from the **measured** noise floor in the sideband band. If
    timestamps are present, timing jitter and drift are characterised too and
    reported as explanatory notes.

    Args:
        recording: Recording to audit.
        line_hz: Supply frequency -- the tone whose skirt does the damage.
        phase: Which phase to measure.
        reference_fault_dbc: Sideband level the detector must be able to see.
        offset_band_hz: Where to measure the floor, as offsets from the carrier.
        drift_cutoff_hz: Boundary between slow drift and fast jitter.
        window: Analysis window.

    Returns:
        A :class:`ClockQuality` describing whether detection is possible, and why.
    """
    n = len(recording)
    notes: list[str] = []

    floor, halfwidth = _floor_and_carrier(
        recording, line_hz=line_hz, phase=phase, offset_band_hz=offset_band_hz, window=window
    )
    headroom = reference_fault_dbc - floor

    have_ts = recording.timestamps is not None
    timing = _analyse_timing(recording, line_hz=line_hz, drift_cutoff_hz=drift_cutoff_hz)
    fs_mean = timing.fs_mean
    fs_span = timing.fs_span
    jitter_rms = timing.jitter_rms
    drift_rms = timing.drift_rms
    relative = timing.relative
    resolution = timing.resolution
    quantization_limited = timing.quantization_limited
    predicted = timing.predicted_floor_dbc
    notes.extend(timing.notes)

    if have_ts:
        if quantization_limited:
            notes.append(
                f"Timestamp resolution ({resolution * 1e6:.0f} us) is coarse relative to the "
                f"sample interval ({1e6 / fs_mean:.0f} us), so the timing figures are an "
                "UPPER bound - some of the apparent jitter is logging quantisation. Re-log "
                "with micros() or a hardware timer to separate the two."
            )
        if drift_rms > 5.0 * jitter_rms and drift_rms > 1e-3:
            notes.append(
                f"Slow time-base drift of {drift_rms * 1e3:.0f} ms rms dominates the timing "
                "error. It does not raise the noise floor, but it does shift the frequency "
                "axis - run statorscope.calibrate.grid_lock before trusting any frequency."
            )
        if math.isfinite(predicted) and predicted > floor + 15.0:
            notes.append(
                f"The white-jitter model predicts a {predicted:+.1f} dBc floor but the "
                f"measured floor is {floor:+.1f} dBc. The timing error is correlated "
                "(drift), not white, so it concentrates near the carrier rather than "
                "raising the broadband floor. The measured value is the one that counts."
            )
    else:
        notes.append(
            "No timestamps supplied, so timing jitter could not be characterised. The "
            "floor was still measured from the signal, so the verdict stands - but if a "
            "fault is missed, pass timestamps to find out whether the clock is why."
        )

    # A recording has to clear two independent hurdles, and the worse one wins:
    #   1. the floor must sit below the fault level (is there room to see it?)
    #   2. the carrier must not be sitting on the sidebands (is there anywhere to look?)
    # Reporting only the first is how a badly smeared recording reads as GOOD.
    if not math.isfinite(floor) and floor < 0:
        verdict = TrustLevel.UNKNOWN
        notes.append("No carrier could be located, so the noise floor is undefined.")
    elif headroom >= _GOOD_HEADROOM_DB:
        verdict = TrustLevel.GOOD
    elif headroom >= _MARGINAL_HEADROOM_DB:
        verdict = TrustLevel.MARGINAL
        notes.append(
            "Only gross faults clear the measured floor. Treat any detection as a screen, "
            "not a diagnosis. A longer record is the cheapest way to lower the floor."
        )
    else:
        verdict = TrustLevel.UNRELIABLE
        notes.append(
            f"The measured floor ({floor:+.1f} dBc) sits ABOVE the level a real fault would "
            f"produce ({reference_fault_dbc:+.1f} dBc). Sidebands found here cannot be "
            "distinguished from acquisition noise. Fix the acquisition, not the maths."
        )

    blind_slip = halfwidth / (2.0 * line_hz)
    if verdict is not TrustLevel.UNKNOWN and blind_slip >= BLIND_SLIP_UNRELIABLE:
        verdict = TrustLevel.UNRELIABLE
        notes.append(
            f"The carrier is smeared across +/-{halfwidth:.2f} Hz, blinding every slip below "
            f"{100 * blind_slip:.2f}%. Induction motors run at 1-5% slip, so this recording "
            "cannot see the sidebands of a normally loaded machine at all - whatever the "
            "noise floor says. This is a smeared time base: see grid_lock and "
            "resample_uniform, and fix the acquisition clock."
        )
    elif verdict is TrustLevel.GOOD and blind_slip >= BLIND_SLIP_MARGINAL:
        verdict = TrustLevel.MARGINAL
        notes.append(
            f"The carrier is smeared across +/-{halfwidth:.2f} Hz, so slip below "
            f"{100 * blind_slip:.2f}% cannot be resolved. Lightly loaded machines will read "
            "as healthy because their sidebands are underneath the carrier."
        )

    return ClockQuality(
        verdict=verdict,
        n_samples=n,
        fs_mean=fs_mean,
        fs_span=fs_span,
        measured_floor_dbc=floor,
        reference_fault_dbc=reference_fault_dbc,
        headroom_db=headroom,
        predicted_jitter_floor_dbc=predicted,
        carrier_halfwidth_hz=halfwidth,
        jitter_rms_s=jitter_rms,
        drift_rms_s=drift_rms,
        relative_jitter=relative,
        timestamp_resolution_s=resolution,
        quantization_limited=quantization_limited,
        timestamps_available=have_ts,
        notes=tuple(notes),
    )
