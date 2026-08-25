"""Fault detection, sensorless slip estimation, and the top-level diagnosis.

The detector's defining rule: **a candidate must be prominent, not merely loud.**

Clock jitter and window leakage both raise the level *around* the fundamental, so
a detector that thresholds on absolute amplitude fires on healthy machines. A real
sideband is a discrete line standing above its own local neighbourhood. Measuring
that -- prominence over a local median floor -- is what separates a fault from a
skirt, and it costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibrate import GridLock, grid_lock
from .faults import FaultSignature, broken_rotor_bar, eccentricity, stator_interturn
from .quality import ClockQuality, TrustLevel, assess_clock
from .signals import Motor, Recording
from .spectrum import DEFAULT_WINDOW, Spectrum, compute_spectrum, suppress_fundamental

#: A candidate must stand at least this far above its local floor to count.
#: White noise alone reaches ~9 dB prominence over a few-bin search window, so a
#: 6 dB threshold false-positives on healthy machines. Measured, not guessed --
#: see ``tests/test_detect.py::TestThresholdCalibration``.
MIN_PROMINENCE_DB = 12.0

#: The slip search evaluates hundreds of candidates, so it carries a much larger
#: multiple-comparison burden than a single lookup and needs a stiffer threshold.
SLIP_CONFIDENCE_DB = 15.0

#: Upper and lower sidebands must match within this many dB to be a credible pair.
MAX_ASYMMETRY_DB = 12.0

#: No component weaker than this is ever claimed as a fault, however prominent it
#: looks. Real broken-bar sidebands sit at -25 to -55 dBc; prominence is measured
#: against a *local* floor, so on near-noiseless data (a simulation, or a very
#: clean rig) numerical residue tens of dB below anything physical still scores
#: high prominence. This is the absolute sanity bound underneath the relative one.
MIN_FAULT_LEVEL_DBC = -70.0

#: Weight given to second-order (k=2) sideband confirmation in the slip search.
#: Without it, slip ``s`` and slip ``2s`` are indistinguishable: the k=1 sidebands
#: of ``2s`` land exactly on the k=2 sidebands of ``s``.
_SECOND_ORDER_WEIGHT = 0.5

#: Severity bands for broken-bar sideband level, in dBc. Field heuristics.
SEVERITY_BANDS: tuple[tuple[float, str], ...] = (
    (-35.0, "severe"),
    (-45.0, "moderate"),
    (-55.0, "incipient"),
)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One interrogated frequency and what was found there."""

    label: str
    frequency_hz: float
    level_dbc: float
    prominence_db: float
    passed: bool

    def __str__(self) -> str:
        mark = "HIT " if self.passed else "  . "
        return (
            f"{mark}{self.label:<22} {self.frequency_hz:8.3f} Hz  "
            f"{self.level_dbc:+7.1f} dBc  prom {self.prominence_db:+5.1f} dB"
        )


@dataclass(frozen=True, slots=True)
class SlipEstimate:
    """Sensorless slip, recovered from sideband geometry."""

    slip: float
    rpm: float
    score_db: float
    """Prominence of the weaker of the two sidebands. Low means low confidence."""

    method: str
    searched: tuple[float, float]

    @property
    def confident(self) -> bool:
        return self.score_db >= SLIP_CONFIDENCE_DB


@dataclass(frozen=True, slots=True)
class FaultResult:
    """Verdict for one fault mechanism."""

    kind: str
    detected: bool
    severity: str
    strongest_dbc: float
    evidence: tuple[Evidence, ...]
    notes: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"{self.kind}: {'DETECTED' if self.detected else 'not detected'}"
        if self.detected:
            head += f" ({self.severity}, strongest {self.strongest_dbc:+.1f} dBc)"
        body = "\n".join(f"    {e}" for e in self.evidence)
        return head + ("\n" + body if body else "")


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """Complete result: what was found, and whether the data supports believing it."""

    name: str
    clock: ClockQuality
    grid: GridLock | None
    slip: SlipEstimate
    faults: tuple[FaultResult, ...]
    spectrum: Spectrum

    @property
    def trust(self) -> TrustLevel:
        return self.clock.verdict

    @property
    def supported(self) -> bool:
        """True when at least one fault fired *and* the clock can back it up."""
        return any(f.detected for f in self.faults) and self.clock.trustworthy

    def summary(self) -> str:
        """Full human-readable report."""
        parts = [f"=== statorscope diagnosis: {self.name or '(unnamed)'} ===", ""]
        parts.append(self.clock.explain())
        parts.append("")
        if self.grid is not None:
            parts.append(self.grid.explain())
            parts.append("")
        conf = "confident" if self.slip.confident else "LOW CONFIDENCE"
        parts.append(
            f"Slip estimate: {self.slip.slip:.4f} ({self.slip.rpm:.0f} rpm) "
            f"via {self.slip.method} [{conf}, score {self.slip.score_db:+.1f} dB]"
        )
        parts.append("")
        for fault in self.faults:
            parts.append(str(fault))
            parts.append("")
        if self.clock.verdict is TrustLevel.UNRELIABLE:
            parts.append(
                "VERDICT: UNSUPPORTED. The clock audit failed, so the findings above "
                "cannot be distinguished from acquisition noise. Do not act on them."
            )
        elif not any(f.detected for f in self.faults):
            parts.append("VERDICT: no fault signature found above the local noise floor.")
        else:
            hits = ", ".join(f.kind for f in self.faults if f.detected)
            parts.append(f"VERDICT: {hits} ({self.clock.verdict} confidence in the data).")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.summary()


def residual_spectrum(
    recording: Recording,
    motor: Motor,
    *,
    phase: int | str = 0,
    n_harmonics: int = 1,
    window: str = DEFAULT_WINDOW,
) -> Spectrum:
    """Spectrum of the current with the supply components fitted out.

    The carrier is removed in the time domain so its leakage disappears entirely,
    but every level is still quoted in dBc **relative to that carrier** -- which
    means measuring it before suppression and re-referencing afterwards. Skipping
    that step silently normalises against the residual's own maximum and makes
    every reported dBc value meaningless.

    Args:
        recording: Recording to analyse.
        motor: Machine under test; supplies the line frequency.
        phase: Which phase to analyse.
        n_harmonics: How many supply harmonics to remove.
        window: Analysis window.

    Returns:
        A :class:`Spectrum` referenced to the true fundamental.
    """
    x = recording.phase(phase)
    carrier = compute_spectrum(x, recording.fs, window=window, reference_hz=motor.line_hz)
    resid = suppress_fundamental(x, recording.fs, motor.line_hz, n_harmonics=n_harmonics)
    spec = compute_spectrum(resid, recording.fs, window=window, reference_hz=motor.line_hz)
    return spec.rereferenced(carrier.reference_hz, carrier.reference_amplitude)


def estimate_slip(
    spectrum: Spectrum,
    motor: Motor,
    *,
    slip_range: tuple[float, float] = (0.002, 0.08),
    n_steps: int = 400,
    tol_hz: float = 0.15,
) -> SlipEstimate:
    """Recover slip from the broken-bar sideband pair, without a tachometer.

    Scans candidate slips and scores each by the **prominence** of the weaker of
    the two ``(1 +/- 2s)f`` sidebands. Scoring on prominence rather than level is
    what stops a monotonic phase-noise skirt from winning the search -- a skirt
    has level everywhere and prominence nowhere.

    Args:
        spectrum: Spectrum of the current, ideally with the fundamental suppressed.
        motor: Machine under test.
        slip_range: Slip bounds to search. Narrow this with nameplate data when
            available.
        n_steps: Search resolution.
        tol_hz: Half-width of the peak search at each candidate frequency.

    Returns:
        A :class:`SlipEstimate`. Check ``.confident`` before trusting it.
    """
    lo, hi = slip_range
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"invalid slip_range {slip_range}")

    f0 = motor.line_hz
    nyquist = float(spectrum.freq[-1])
    candidates = np.linspace(lo, hi, n_steps)
    best_slip, best_score, best_primary = lo, -np.inf, -np.inf

    # Anything inside the carrier's own occupied bandwidth is carrier residue, not
    # a sideband. suppress_fundamental removes a single sinusoid; a carrier smeared
    # by time-base drift is not one, so a large residue survives right beside it and
    # is locally prominent. Searching there finds that residue every time and
    # returns an arbitrarily small slip with a huge score.
    guard_hz = spectrum.carrier_halfwidth_hz()

    def pair_prominence(order: int, s: float) -> float:
        """Weakest of the two order-``k`` sidebands, or -inf if unusable."""
        offset = 2.0 * order * s * f0
        if offset <= guard_hz:
            return -np.inf
        f_low, f_high = (1.0 - 2.0 * order * s) * f0, (1.0 + 2.0 * order * s) * f0
        if f_low <= 0 or f_high >= nyquist:
            return -np.inf
        l_low = spectrum.level_dbc(f_low, tol_hz)
        l_high = spectrum.level_dbc(f_high, tol_hz)
        if abs(l_low - l_high) > MAX_ASYMMETRY_DB:
            return -np.inf
        return min(
            spectrum.prominence_db(f_low, tol_hz=tol_hz),
            spectrum.prominence_db(f_high, tol_hz=tol_hz),
        )

    for s in candidates:
        primary = pair_prominence(1, float(s))
        if not np.isfinite(primary):
            continue
        # Second-order confirmation breaks the s / 2s ambiguity: the k=1 sidebands
        # of a doubled slip coincide with the k=2 sidebands of the true slip, so
        # only the true slip explains both orders at once.
        second = pair_prominence(2, float(s))
        score = primary + _SECOND_ORDER_WEIGHT * max(0.0, second if np.isfinite(second) else 0.0)
        if score > best_score:
            best_score, best_primary, best_slip = score, primary, float(s)

    best_score = best_primary

    # Refine: the coarse grid only localises the sidebands to within tol_hz. The
    # interpolated peak positions give slip directly from their separation,
    # 2s*f0 either side of the carrier, which is an order of magnitude tighter.
    method = "sideband-pair prominence search"
    if not np.isfinite(best_score):
        # Every candidate sat inside the carrier. The recording cannot resolve any
        # slip in the searched range, which is a real answer, not an error.
        return SlipEstimate(
            slip=lo,
            rpm=motor.synchronous_rpm * (1.0 - lo),
            score_db=0.0,
            method=(
                f"no usable sideband: carrier occupies +/-{guard_hz:.2f} Hz, which "
                f"swallows every slip below {guard_hz / (2 * f0):.4f}"
            ),
            searched=slip_range,
        )
    if best_score >= SLIP_CONFIDENCE_DB:
        f_low_hat, _ = spectrum.peak_near((1.0 - 2.0 * best_slip) * f0, tol_hz)
        f_high_hat, _ = spectrum.peak_near((1.0 + 2.0 * best_slip) * f0, tol_hz)
        refined = (f_high_hat - f_low_hat) / (4.0 * f0)
        if lo <= refined <= hi:
            best_slip = float(refined)
            method += " + peak-separation refinement"

    return SlipEstimate(
        slip=best_slip,
        rpm=motor.synchronous_rpm * (1.0 - best_slip),
        score_db=float(best_score) if np.isfinite(best_score) else 0.0,
        method=method,
        searched=slip_range,
    )


def _severity(level_dbc: float) -> str:
    for threshold, name in SEVERITY_BANDS:
        if level_dbc >= threshold:
            return name
    return "trace"


def evaluate_signature(
    spectrum: Spectrum,
    signature: FaultSignature,
    *,
    min_prominence_db: float = MIN_PROMINENCE_DB,
    tol_hz: float = 0.15,
    floor_dbc: float = -np.inf,
    min_level_dbc: float = MIN_FAULT_LEVEL_DBC,
) -> FaultResult:
    """Interrogate every frequency in a signature and decide whether it fired.

    Args:
        spectrum: Spectrum to probe.
        signature: Frequencies to test.
        min_prominence_db: Required rise above the local floor.
        tol_hz: Peak search half-width.
        floor_dbc: Hard floor from the clock audit; candidates below this are
            rejected regardless of prominence.
        min_level_dbc: Absolute sanity bound. Nothing weaker is claimed as a fault
            no matter how prominent it is locally.

    Returns:
        A :class:`FaultResult` carrying per-frequency evidence.
    """
    nyquist = float(spectrum.freq[-1])
    usable = signature.within(0.0, nyquist)
    notes: list[str] = []
    if len(usable) < len(signature):
        notes.append(
            f"{len(signature) - len(usable)} of {len(signature)} signature frequencies "
            f"lie above Nyquist ({nyquist:.1f} Hz) and were not tested."
        )

    evidence: list[Evidence] = []
    for f_hz, label in zip(usable.frequencies, usable.labels, strict=True):
        level = spectrum.level_dbc(float(f_hz), tol_hz)
        prom = spectrum.prominence_db(float(f_hz), tol_hz=tol_hz)
        passed = bool(prom >= min_prominence_db and level > floor_dbc and level >= min_level_dbc)
        evidence.append(Evidence(label, float(f_hz), level, prom, passed))

    hits = [e for e in evidence if e.passed]
    strongest = max((e.level_dbc for e in hits), default=-np.inf)
    return FaultResult(
        kind=signature.kind,
        detected=bool(hits),
        severity=_severity(strongest) if hits else "healthy",
        strongest_dbc=strongest,
        evidence=tuple(evidence),
        notes=tuple(notes),
    )


def diagnose(
    recording: Recording,
    motor: Motor,
    *,
    slip: float | None = None,
    calibrate: bool = True,
    phase: int | str = 0,
    suppress_harmonics: int = 1,
    min_prominence_db: float = MIN_PROMINENCE_DB,
    window: str = DEFAULT_WINDOW,
) -> Diagnosis:
    """Run the full pipeline: audit, calibrate, estimate slip, detect, report.

    Args:
        recording: Current recording.
        motor: Machine under test.
        slip: Known slip. If ``None``, it is estimated from the spectrum.
        calibrate: Grid-lock the sample rate before analysis. Leave on unless the
            acquisition has a verified hardware clock.
        phase: Which phase to analyse.
        suppress_harmonics: How many supply harmonics to fit out before analysis.
        min_prominence_db: Detection threshold above the local floor.
        window: Analysis window.

    Returns:
        A :class:`Diagnosis`. Print it, or inspect ``.faults`` programmatically.

    Example:
        >>> from statorscope import Motor, diagnose, synthesize
        >>> rec, _ = synthesize(Motor(pole_pairs=2), slip=0.03, broken_bar_dbc=-42)
        >>> report = diagnose(rec, Motor(pole_pairs=2))
        >>> report.faults[0].kind
        'broken_rotor_bar'
    """
    clock = assess_clock(recording, line_hz=motor.line_hz, phase=phase)

    lock: GridLock | None = None
    work = recording
    if calibrate:
        lock = grid_lock(recording, nominal_hz=motor.line_hz, phase=phase, window=window)
        work = lock.recording

    spec = residual_spectrum(
        work, motor, phase=phase, n_harmonics=suppress_harmonics, window=window
    )

    slip_est = (
        SlipEstimate(slip, motor.synchronous_rpm * (1 - slip), np.inf, "supplied", (slip, slip))
        if slip is not None
        else estimate_slip(spec, motor)
    )
    s = slip_est.slip

    floor = clock.measured_floor_dbc if np.isfinite(clock.measured_floor_dbc) else -np.inf
    signatures = [
        broken_rotor_bar(motor, s),
        stator_interturn(motor, s),
        eccentricity(motor, s),
    ]
    results = [
        evaluate_signature(spec, sig, min_prominence_db=min_prominence_db, floor_dbc=floor)
        for sig in signatures
    ]

    # Every signature is evaluated *at the estimated slip*. If the slip search
    # never found a credible sideband pair, that slip is a noise artefact and
    # nothing evaluated at it means anything -- so say so rather than reporting
    # whatever happened to land near a spurious frequency.
    if not slip_est.confident:
        note = (
            f"Slip could not be established (best sideband-pair prominence "
            f"{slip_est.score_db:+.1f} dB, need {SLIP_CONFIDENCE_DB:+.1f} dB). "
            "Signature frequencies are therefore unknown and no detection is claimed."
        )
        results = [
            FaultResult(
                kind=r.kind,
                detected=False,
                severity="healthy",
                strongest_dbc=-np.inf,
                # Clear the per-frequency hits too: they were evaluated at a slip
                # that turned out to be meaningless, so showing them as passing
                # would contradict the verdict directly above them.
                evidence=tuple(
                    Evidence(e.label, e.frequency_hz, e.level_dbc, e.prominence_db, passed=False)
                    for e in r.evidence
                ),
                notes=(*r.notes, note),
            )
            for r in results
        ]
    faults = tuple(results)

    return Diagnosis(
        name=recording.name,
        clock=clock,
        grid=lock,
        slip=slip_est,
        faults=faults,
        spectrum=spec,
    )
