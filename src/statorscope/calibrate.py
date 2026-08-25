"""Sample-rate calibration against the mains, and uniform resampling.

Any acquisition whose sample rate is derived from device timestamps
(``millis()``, ``micros()``, a USB packet counter) has an unknown scale error on
its frequency axis. That error moves every fault frequency you are looking for.

The fix is free and nobody uses it: **the grid is a calibration tone**. Utility
frequency is regulated to roughly +/-0.05 Hz of nominal, so any larger deviation
in your measured fundamental is your own time base being wrong. Measure it, and
divide it out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signals import FloatArray, Recording
from .spectrum import DEFAULT_WINDOW, estimate_fundamental

#: Utility frequency regulation band (Hz). Deviations beyond this are time-base error.
GRID_TOLERANCE_HZ = 0.05


@dataclass(frozen=True, slots=True)
class GridLock:
    """Result of calibrating a time base against the supply frequency."""

    recording: Recording
    """The corrected recording. Use this for all downstream analysis."""

    fs_before: float
    fs_after: float
    measured_hz: float
    """Fundamental as it appeared on the uncalibrated time base."""

    nominal_hz: float
    correction_ppm: float
    within_grid_tolerance: bool
    """True if the original time base was already good enough to skip this."""

    def explain(self) -> str:
        direction = "fast" if self.correction_ppm > 0 else "slow"
        lines = [
            "Grid-lock calibration",
            f"  measured fundamental : {self.measured_hz:.4f} Hz "
            f"(nominal {self.nominal_hz:.1f} Hz)",
            f"  sample rate          : {self.fs_before:.3f} -> {self.fs_after:.3f} Hz",
            f"  time-base error      : {self.correction_ppm:+,.0f} ppm "
            f"({abs(self.correction_ppm) / 1e4:.3f}% {direction})",
        ]
        if self.within_grid_tolerance:
            lines.append("  ! Already within grid tolerance - correction was not needed.")
        else:
            lines.append(
                f"  ! Uncalibrated, every fault frequency would have been offset by "
                f"{abs(self.correction_ppm) / 1e4:.3f}%."
            )
        return "\n".join(lines)


def grid_lock(
    recording: Recording,
    *,
    nominal_hz: float = 50.0,
    phase: int | str = 0,
    search_hz: float = 5.0,
    window: str = DEFAULT_WINDOW,
) -> GridLock:
    """Calibrate a recording's sample rate using the supply frequency as reference.

    Args:
        recording: Recording whose ``fs`` may be wrong.
        nominal_hz: True supply frequency, 50.0 or 60.0.
        phase: Which phase to measure the fundamental on.
        search_hz: How far from nominal to search. Widen only if the time base is
            badly wrong; too wide risks locking onto a harmonic of something else.
        window: Analysis window.

    Returns:
        A :class:`GridLock` whose ``.recording`` carries the corrected ``fs``.

    Example:
        >>> locked = grid_lock(rec, nominal_hz=50.0)      # doctest: +SKIP
        >>> rec = locked.recording                        # doctest: +SKIP
    """
    x = recording.phase(phase)
    measured = estimate_fundamental(
        x, recording.fs, nominal_hz=nominal_hz, tol_hz=search_hz, window=window
    )
    if measured <= 0:
        raise ValueError("could not locate a fundamental to calibrate against")

    fs_true = recording.fs * nominal_hz / measured
    ppm = (measured / nominal_hz - 1.0) * 1e6

    return GridLock(
        recording=recording.with_fs(fs_true),
        fs_before=recording.fs,
        fs_after=fs_true,
        measured_hz=measured,
        nominal_hz=nominal_hz,
        correction_ppm=ppm,
        within_grid_tolerance=abs(measured - nominal_hz) <= GRID_TOLERANCE_HZ,
    )


def resample_uniform(
    recording: Recording,
    *,
    fs: float | None = None,
) -> Recording:
    """Interpolate non-uniformly sampled data onto a uniform time grid.

    Args:
        recording: Must carry ``timestamps``.
        fs: Target rate. Defaults to the mean rate of the original.

    Returns:
        A new recording on a uniform grid, with ``timestamps`` dropped.

    Notes:
        This helps **only when the timestamps are more accurate than the sampling
        is uniform** -- that is, you know *when* each sample was taken even though
        the intervals vary. It does not help when the timestamps themselves are
        coarsely quantised, because then the true sample instants are unknown and
        the interpolation inherits the same error. Check
        :attr:`~statorscope.quality.ClockQuality.quantization_limited` first.
    """
    if recording.timestamps is None:
        raise ValueError("resample_uniform requires timestamps")

    ts = recording.timestamps
    span = float(ts[-1] - ts[0])
    if span <= 0:
        raise ValueError("timestamps are not increasing")

    target_fs = fs if fs is not None else len(recording) / span
    n_out = int(np.floor(span * target_fs)) + 1
    grid = ts[0] + np.arange(n_out, dtype=np.float64) / target_fs

    src = recording.current
    if src.ndim == 1:
        out: FloatArray = np.interp(grid, ts, src)
    else:
        out = np.column_stack([np.interp(grid, ts, src[:, i]) for i in range(src.shape[1])])

    return Recording(current=out, fs=target_fs, timestamps=None, name=recording.name)
