"""Core data containers: the motor under test and the current recording."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

PHASE_NAMES = ("R", "S", "T")


@dataclass(frozen=True, slots=True)
class Motor:
    """Nameplate data for the machine under test.

    Args:
        pole_pairs: Pole *pairs*, not poles. A 4-pole motor has ``pole_pairs=2``.
            This is the single most common unit error in MCSA code -- the field is
            named explicitly so it cannot be confused with pole count.
        rotor_bars: Number of rotor bars/slots. Required for eccentricity and
            rotor-slot-harmonic models; optional otherwise.
        line_hz: Supply frequency, 50.0 or 60.0.
        rated_rpm: Nameplate full-load speed, used only to bound the slip search.
    """

    pole_pairs: int
    rotor_bars: int | None = None
    line_hz: float = 50.0
    rated_rpm: float | None = None

    def __post_init__(self) -> None:
        if self.pole_pairs < 1:
            raise ValueError(f"pole_pairs must be >= 1, got {self.pole_pairs}")
        if self.rotor_bars is not None and self.rotor_bars < 1:
            raise ValueError(f"rotor_bars must be >= 1, got {self.rotor_bars}")
        if self.line_hz <= 0:
            raise ValueError(f"line_hz must be > 0, got {self.line_hz}")

    @property
    def synchronous_rpm(self) -> float:
        """Synchronous speed in RPM: ``60 * f / pole_pairs``."""
        return 60.0 * self.line_hz / self.pole_pairs

    @property
    def rated_slip(self) -> float | None:
        """Slip at nameplate speed, if ``rated_rpm`` was given."""
        if self.rated_rpm is None:
            return None
        return (self.synchronous_rpm - self.rated_rpm) / self.synchronous_rpm

    def rotor_hz(self, slip: float) -> float:
        """Mechanical rotation frequency in Hz at a given slip."""
        return self.line_hz * (1.0 - slip) / self.pole_pairs


@dataclass(frozen=True, slots=True)
class Recording:
    """A three-phase (or single-phase) current recording.

    Args:
        current: Shape ``(n_samples,)`` or ``(n_samples, n_phases)``. Units are
            arbitrary -- every result in this library is a *ratio* (dBc), so an
            uncalibrated ADC scale is fine.
        fs: Sample rate in Hz. If it was derived from device timestamps rather
            than a real clock, pass ``timestamps`` too so the clock can be audited.
        timestamps: Optional per-sample time in **seconds**. Supplying this is what
            enables :meth:`clock_quality` to detect the jitter that silently
            fabricates fault sidebands.
        name: Free-form label used in reports.
    """

    current: FloatArray
    fs: float
    timestamps: FloatArray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.current.ndim not in (1, 2):
            raise ValueError(f"current must be 1-D or 2-D, got {self.current.ndim}-D")
        if self.fs <= 0:
            raise ValueError(f"fs must be > 0, got {self.fs}")
        if self.timestamps is not None and len(self.timestamps) != len(self.current):
            raise ValueError(
                f"timestamps length {len(self.timestamps)} != samples {len(self.current)}"
            )

    def __len__(self) -> int:
        return int(self.current.shape[0])

    @property
    def n_phases(self) -> int:
        return 1 if self.current.ndim == 1 else int(self.current.shape[1])

    @property
    def duration_s(self) -> float:
        return len(self) / self.fs

    @property
    def freq_resolution_hz(self) -> float:
        """Raw FFT bin spacing. Windowing widens the effective resolution ~4x."""
        return self.fs / len(self)

    def phase(self, index: int | str = 0) -> FloatArray:
        """Return one phase as a mean-removed 1-D array.

        Args:
            index: Column index, or one of ``"R"``, ``"S"``, ``"T"``.
        """
        if isinstance(index, str):
            key = index.upper()
            if key not in PHASE_NAMES:
                raise KeyError(f"phase must be one of {PHASE_NAMES} or an int, got {index!r}")
            index = PHASE_NAMES.index(key)
        col = self.current if self.current.ndim == 1 else self.current[:, index]
        return np.asarray(col, dtype=np.float64) - float(np.mean(col))

    def with_fs(self, fs: float) -> Recording:
        """Return a copy with a corrected sample rate (see :mod:`statorscope.calibrate`)."""
        return replace(self, fs=fs)

    # ---------------------------------------------------------------- loaders

    @classmethod
    def from_array(
        cls,
        current: npt.ArrayLike,
        fs: float,
        *,
        timestamps: npt.ArrayLike | None = None,
        name: str = "",
    ) -> Recording:
        """Build a recording from any array-like."""
        ts = None if timestamps is None else np.asarray(timestamps, dtype=np.float64)
        return cls(np.asarray(current, dtype=np.float64), fs, ts, name)

    @classmethod
    def from_text(
        cls,
        path: str | Path,
        *,
        time_column: int | None = 0,
        current_columns: tuple[int, ...] | None = None,
        time_unit: Literal["s", "ms", "us"] = "ms",
        fs: float | None = None,
        name: str | None = None,
        **loadtxt_kwargs: Any,
    ) -> Recording:
        """Load a whitespace/CSV-delimited log of ``[time, phase...]`` columns.

        If ``time_column`` is given, ``fs`` is derived from the elapsed time and the
        timestamps are retained so the clock can be audited. This is the common shape
        for Arduino/ESP32 serial logs.

        Args:
            path: File to read.
            time_column: Index of the timestamp column, or ``None`` if there isn't one.
            current_columns: Indices of the phase columns. Defaults to every column
                except ``time_column``.
            time_unit: Units of the timestamp column.
            fs: Sample rate, required only when ``time_column is None``.
            name: Report label; defaults to the file stem.
        """
        p = Path(path)
        data = np.loadtxt(p, dtype=np.float64, **loadtxt_kwargs)
        if data.ndim == 1:
            data = data[:, None]

        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6}[time_unit]
        if current_columns is None:
            keep = [i for i in range(data.shape[1]) if i != time_column]
            current_columns = tuple(keep)
        current = data[:, list(current_columns)]

        if time_column is None:
            if fs is None:
                raise ValueError("fs is required when time_column is None")
            return cls(current, fs, None, name or p.stem)

        ts = data[:, time_column] * scale
        span = float(ts[-1] - ts[0])
        if span <= 0:
            raise ValueError(f"{p.name}: timestamps are not increasing (span={span})")
        derived = len(ts) / span
        return cls(current, fs if fs is not None else derived, ts - ts[0], name or p.stem)
