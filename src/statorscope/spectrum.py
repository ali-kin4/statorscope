"""Spectral estimation tuned for MCSA.

MCSA is a dynamic-range problem, not a resolution problem. The fault sidebands sit
0.5-5 Hz from a fundamental that is 40-60 dB larger, so the two things that matter
are (a) a window whose sidelobes fall below the fault level and (b) removing the
fundamental outright so its skirt stops masking its own neighbourhood.

The default window is Blackman-Harris (-92 dB sidelobes). A Hann window, the usual
default elsewhere, has -31 dB sidelobes and will manufacture "sidebands" at exactly
the offsets a broken-bar detector searches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window

from .signals import FloatArray

#: Default analysis window. Sidelobes must sit below the fault level being claimed.
DEFAULT_WINDOW = "blackmanharris"


@dataclass(frozen=True, slots=True)
class Spectrum:
    """A one-sided amplitude spectrum with a carrier reference for dBc maths."""

    freq: FloatArray
    """Frequency axis in Hz."""

    amplitude: FloatArray
    """Linear amplitude, window-corrected so a pure tone reads its true peak value."""

    fs: float
    n_samples: int
    reference_hz: float
    """Carrier frequency that ``dbc`` is measured against (the fundamental)."""

    reference_amplitude: float

    @property
    def dbc(self) -> FloatArray:
        """Amplitude relative to the carrier, in dB."""
        ref = self.reference_amplitude if self.reference_amplitude > 0 else 1.0
        return 20.0 * np.log10(self.amplitude / ref + 1e-300)

    @property
    def resolution_hz(self) -> float:
        """Raw bin spacing. The window widens the effective main lobe ~4x."""
        return float(self.freq[1] - self.freq[0])

    def rereferenced(self, hz: float, amplitude: float) -> Spectrum:
        """Return a copy whose dBc is measured against a different carrier.

        Needed after :func:`suppress_fundamental`: the residual no longer contains
        the carrier, but every level must still be quoted relative to it.
        """
        return Spectrum(
            freq=self.freq,
            amplitude=self.amplitude,
            fs=self.fs,
            n_samples=self.n_samples,
            reference_hz=hz,
            reference_amplitude=amplitude,
        )

    def _slice(self, hz: float, tol_hz: float) -> slice:
        lo = int(np.searchsorted(self.freq, hz - tol_hz))
        hi = int(np.searchsorted(self.freq, hz + tol_hz)) + 1
        return slice(max(0, lo), min(len(self.freq), max(hi, lo + 1)))

    def peak_near(self, hz: float, tol_hz: float = 0.25) -> tuple[float, float]:
        """Locate the strongest component within ``+/- tol_hz`` of ``hz``.

        Returns:
            ``(frequency_hz, level_dbc)`` with sub-bin frequency interpolation.
        """
        sl = self._slice(hz, tol_hz)
        seg = self.dbc[sl]
        if seg.size == 0:
            return hz, -np.inf
        k_local = int(np.argmax(seg))
        k = sl.start + k_local
        f_hat, y_hat = _parabolic_refine(self.freq, self.dbc, k)
        return f_hat, y_hat

    def level_dbc(self, hz: float, tol_hz: float = 0.25) -> float:
        """Peak level within ``+/- tol_hz`` of ``hz``, in dBc."""
        return self.peak_near(hz, tol_hz)[1]

    def local_floor_dbc(
        self,
        hz: float,
        *,
        span_hz: float = 4.0,
        exclude_hz: float = 0.5,
    ) -> float:
        """Median noise floor in a neighbourhood, excluding the component itself.

        This is the reference a candidate must stand *above*. Using an absolute
        threshold instead is what lets a phase-noise skirt read as a fault.
        """
        sl = self._slice(hz, span_hz)
        f_seg = self.freq[sl]
        d_seg = self.dbc[sl]
        keep = np.abs(f_seg - hz) > exclude_hz
        if not np.any(keep):
            return float(np.median(d_seg)) if d_seg.size else -np.inf
        return float(np.median(d_seg[keep]))

    def carrier_halfwidth_hz(
        self,
        *,
        far_field_hz: tuple[float, float] = (8.0, 20.0),
        margin_db: float = 6.0,
        max_offset_hz: float = 8.0,
    ) -> float:
        """How far the carrier's own energy extends either side of its peak.

        A pure tone occupies one window main lobe. A carrier smeared by time-base
        drift occupies far more, and :func:`suppress_fundamental` -- which fits and
        removes a *single* sinusoid -- cannot remove the smeared part. What is left
        is a large residue immediately beside the carrier that looks exactly like a
        low-slip fault sideband to any detector that goes looking there.

        This measures the damage directly: walk outward until the carrier's skirt
        drops to within ``margin_db`` of the far-field floor on both sides and
        stays there.

        Args:
            far_field_hz: Offset band used as the clean reference floor.
            margin_db: How close to that floor counts as "the carrier has ended".
            max_offset_hz: Give up beyond this offset.

        Returns:
            Half-width in Hz. Roughly one bin for a clean tone; much larger when
            the time base drifted. Frequencies within this of the carrier carry no
            usable fault information.
        """
        f0 = self.reference_hz
        offset = np.abs(self.freq - f0)
        dbc = self.dbc
        far = (offset >= far_field_hz[0]) & (offset <= far_field_hz[1])
        if not np.any(far):
            return float(self.resolution_hz)
        threshold = float(np.median(dbc[far])) + margin_db

        # Compare the *median* level in a ring at each offset, not the peak. Peak
        # comparison is hopeless against noise: the maximum of a handful of Rayleigh
        # bins clears any median by well over 6 dB, so a clean spectrum would look
        # like an endlessly wide carrier. A median tracks the skirt and ignores the
        # spikes.
        step = max(self.resolution_hz, max_offset_hz / 200.0)
        ring = max(step, 3.0 * self.resolution_hz)
        below = 0
        needed = 3  # sustained, so one lucky dip inside the skirt does not end it
        probe = step
        while probe <= max_offset_hz:
            in_ring = np.abs(offset - probe) <= ring
            if np.any(in_ring) and float(np.median(dbc[in_ring])) <= threshold:
                below += 1
                if below >= needed:
                    return float(max(probe - (needed - 1) * step, self.resolution_hz))
            else:
                below = 0
            probe += step
        return float(max_offset_hz)

    def prominence_db(
        self,
        hz: float,
        *,
        tol_hz: float = 0.25,
        span_hz: float = 4.0,
    ) -> float:
        """How far a component rises above its own local floor, in dB.

        A discrete sideband has high prominence. A jitter skirt has a level but
        almost no prominence, because its neighbourhood is elevated too. This is
        the single most useful discriminator in the library.
        """
        level = self.level_dbc(hz, tol_hz)
        floor = self.local_floor_dbc(hz, span_hz=span_hz, exclude_hz=tol_hz * 2)
        if not np.isfinite(level) or not np.isfinite(floor):
            return 0.0
        return float(level - floor)


def _parabolic_refine(freq: FloatArray, values_db: FloatArray, k: int) -> tuple[float, float]:
    """Refine a discrete peak to sub-bin accuracy by fitting a parabola in dB."""
    if k <= 0 or k >= len(values_db) - 1:
        return float(freq[k]), float(values_db[k])
    y0, y1, y2 = values_db[k - 1], values_db[k], values_db[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freq[k]), float(y1)
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    df = float(freq[1] - freq[0])
    return float(freq[k] + delta * df), float(y1 - 0.25 * (y0 - y2) * delta)


def compute_spectrum(
    x: FloatArray,
    fs: float,
    *,
    window: str = DEFAULT_WINDOW,
    reference_hz: float | None = None,
    reference_tol_hz: float = 5.0,
) -> Spectrum:
    """Compute a window-corrected one-sided amplitude spectrum.

    Args:
        x: Real signal, mean removed internally.
        fs: Sample rate in Hz.
        window: Any :func:`scipy.signal.get_window` name. Leave at the default
            unless you know why you are changing it.
        reference_hz: Nominal carrier for dBc normalisation, typically the supply
            frequency. The true peak is located within ``reference_tol_hz`` of it.
            If ``None``, the global maximum is used.
        reference_tol_hz: Search half-width for the carrier.

    Returns:
        A :class:`Spectrum`.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.mean(x))
    n = x.size
    if n < 8:
        raise ValueError(f"need at least 8 samples, got {n}")

    w = get_window(window, n, fftbins=True)
    spec = np.fft.rfft(x * w)
    freq = np.fft.rfftfreq(n, 1.0 / fs)
    amp = 2.0 * np.abs(spec) / float(np.sum(w))

    if reference_hz is None:
        ref_amp = float(np.max(amp))
        ref_hz = float(freq[int(np.argmax(amp))])
    else:
        lo = int(np.searchsorted(freq, reference_hz - reference_tol_hz))
        hi = int(np.searchsorted(freq, reference_hz + reference_tol_hz)) + 1
        lo, hi = max(0, lo), min(len(freq), max(hi, lo + 1))
        k = lo + int(np.argmax(amp[lo:hi]))
        ref_amp = float(amp[k])
        db = 20.0 * np.log10(amp / (ref_amp or 1.0) + 1e-300)
        ref_hz, _ = _parabolic_refine(freq, db, k)

    return Spectrum(
        freq=freq,
        amplitude=amp,
        fs=fs,
        n_samples=n,
        reference_hz=ref_hz,
        reference_amplitude=ref_amp,
    )


def estimate_fundamental(
    x: FloatArray,
    fs: float,
    *,
    nominal_hz: float = 50.0,
    tol_hz: float = 5.0,
    window: str = DEFAULT_WINDOW,
) -> float:
    """Estimate the supply fundamental to sub-bin accuracy.

    Args:
        x: Real signal.
        fs: Sample rate in Hz.
        nominal_hz: Where to look (50 or 60).
        tol_hz: Search half-width.
        window: Analysis window.

    Returns:
        Fundamental frequency in Hz, as measured on the given time base.
    """
    return compute_spectrum(
        x, fs, window=window, reference_hz=nominal_hz, reference_tol_hz=tol_hz
    ).reference_hz


def suppress_fundamental(
    x: FloatArray,
    fs: float,
    f0: float,
    *,
    n_harmonics: int = 1,
) -> FloatArray:
    """Least-squares fit and subtract the fundamental (and optional harmonics).

    Removing the carrier in the *time* domain eliminates its spectral leakage
    entirely, rather than attenuating it. Fault sidebands are untouched because
    they are at different frequencies and are orthogonal to the fitted basis over
    a long record.

    Args:
        x: Real signal.
        fs: Sample rate in Hz.
        f0: Fundamental frequency, ideally from :func:`estimate_fundamental`.
        n_harmonics: How many harmonics to remove (1 = fundamental only).

    Returns:
        The residual signal with the supply components removed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    t = np.arange(n, dtype=np.float64) / fs
    cols = []
    for h in range(1, max(1, n_harmonics) + 1):
        cols.append(np.cos(2.0 * np.pi * f0 * h * t))
        cols.append(np.sin(2.0 * np.pi * f0 * h * t))
    basis = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(basis, x, rcond=None)
    return np.asarray(x - basis @ coeffs, dtype=np.float64)
