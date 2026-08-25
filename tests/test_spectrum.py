"""Spectral estimation and calibration tests."""

from __future__ import annotations

import numpy as np
import pytest

from statorscope import (
    Motor,
    Recording,
    compute_spectrum,
    estimate_fundamental,
    grid_lock,
    resample_uniform,
    suppress_fundamental,
    synthesize,
)

MOTOR = Motor(pole_pairs=2, line_hz=50.0)


def _tone(f_hz: float, fs: float, n: int, amp: float = 1.0, phase: float = 0.0):
    t = np.arange(n) / fs
    return amp * np.cos(2 * np.pi * f_hz * t + phase)


class TestSpectrum:
    def test_locates_a_pure_tone(self):
        x = _tone(50.0, 1000.0, 10_000)
        spec = compute_spectrum(x, 1000.0, reference_hz=50.0)
        assert spec.reference_hz == pytest.approx(50.0, abs=0.01)

    def test_amplitude_is_window_corrected(self):
        x = _tone(50.0, 1000.0, 10_000, amp=3.0)
        spec = compute_spectrum(x, 1000.0, reference_hz=50.0)
        assert spec.reference_amplitude == pytest.approx(3.0, rel=0.02)

    def test_dbc_is_relative_to_the_carrier(self):
        fs, n = 2000.0, 20_000
        x = _tone(50.0, fs, n) + _tone(53.0, fs, n, amp=10 ** (-40 / 20))
        spec = compute_spectrum(x, fs, reference_hz=50.0)
        assert spec.level_dbc(53.0) == pytest.approx(-40.0, abs=0.5)

    def test_sub_bin_frequency_interpolation(self):
        """A tone deliberately placed between bins is still located accurately."""
        fs, n = 1000.0, 8192
        offset = 50.0 + 0.5 * (fs / n)
        x = _tone(offset, fs, n)
        assert estimate_fundamental(x, fs, nominal_hz=50.0) == pytest.approx(offset, abs=0.005)

    def test_rejects_too_short_input(self):
        with pytest.raises(ValueError, match="at least 8"):
            compute_spectrum(np.zeros(4), 1000.0)


class TestProminence:
    def test_discrete_tone_is_prominent(self):
        fs, n = 2000.0, 20_000
        x = _tone(50.0, fs, n) + _tone(53.0, fs, n, amp=10 ** (-40 / 20))
        spec = compute_spectrum(x, fs, reference_hz=50.0)
        assert spec.prominence_db(53.0) > 20.0

    def test_broadband_noise_has_no_prominence(self):
        """The discriminator that stops a phase-noise skirt reading as a fault."""
        rng = np.random.default_rng(0)
        fs, n = 2000.0, 20_000
        x = _tone(50.0, fs, n) + rng.normal(0, 0.01, n)
        spec = compute_spectrum(x, fs, reference_hz=50.0)
        assert spec.prominence_db(53.0) < 15.0

    def test_local_floor_excludes_the_component_itself(self):
        fs, n = 2000.0, 20_000
        x = _tone(50.0, fs, n) + _tone(53.0, fs, n, amp=10 ** (-20 / 20))
        spec = compute_spectrum(x, fs, reference_hz=50.0)
        floor = spec.local_floor_dbc(53.0, span_hz=4.0, exclude_hz=0.5)
        assert floor < spec.level_dbc(53.0) - 20.0


class TestSuppressFundamental:
    def test_removes_the_carrier(self):
        fs, n = 1000.0, 10_000
        x = _tone(50.0, fs, n)
        resid = suppress_fundamental(x, fs, 50.0)
        assert np.std(resid) < 1e-9 * np.std(x) + 1e-9

    def test_leaves_sidebands_intact(self):
        fs, n = 2000.0, 40_000
        side = 10 ** (-40 / 20)
        x = _tone(50.0, fs, n) + _tone(47.0, fs, n, amp=side)
        resid = suppress_fundamental(x, fs, 50.0)
        spec = compute_spectrum(resid, fs, reference_hz=50.0)
        # dBc reference is now the residual's own max, so compare linear amplitude.
        idx = int(np.argmin(np.abs(spec.freq - 47.0)))
        assert spec.amplitude[idx] == pytest.approx(side, rel=0.1)

    def test_removes_harmonics_when_asked(self):
        fs, n = 2000.0, 20_000
        x = _tone(50.0, fs, n) + _tone(150.0, fs, n, amp=0.1)
        resid = suppress_fundamental(x, fs, 50.0, n_harmonics=3)
        assert np.std(resid) < 1e-6


class TestGridLock:
    def test_recovers_a_known_sample_rate_error(self):
        rec, truth = synthesize(
            MOTOR, fs=2000.0, duration_s=10, fs_error_ppm=13_500.0, jitter_rms_s=0.0
        )
        lock = grid_lock(rec, nominal_hz=50.0)
        assert lock.fs_after == pytest.approx(truth.fs_true, rel=1e-3)
        assert lock.correction_ppm == pytest.approx(13_500.0, rel=0.05)

    def test_clean_recording_is_reported_as_already_locked(self):
        rec, _ = synthesize(MOTOR, fs=2000.0, duration_s=10, fs_error_ppm=0.0, jitter_rms_s=0.0)
        lock = grid_lock(rec, nominal_hz=50.0)
        assert lock.within_grid_tolerance

    def test_explain_mentions_ppm(self):
        rec, _ = synthesize(MOTOR, fs=2000.0, duration_s=5, fs_error_ppm=5000.0)
        assert "ppm" in grid_lock(rec, nominal_hz=50.0).explain()


class TestResampleUniform:
    def test_produces_a_uniform_grid(self):
        rec, _ = synthesize(MOTOR, fs=1000.0, duration_s=5, jitter_rms_s=2e-4)
        out = resample_uniform(rec)
        assert out.timestamps is None
        assert out.n_phases == rec.n_phases

    def test_requires_timestamps(self):
        rec = Recording.from_array(np.zeros((100, 3)), fs=100.0)
        with pytest.raises(ValueError, match="timestamps"):
            resample_uniform(rec)
