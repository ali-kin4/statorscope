"""Audit tests: the measured floor drives the verdict, the jitter model explains it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statorscope import (
    Motor,
    Recording,
    TrustLevel,
    assess_clock,
    jitter_noise_floor_dbc,
    measure_noise_floor_dbc,
    synthesize,
)

MOTOR = Motor(pole_pairs=2, line_hz=50.0)


class TestJitterFloorModel:
    """The predictive model. Diagnostic only -- it does not drive the verdict."""

    def test_perfect_clock_has_no_floor(self):
        assert jitter_noise_floor_dbc(0.0, 50.0, 32768) == -math.inf

    def test_more_jitter_raises_the_floor(self):
        assert jitter_noise_floor_dbc(1e-3, 50.0, 32768) > jitter_noise_floor_dbc(1e-6, 50.0, 32768)

    def test_longer_records_lower_the_floor(self):
        assert jitter_noise_floor_dbc(1e-4, 50.0, 65536) < jitter_noise_floor_dbc(1e-4, 50.0, 4096)

    def test_higher_carrier_raises_the_floor_proportionally(self):
        """Jitter converts to phase noise in proportion to frequency: 10x -> +20 dB."""
        at_50 = jitter_noise_floor_dbc(1e-4, 50.0, 32768)
        at_500 = jitter_noise_floor_dbc(1e-4, 500.0, 32768)
        assert at_500 == pytest.approx(at_50 + 20.0, abs=0.1)

    def test_hand_computed_anchor(self):
        """0.49 ms rms at 50 Hz over 32768 samples.

        -(-20*log10(2*pi*50*4.9e-4) + 10*log10(16384)) = -58.4 dBc.
        """
        assert jitter_noise_floor_dbc(0.49e-3, 50.0, 32768) == pytest.approx(-58.4, abs=1.0)

    def test_rejects_nonsense_input(self):
        with pytest.raises(ValueError):
            jitter_noise_floor_dbc(1e-4, 0.0, 1000)
        with pytest.raises(ValueError):
            jitter_noise_floor_dbc(1e-4, 50.0, 1)


class TestMeasuredFloor:
    def test_tracks_the_injected_noise_floor(self):
        rec, _ = synthesize(MOTOR, fs=5000, duration_s=10, noise_floor_dbc=-80.0)
        assert measure_noise_floor_dbc(rec, line_hz=50.0) == pytest.approx(-80.0, abs=5.0)

    def test_a_noisier_recording_measures_higher(self):
        quiet, _ = synthesize(MOTOR, fs=5000, duration_s=10, noise_floor_dbc=-90.0)
        loud, _ = synthesize(MOTOR, fs=5000, duration_s=10, noise_floor_dbc=-60.0)
        assert measure_noise_floor_dbc(loud) > measure_noise_floor_dbc(quiet) + 15.0

    def test_longer_records_measure_a_lower_floor(self):
        """The cheapest way to reach a weak sideband is to record for longer."""
        short, _ = synthesize(MOTOR, fs=5000, duration_s=4, noise_floor_dbc=-70.0, seed=1)
        long, _ = synthesize(MOTOR, fs=5000, duration_s=40, noise_floor_dbc=-70.0, seed=1)
        assert measure_noise_floor_dbc(long) < measure_noise_floor_dbc(short)


class TestVerdicts:
    def test_clean_recording_is_good(self):
        rec, _ = synthesize(MOTOR, fs=5000, duration_s=10, jitter_rms_s=0.0)
        q = assess_clock(rec, line_hz=50.0)
        assert q.verdict is TrustLevel.GOOD
        assert q.trustworthy

    def test_millis_style_acquisition_is_unreliable(self):
        """Accumulating interval error wrecks the near-carrier region."""
        rec, _ = synthesize(
            MOTOR,
            fs=620,
            duration_s=52,
            jitter_rms_s=0.5e-3,
            jitter_model="interval",
            timestamp_resolution_s=1e-3,
        )
        q = assess_clock(rec, line_hz=50.0)
        assert q.verdict is TrustLevel.UNRELIABLE
        assert not q.trustworthy
        assert q.headroom_db < 0

    def test_headroom_is_fault_level_minus_measured_floor(self):
        rec, _ = synthesize(MOTOR, fs=5000, duration_s=10)
        q = assess_clock(rec, reference_fault_dbc=-50.0)
        assert q.headroom_db == pytest.approx(-50.0 - q.measured_floor_dbc, abs=1e-9)

    def test_missing_timestamps_still_yields_a_verdict(self):
        """The floor comes from the signal, so the audit is not blocked."""
        rec, _ = synthesize(MOTOR, fs=5000, duration_s=10)
        stripped = Recording.from_array(rec.current, fs=rec.fs)
        q = assess_clock(stripped)
        assert not q.timestamps_available
        assert q.verdict is TrustLevel.GOOD
        assert any("No timestamps supplied" in n for n in q.notes)

    def test_explain_reports_the_measured_floor(self):
        rec, _ = synthesize(MOTOR, fs=5000, duration_s=5)
        text = assess_clock(rec).explain()
        assert "Clock audit" in text
        assert "MEASURED floor" in text
        assert "headroom" in text


class TestTimingDiagnostics:
    def test_quantized_timestamps_are_flagged_as_an_upper_bound(self):
        rec, _ = synthesize(
            MOTOR, fs=620, duration_s=30, jitter_rms_s=0.4e-3, timestamp_resolution_s=1e-3
        )
        q = assess_clock(rec)
        assert q.quantization_limited
        assert any("UPPER" in n for n in q.notes)

    def test_drift_is_separated_from_jitter(self):
        """Interval jitter accumulates: drift must dominate the fast component."""
        rec, _ = synthesize(
            MOTOR, fs=1000, duration_s=30, jitter_rms_s=2e-4, jitter_model="interval"
        )
        q = assess_clock(rec)
        assert q.drift_rms_s > q.jitter_rms_s

    def test_model_measurement_disagreement_is_reported(self):
        """Correlated drift makes the white-jitter model pessimistic; say so."""
        rec, _ = synthesize(
            MOTOR,
            fs=1000,
            duration_s=30,
            jitter_rms_s=2e-4,
            jitter_model="interval",
            noise_floor_dbc=-95.0,
        )
        q = assess_clock(rec)
        if q.predicted_jitter_floor_dbc > q.measured_floor_dbc + 15.0:
            assert any("correlated" in n for n in q.notes)

    def test_aperture_jitter_does_not_drift(self):
        rec, _ = synthesize(
            MOTOR, fs=1000, duration_s=30, jitter_rms_s=2e-4, jitter_model="aperture"
        )
        q = assess_clock(rec)
        assert q.drift_rms_s < q.jitter_rms_s

    def test_rejects_unknown_jitter_model(self):
        with pytest.raises(ValueError, match="jitter_model"):
            synthesize(MOTOR, fs=1000, duration_s=1, jitter_rms_s=1e-4, jitter_model="nonsense")  # type: ignore[arg-type]

    def test_non_monotonic_timestamps_are_flagged(self):
        rec, _ = synthesize(MOTOR, fs=1000, duration_s=5)
        ts = np.array(rec.timestamps, copy=True)
        ts[100] = ts[99] - 1e-3
        broken = Recording(rec.current, rec.fs, ts, rec.name)
        assert any("Non-monotonic" in n for n in assess_clock(broken).notes)
