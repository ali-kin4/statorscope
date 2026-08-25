"""Detection tests, including the false-positive regression that motivates the library."""

from __future__ import annotations

import numpy as np
import pytest

from statorscope import (
    Motor,
    TrustLevel,
    diagnose,
    estimate_slip,
    millis_jitter_recording,
    synthesize,
)
from statorscope.detect import MIN_FAULT_LEVEL_DBC, MIN_PROMINENCE_DB, residual_spectrum
from statorscope.quality import INCIPIENT_FAULT_DBC

MOTOR = Motor(pole_pairs=2, rotor_bars=28, line_hz=50.0)


def _residual_spectrum(rec, motor=MOTOR):
    return residual_spectrum(rec, motor)


class TestBrokenBarDetection:
    def test_detects_injected_fault(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-42, fs=5000, duration_s=10)
        report = diagnose(rec, MOTOR)
        brb = next(f for f in report.faults if f.kind == "broken_rotor_bar")
        assert brb.detected
        assert report.supported

    def test_healthy_motor_is_not_flagged(self):
        rec, truth = synthesize(MOTOR, slip=0.03, broken_bar_dbc=None, fs=5000, duration_s=10)
        assert truth.healthy
        report = diagnose(rec, MOTOR)
        brb = next(f for f in report.faults if f.kind == "broken_rotor_bar")
        assert not brb.detected, "healthy machine reported as faulty"

    @pytest.mark.parametrize("level", [-35.0, -42.0, -50.0])
    def test_severity_tracks_injected_level(self, level: float):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=level, fs=5000, duration_s=15)
        report = diagnose(rec, MOTOR)
        brb = next(f for f in report.faults if f.kind == "broken_rotor_bar")
        assert brb.detected
        # Recovered level should be within a few dB of what was injected.
        assert brb.strongest_dbc == pytest.approx(level, abs=4.0)

    def test_sidebands_land_at_the_predicted_frequencies(self):
        slip = 0.025
        rec, _ = synthesize(MOTOR, slip=slip, broken_bar_dbc=-40, fs=5000, duration_s=15)
        report = diagnose(rec, MOTOR)
        brb = next(f for f in report.faults if f.kind == "broken_rotor_bar")
        hits = [e for e in brb.evidence if e.passed]
        assert hits
        expected_lower = (1 - 2 * slip) * MOTOR.line_hz
        assert any(abs(e.frequency_hz - expected_lower) < 0.1 for e in hits)


class TestJitterFalsePositive:
    """The regression this library exists for.

    A healthy machine sampled through a millis()-timestamped Arduino chain
    produces a phase-noise skirt where the broken-bar sidebands live. Amplitude
    thresholding reads it as a fault. Both guards must hold: the clock audit
    refuses the recording, and prominence scoring declines to call the skirt a
    sideband.
    """

    def test_audit_refuses_millis_acquisition(self):
        rec, truth = millis_jitter_recording(MOTOR, slip=0.03, broken_bar_dbc=None)
        assert truth.healthy
        report = diagnose(rec, MOTOR)
        assert report.clock.verdict is TrustLevel.UNRELIABLE
        assert not report.clock.trustworthy

    def test_healthy_jittered_motor_is_never_reported_as_supported(self):
        rec, truth = millis_jitter_recording(MOTOR, slip=0.03, broken_bar_dbc=None)
        assert truth.healthy
        report = diagnose(rec, MOTOR)
        assert not report.supported, (
            "a healthy machine with a bad clock was reported as a supported fault - "
            "this is the exact failure the library is built to prevent"
        )

    def test_summary_says_unsupported_out_loud(self):
        rec, _ = millis_jitter_recording(MOTOR, slip=0.03, broken_bar_dbc=None)
        text = diagnose(rec, MOTOR).summary()
        assert "UNSUPPORTED" in text

    def test_quantization_is_flagged(self):
        rec, _ = millis_jitter_recording(MOTOR, slip=0.03)
        report = diagnose(rec, MOTOR)
        assert report.clock.quantization_limited
        assert any("quantisation" in n for n in report.clock.notes)


class TestSlipEstimation:
    @pytest.mark.parametrize("slip", [0.010, 0.025, 0.040])
    def test_recovers_slip_without_a_tachometer(self, slip: float):
        rec, _ = synthesize(MOTOR, slip=slip, broken_bar_dbc=-40, fs=5000, duration_s=20)
        est = estimate_slip(_residual_spectrum(rec), MOTOR)
        assert est.confident
        assert est.slip == pytest.approx(slip, abs=0.002)

    def test_rpm_follows_from_slip(self):
        rec, truth = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-40, fs=5000, duration_s=20)
        est = estimate_slip(_residual_spectrum(rec), MOTOR)
        assert est.rpm == pytest.approx(truth.rotor_rpm, rel=0.01)

    def test_low_confidence_on_a_healthy_machine(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=None, fs=5000, duration_s=10)
        est = estimate_slip(_residual_spectrum(rec), MOTOR)
        assert not est.confident, "slip search locked onto noise on a healthy machine"


class TestDiagnosisPlumbing:
    def test_supplied_slip_bypasses_estimation(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-40, fs=5000, duration_s=10)
        report = diagnose(rec, MOTOR, slip=0.03)
        assert report.slip.method == "supplied"
        assert report.slip.slip == 0.03

    def test_signature_above_nyquist_is_reported_not_silently_dropped(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-40, fs=400, duration_s=20)
        report = diagnose(rec, MOTOR)
        stator = next(f for f in report.faults if f.kind == "stator_interturn")
        assert any("above Nyquist" in n for n in stator.notes)

    def test_summary_is_printable(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-40, fs=5000, duration_s=5)
        text = diagnose(rec, MOTOR).summary()
        assert "statorscope diagnosis" in text
        assert "Clock audit" in text


class TestThresholdCalibration:
    """Empirical justification for MIN_PROMINENCE_DB.

    The threshold is not a taste call. White noise alone produces prominence,
    because the peak of a few bins always stands above the median of many. If the
    threshold sits inside that distribution, every healthy machine reports a fault.
    These tests pin the measurement so a future change has to re-justify itself.
    """

    def test_noise_only_prominence_stays_below_the_threshold(self):
        worst = 0.0
        for seed in range(5):
            rec, _ = synthesize(
                MOTOR, slip=0.03, broken_bar_dbc=None, fs=5000, duration_s=10, seed=seed
            )
            spec = _residual_spectrum(rec)
            proms = [spec.prominence_db(f, tol_hz=0.15) for f in np.linspace(44, 56, 200)]
            worst = max(worst, *proms)
        assert worst < MIN_PROMINENCE_DB, (
            f"noise reached {worst:.1f} dB prominence but the detection threshold is "
            f"{MIN_PROMINENCE_DB} dB - healthy machines will be flagged"
        )

    def test_a_real_sideband_clears_the_threshold_with_margin(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-50, fs=5000, duration_s=20)
        spec = _residual_spectrum(rec)
        assert spec.prominence_db(47.0, tol_hz=0.15) > MIN_PROMINENCE_DB + 10.0


class TestSmearedCarrier:
    """Regression: a drift-smeared carrier must not be mistaken for a fault.

    suppress_fundamental fits and removes a single sinusoid. A carrier smeared by
    time-base drift is not one, so a large residue survives immediately beside it
    -- and that residue is locally prominent, so it looks exactly like a low-slip
    broken-bar sideband.

    This was found by running the library on the real 2018 recordings that
    motivated it, where it reported all three fault types on every file. The
    synthetic suite missed it because a clean synthetic carrier suppresses
    perfectly, leaving no residue to trip over.
    """

    def test_clean_carrier_occupies_about_one_bin(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-42, fs=5000, duration_s=20)
        halfwidth = _residual_spectrum(rec).carrier_halfwidth_hz()
        assert halfwidth < 0.5, f"clean carrier measured {halfwidth:.3f} Hz wide"

    def test_smeared_carrier_is_measured_as_wide(self):
        rec, _ = millis_jitter_recording(MOTOR, slip=0.03, broken_bar_dbc=None)
        halfwidth = _residual_spectrum(rec).carrier_halfwidth_hz()
        assert halfwidth > 1.0, f"smeared carrier measured only {halfwidth:.3f} Hz wide"

    def test_slip_search_refuses_to_look_inside_the_carrier(self):
        """The original failure: slip pinned to the bottom of the range, score +40 dB."""
        rec, truth = millis_jitter_recording(MOTOR, slip=0.03, broken_bar_dbc=None)
        assert truth.healthy
        est = estimate_slip(_residual_spectrum(rec), MOTOR)
        assert not est.confident, (
            f"slip search locked onto carrier residue at slip={est.slip:.4f} "
            f"with score {est.score_db:+.1f} dB"
        )

    def test_smeared_healthy_recording_claims_nothing(self):
        rec, truth = millis_jitter_recording(MOTOR, slip=0.03, broken_bar_dbc=None)
        assert truth.healthy
        report = diagnose(rec, MOTOR)
        assert not report.supported
        assert not any(f.detected for f in report.faults), (
            "smeared carrier produced fault detections on a healthy machine"
        )

    def test_audit_refuses_when_the_carrier_blinds_normal_slip(self):
        rec, _ = millis_jitter_recording(MOTOR, slip=0.03, broken_bar_dbc=None)
        q = diagnose(rec, MOTOR).clock
        assert q.carrier_halfwidth_hz > 1.0
        assert q.verdict is TrustLevel.UNRELIABLE
        assert not q.trustworthy

    def test_a_real_fault_on_a_clean_carrier_is_still_found(self):
        """The guard must not be so wide that it suppresses genuine detections."""
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-45, fs=5000, duration_s=20)
        report = diagnose(rec, MOTOR)
        brb = next(f for f in report.faults if f.kind == "broken_rotor_bar")
        assert brb.detected
        assert report.supported

    @pytest.mark.parametrize("slip", [0.02, 0.03, 0.05])
    def test_guard_does_not_block_normal_operating_slip(self, slip: float):
        rec, _ = synthesize(MOTOR, slip=slip, broken_bar_dbc=-42, fs=5000, duration_s=20)
        est = estimate_slip(_residual_spectrum(rec), MOTOR)
        assert est.confident
        assert est.slip == pytest.approx(slip, abs=0.002)


class TestAbsoluteFaultFloor:
    """Regression: prominence alone is not enough on near-noiseless data.

    Prominence is measured against a *local* floor. On a simulation or a very clean
    rig, numerical residue tens of dB below anything physical still scores high
    prominence. Found by benchmarking against the public BBIM2023 dataset, where a
    healthy motor was reported faulty at -148 dBc.
    """

    def test_absurdly_weak_component_is_never_a_fault(self):
        rec, truth = synthesize(
            MOTOR, slip=0.03, broken_bar_dbc=None, fs=5000, duration_s=20, noise_floor_dbc=-200.0
        )
        assert truth.healthy
        report = diagnose(rec, MOTOR)
        assert not any(f.detected for f in report.faults), (
            "numerical residue on a noiseless healthy machine was reported as a fault"
        )

    def test_the_bound_sits_below_any_real_fault(self):
        """It must reject numerical noise without rejecting weak genuine faults."""
        assert MIN_FAULT_LEVEL_DBC < INCIPIENT_FAULT_DBC

    def test_an_incipient_fault_still_clears_the_bound(self):
        rec, _ = synthesize(MOTOR, slip=0.03, broken_bar_dbc=-55, fs=5000, duration_s=30)
        report = diagnose(rec, MOTOR)
        brb = next(f for f in report.faults if f.kind == "broken_rotor_bar")
        assert brb.detected


class TestNoiselessCarrierWidth:
    """Regression: the carrier-width walk must not saturate on clean data.

    The walk stopped only when the skirt reached the noise floor. On near-noiseless
    data that floor is hundreds of dB down, so it never stopped, saturated at its
    cap, and blinded the entire slip range -- turning every real fault into a miss.
    """

    def test_clean_carrier_does_not_saturate(self):
        rec, _ = synthesize(
            MOTOR,
            slip=0.04,
            broken_bar_dbc=-40,
            fs=2000,
            duration_s=2.4,
            noise_floor_dbc=-200.0,
        )
        halfwidth = _residual_spectrum(rec).carrier_halfwidth_hz()
        assert halfwidth < 8.0, f"carrier width saturated at {halfwidth:.2f} Hz"

    def test_fault_on_a_noiseless_short_record_is_still_found(self):
        """The BBIM2023 conditions: 2 kHz, 2.4 s, ~4% slip, no noise."""
        rec, _ = synthesize(
            MOTOR,
            slip=0.04,
            broken_bar_dbc=-38,
            fs=2000,
            duration_s=2.4,
            noise_floor_dbc=-200.0,
        )
        est = estimate_slip(_residual_spectrum(rec), MOTOR)
        assert est.confident
        assert est.slip == pytest.approx(0.04, abs=0.003)
