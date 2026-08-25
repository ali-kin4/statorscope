"""Fault-frequency model tests, checked against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from statorscope import (
    FaultSignature,
    Motor,
    bearing_defect,
    broken_rotor_bar,
    eccentricity,
    rotor_slot_harmonics,
    stator_interturn,
)

MOTOR = Motor(pole_pairs=2, rotor_bars=28, line_hz=50.0)


class TestMotor:
    def test_synchronous_speed(self):
        """4-pole (2 pole pairs) on 50 Hz is 1500 rpm."""
        assert MOTOR.synchronous_rpm == pytest.approx(1500.0)
        assert Motor(pole_pairs=1, line_hz=50.0).synchronous_rpm == pytest.approx(3000.0)
        assert Motor(pole_pairs=2, line_hz=60.0).synchronous_rpm == pytest.approx(1800.0)

    def test_rotor_frequency(self):
        """At 3% slip a 4-pole 50 Hz machine turns at 24.25 Hz."""
        assert MOTOR.rotor_hz(0.03) == pytest.approx(24.25)

    def test_rated_slip_from_nameplate(self):
        m = Motor(pole_pairs=2, line_hz=50.0, rated_rpm=1455.0)
        assert m.rated_slip == pytest.approx(0.03)

    def test_rated_slip_is_none_without_nameplate(self):
        assert MOTOR.rated_slip is None

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_invalid_pole_pairs(self, bad: int):
        with pytest.raises(ValueError, match="pole_pairs"):
            Motor(pole_pairs=bad)


class TestBrokenRotorBar:
    def test_first_order_sidebands(self):
        """At 3% slip: 50*(1-0.06)=47 Hz and 50*(1+0.06)=53 Hz."""
        sig = broken_rotor_bar(MOTOR, 0.03, n_sidebands=1)
        assert sig.frequencies == pytest.approx([47.0, 53.0])
        assert sig.labels == ("(1-2s)f", "(1+2s)f")

    def test_higher_orders(self):
        sig = broken_rotor_bar(MOTOR, 0.03, n_sidebands=2)
        assert sig.frequencies == pytest.approx([47.0, 53.0, 44.0, 56.0])

    def test_independent_of_pole_count(self):
        a = broken_rotor_bar(Motor(pole_pairs=1), 0.03, n_sidebands=1)
        b = broken_rotor_bar(Motor(pole_pairs=4), 0.03, n_sidebands=1)
        assert a.frequencies == pytest.approx(b.frequencies)

    def test_zero_slip_collapses_onto_the_fundamental(self):
        sig = broken_rotor_bar(MOTOR, 0.0, n_sidebands=1)
        assert sig.frequencies == pytest.approx([50.0, 50.0])

    @pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
    def test_rejects_invalid_slip(self, bad: float):
        with pytest.raises(ValueError, match="slip"):
            broken_rotor_bar(MOTOR, bad)


class TestEccentricity:
    def test_sidebands_at_rotor_frequency(self):
        """f +/- f_r with f_r = 24.25 Hz at 3% slip."""
        sig = eccentricity(MOTOR, 0.03, max_order=1)
        assert sig.frequencies == pytest.approx([25.75, 74.25])

    def test_negative_frequencies_are_dropped(self):
        sig = eccentricity(MOTOR, 0.03, max_order=3)
        assert np.all(sig.frequencies > 0)


class TestStatorInterturn:
    def test_all_frequencies_positive(self):
        sig = stator_interturn(MOTOR, 0.03)
        assert np.all(sig.frequencies > 0)

    def test_labels_match_frequency_count(self):
        sig = stator_interturn(MOTOR, 0.03)
        assert len(sig.labels) == len(sig.frequencies)

    def test_known_component(self):
        """n=1, k=1, p=2, s=0.03 -> 50*(1*0.97/2 + 1) = 74.25 Hz."""
        sig = stator_interturn(MOTOR, 0.03, max_k=1)
        assert pytest.approx(min(sig.frequencies, key=lambda f: abs(f - 74.25))) == 74.25


class TestRotorSlotHarmonics:
    def test_requires_rotor_bars(self):
        with pytest.raises(ValueError, match="rotor_bars"):
            rotor_slot_harmonics(Motor(pole_pairs=2), 0.03)

    def test_produces_high_frequency_components(self):
        sig = rotor_slot_harmonics(MOTOR, 0.03)
        assert len(sig) > 0
        assert sig.frequencies.max() > 500.0


class TestBearingDefect:
    def test_geometry_and_approximation_are_in_the_same_ballpark(self):
        exact = bearing_defect(
            MOTOR, 0.03, n_balls=8, ball_diameter=7.94, pitch_diameter=38.5, max_order=1
        )
        approx = bearing_defect(MOTOR, 0.03, n_balls=8, max_order=1)
        assert len(exact) == len(approx)
        assert exact.frequencies == pytest.approx(approx.frequencies, rel=0.35)

    def test_all_positive(self):
        sig = bearing_defect(MOTOR, 0.03, max_order=2)
        assert np.all(sig.frequencies > 0)


class TestSignatureContainer:
    def test_within_filters_and_keeps_labels_aligned(self):
        sig = broken_rotor_bar(MOTOR, 0.03, n_sidebands=3)
        clipped = sig.within(48.0, 54.0)
        assert len(clipped.frequencies) == len(clipped.labels)
        assert np.all(clipped.frequencies >= 48.0)
        assert np.all(clipped.frequencies <= 54.0)

    def test_mismatched_labels_rejected(self):
        with pytest.raises(ValueError, match="labels"):
            FaultSignature("x", np.array([1.0, 2.0]), ("only-one",))
