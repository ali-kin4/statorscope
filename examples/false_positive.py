"""The failure this library exists to prevent, in 40 lines.

A healthy motor, sampled through a cheap software-timed acquisition chain, is
reported as having a broken rotor bar by naive amplitude thresholding. Run it:

    uv run python examples/false_positive.py
"""

from __future__ import annotations

from statorscope import (
    Motor,
    Spectrum,
    diagnose,
    millis_jitter_recording,
    residual_spectrum,
    synthesize,
)

motor = Motor(pole_pairs=2, rotor_bars=28, line_hz=50.0)
SLIP = 0.03


def naive_detector(spectrum: Spectrum, slip: float) -> tuple[bool, float]:
    """What most MCSA code does: look up the sideband, threshold on amplitude."""
    lower = (1 - 2 * slip) * 50.0
    level = spectrum.level_dbc(lower, 0.2)
    return level > -50.0, level


def main() -> None:
    print("=" * 74)
    print("A HEALTHY motor. No broken bars. Two acquisition chains.")
    print("=" * 74)

    clean, _ = synthesize(motor, slip=SLIP, broken_bar_dbc=None, fs=5000, duration_s=20)
    cheap, truth = millis_jitter_recording(motor, slip=SLIP, broken_bar_dbc=None)
    assert truth.healthy

    for label, rec in (("hardware clock", clean), ("millis() logging", cheap)):
        spec = residual_spectrum(rec, motor)
        fired, level = naive_detector(spec, SLIP)
        report = diagnose(rec, motor)
        print(f"\n--- {label} ---")
        print(
            f"  naive amplitude detector : {'BROKEN BAR' if fired else 'clear'} ({level:+.1f} dBc)"
        )
        print(f"  statorscope              : {'faults found' if report.supported else 'clear'}")
        print(f"  clock verdict            : {report.clock.verdict.upper()}")
        print(f"  measured floor           : {report.clock.measured_floor_dbc:+.1f} dBc")

    print(
        "\nThe naive detector fires on a healthy machine as soon as the clock is cheap.\n"
        "statorscope measures the floor first and refuses to claim anything it\n"
        "cannot separate from acquisition noise."
    )


if __name__ == "__main__":
    main()
