"""Analytic fault-frequency models for induction machines.

These are the textbook MCSA signatures (Thomson & Fenger and the standard
condition-monitoring literature), implemented once, correctly, with the
pole-pair convention stated explicitly at every call site.

Every function returns a :class:`FaultSignature`: the frequencies to interrogate
plus a human-readable label for each, so a report can say *which* component fired
rather than just "fault".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signals import FloatArray, Motor


@dataclass(frozen=True, slots=True)
class FaultSignature:
    """A set of frequencies that a specific fault mechanism would excite."""

    kind: str
    frequencies: FloatArray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.frequencies) != len(self.labels):
            raise ValueError(
                f"{self.kind}: {len(self.frequencies)} frequencies vs {len(self.labels)} labels"
            )

    def __len__(self) -> int:
        return int(self.frequencies.size)

    def within(self, low_hz: float, high_hz: float) -> FaultSignature:
        """Restrict the signature to a frequency band (e.g. below Nyquist)."""
        keep = (self.frequencies >= low_hz) & (self.frequencies <= high_hz)
        return FaultSignature(
            kind=self.kind,
            frequencies=self.frequencies[keep],
            labels=tuple(lab for lab, k in zip(self.labels, keep, strict=True) if k),
        )


def broken_rotor_bar(motor: Motor, slip: float, *, n_sidebands: int = 3) -> FaultSignature:
    """Broken/cracked rotor bar sidebands: ``f * (1 +/- 2ks)``.

    The classic signature. Independent of pole count. The lower sideband
    ``(1-2s)f`` is the primary indicator; the upper ``(1+2s)f`` arises from the
    consequent speed ripple and confirms it.

    Args:
        motor: Machine under test.
        slip: Per-unit slip (0 = synchronous, 1 = standstill).
        n_sidebands: Sideband order ``k`` to generate, 1..n.

    Returns:
        Frequencies ordered ``(1-2s)f, (1+2s)f, (1-4s)f, (1+4s)f, ...``
    """
    _check_slip(slip)
    f = motor.line_hz
    freqs: list[float] = []
    labels: list[str] = []
    for k in range(1, n_sidebands + 1):
        for sign, sym in ((-1, "-"), (1, "+")):
            freqs.append((1.0 + sign * 2.0 * k * slip) * f)
            labels.append(f"(1{sym}{2 * k}s)f")
    return FaultSignature("broken_rotor_bar", np.asarray(freqs, dtype=np.float64), tuple(labels))


def stator_interturn(motor: Motor, slip: float, *, max_k: int = 5) -> FaultSignature:
    """Stator inter-turn short-circuit: ``f * [n(1-s)/p +/- k]``.

    Args:
        motor: Machine under test.
        slip: Per-unit slip.
        max_k: Highest odd supply-harmonic order ``k`` to include (1, 3, 5, ...).

    Returns:
        Positive frequencies only; the formula generates negative values that
        alias, and those are dropped.
    """
    _check_slip(slip)
    f, p = motor.line_hz, motor.pole_pairs
    freqs: list[float] = []
    labels: list[str] = []
    for k in range(1, max_k + 1, 2):
        for n in range(1, 2 * p):
            for sign, sym in ((-1, "-"), (1, "+")):
                value = f * (n * (1.0 - slip) / p + sign * k)
                if value > 0:
                    freqs.append(value)
                    labels.append(f"[{n}(1-s)/p{sym}{k}]f")
    return FaultSignature("stator_interturn", np.asarray(freqs, dtype=np.float64), tuple(labels))


def eccentricity(
    motor: Motor,
    slip: float,
    *,
    max_order: int = 3,
) -> FaultSignature:
    """Mixed air-gap eccentricity: ``f +/- k * f_rotor``.

    Mixed (static + dynamic) eccentricity modulates the air-gap permeance at the
    rotor rotational frequency, producing low-order sidebands around the
    fundamental. This is the form that needs no rotor-slot count.

    Args:
        motor: Machine under test.
        slip: Per-unit slip.
        max_order: Highest sideband order ``k``.
    """
    _check_slip(slip)
    f = motor.line_hz
    f_r = motor.rotor_hz(slip)
    freqs: list[float] = []
    labels: list[str] = []
    for k in range(1, max_order + 1):
        for sign, sym in ((-1, "-"), (1, "+")):
            value = f + sign * k * f_r
            if value > 0:
                freqs.append(value)
                labels.append(f"f{sym}{k}f_r")
    return FaultSignature("eccentricity", np.asarray(freqs, dtype=np.float64), tuple(labels))


def rotor_slot_harmonics(
    motor: Motor,
    slip: float,
    *,
    eccentricity_orders: tuple[int, ...] = (0, 1),
    stator_harmonics: tuple[int, ...] = (1, 3, 5),
) -> FaultSignature:
    """Rotor slot harmonics: ``f * [(R +/- n_d)(1-s)/p +/- n_ws]``.

    Requires ``motor.rotor_bars``. These components are strong and their position
    depends sharply on slip, which makes them the best available handle for
    *sensorless* speed estimation -- see :func:`statorscope.detect.estimate_slip`.

    Args:
        motor: Machine under test; ``rotor_bars`` must be set.
        slip: Per-unit slip.
        eccentricity_orders: ``n_d``; 0 is static, 1+ dynamic eccentricity.
        stator_harmonics: ``n_ws``, odd stator MMF harmonic orders.

    Raises:
        ValueError: If ``motor.rotor_bars`` is not set.
    """
    _check_slip(slip)
    if motor.rotor_bars is None:
        raise ValueError("rotor_slot_harmonics requires Motor(rotor_bars=...)")
    f, p, R = motor.line_hz, motor.pole_pairs, motor.rotor_bars
    freqs: list[float] = []
    labels: list[str] = []
    for n_d in eccentricity_orders:
        for n_ws in stator_harmonics:
            for sign, sym in ((-1, "-"), (1, "+")):
                value = f * ((R + n_d) * (1.0 - slip) / p + sign * n_ws)
                if value > 0:
                    freqs.append(value)
                    labels.append(f"[(R+{n_d})(1-s)/p{sym}{n_ws}]f")
    return FaultSignature(
        "rotor_slot_harmonics", np.asarray(freqs, dtype=np.float64), tuple(labels)
    )


def bearing_defect(
    motor: Motor,
    slip: float,
    *,
    n_balls: int = 8,
    ball_diameter: float | None = None,
    pitch_diameter: float | None = None,
    contact_angle_deg: float = 0.0,
    max_order: int = 2,
) -> FaultSignature:
    """Bearing race defects reflected into the current: ``|f +/- m * f_defect|``.

    If bearing geometry is supplied the exact BPFO/BPFI are used; otherwise the
    standard approximations ``BPFO ~ 0.4 * n * f_r`` and ``BPFI ~ 0.6 * n * f_r``
    are applied, which hold for most common bearings.

    Args:
        motor: Machine under test.
        slip: Per-unit slip.
        n_balls: Rolling element count.
        ball_diameter: Rolling element diameter, same units as ``pitch_diameter``.
        pitch_diameter: Bearing pitch diameter.
        contact_angle_deg: Contact angle in degrees.
        max_order: Highest modulation order ``m``.
    """
    _check_slip(slip)
    f = motor.line_hz
    f_r = motor.rotor_hz(slip)

    if ball_diameter is not None and pitch_diameter is not None and pitch_diameter > 0:
        ratio = (ball_diameter / pitch_diameter) * np.cos(np.deg2rad(contact_angle_deg))
        bpfo = (n_balls / 2.0) * f_r * (1.0 - ratio)
        bpfi = (n_balls / 2.0) * f_r * (1.0 + ratio)
    else:
        bpfo = 0.4 * n_balls * f_r
        bpfi = 0.6 * n_balls * f_r

    freqs: list[float] = []
    labels: list[str] = []
    for name, f_defect in (("BPFO", bpfo), ("BPFI", bpfi)):
        for m in range(1, max_order + 1):
            for sign, sym in ((-1, "-"), (1, "+")):
                value = abs(f + sign * m * f_defect)
                if value > 0:
                    freqs.append(value)
                    labels.append(f"|f{sym}{m}*{name}|")
    return FaultSignature("bearing_defect", np.asarray(freqs, dtype=np.float64), tuple(labels))


def _check_slip(slip: float) -> None:
    if not 0.0 <= slip < 1.0:
        raise ValueError(f"slip must be in [0, 1), got {slip}")
