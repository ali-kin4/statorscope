"""statorscope - Motor Current Signature Analysis that knows when to refuse.

Detect broken rotor bars, stator inter-turn faults and air-gap eccentricity from
three-phase current, and get told when your acquisition cannot support the answer.

Quickstart:
    >>> from statorscope import Motor, diagnose, synthesize
    >>> motor = Motor(pole_pairs=2, rotor_bars=28, line_hz=50.0)
    >>> rec, truth = synthesize(motor, slip=0.03, broken_bar_dbc=-42)
    >>> report = diagnose(rec, motor)
    >>> report.supported
    True
"""

from __future__ import annotations

from .calibrate import GridLock, grid_lock, resample_uniform
from .detect import (
    Diagnosis,
    Evidence,
    FaultResult,
    SlipEstimate,
    diagnose,
    estimate_slip,
    evaluate_signature,
    residual_spectrum,
)
from .faults import (
    FaultSignature,
    bearing_defect,
    broken_rotor_bar,
    eccentricity,
    rotor_slot_harmonics,
    stator_interturn,
)
from .quality import (
    ClockQuality,
    TrustLevel,
    assess_clock,
    jitter_noise_floor_dbc,
    measure_noise_floor_dbc,
)
from .signals import Motor, Recording
from .spectrum import (
    Spectrum,
    compute_spectrum,
    estimate_fundamental,
    suppress_fundamental,
)
from .synth import GroundTruth, millis_jitter_recording, synthesize

__version__ = "0.1.0"

__all__ = [
    "ClockQuality",
    "Diagnosis",
    "Evidence",
    "FaultResult",
    "FaultSignature",
    "GridLock",
    "GroundTruth",
    "Motor",
    "Recording",
    "SlipEstimate",
    "Spectrum",
    "TrustLevel",
    "__version__",
    "assess_clock",
    "bearing_defect",
    "broken_rotor_bar",
    "compute_spectrum",
    "diagnose",
    "eccentricity",
    "estimate_fundamental",
    "estimate_slip",
    "evaluate_signature",
    "grid_lock",
    "jitter_noise_floor_dbc",
    "measure_noise_floor_dbc",
    "millis_jitter_recording",
    "resample_uniform",
    "residual_spectrum",
    "rotor_slot_harmonics",
    "stator_interturn",
    "suppress_fundamental",
    "synthesize",
]
