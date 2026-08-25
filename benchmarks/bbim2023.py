"""Benchmark statorscope against the public BBIM2023 labelled dataset.

Reproduce:

    # download both workbooks (CC BY 4.0) from
    #   https://data.mendeley.com/datasets/f5ksvkrp73/1
    uv run --with openpyxl python benchmarks/bbim2023.py path/to/*.xlsx

Reports detection against ground-truth labels, and -- where the workbook carries a
rotor-speed channel -- the error of the sensorless slip estimate against the
simulator's true slip.
"""

from __future__ import annotations

import sys
from pathlib import Path

from statorscope import diagnose
from statorscope.datasets import DATASETS, load_bbim2023

HEADER = (
    f"{'case':22s} {'label':14s} {'true slip':>9s} {'est slip':>9s} "
    f"{'err':>7s} {'BRB':>4s} {'level':>9s} {'severity':>10s} {'result':>7s}"
)


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__)
        return 2

    info = DATASETS["bbim2023"]
    print(f"{info.name}  [{info.licence}]")
    print(f"{info.url}")
    print("simulated data - validates the analysis, not the sensor chain\n")
    print(HEADER)

    correct = total = 0
    slip_errors: list[float] = []

    for path in paths:
        for item in load_bbim2023(path):
            report = diagnose(item.recording, item.motor, calibrate=False)
            brb = next(f for f in report.faults if f.kind == "broken_rotor_bar")

            hit = brb.detected == (not item.healthy)
            correct += hit
            total += 1

            if item.true_slip is not None:
                err = abs(report.slip.slip - item.true_slip)
                if not item.healthy:
                    slip_errors.append(err)
                true_s, err_s = f"{item.true_slip:.4f}", f"{err:.4f}"
            else:
                true_s, err_s = "n/a", "-"

            label = "healthy" if item.healthy else f"{item.broken_bars} broken bar(s)"
            level = f"{brb.strongest_dbc:+.1f}" if brb.detected else "-"
            print(
                f"{item.recording.name:22s} {label:14s} {true_s:>9s} "
                f"{report.slip.slip:9.4f} {err_s:>7s} "
                f"{('YES' if brb.detected else 'no'):>4s} {level:>9s} "
                f"{(brb.severity if brb.detected else '-'):>10s} "
                f"{('OK' if hit else 'WRONG'):>7s}"
            )

    print(f"\ndetection: {correct}/{total} correct")
    if slip_errors:
        worst = max(slip_errors)
        mean = sum(slip_errors) / len(slip_errors)
        print(f"sensorless slip error vs ground truth: mean {mean:.4f}, worst {worst:.4f}")
    print(f"\ncite: {info.citation}")
    return 0 if correct == total else 1


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if Path(a).exists()]
    raise SystemExit(main(args))
