"""Command-line interface. Install with ``pip install "statorscope[cli]"``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ModuleNotFoundError as exc:  # pragma: no cover - import-guard only
    raise SystemExit(
        "statorscope CLI needs extra packages. Install with:\n\n"
        '    pip install "statorscope[cli]"\n'
    ) from exc

from . import __version__
from .detect import diagnose as run_diagnose
from .quality import TrustLevel, assess_clock
from .signals import Motor, Recording
from .synth import synthesize

app = typer.Typer(
    name="statorscope",
    help="Motor Current Signature Analysis that knows when to refuse.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_TRUST_STYLE = {
    TrustLevel.GOOD: "bold green",
    TrustLevel.MARGINAL: "bold yellow",
    TrustLevel.UNRELIABLE: "bold red",
    TrustLevel.UNKNOWN: "bold blue",
}


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"statorscope {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Motor Current Signature Analysis that knows when to refuse."""


@app.command()
def analyse(
    path: Annotated[Path, typer.Argument(help="Whitespace/CSV log of [time, R, S, T].")],
    pole_pairs: Annotated[int, typer.Option("--pole-pairs", "-p", help="Pole PAIRS, not poles.")],
    rotor_bars: Annotated[int | None, typer.Option("--rotor-bars", "-b")] = None,
    line_hz: Annotated[float, typer.Option("--line-hz", "-f", help="50 or 60.")] = 50.0,
    slip: Annotated[
        float | None, typer.Option("--slip", help="Known slip; else estimated.")
    ] = None,
    time_column: Annotated[int, typer.Option("--time-column")] = 0,
    time_unit: Annotated[str, typer.Option("--time-unit", help="s, ms or us.")] = "ms",
    fs: Annotated[float | None, typer.Option("--fs", help="Override sample rate.")] = None,
    phase: Annotated[str, typer.Option("--phase", help="R, S or T.")] = "R",
    no_calibrate: Annotated[bool, typer.Option("--no-calibrate")] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Report findings even if the clock audit fails.")
    ] = False,
) -> None:
    """Diagnose a recording and print a full report."""
    if not path.exists():
        console.print(f"[bold red]No such file:[/] {path}")
        raise typer.Exit(code=2)

    motor = Motor(pole_pairs=pole_pairs, rotor_bars=rotor_bars, line_hz=line_hz)
    rec = Recording.from_text(
        path,
        time_column=time_column,
        time_unit=time_unit,  # type: ignore[arg-type]
        fs=fs,
    )
    report = run_diagnose(rec, motor, slip=slip, calibrate=not no_calibrate, phase=phase)

    style = _TRUST_STYLE[report.clock.verdict]
    console.print(
        Panel(
            report.clock.explain(),
            title=f"[{style}]clock: {report.clock.verdict.upper()}[/]",
            border_style=style,
        )
    )
    if report.grid is not None:
        console.print(Panel(report.grid.explain(), title="calibration", border_style="cyan"))

    console.print(
        f"\n[bold]slip[/] {report.slip.slip:.4f}  "
        f"([bold]{report.slip.rpm:.0f} rpm[/])  "
        f"{'confident' if report.slip.confident else '[yellow]low confidence[/]'}\n"
    )

    table = Table(title="fault signatures", header_style="bold")
    table.add_column("mechanism")
    table.add_column("verdict")
    table.add_column("severity")
    table.add_column("strongest")
    for fault in report.faults:
        table.add_row(
            fault.kind,
            "[bold red]DETECTED[/]" if fault.detected else "[green]clear[/]",
            fault.severity,
            f"{fault.strongest_dbc:+.1f} dBc" if fault.detected else "-",
        )
    console.print(table)

    if report.clock.verdict is TrustLevel.UNRELIABLE and not force:
        console.print(
            Panel(
                "The audit failed: the measured noise floor sits above the level a real "
                "fault would produce, so nothing above can be distinguished from "
                "acquisition noise.\n\nFix the acquisition, or re-run with --force to "
                "see the findings anyway (they are not evidence).",
                title="[bold red]UNSUPPORTED[/]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(f"\n[{style}]verdict:[/] " + ("faults found" if report.supported else "clear"))


@app.command()
def audit(
    path: Annotated[Path, typer.Argument(help="Recording to audit.")],
    line_hz: Annotated[float, typer.Option("--line-hz", "-f")] = 50.0,
    time_column: Annotated[int, typer.Option("--time-column")] = 0,
    time_unit: Annotated[str, typer.Option("--time-unit")] = "ms",
) -> None:
    """Audit a recording's sample clock without running detection.

    Use this before you trust anything else: it answers whether the data can
    support a fault claim at all.
    """
    if not path.exists():
        console.print(f"[bold red]No such file:[/] {path}")
        raise typer.Exit(code=2)

    rec = Recording.from_text(path, time_column=time_column, time_unit=time_unit)  # type: ignore[arg-type]
    q = assess_clock(rec, line_hz=line_hz)
    style = _TRUST_STYLE[q.verdict]
    console.print(Panel(q.explain(), title=f"[{style}]{q.verdict.upper()}[/]", border_style=style))
    raise typer.Exit(code=0 if q.trustworthy else 1)


@app.command()
def demo(
    fault_dbc: Annotated[
        float, typer.Option("--fault", help="Inject a broken bar at this dBc level.")
    ] = -42.0,
    healthy: Annotated[bool, typer.Option("--healthy", help="Inject no fault at all.")] = False,
    jitter: Annotated[
        bool, typer.Option("--jitter", help="Simulate a millis()-logged Arduino clock.")
    ] = False,
) -> None:
    """Run the pipeline on synthetic data with known ground truth.

    ``statorscope demo --healthy --jitter`` reproduces the false positive this
    library exists to prevent: a healthy machine that naive MCSA calls faulty.
    """
    motor = Motor(pole_pairs=2, rotor_bars=28, line_hz=50.0)
    kwargs: dict[str, object] = {
        "slip": 0.03,
        "broken_bar_dbc": None if healthy else fault_dbc,
    }
    if jitter:
        kwargs |= {
            "fs": 620.0,
            "duration_s": 52.0,
            "jitter_rms_s": 0.5e-3,
            "jitter_model": "interval",
            "timestamp_resolution_s": 1e-3,
        }
    else:
        kwargs |= {"fs": 5000.0, "duration_s": 20.0}

    rec, truth = synthesize(motor, **kwargs)  # type: ignore[arg-type]
    injected = "healthy" if truth.healthy else f"broken bar @ {truth.broken_bar_dbc} dBc"
    console.print(
        Panel(
            f"injected: {injected}"
            f"\nslip: {truth.slip}  ({truth.rotor_rpm:.0f} rpm)"
            f"\nclock: {'millis() jitter' if jitter else 'clean'}",
            title="ground truth",
            border_style="magenta",
        )
    )
    console.print(run_diagnose(rec, motor).summary())


if __name__ == "__main__":  # pragma: no cover
    app()
