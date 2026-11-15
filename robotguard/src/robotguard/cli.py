"""CLI entry point: 'robotguard demo' or 'robotguard check ...'."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from importlib import resources
import sys

from .core import ConfigurationError, RobotSpec, validate_trace
from .report import write_report
from .simulation import SimulationError, simulate_trace, write_simulation
from .integration import PolicyContractError
from .showcase import ComparisonError


def _load_jsonl(path: Path) -> list[dict]:
    actions = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                action = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-standard numeric literal {x}")))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            actions.append(action)
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotguard", description="Pre-flight validation for robot joint-position trajectories")
    subs = parser.add_subparsers(dest="command", required=True)
    check = subs.add_parser("check", help="Validate a JSONL joint-position action trace")
    check.add_argument("trace", type=Path)
    check.add_argument("--spec", required=True, type=Path, help="Robot limits JSON")
    check.add_argument("--html", type=Path, help="Write self-contained HTML report")
    check.add_argument("--json", dest="json_path", type=Path, help="Write machine-readable JSON report")
    subs.add_parser("demo", help="Generate the built-in unsafe-action demonstration and report")
    sim = subs.add_parser("simulate", help="Pre-flight then replay an approved trace in MuJoCo")
    sim.add_argument("trace", type=Path)
    sim.add_argument("--spec", required=True, type=Path)
    sim.add_argument("--model", required=True, type=Path, help="MuJoCo MJCF XML file")
    sim.add_argument("--out", type=Path, default=Path("robotguard-output"))
    demo_sim = subs.add_parser("mujoco-demo", help="Run the packaged real-physics MuJoCo example")
    play = subs.add_parser("playground", help="Open-source offline robot replay: measured MuJoCo physics and browser controls")
    play.add_argument("--out", type=Path, default=Path("robotguard-output"))
    demo_sim.add_argument("--out", type=Path, default=Path("robotguard-output"))
    showcase = subs.add_parser("showcase", help="Run two deterministic policies through actual MuJoCo and open an offline comparison dashboard")
    showcase.add_argument("--out", type=Path, default=Path("robotguard-showcase"))
    args = parser.parse_args(argv)
    try:
        if args.command == "showcase":
            from .showcase import run_showcase
            result = run_showcase(args.out)
            print("Policy contract -> preflight -> real MuJoCo -> measured replay: completed")
            for name, meta in result["policies"].items():
                print(f"  {name}: RMSE={meta['metrics']['rmse_tracking_error_rad']:.6f} rad; physics steps={meta['integration_steps']}")
            print(f"Rejected input: {result['rejected_violations']} diagnostic(s); not simulated")
            print(f"Dashboard: {(args.out / 'index.html').resolve()}")
            return 0
        if args.command in ("mujoco-demo", "playground"):
            assets = resources.files("robotguard").joinpath("assets")
            spec = RobotSpec.from_dict(json.loads(assets.joinpath("demo_spec.json").read_text(encoding="utf-8")))
            xml = assets.joinpath("demo_arm.xml").read_text(encoding="utf-8")
            safe = [json.loads(line) for line in assets.joinpath("demo_safe.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            unsafe = [json.loads(line) for line in assets.joinpath("demo_unsafe.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            args.out.mkdir(parents=True, exist_ok=True)
            rejected = validate_trace(spec, unsafe)
            if rejected.ok:
                raise SimulationError("Unsafe fixture unexpectedly passed; refusing to report success")
            (args.out / "rejected.json").write_text(json.dumps(rejected.as_dict(), indent=2) + "\n", encoding="utf-8")
            write_report(args.out / "rejected.html", spec, rejected)
            simulated = simulate_trace(spec, safe, xml)
            write_simulation(args.out, simulated)
            print(f"Unsafe trajectory rejected ({len(rejected.violations)} pre-flight findings); no unsafe physics run")
            print(f"Safe trajectory: {simulated['status']} in MuJoCo {simulated['mujoco_version']}")
            print(f"Reproducible reports: {args.out.resolve()}/simulation.html and rejected.html")
            if (args.out / "playground.html").is_file():
                print(f"Browser playground: {(args.out / 'playground.html').resolve()}")
            return 0 if simulated["status"] == "SIMULATION_COMPLETED" else 1
        if args.command == "simulate":
            spec = RobotSpec.from_file(args.spec)
            actions = _load_jsonl(args.trace)
            audit = validate_trace(spec, actions)
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "preflight.json").write_text(json.dumps(audit.as_dict(), indent=2) + "\n", encoding="utf-8")
            write_report(args.out / "preflight.html", spec, audit)
            if not audit.ok:
                print(f"Pre-flight FAIL: {len(audit.violations)} violation(s); MuJoCo was not invoked")
                return 1
            simulated = simulate_trace(spec, actions, args.model.read_text(encoding="utf-8"))
            write_simulation(args.out, simulated)
            print(f"Pre-flight PASS; MuJoCo {simulated['mujoco_version']}: {simulated['status']}")
            print(f"View report: {(args.out / 'simulation.html').resolve()}")
            return 0 if simulated["status"] == "SIMULATION_COMPLETED" else 1
        if args.command == "demo":
            spec = RobotSpec.from_dict({"name": "DemoArm-3", "joint_names": ["shoulder", "elbow", "wrist"],
                "position_limits": [[-2, 2], [-1.5, 1.5], [-3, 3]],
                "max_velocity": [1, 1.2, 1.5], "max_acceleration": [4, 5, 6],
                "units": "rad", "frame": "joint", "max_interval_s": 0.15})
            actions = [
                {"t": 0.0, "position": [0, 0, 0], "units": "rad", "frame": "joint"},
                {"t": 0.1, "position": [0.03, 0.04, 0.02], "units": "rad", "frame": "joint"},
                {"t": 0.2, "position": [1.8, 0.05, 0], "units": "rad", "frame": "joint"},
                {"t": 0.3, "position": [2.7, 0.1, 0.01], "units": "rad", "frame": "joint"},
                {"t": 0.4, "position": [2.8, 0.1, 0], "units": "deg", "frame": "joint"},
            ]
            output = Path("robotguard-demo.html")
            report = validate_trace(spec, actions)
            write_report(output, spec, report)
            print(f"Demo: {len(report.violations)} expected violations detected; report: {output.resolve()}")
            return 0  # Showing a deliberately invalid demo is a successful demo command.
        spec = RobotSpec.from_file(args.spec)
        actions = _load_jsonl(args.trace)
        report = validate_trace(spec, actions)
        print(f"RobotGuard {('PASS' if report.ok else 'FAIL')}: {report.samples} samples, {len(report.violations)} violation(s)")
        for issue in report.violations:
            print(f"  [{issue.code}] sample {issue.sample}: {issue.message}")
        if args.html:
            write_report(args.html, spec, report)
            print(f"HTML report: {args.html}")
        if args.json_path:
            args.json_path.parent.mkdir(parents=True, exist_ok=True)
            args.json_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
            print(f"JSON report: {args.json_path}")
        return 0 if report.ok else 1
    except (ConfigurationError, ValueError, OSError, json.JSONDecodeError, SimulationError, PolicyContractError, ComparisonError) as exc:
        print(f"RobotGuard error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
