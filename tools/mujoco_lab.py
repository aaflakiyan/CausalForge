#!/usr/bin/env python3
"""CausalForge MuJoCo Lab: runnable dual-UR5e scene, viewer, animation and tests.

This uses CausalForge's exact validated scene / IK / action checks rather than a
second approximate robot implementation. Tested against the pinned MuJoCo
Menagerie model bundled with the release archive.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUNDLE = ROOT.parent if (ROOT.parent / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml").is_file() else ROOT
# Work both inside a normal clone (ROOT/src) and in the downloadable offline
# bundle (ROOT/CausalForge/src).
SRC = ROOT / "src" if (ROOT / "src").is_dir() else ROOT / "CausalForge" / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


def menagerie_path(p: str | None) -> Path:
    if p:
        result = Path(p).expanduser().resolve()
    else:
        choices = [
            BUNDLE / "mujoco_menagerie",
            ROOT / "mujoco_menagerie",
            ROOT / "third_party" / "mujoco_menagerie",
            HERE.parent.parent / "third_party" / "mujoco_menagerie",
        ]
        result = next((x for x in choices if (x / "universal_robots_ur5e" / "ur5e.xml").is_file()), choices[0])
    if not (result / "universal_robots_ur5e" / "ur5e.xml").is_file():
        raise FileNotFoundError(
            f"Official Menagerie UR5e XML missing in {result}. "
            "Use the complete release bundle, or download the pinned official Menagerie model."
        )
    return result


def record_report(dest: Path, report: dict) -> None:
    from vla_council.causalforge.quickstart import html_report
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (dest / "report.html").write_text(html_report(report), encoding="utf-8")


def do_check(root: Path, out: Path, trace: bool = True) -> dict:
    from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy, evaluate
    from vla_council.causalforge.dual_ur5e_reach import load_reach_model, reach_arena
    import mujoco
    from vla_council.causalforge.dual_ur5e import ids
    model = load_reach_model(root)
    q, actuator, site = ids(model)
    if model.nq != 12 or model.nu != 12 or q.shape != (2, 6) or actuator.shape != (2, 6):
        raise AssertionError("The scene is not the genuine 12-actuator dual UR5e model")
    # Verify that the inspectable scene shipped in the kit is the same scene
    # used by our physics model builder.
    asset_path = (ROOT / "assets" / "mujoco" / "dual_ur5e_reach_arena.xml")
    if not asset_path.is_file():
        asset_path = ROOT / "CausalForge" / "assets" / "mujoco" / "dual_ur5e_reach_arena.xml"
    if asset_path.is_file() and asset_path.read_text(encoding="utf-8") != reach_arena():
        raise AssertionError("Bundled arena XML differs from the scene actually used by MuJoCo")
    report = evaluate(BuiltinReachPolicy(model), menagerie=root, record_trace=trace)
    record_report(out, report)
    print("Actual dual-UR5e MuJoCo task:", "PASS" if report["task_success"] else "FAIL")
    print("  Joint controls:", model.nu, " | scene timesteps:", model.opt.timestep)
    print("  Physical end-effector errors (mm):", [round(1000 * x, 2) for x in report["tool_error_m"]])
    print("  Arm contacts:", report["inter_arm_contact_steps"],
          "| Table contacts:", report["table_contact_steps"])
    print("  Minimum inter-tool site distance (m):", round(report["minimum_tool_site_separation_m"], 4))
    print("  HTML report:", (out / "report.html").resolve())
    if not report["task_success"]:
        raise SystemExit("Physical reach acceptance tests failed; see metrics.json")
    return report


def do_render(root: Path, out: Path, *, duration: float = 3.0) -> None:
    from vla_council.causalforge.dual_ur5e_reach import load_reach_model, execute
    import imageio.v2 as imageio
    if not duration > 0:
        raise ValueError("Duration must be positive")
    out.mkdir(parents=True, exist_ok=True)
    model = load_reach_model(root)
    report, frames = execute(model, duration=duration, render=True, record_trace=True)
    if len(frames) < 10:
        raise AssertionError("MuJoCo renderer produced too few frames")
    imageio.mimsave(out / "dual_ur5e_reach.gif", frames, duration=duration/len(frames), loop=0)
    (out / "render_metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Actual MuJoCo frames:", len(frames))
    print("  GIF:", (out / "dual_ur5e_reach.gif").resolve())
    print("  Rendered tool errors (mm):", [round(1000*x,2) for x in report["final_goal_errors_m"]])
    if not all(x < .07 for x in report["final_goal_errors_m"]):
        raise SystemExit("Rendered controller did not reach its targets")


def do_view(root: Path, *, duration: float = 3.0, repeat: bool = False) -> None:
    """GUI playback with actual controller commands at real 50 Hz."""
    import mujoco
    import mujoco.viewer
    import numpy as np
    from vla_council.causalforge.dual_ur5e_policy import (
        BuiltinReachPolicy, _validate, DT,
    )
    from vla_council.causalforge.dual_ur5e_reach import (
        load_reach_model, reset, GOALS,
    )
    from vla_council.causalforge.dual_ur5e import ids
    model = load_reach_model(root)
    q, actuator, _ = ids(model)
    ref = BuiltinReachPolicy(model, duration=duration)
    data = reset(model)
    plan = ref.predict({"goal_xyz_m": GOALS, "joint_positions_rad": data.qpos[q].copy()})
    actions, substeps = _validate(plan, model, data.qpos[q].copy())
    print("Opening MuJoCo viewer. Close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Keep the exact MjData instance handed to launch_passive.
            # Reset its state in-place; never swap in a different MjData.
            mujoco.mj_resetData(model, data)
            from vla_council.causalforge.dual_ur5e import HOME
            for side in range(2):
                data.qpos[q[side]] = HOME
                data.ctrl[actuator[side]] = HOME
            mujoco.mj_forward(model, data)
            for cmd in actions:
                if not viewer.is_running():
                    return
                t0=time.monotonic()
                data.ctrl[actuator.ravel()] = cmd
                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(max(0., DT - (time.monotonic() - t0)))
            if not repeat:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(.05)
                return


def do_stress(root: Path, out: Path, *, full: bool = False) -> None:
    from vla_council.causalforge.latency_audit import audit
    out.mkdir(parents=True, exist_ok=True)
    seeds=(41,42,43) if full else (42,)
    goals=4 if full else 2
    report=audit(root, seeds=seeds, goals_per_seed=goals,
                 delay_ticks=(0,2,5,10), out=out / "latency_stress.json")
    print("Latency stress run:", len(report["trials"]), "actual MuJoCo physics trials")
    for result in report["results"].values():
        print(f"  {result['delay_ms']:3d} ms: "
              f"{1000*result['mean_tool_error_m']:.2f} mm mean error, "
              f"{result['task_success_rate']*100:.1f}% success")


def do_student(root: Path, out: Path, *, full: bool = False) -> None:
    from vla_council.causalforge.ur5e_teacher_student import run
    out.mkdir(parents=True, exist_ok=True)
    result=run(root, out / "student", train=10 if full else 4,
               test=4 if full else 2, seed=42)
    print("Student trained and evaluated on held-out MuJoCo goals:")
    print("  Success rate:", result["held_out_student_success_rate"])
    print("  Mean error (mm):", 1000*result["held_out_student_mean_goal_error_m"])


def do_unittests(root: Path) -> None:
    import subprocess
    repo = ROOT if (ROOT / "tests").is_dir() else ROOT / "CausalForge"
    env=os.environ.copy()
    env["CAUSALFORGE_MENAGERIE"]=str(root)
    # Restrict to the tested, real-MuJoCo smoke and safety contract tests:
    for name in ("test_dual_ur5e_reach.py","test_dual_ur5e_policy.py",
                 "test_latency_audit.py"):
        print("Running:", name, flush=True)
        subprocess.run([
            sys.executable, "-m", "unittest", "discover",
            "-s", str(repo / "tests"), "-p", name, "-v"
        ], check=True, env=env)


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check","render","view","stress","student","tests","all"))
    parser.add_argument("--menagerie", type=str, default=None,
                        help="Folder containing official universal_robots_ur5e model")
    parser.add_argument("--out", type=Path, default=BUNDLE / "results")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--repeat", action="store_true",
                        help="Loop physical motion inside interactive GUI viewer")
    parser.add_argument("--full", action="store_true",
                        help="Use more independent seeds and a longer training trial")
    args=parser.parse_args(argv)
    if args.command in ("view","render","all") and os.environ.get("MUJOCO_GL")=="osmesa" and args.command=="view":
        raise RuntimeError("GUI viewer needs a display. Unset MUJOCO_GL=osmesa before 'view'.")
    root=menagerie_path(args.menagerie)
    if args.command in ("check","all"):
        do_check(root, args.out / "reach", trace=True)
    if args.command in ("render","all"):
        do_render(root, args.out / "render", duration=args.duration)
    if args.command=="view":
        do_view(root, duration=args.duration, repeat=args.repeat)
    if args.command in ("stress","all"):
        do_stress(root, args.out / "stress", full=args.full)
    if args.command in ("student","all"):
        do_student(root, args.out / "distillation", full=args.full)
    if args.command=="tests":
        do_unittests(root)
    print("Finished:", args.command)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
