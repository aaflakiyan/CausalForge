"""CausalForge command-line entry point: reproducible CPU MuJoCo smoke test.

Installation: pip install -e '.[mujoco]'
Execution:    causalforge demo
              causalforge demo --menagerie /path/to/mujoco_menagerie

Upstream UR5e meshes are cached from their pinned source revision.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from html import escape
from pathlib import Path

MENAGERIE_REVISION = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def locate_menagerie(path: Path | None = None, cache: Path | None = None) -> Path:
    """Fetch official pinned Menagerie model once; reuse local cache afterwards."""
    root = Path(path).expanduser().resolve() if path is not None else (
        Path(cache).expanduser().resolve() if cache is not None
        else (Path.home()/".cache"/"causalforge"/"mujoco_menagerie").resolve()
    )
    xml = root/"universal_robots_ur5e"/"ur5e.xml"
    if path is not None:
        if not xml.is_file():
            raise FileNotFoundError(f"UR5e MJCF missing: {xml}")
        return root
    if xml.is_file():
        revision = root/".git"
        if not revision.exists():
            raise RuntimeError(f"Cached robot model has no source revision: {root}")
        result = subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
                                capture_output=True,text=True,check=True)
        if result.stdout.strip() != MENAGERIE_REVISION:
            raise RuntimeError("Cached Menagerie revision differs from required pin; remove cache to redownload")
        return root
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Non-empty asset cache without UR5e model: {root}")
    root.parent.mkdir(parents=True,exist_ok=True)
    def run(*args):
        try:
            subprocess.run(list(args),check=True,stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,text=True,timeout=180)
        except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Could not fetch pinned UR5e model: {exc}. "
                "You can download MuJoCo Menagerie yourself and pass --menagerie PATH."
            ) from exc
    run("git","clone","--no-checkout","--filter=blob:none",MENAGERIE_URL,str(root))
    run("git","-C",str(root),"sparse-checkout","init","--cone")
    run("git","-C",str(root),"sparse-checkout","set","universal_robots_ur5e")
    run("git","-C",str(root),"fetch","--depth","1","origin",MENAGERIE_REVISION)
    run("git","-C",str(root),"checkout","--detach","FETCH_HEAD")
    if not xml.is_file():
        raise FileNotFoundError(f"Downloaded UR5e model missing at {xml}")
    return root


def html_report(report: dict) -> str:
    """Self-contained human-readable report from actual MuJoCo metrics."""
    errors = report["tool_error_m"]
    motion = report["tool_displacement_m"]
    passed = bool(report["task_success"])
    status = "PASS" if passed else "FAIL"
    cells = "".join(
        f"<tr><td>{escape(name)}</td><td>{value:.2f}</td></tr>"
        for name, value in (
            ("Left tool error (mm)", 1000 * errors[0]),
            ("Right tool error (mm)", 1000 * errors[1]),
            ("Left tool displacement (mm)", 1000 * motion[0]),
            ("Right tool displacement (mm)", 1000 * motion[1]),
            ("Arm–arm contact steps", report["inter_arm_contact_steps"]),
            ("Arm–table contact steps", report["table_contact_steps"]),
            ("Validated action samples", report["actions_checked_before_execution"]),
            ("Physics substeps per sample", report["physics_substeps_per_action"]),
        )
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>CausalForge | Verified dual UR5e run</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:850px;margin:5vh auto;"
        "padding:0 22px;background:#101827;color:#f2f6fc}h1{margin-bottom:4px}"
        ".muted{color:#a7b4c6}.card{background:#1b283a;border:1px solid #35445a;"
        "border-radius:12px;padding:22px;margin:24px 0}table{border-collapse:collapse;"
        "width:100%}td{padding:12px 8px;border-bottom:1px solid #35445a}"
        "td:last-child{text-align:right;font-variant-numeric:tabular-nums}"
        ".status{font-weight:700;color:" + ("#75da99" if passed else "#ff8c8c") + "}"
        "</style></head><body><h1>CausalForge</h1>"
        "<p class='muted'>Actual MuJoCo simulation • official dual UR5e • absolute joint control</p>"
        f"<div class='card'><h2>Reach validation: <span class='status'>{status}</span></h2>"
        "<p>This result validates the simulated joint-policy command boundary only, "
        "not real-hardware safety or pretrained VLA inference.</p></div>"
        "<div class='card'><h2>Measured physical results</h2><table>"
        + cells + "</table></div>"
        "<p class='muted'>Raw evidence: metrics.json • full policy metadata and optional trace</p>"
        "</body></html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalforge",
                                     description="Verified Physical AI simulation runner")
    sub = parser.add_subparsers(dest="command",required=True)
    demo = sub.add_parser("demo",help="Run and validate a real dual UR5e MuJoCo reach")
    demo.add_argument("--menagerie",type=Path,help="Use an existing MuJoCo Menagerie checkout")
    demo.add_argument("--cache",type=Path,help="Change the automatic robot-asset cache")
    demo.add_argument("--out",type=Path,default=Path("runs/quickstart"))
    demo.add_argument("--trace",action="store_true",help="Save each measured control step")
    vla_demo = sub.add_parser(
        "demo-vlas",
        help="Run three MOCK VLA inference interfaces against real MuJoCo UR5e physics"
    )
    vla_demo.add_argument("--menagerie",type=Path)
    vla_demo.add_argument("--cache",type=Path)
    vla_demo.add_argument("--out",type=Path,default=Path("runs/vla-bridge-demo"))
    distill = sub.add_parser(
        "distill-demo", help="Train a SYNTHETIC multi-teacher, multi-embodiment student without GPU"
    )
    distill.add_argument("--out",type=Path,default=Path("runs/universal-student"))
    distill.add_argument("--episodes",type=int,default=80)
    distill.add_argument("--seed",type=int,default=42)
    native = sub.add_parser(
        "eval-pretrained",
        help="Run REAL pre-trained SmolVLA versus ACT in their native PushT simulator"
    )
    native.add_argument("--episodes",type=int,default=2)
    native.add_argument("--seed",type=int,default=42)
    native.add_argument("--out",type=Path,default=Path("runs/real-checkpoint-pusht"))
    native.add_argument("--timeout",type=int,default=1200)
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .dual_ur5e_policy import BuiltinReachPolicy,evaluate
        from .dual_ur5e_reach import load_reach_model
        root = locate_menagerie(args.menagerie,args.cache)
        model = load_reach_model(root)
        report = evaluate(BuiltinReachPolicy(model),menagerie=root,record_trace=args.trace)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        (args.out/"report.html").write_text(html_report(report),encoding="utf-8")
        print("CausalForge dual-UR5e / reach")
        print(f"  12 joints validated | {report['actions_checked_before_execution']} command samples")
        print(f"  Tool errors: {[round(x*1000,2) for x in report['tool_error_m']]} mm")
        print(f"  Contacts: arm={report['inter_arm_contact_steps']}, table={report['table_contact_steps']}")
        print(f"  Result: {'PASS' if report['task_success'] else 'FAIL'}")
        print(f"  Evidence: {args.out/'metrics.json'}")
        print(f"  Report: {args.out/'report.html'}")
        return 0 if report["task_success"] else 1
    if args.command == "demo-vlas":
        from .vla_bridge_demo import run
        root = locate_menagerie(args.menagerie, args.cache)
        report = run(root)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        ok = not report["rejected"] and all(v["task_success"] for v in report["teachers"].values())
        print("CausalForge multi-VLA protocol compatibility (MOCK PROVIDERS, no pretrained inference)")
        for name, result in report["teachers"].items():
            print(f"  {name}: {'PASS' if result['task_success'] else 'FAIL'}"
                  f", max goal error {max(result['tool_error_m'])*1000:.2f} mm")
        if report["rejected"]:
            print(f"  Rejected: {report['rejected']}")
        print(f"  Evidence: {args.out/'metrics.json'}")
        return 0 if ok else 1
    if args.command == "distill-demo":
        from .universal_demo import run
        report=run(args.out,seed=args.seed,n=args.episodes)
        print("CausalForge UNIVERSAL student demo (SYNTHETIC teachers, NO pretrained models)")
        for robot,score in report["scores"]["per_embodiment"].items():
            print(f"  {robot}: held-out teacher-label MSE {score['mean_mse']:.6f}")
        print(f"  Saved: {args.out/'student.json'}")
        print(f"  Evidence: {args.out/'report.json'}")
        return 0
    if args.command == "eval-pretrained":
        from .native_vla_benchmark import evaluate_native, DEFAULT_MODELS
        result = evaluate_native(
            models=DEFAULT_MODELS,episodes=args.episodes,seed=args.seed,
            out=args.out,timeout_s=args.timeout)
        print("REAL checkpoint native PushT comparison (not UR5e MuJoCo)")
        for model,row in result["policies"].items():
            print(f"  {model}: "
                  + (f"{row['pc_success']:.1f}% success; mean overlap {row['avg_max_reward']:.3f}"
                     if row["completed"] else f"FAILED: {row.get('error','unknown')}"))
        print(f"  Evidence: {args.out/'comparison.json'}")
        if not result["all_models_ran"]:
            raise SystemExit("A native checkpoint failed. Review evidence; no fake score substituted.")
        return 0
    return 2

if __name__=="__main__":
    raise SystemExit(main())
