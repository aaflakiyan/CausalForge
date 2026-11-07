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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="causalforge",
                                     description="Verified Physical AI simulation runner")
    sub = parser.add_subparsers(dest="command",required=True)
    demo = sub.add_parser("demo",help="Run and validate a real dual UR5e MuJoCo reach")
    demo.add_argument("--menagerie",type=Path,help="Use an existing MuJoCo Menagerie checkout")
    demo.add_argument("--cache",type=Path,help="Change the automatic robot-asset cache")
    demo.add_argument("--out",type=Path,default=Path("runs/quickstart"))
    demo.add_argument("--trace",action="store_true",help="Save each measured control step")
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .dual_ur5e_policy import BuiltinReachPolicy,evaluate
        from .dual_ur5e_reach import load_reach_model
        root = locate_menagerie(args.menagerie,args.cache)
        model = load_reach_model(root)
        report = evaluate(BuiltinReachPolicy(model),menagerie=root,record_trace=args.trace)
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print("CausalForge dual-UR5e / reach")
        print(f"  12 joints validated | {report['actions_checked_before_execution']} command samples")
        print(f"  Tool errors: {[round(x*1000,2) for x in report['tool_error_m']]} mm")
        print(f"  Contacts: arm={report['inter_arm_contact_steps']}, table={report['table_contact_steps']}")
        print(f"  Result: {'PASS' if report['task_success'] else 'FAIL'}")
        print(f"  Evidence: {args.out/'metrics.json'}")
        return 0 if report["task_success"] else 1
    return 2

if __name__=="__main__":
    raise SystemExit(main())
