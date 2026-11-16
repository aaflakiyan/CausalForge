"""Headless benchmark for the interactive lab."""
import argparse
import json
from pathlib import Path
from .engine import ReachingLab


def main():
    p = argparse.ArgumentParser(description="Evaluate MuJoCo reaching across fixed targets")
    p.add_argument("--out", default="results/lab01.json")
    p.add_argument("--steps", type=int, default=1200)
    args = p.parse_args()
    if not 1 <= args.steps <= 20000:
        p.error("--steps must be between 1 and 20000")
    lab = ReachingLab()
    results = []
    for i, target in enumerate(((0.45, 0.35), (0.35, 0.4), (0.52, 0.18), (0.3, -0.35))):
        lab.configure(target=target, noise=0, gain=2.5)
        lab.reset(seed=42 + i)
        for _ in range((args.steps + 9)//10):
            lab.step(min(10, args.steps - 10*_))
        state = lab.snapshot()
        results.append({"target": target, "final_error_m": state["error_m"], "success": state["success"]})
    report = {"physics_engine": "MuJoCo", "episode_steps": args.steps, "episodes": results,
              "success_rate": sum(r["success"] for r in results)/len(results)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
