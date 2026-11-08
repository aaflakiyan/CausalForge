"""Strict native-checkpoint comparison: SmolVLA against ACT on PushT.

This RUNS actual, externally hosted pretrained checkpoints using the official
LeRobot evaluator and their native PushT simulator. No MuJoCo UR5e transfer
is claimed: PushT uses Pymunk, and its 2D action space is NOT a UR5e command.

Every participant must produce a real eval_info.json. Missing checkpoints,
failed inference or incompatible schemas fail the run, never turn into zeros.
No benchmark numbers are hard-coded. The checkpoint path and seed are saved.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

DEFAULT_MODELS = ("n3puiol/smolvla_pusht","aadarshram/act_pusht")
SUPPORTED = {
    "n3puiol/smolvla_pusht": "SmolVLA vision-language-action",
    "aadarshram/act_pusht": "ACT action-chunking imitation baseline",
    "lerobot/diffusion_pusht": "Diffusion Policy imitation baseline",
}

def parse_native_info(info:dict, n_episodes:int)->dict:
    """Require observed, finite, correctly sized native physics statistics."""
    if not isinstance(info,dict):
        raise ValueError("Expected LeRobot eval_info JSON object")
    # LeRobot 0.4.x may write stats directly or nested under overall.
    if "aggregated" in info and "per_episode" in info:
        aggregate=info["aggregated"]
        episodes=info["per_episode"]
        if len(episodes)!=n_episodes:
            raise ValueError("Native evaluator did not finish expected episode count")
        for entry in episodes:
            if not all(k in entry for k in ("seed","max_reward","success")):
                raise ValueError("Missing native episode measurements")
        data={
            "avg_max_reward":aggregate["avg_max_reward"],
            "pc_success":aggregate["pc_success"],
            "episodes":[{"seed":int(x["seed"]),"max_reward":float(x["max_reward"]),
                         "success":bool(x["success"])} for x in episodes],
        }
    elif "overall" in info:
        aggregate=info["overall"]
        # Aggregate only counts as a measured run when the evaluator reports
        # exactly the requested number of per-task episodes.
        details=info.get("per_task", [])
        if not details:
            raise ValueError("Native evaluator returned no per-task evidence")
        episodes=[]
        for task in details:
            metrics=task.get("metrics",{})
            rewards=metrics.get("max_rewards",[])
            outcomes=metrics.get("successes",[])
            if len(rewards)!=len(outcomes):
                raise ValueError("Missing task-level success or reward")
            episodes.extend([{"seed":None,"max_reward":float(r),"success":bool(s)}
                for r,s in zip(rewards,outcomes)])
        if len(episodes)!=n_episodes:
            raise ValueError("Wrong number of evaluation episodes")
        data={
            "avg_max_reward":aggregate["avg_max_reward"],
            "pc_success":aggregate["pc_success"],
            "episodes":episodes,
        }
    else:
        raise ValueError("Unknown LeRobot eval_info format")
    for key in ("avg_max_reward","pc_success"):
        value=float(data[key])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite measured result {key}")
        data[key]=value
    if data["pc_success"]<0 or data["pc_success"]>100:
        raise ValueError("Impossible success percentage")
    for x in data["episodes"]:
        if not math.isfinite(x["max_reward"]):
            raise ValueError("Non-finite episode reward")
    return data


def evaluate_native(*, models, out:Path, episodes=2, seed=42, timeout_s=1200,
                    executable="lerobot-eval") -> dict:
    if episodes<1 or episodes>1000 or timeout_s<10:
        raise ValueError("Invalid episode count or timeout")
    if len(set(models))!=len(models) or not models:
        raise ValueError("Require unique named model checkpoints")
    for model in models:
        if model not in SUPPORTED:
            raise ValueError(f"Unsupported checkpoint {model}; add a reviewed model manifest first")
    out.mkdir(parents=True,exist_ok=True)
    report=dict(
        benchmark="native_pretrained_policy_on_pusht",
        environment="LeRobot PushT (Pymunk; NOT MuJoCo UR5e)",
        episodes=episodes,seed=seed,
        runtime=dict(python=sys.version.split()[0],platform=platform.platform(),
                     evaluated_utc=datetime.now(timezone.utc).isoformat()),
        policies={}, all_models_ran=False,
        warnings=[
          "SmolVLA is a VLA; ACT and diffusion are imitation baselines, not VLAs.",
          "These checkpoints were trained on PushT and do not directly control the UR5e.",
          "Small CPU smoke tests cannot establish statistical performance superiority.",
        ],
    )
    for model in models:
        name=model.replace("/","__")
        directory=out/name
        directory.mkdir(exist_ok=True)
        command=[
            executable,f"--policy.path={model}","--env.type=pusht",
            "--eval.batch_size=1",f"--eval.n_episodes={episodes}",
            f"--seed={seed}","--policy.device=cpu","--policy.use_amp=false",
            f"--output_dir={directory.resolve()}",
        ]
        started=time.perf_counter()
        status=dict(model=model, architecture=SUPPORTED[model], command=command,
                    completed=False,checkpoint_inference=False)
        report["policies"][model]=status
        try:
            proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,timeout=timeout_s,check=False)
            status["exit_code"]=proc.returncode
            (directory/"native_log.txt").write_text(proc.stdout,encoding="utf-8")
            if proc.returncode:
                raise RuntimeError("Native LeRobot evaluator failed; inspect native_log.txt")
            resultfile=directory/"eval_info.json"
            if not resultfile.is_file():
                raise RuntimeError("Native checkpoint evaluation produced no eval_info.json")
            stats=parse_native_info(json.loads(resultfile.read_text()),episodes)
            status.update(stats)
            status["completed"]=True
            status["checkpoint_inference"]=True
        except (OSError,RuntimeError,subprocess.TimeoutExpired,ValueError,KeyError) as exc:
            status["error"]=str(exc)
        finally:
            status["elapsed_s"]=round(time.perf_counter()-started,2)
            (out/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    report["all_models_ran"]=all(x["completed"] for x in report["policies"].values())
    if report["all_models_ran"]:
        rewards={m:x["avg_max_reward"] for m,x in report["policies"].items()}
        report["best_mean_overlap_model"]=max(rewards,key=rewards.get)
    (out/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",action="append",dest="models",choices=sorted(SUPPORTED),
                   help="Repeat to compare multiple checkpoints")
    p.add_argument("--episodes",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--out",type=Path,default=Path("runs/real-checkpoint-pusht"))
    p.add_argument("--timeout",type=int,default=1200)
    a=p.parse_args()
    report=evaluate_native(models=a.models or DEFAULT_MODELS,episodes=a.episodes,
                           seed=a.seed,out=a.out,timeout_s=a.timeout)
    print(json.dumps({
        "benchmark":report["benchmark"],
        "all_models_ran":report["all_models_ran"],
        "policies":{m:{k:v for k,v in x.items() if k in
            ("completed","pc_success","avg_max_reward","elapsed_s","error")}
            for m,x in report["policies"].items()},
    },indent=2))
    if not report["all_models_ran"]:
        raise SystemExit("At least one pretrained checkpoint did NOT execute; inspect the report and logs.")


if __name__=="__main__":
    main()
