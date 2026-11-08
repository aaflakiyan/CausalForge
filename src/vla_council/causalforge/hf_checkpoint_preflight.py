"""Inspect genuine Hugging Face VLA checkpoint configs before robot deployment.

Downloads only checkpoint config metadata; it does not run the neural model.
Determines if action dimensions and declared action mode can be fed directly to
the dual-UR5e 12-joint controller. Reject unknown encodings by default.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .dual_ur5e_policy import SPEC

def inspect_config(model_id:str, config:dict) -> dict:
    if not isinstance(config,dict): raise ValueError("Checkpoint config must be a JSON object")
    output=config.get("output_features") or {}
    action=output.get("action") or {}
    shape=action.get("shape")
    action_dims=None
    if isinstance(shape,list) and len(shape)==1 and type(shape[0]) is int:
        action_dims=shape[0]
    # Never infer joint-position control mode from action dimensionality.
    # In LeRobot tasks actions may mean absolute 2D cursor positions.
    direct_ur5e_compatible=False
    reasons=[]
    if action_dims != len(SPEC.ordered_axes):
        reasons.append(f"Action dimensions {action_dims} != dual UR5e {len(SPEC.ordered_axes)}")
    reasons.append("Checkpoint lacks verified 12 named UR5e joint axes, radians, joint frame and 20 ms contract")
    return dict(model=model_id,pretrained_config_inspected=True,
                action_dimension=action_dims,
                declared_model_type=config.get("type"),
                native_input_features=sorted((config.get("input_features") or {}).keys()),
                direct_ur5e_compatible=direct_ur5e_compatible,
                refused_reasons=reasons)

def preflight(models,output:Path):
    from huggingface_hub import hf_hub_download,HfApi
    api=HfApi()
    results={}
    for model in models:
        info=api.model_info(model)
        config_path=hf_hub_download(repo_id=model,filename="config.json",revision=info.sha)
        config=json.loads(Path(config_path).read_text())
        record=inspect_config(model,config)
        record["checkpoint_commit"]=info.sha
        results[model]=record
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(models=results),indent=2)+"\n")
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",action="append",required=True)
    p.add_argument("--out",type=Path,default=Path("runs/pretrained-preflight.json"))
    args=p.parse_args()
    results=preflight(args.model,args.out)
    print(json.dumps(results,indent=2))
    # Refusing unsafe application is expected success of this diagnostic.
if __name__=="__main__":
    main()
