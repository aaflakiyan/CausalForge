"""Diagnostics for dual UR5e workspace and motion adequacy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .dual_ur5e import load_model, initial, ids, targets, rollout
def main():
    import argparse, mujoco
    p=argparse.ArgumentParser()
    p.add_argument("--menagerie",type=Path,required=True)
    a=p.parse_args()
    model=load_model(a.menagerie)
    data=initial(model)
    q, act, sites=ids(model)
    goal,xyz=targets(model,.26)
    report={"home_sites":data.site_xpos[sites].tolist(),"goal_sites":xyz.tolist(),
    "distance_home_to_goal_m":np.linalg.norm(xyz-data.site_xpos[sites],axis=1).tolist(),
    "home_q":data.qpos[q].tolist(),"base_names":["left_","right_"]}
    for name in ["synchronous","compensate_right"]:
        outcome,_=rollout(model,name,goal,xyz,right_scale=1.0,duration=1.5)
        report[name]=outcome
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
