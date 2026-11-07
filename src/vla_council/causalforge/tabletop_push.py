"""3D tabletop contact simulation with gravity and a free-body payload.

Standalone research environment; intentionally not yet used to train a VLA.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .mujoco_push import mujoco

SCENE_PATH = Path(__file__).resolve().parents[3] / 'assets' / 'mujoco' / 'tabletop_push.xml'


def simulate(force: float = 7.0, friction: float = 0.8, payload_mass: float = 0.5,
             duration: float = 1.0, stride: int = 10) -> tuple[list[dict], dict]:
    if mujoco is None:
        raise RuntimeError("MuJoCo is required: pip install -e '.[mujoco]'")
    values = [force, friction, payload_mass, duration]
    if not np.all(np.isfinite(values)) or abs(force) > 12 or friction <= 0 or payload_mass <= 0 or duration <= 0 or stride < 1:
        raise ValueError('Invalid physical parameters or sampling interval')
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    model.geom_friction[model.geom('floor').id, 0] = friction
    model.body_mass[model.body('payload').id] = payload_mass
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    pusher_q = int(np.asarray(model.joint('pusher_x').qposadr).item())
    payload_q = int(np.asarray(model.joint('payload_free').qposadr).item())
    initial_x = float(data.qpos[payload_q])
    steps = round(duration / model.opt.timestep)
    if steps < 1:
        raise ValueError('Duration shorter than one physics step')
    samples = []
    for step in range(steps + 1):
        if step % stride == 0 or step == steps:
            samples.append({
                'time_s': float(data.time), 'pusher_x_m': float(data.qpos[pusher_q]),
                'payload_x_m': float(data.qpos[payload_q]),
                'payload_y_m': float(data.qpos[payload_q + 1]),
                'payload_z_m': float(data.qpos[payload_q + 2]),
                'contact_pairs': int(data.ncon), 'motor_ctrl': float(force),
            })
        if step < steps:
            data.ctrl[0] = force
            mujoco.mj_step(model, data)
    return samples, {
        'environment': 'causalforge_tabletop_push',
        'scene_file': 'assets/mujoco/tabletop_push.xml',
        'gravity_m_s2': -9.81,
        'floor_friction_sliding_coefficient': friction,
        'payload_mass_kg': payload_mass,
        'motor_command': force,
        'duration_s': float(data.time),
        'payload_displacement_x_m': float(data.qpos[payload_q]) - initial_x,
        'final_payload_xyz_m': [float(x) for x in data.qpos[payload_q:payload_q + 3]],
        'samples': len(samples),
        'limitations': 'One actuated pusher; no grasping robot, cameras or learned VLA policy',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', type=float, default=7.0)
    parser.add_argument('--friction', type=float, default=0.8)
    parser.add_argument('--payload-mass', type=float, default=0.5)
    parser.add_argument('--duration', type=float, default=1.0)
    parser.add_argument('--out', type=Path, default=Path('runs/tabletop'))
    args = parser.parse_args()
    rows, report = simulate(args.force, args.friction, args.payload_mass, args.duration)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
