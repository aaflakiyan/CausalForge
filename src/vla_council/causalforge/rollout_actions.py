"""Replay validated recorded policy actuator commands in the MuJoCo pusher.

The interface is an integration test, NOT pretrained VLA inference.
Controls are dimensionless actuator values, not forces in newtons.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .mujoco_push import XML, Physics, mujoco
from .vla_interface import ActionChunk, ActionSpec

PUSHER_SPEC = ActionSpec(
    robot='causalforge_pusher', representation='motor_control',
    ordered_axes=('pusher_motor',), units=('actuator_ctrl',),
    frame='actuator', dt=0.02,
)


def parse_record(path: Path) -> ActionChunk:
    data = json.loads(path.read_text(encoding='utf-8'))
    try:
        raw = data['action_spec']
        spec = ActionSpec(raw['robot'], raw['representation'], tuple(raw['ordered_axes']),
                          tuple(raw['units']), raw['frame'], float(raw['dt']))
        chunk = ActionChunk(np.asarray(data['actions'], dtype=float), spec, str(data['model_name']))
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing or malformed action record') from exc
    if spec != PUSHER_SPEC:
        raise ValueError('Incompatible action representation: explicit validated conversion required')
    if not np.all((chunk.actions >= -12) & (chunk.actions <= 12)):
        raise ValueError('Pusher motor commands must be in [-12,12]')
    return chunk


def execute(chunk: ActionChunk, physics: Physics = Physics()) -> dict:
    if mujoco is None:
        raise RuntimeError("MuJoCo required: pip install -e '.[mujoco]'")
    if chunk.spec != PUSHER_SPEC:
        raise ValueError('Action specification incompatible with the MuJoCo pusher')
    if not np.isfinite(chunk.actions).all() or np.any(np.abs(chunk.actions) > 12):
        raise ValueError('Invalid actuator commands')
    if not np.isfinite(physics.payload_mass) or not np.isfinite(physics.motor_scale) or physics.payload_mass <= 0 or physics.motor_scale <= 0:
        raise ValueError('Invalid physics configuration')
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    model.actuator_gear[0, 0] = physics.motor_scale
    model.body_mass[model.body('payload').id] = physics.payload_mass
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    repeat = chunk.spec.dt / model.opt.timestep
    if abs(repeat - round(repeat)) > 1e-8:
        raise ValueError('Action period must be an integer multiple of the physics timestep')
    repeat = int(round(repeat))
    pusher_idx = int(np.asarray(model.joint('pusher_x').qposadr).item())
    payload_idx = int(np.asarray(model.joint('payload_x').qposadr).item())
    start = data.qpos[[pusher_idx, payload_idx]].copy()
    samples = []
    for action in chunk.actions:
        data.ctrl[0] = float(action[0])
        for _ in range(repeat):
            mujoco.mj_step(model, data)
        samples.append({
            'time_s': float(data.time),
            'motor_ctrl': float(action[0]),
            'pusher_position_m': float(data.qpos[pusher_idx]),
            'payload_position_m': float(data.qpos[payload_idx]),
        })
    result = data.qpos[[pusher_idx, payload_idx]] - start
    return {
        'model_name': chunk.model_name,
        'real_vla_checkpoint_verified': False,
        'model': 'causalforge_push',
        'action_spec': {
            'robot': PUSHER_SPEC.robot, 'representation': PUSHER_SPEC.representation,
            'axes': list(PUSHER_SPEC.ordered_axes), 'units': list(PUSHER_SPEC.units),
            'frame': PUSHER_SPEC.frame, 'dt': PUSHER_SPEC.dt,
        },
        'physics': {'motor_scale': physics.motor_scale, 'payload_mass': physics.payload_mass},
        'pusher_displacement_m': float(result[0]),
        'payload_displacement_m': float(result[1]),
        'samples': samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path, help='JSON containing model_name, action_spec, actions')
    parser.add_argument('--out', type=Path, default=Path('runs/policy-rollout.json'))
    parser.add_argument('--motor-scale', type=float, default=1.0)
    parser.add_argument('--payload-mass', type=float, default=0.5)
    args = parser.parse_args()
    report = execute(parse_record(args.record), Physics(args.motor_scale, args.payload_mass))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
