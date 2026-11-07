"""Export and run the exact CausalForge MuJoCo 1D contact scene.

Use --render only on a machine with a working OpenGL / EGL context.
The experiment is a constrained contact diagnostic, not humanoid robotics.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .mujoco_push import XML, Physics, mujoco


def export_scene(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(XML.rstrip() + '\n', encoding='utf-8')
    return path


def simulate(physics: Physics, force: float, duration: float, stride: int = 10,
             render_dir: Path | None = None) -> tuple[list[dict], dict]:
    if mujoco is None:
        raise RuntimeError("MuJoCo is required. Install: pip install -e '.[mujoco]'")
    if not (np.isfinite(force) and np.isfinite(duration)) or abs(force) > 12 or duration <= 0:
        raise ValueError('force must be finite and within [-12,12]; duration must be positive')
    if not np.isfinite(physics.motor_scale) or not np.isfinite(physics.payload_mass) or physics.motor_scale <= 0 or physics.payload_mass <= 0:
        raise ValueError('physical parameters must be finite positive numbers')
    if stride < 1:
        raise ValueError('stride must be >= 1')
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    model.actuator_gear[0, 0] = physics.motor_scale
    model.body_mass[model.body('payload').id] = physics.payload_mass
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    pusher_q = model.joint('pusher_x').qposadr
    payload_q = model.joint('payload_x').qposadr
    pusher_v = model.joint('pusher_x').dofadr
    payload_v = model.joint('payload_x').dofadr
    steps = round(duration / model.opt.timestep)
    if steps < 1:
        raise ValueError('duration must be at least one physics step')
    initial = (float(data.qpos[pusher_q]), float(data.qpos[payload_q]))
    renderer = None
    if render_dir is not None:
        try:
            import imageio.v3 as iio
        except ImportError as e:
            raise RuntimeError('To render frames: pip install imageio') from e
        render_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=360, width=640)
    rows: list[dict] = []
    try:
        for step in range(steps + 1):
            if step % stride == 0 or step == steps:
                rows.append({
                    'time_s': round(float(data.time), 8),
                    'pusher_x_m': float(data.qpos[pusher_q]),
                    'payload_x_m': float(data.qpos[payload_q]),
                    'pusher_velocity_m_s': float(data.qvel[pusher_v]),
                    'payload_velocity_m_s': float(data.qvel[payload_v]),
                    'command_force': float(force),
                })
                if renderer is not None:
                    renderer.update_scene(data)
                    iio.imwrite(render_dir / f'frame_{step:05d}.png', renderer.render())
            if step < steps:
                data.ctrl[0] = force
                mujoco.mj_step(model, data)
    finally:
        if renderer is not None:
            renderer.close()
    summary = {
        'model': 'causalforge_push',
        'limitations': '1D slider; zero gravity; damping rather than floor friction; synthetic controls only',
        'motor_scale': physics.motor_scale,
        'payload_mass_kg': physics.payload_mass,
        'command_force': force,
        'requested_duration_s': duration,
        'simulated_duration_s': float(data.time),
        'physics_timestep_s': float(model.opt.timestep),
        'pusher_displacement_m': float(data.qpos[pusher_q]) - initial[0],
        'payload_displacement_m': float(data.qpos[payload_q]) - initial[1],
        'trajectory_points': len(rows),
    }
    return rows, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xml', type=Path, help='Export the canonical MuJoCo XML (also runs the experiment)')
    p.add_argument('--force', type=float, default=7.0)
    p.add_argument('--duration', type=float, default=0.8)
    p.add_argument('--motor-scale', type=float, default=1.0)
    p.add_argument('--payload-mass', type=float, default=0.5)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--out', type=Path, default=Path('runs/scene'))
    p.add_argument('--render', action='store_true', help='Write numbered PNG frames; requires OpenGL and imageio')
    a = p.parse_args()
    if a.xml:
        export_scene(a.xml)
    rows, summary = simulate(Physics(a.motor_scale, a.payload_mass), a.force,
                             a.duration, a.stride, a.out / 'frames' if a.render else None)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / 'trajectory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
