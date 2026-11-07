"""MuJoCo physical benchmark: a motor-driven pusher and sliding payload.

This is a deliberately restricted one-axis contact problem, not a VLA or a
humanoid. The payload slider has damping, not realistic Coulomb floor friction.
All trials start from independent MuJoCo states for reproducibility.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Lazy import: original toy VLA Council remains usable without MuJoCo.
try:
    import mujoco
except ImportError:
    mujoco = None


XML = '''<mujoco model="causalforge_push">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <geom name="visual_rail" type="box" size="0.85 0.09 0.005" pos="0.25 0 0.005" rgba="0.6 0.65 0.7 1" contype="0" conaffinity="0"/>
    <body name="pusher" pos="-0.12 0 0.07">
      <joint name="pusher_x" type="slide" axis="1 0 0" limited="true" range="-0.2 1.05" damping="0.15"/>
      <geom name="pusher_geom" type="box" size="0.025 0.06 0.045" mass="0.5" rgba="0.2 0.45 0.9 1" friction="0.8 0.005 0.0001"/>
    </body>
    <body name="payload" pos="0.12 0 0.07">
      <joint name="payload_x" type="slide" axis="1 0 0" limited="true" range="-0.1 1.05" damping="0.8"/>
      <geom name="payload_geom" type="box" size="0.04 0.06 0.045" mass="0.5" rgba="0.95 0.5 0.2 1" friction="0.8 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="pusher_motor" joint="pusher_x" gear="1" ctrllimited="true" ctrlrange="-12 12"/>
  </actuator>
</mujoco>'''


@dataclass(frozen=True)
class Physics:
    motor_scale: float = 1.0
    payload_mass: float = 0.5


HYPOTHESES = {
    'weak_motor': Physics(motor_scale=0.42, payload_mass=0.5),
    'heavy_payload': Physics(motor_scale=1.0, payload_mass=3.0),
}


@dataclass(frozen=True)
class Result:
    pusher_displacement: float
    payload_displacement: float
    max_abs_ctrl: float

    def sensors(self) -> np.ndarray:
        # Two simulated encoder readings. A real system needs sensor noise and delay.
        return np.array([self.pusher_displacement, self.payload_displacement], dtype=float)


def run(physics: Physics, force: float, duration: float) -> Result:
    if mujoco is None:
        raise RuntimeError('MuJoCo is not installed: pip install mujoco')
    if duration <= 0 or abs(force) > 12 or physics.motor_scale <= 0 or physics.payload_mass <= 0:
        raise ValueError('invalid simulation parameters')
    m = mujoco.MjModel.from_xml_string(XML)
    m.actuator_gear[0, 0] = physics.motor_scale
    m.body_mass[m.body('payload').id] = physics.payload_mass
    d = mujoco.MjData(m)
    # Body masses and actuator gears are modified after compilation. Refresh
    # mass-related model constants before the first dynamics evaluation.
    mujoco.mj_setConst(m, d)
    mujoco.mj_forward(m, d)
    j_pusher = int(m.joint('pusher_x').qposadr)
    j_payload = int(m.joint('payload_x').qposadr)
    initial = d.qpos[[j_pusher, j_payload]].copy()
    for _ in range(round(duration/m.opt.timestep)):
        d.ctrl[0] = force
        mujoco.mj_step(m, d)
    movement = d.qpos[[j_pusher, j_payload]] - initial
    return Result(float(movement[0]), float(movement[1]), abs(force))


def predict(hypothesis: str, probe) -> np.ndarray:
    return run(HYPOTHESES[hypothesis], probe.force, probe.duration).sensors()


def select_command(physics: Physics, target: float = 0.14) -> tuple[float, dict[str, float]]:
    """Choose from three synthetic teacher controls using simulated outcomes.

    This uses no VLA weights, no student training, and no real-world commands.
    Safety/risk is not established by this simple target-based score.
    """
    candidates = {'cautious': 2.5, 'nominal': 4.0, 'strong': 7.0}
    horizon = 0.42
    scores = {}
    for teacher, command in candidates.items():
        sim = run(physics, command, horizon)
        scores[teacher] = abs(target-sim.payload_displacement) + max(0.0, sim.payload_displacement-target)*0.5
    winner = min(scores, key=scores.get)
    return candidates[winner], scores
