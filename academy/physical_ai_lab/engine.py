"""MuJoCo physics engine: task-level DLS commands -> joint position servos."""
from pathlib import Path
import numpy as np
from .kinematics import LINKS, dls_velocity, forward, reachable

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "planar_arm.xml"


class ReachingLab:
    """One instance per single-user teaching session; not a multiuser service."""

    def __init__(self, target=(0.45, 0.35), seed=42):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.gain = 2.5
        self.noise = 0.0
        self.target = np.asarray(target, dtype=float)
        self.reset()

    def reset(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = [-0.65, 1.35]
        self.data.ctrl[:2] = self.data.qpos[:2]
        self.model.site_pos[0, :2] = self.target
        self.mujoco.mj_forward(self.model, self.data)
        self.history = []
        return self.snapshot()

    def configure(self, *, target=None, gain=None, noise=None):
        if target is not None:
            t = np.asarray(target, dtype=float)
            if not reachable(t):
                raise ValueError("target must be reachable: 0.1 to 0.7 m from base")
            self.target = t
            self.model.site_pos[0, :2] = t
        if gain is not None:
            if not np.isfinite(gain) or not 0.1 <= gain <= 6.0:
                raise ValueError("gain must be in [0.1, 6.0]")
            self.gain = float(gain)
        if noise is not None:
            if not np.isfinite(noise) or not 0 <= noise <= 0.06:
                raise ValueError("noise must be in [0, 0.06] metres")
            self.noise = float(noise)
        self.mujoco.mj_forward(self.model, self.data)
        return self.snapshot()

    def step(self, steps=10):
        if not isinstance(steps, int) or not 1 <= steps <= 100:
            raise ValueError("steps must be an integer between 1 and 100")
        dt = float(self.model.opt.timestep)
        for index in range(steps):
            if index % 5 == 0:
                q = self.data.qpos[:2].copy()
                # Noise perturbs the *observed tip position*, not the physics.
                observed = forward(q) + self.rng.normal(0, self.noise, size=2)
                perceived_target = self.target + (forward(q) - observed)
                # For noisy estimates a target may become unreachable; clip its radius.
                radius = np.linalg.norm(perceived_target)
                if radius > 0:
                    perceived_target *= np.clip(radius, 0.101, sum(LINKS) - 0.001) / radius
                dq = dls_velocity(q, perceived_target, gain=self.gain)
                self.data.ctrl[:2] = np.clip(q + 5*dt*dq, -3.12, 3.12)
            self.mujoco.mj_step(self.model, self.data)
        state = self.snapshot()
        self.history.append({"t": state["time_s"], "error": state["error_m"]})
        self.history = self.history[-180:]
        state["history"] = self.history
        return state

    def snapshot(self):
        tip = self.data.site("tip").xpos[:2].copy()
        q = self.data.qpos[:2].copy()
        return {
            "engine": "mujoco", "time_s": round(float(self.data.time), 4),
            "q_rad": q.tolist(), "tip_xy_m": tip.tolist(),
            "target_xy_m": self.target.tolist(),
            "elbow_xy_m": [float(LINKS[0]*np.cos(q[0])), float(LINKS[0]*np.sin(q[0]))],
            "error_m": round(float(np.linalg.norm(self.target-tip)), 6),
            "success": bool(np.linalg.norm(self.target-tip) < 0.025),
            "gain": self.gain, "noise_m": self.noise,
            "timestep_s": float(self.model.opt.timestep),
            "history": getattr(self, "history", []),
        }
