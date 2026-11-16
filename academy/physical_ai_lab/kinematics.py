"""Planar two-link arm geometry (SI units; joint angles in radians)."""
import numpy as np

LINKS = (0.40, 0.30)


def forward(q):
    """Cartesian end-effector [x,y] in metres."""
    q1, q2 = np.asarray(q, dtype=float)
    a, b = LINKS
    return np.array([a * np.cos(q1) + b * np.cos(q1 + q2),
                     a * np.sin(q1) + b * np.sin(q1 + q2)])


def jacobian(q):
    """2x2 matrix mapping joint velocity [rad/s] to tip velocity [m/s]."""
    q1, q2 = np.asarray(q, dtype=float)
    a, b = LINKS
    return np.array([[-a*np.sin(q1)-b*np.sin(q1+q2), -b*np.sin(q1+q2)],
                     [ a*np.cos(q1)+b*np.cos(q1+q2),  b*np.cos(q1+q2)]])


def reachable(target):
    t = np.asarray(target, dtype=float)
    if t.shape != (2,) or not np.all(np.isfinite(t)):
        return False
    r = np.linalg.norm(t)
    return abs(LINKS[0]-LINKS[1])-1e-9 <= r <= sum(LINKS)+1e-9


def dls_velocity(q, target, gain=2.5, damping=0.08, limit=2.0):
    """Damped least-squares IK velocity, bounded per joint [rad/s]."""
    if not all(np.isfinite(v) and v > 0 for v in (gain, damping, limit)):
        raise ValueError("gain, damping and limit must be finite and positive")
    t = np.asarray(target, dtype=float)
    if not reachable(t):
        raise ValueError("target must be reachable and finite")
    e = t - forward(q)
    j = jacobian(q)
    dq = j.T @ np.linalg.solve(j @ j.T + damping*damping*np.eye(2), gain*e)
    return np.clip(dq, -limit, limit)
