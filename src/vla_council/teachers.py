"""Synthetic specialist teachers (NOT actual GR00T/openpi/OpenVLA adapters)."""
from __future__ import annotations
import numpy as np
from .core import Observation, Proposal, bounded

class DirectTeacher:
    name = "direct"
    def propose(self, o: Observation) -> Proposal:
        return Proposal(self.name, tuple(bounded((o.gx-o.x, o.gy-o.y))),
                        "Shortest straight-line approach")

class ClearanceTeacher:
    name = "clearance"
    def propose(self, o: Observation) -> Proposal:
        pos = np.array([o.x, o.y])
        goal = np.array([o.gx, o.gy])
        centre = np.array([o.ox, o.oy])
        to_goal = goal - pos
        # Potential-field style obstacle repulsion with tangential component.
        relative = pos - centre
        dist = np.linalg.norm(relative)
        direction = to_goal / max(np.linalg.norm(to_goal), 1e-8)
        if dist < o.obstacle_radius + .28:
            radial = relative / max(dist, 1e-8)
            side = 1. if np.cross(np.r_[direction, 0.], np.r_[centre-pos, 0.])[2] <= 0 else -1.
            tangent = side * np.array([-radial[1], radial[0]])
            direction = direction + (o.obstacle_radius+.28-dist)/.28 * (1.4 * radial + .9 * tangent)
        return Proposal(self.name, tuple(bounded(direction)), "Clearance-aware direction")

class DetourTeacher:
    name = "detour"
    def propose(self, o: Observation) -> Proposal:
        # Two fixed, symmetric obstacle waypoints; choose one using current y.
        if o.x < o.ox + .15 and abs(o.y - o.oy) < o.obstacle_radius + .12:
            side = 1. if o.y >= o.oy else -1.
            waypoint = np.array([o.ox, o.oy + side * (o.obstacle_radius + .20)])
        else:
            waypoint = np.array([o.gx, o.gy])
        return Proposal(self.name, tuple(bounded(waypoint - [o.x, o.y])),
                        "Reach a clearance waypoint, then return to target")

def default_teachers():
    return [DirectTeacher(), ClearanceTeacher(), DetourTeacher()]
