import importlib.util
import numpy as np
import pytest
from physical_ai_lab.kinematics import forward

mujoco_installed = importlib.util.find_spec("mujoco") is not None


@pytest.mark.skipif(not mujoco_installed, reason="Install mujoco-core to run physics integration")
def test_real_physics_and_target():
    from physical_ai_lab.engine import ReachingLab
    lab = ReachingLab()
    start = lab.snapshot()
    assert start["engine"] == "mujoco"
    assert np.allclose(start["tip_xy_m"], forward(start["q_rad"]), atol=1e-5)
    for _ in range(160):
        lab.step(10)
    end = lab.snapshot()
    assert end["time_s"] > 0
    assert np.isfinite(end["error_m"])
    assert end["error_m"] < start["error_m"]
