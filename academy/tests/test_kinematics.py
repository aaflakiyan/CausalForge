import numpy as np
import pytest
from physical_ai_lab.kinematics import forward, jacobian, dls_velocity, reachable


def test_forward_zero():
    assert np.allclose(forward([0, 0]), [0.7, 0])


def test_forward_90():
    assert np.allclose(forward([np.pi/2, 0]), [0, 0.7], atol=1e-8)


def test_jacobian_numerical():
    q = np.array([0.3, -0.7]); eps = 1e-7
    finite_difference = np.column_stack([(forward(q + np.eye(2)[i]*eps) - forward(q - np.eye(2)[i]*eps))/(2*eps) for i in range(2)])
    assert np.allclose(jacobian(q), finite_difference, atol=1e-7)


def test_reachability():
    assert reachable([0.45, 0.35])
    assert not reachable([0.8, 0.2])
    assert not reachable([float("nan"), 0])


def test_dls_improves_local_error():
    q = np.array([-0.6, 1.4]); target = np.array([0.45, 0.35])
    dq = dls_velocity(q, target)
    assert np.linalg.norm(target-forward(q + 0.01*dq)) < np.linalg.norm(target-forward(q))


def test_bad_gain():
    with pytest.raises(ValueError):
        dls_velocity([0, 1], [0.4, 0.3], gain=0)


def test_bad_target():
    with pytest.raises(ValueError):
        dls_velocity([0, 1], [1.2, 0])
