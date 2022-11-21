import numpy as np
import pytest as pt

from scenario_gym.trajectory import Trajectory, _resolve_heading


def test_trajectory():
    """Test that trajectories are constructed correctly."""
    data = np.repeat(np.arange(10, dtype=np.float32)[:, None], 4, axis=1)  # (10, 4)
    traj = Trajectory(data, fields=["t", "x", "y", "h"])
    assert traj.h is not None
    assert traj.max_t == 9, f"Incorrect max time: {traj.max_t}."
    assert np.allclose(
        traj.arclength, 9 * np.sqrt(2)
    ), f"Incorrect arclength: {traj.arclength}."


def test_invalid_create():
    """Test that we cannot create without txy or with an invalid shape."""
    with pt.raises(ValueError):
        Trajectory(np.empty((3, 2)), fields=["x", "y"])
        Trajectory(np.empty((3, 2)), fields=["t", "y"])
        Trajectory(np.empty((3, 2)), fields=["t", "x"])
        Trajectory(np.empty((2)))
        Trajectory(np.empty((2, 2, 2)))
        Trajectory(np.empty((2, 4)), fields=["t", "x", "y"])
        Trajectory(np.empty((2, 2)))


def test_filled_headings():
    """Test that headings are estimated when not provided."""
    data = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )
    traj = Trajectory(data, fields=["t", "x", "y"])
    true_h = np.ones(3) * np.pi / 4
    assert np.allclose(traj.h, true_h), "Headings should be estimated from xy."


def test_filled_zrp():
    """Test that headings are estimated when not provided."""
    data = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )
    traj = Trajectory(data, fields=["t", "x", "y"])
    assert np.allclose(traj.z, np.zeros(3)), "z not filled correctly."
    assert np.allclose(traj.z, np.zeros(3)), "r not filled correctly."
    assert np.allclose(traj.z, np.zeros(3)), "p not filled correctly."


def test_single_cp():
    """Test loading a trajectory with 1 control point."""
    data = np.array(
        [
            [
                0.0,
                1.0,
                1.0,
            ]
        ]
    )
    traj = Trajectory(data, fields=["t", "x", "y"])
    assert np.allclose(traj.h, np.zeros(1)), "Should be assigned 0 heading."
    assert np.allclose(
        traj.position_at_t(10.0)[:2], np.ones(2)
    ), "Should stay fixed."
    assert np.allclose(traj.max_t, 0.0), "Max t should be 0."


def test_resolve_heading():
    """Test that resolve headings correctly reduces the gap between headings."""
    data = np.random.randn(100)
    resolved = _resolve_heading(data)

    delta = (data - resolved) % (2 * np.pi)
    delta = np.where(delta >= np.pi, delta - 2 * np.pi, delta)
    assert np.allclose(
        delta, np.zeros(data.shape[0])
    ), "Difference should be 0 modulo 2pi."
    assert (
        np.abs(np.diff(delta)).sum() <= np.abs(np.diff(data)).sum()
    ), "Stepwise distance should be reduced."


def test_immutable_traj():
    """Test that trajectories are copied correctly."""
    data = np.repeat(np.arange(10, dtype=np.float32)[:, None], 4, axis=1)  # (10, 4)
    traj = Trajectory(data, fields=["t", "x", "y", "h"])

    with pt.raises(ValueError):
        traj.data[-1][0] = 11


def test_copy_traj():
    """Test copying a trajectory."""
    data = np.repeat(np.arange(10, dtype=np.float32)[:, None], 4, axis=1)  # (10, 4)
    traj = Trajectory(data, fields=["t", "x", "y", "h"])

    traj_new = traj.copy()
    assert id(traj) != id(traj_new), "Should have different memory."
    assert id(traj.data) != id(traj_new.data), "Should have different memory."
    assert np.allclose(traj.data, traj_new.data), "Should have equal data."
