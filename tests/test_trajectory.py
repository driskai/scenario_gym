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
        Trajectory(np.array([[0, np.nan, 0]]), fields=["t", "x", "y"])
        Trajectory(np.array([[0, 0.0, np.nan, 0]]), fields=["t", "h", "x", "y"])


def test_unordered_create():
    """Test creating a trajectory with fields in a different order."""
    data = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
        ]
    )
    traj = Trajectory(data, fields=["y", "x", "t"])
    assert np.allclose(traj.t, [0.0, 1.0, 2.0, 3.0]), "t not set correctly."
    assert np.allclose(traj.x, [0.0, 0.0, 1.0, 2.0]), "x not set correctly."
    assert np.allclose(traj.y, [0.0, 0.0, 0.0, 1.0]), "y not set correctly."


def test_create_with_duplicate_nan():
    """Test creating with duplicate rows containing nan values."""
    traj = Trajectory(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, np.nan],
                [1.0, 0.0, 0.0, np.nan],
                [2.0, 0.0, 0.0, 0.0],
            ]
        ),
        fields=["t", "x", "y", "h"],
    )
    assert traj.data.shape[0] == 3, "Duplicate rows should be removed."


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
    """Test that angles are estimated when not provided."""
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


def test_position_at_t():
    """Test the position at t method."""
    data = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    traj = Trajectory(data, fields=["t", "x", "y"])
    assert np.allclose(traj.position_at_t(0.5)[:2], [0.5, 0.5]), "Should be 0.5."
    assert np.allclose(traj.position_at_t(1.5)[:2], [1.5, 1.5]), "Should be 1.5."
    assert np.allclose(traj.position_at_t(2.5)[:2], [2.0, 2.0]), "Should be 2.0."
    assert np.allclose(traj.position_at_t(-1.0)[:2], [-1.0, -1.0]), "Should be -1."
    assert traj.position_at_t(-1.0, extrapolate=False) is None, "Should be None."
    assert traj.position_at_t(3.0, extrapolate=False) is None, "Should be None."
    assert np.allclose(
        traj.position_at_t(data[:, 0])[:, :2], data[:, 1:]
    ), "Incorrect broadcasting."
    assert np.allclose(
        traj.position_at_t(np.array([-1.0, 3.0]))[:, :2],
        np.array([[-1.0, -1.0], [2.0, 2.0]]),
    ), "Incorrect extrapolation."


def test_subsample():
    """Test subsampling a trajectory."""
    traj = Trajectory(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
                [4.0, 4.0, 0.0],
            ]
        ),
        fields=["t", "x", "y"],
    )
    subsample_t = traj.subsample(points_per_t=0.5)
    assert all(
        (
            subsample_t.min_t == 0.0,
            subsample_t.max_t == 4.0,
            subsample_t.arclength == 4.0,
        )
    ), "Incorrect trajectory produced."

    subsample_s = traj.subsample(points_per_s=0.5)
    assert all(
        (
            subsample_s.min_t == 0.0,
            subsample_s.max_t == 4.0,
            subsample_s.arclength == 4.0,
        )
    ), "Incorrect trajectory produced."


def test_curvature_subsample():
    """Test subsampling a trajectory."""
    base_traj = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.9, 1.9, 0.0]],
    )
    mid_y = np.linspace(0, 2, 20)
    mid_x = 2 + np.sqrt(1 - (mid_y - 1) ** 2)
    middle = np.array([np.linspace(2.0, 4.0, 20), mid_x, mid_y]).T
    end = np.array(
        [[4.1, 1.9, 2.0], [6.0, 0.0, 2.0]],
    )
    traj = Trajectory(np.vstack([base_traj, middle, end]), fields=["t", "x", "y"])
    subsampled = traj.curvature_subsample(points_per_s=5)
    assert all(
        (
            subsampled.min_t == 0.0,
            subsampled.max_t == 6.0,
        )
    ), "Incorrect trajectory produced."
    assert (subsampled.x >= 2).sum() > 0.5 * subsampled.x.shape[
        0
    ], "More points should be in the final part of the trajectory."
