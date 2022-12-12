import numpy as np
import pytest as pt

from scenario_gym.scenario_gym import ScenarioGym


@pt.fixture
def scenario_path(all_scenarios):
    """Get a path for a scenario to test."""
    return all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]


def test_poses(scenario_path):
    """Test the basic pose data recoreded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.load_scenario(scenario_path)

    assert gym.state.t == 0.0
    assert gym.state.poses
    assert all(
        (
            np.allclose(
                v, (gym.state.poses[e] - gym.state.prev_poses[e]) / gym.state.dt
            )
            for e, v in gym.state.velocities.items()
        )
    ), "Velocities not correct."
    assert all(
        (len(poses) == 2 for poses in gym.state.recorded_poses().values())
    ), "Wrong number of recorded poses."

    gym.step()

    assert gym.state.poses
    assert all(
        (
            np.allclose(
                v, (gym.state.poses[e] - gym.state.prev_poses[e]) / gym.state.dt
            )
            for e, v in gym.state.velocities.items()
        )
    ), "Velocities not correct."
    assert all(
        (len(poses) == 3 for poses in gym.state.recorded_poses().values())
    ), "Wrong number of recorded poses."


def test_state_info(scenario_path):
    """Test running a scenario and getting data from the state."""
    gym = ScenarioGym(timestep=0.1)
    gym.load_scenario(scenario_path)

    for _ in range(10):
        gym.step()

    e = gym.state.scenario.entities[0]
    pose = gym.state.poses[e]
    distances = [
        np.linalg.norm(pose_[:3] - pose[:3])
        for e_, pose_ in gym.state.poses.items()
        if e_ != e
    ]

    assert (
        len(gym.state.get_entities_in_radius(*pose[:2], np.min(distances) - 0.1))
        == 1
    ) and (
        len(gym.state.get_entities_in_radius(*pose[:2], np.max(distances) + 1))
        == 1 + len(distances)
    ), "Incorrect entities returned."

    names, _ = gym.state.get_road_info_at_entity(e)
    assert "Road" in names, "Entity is on the road."


def test_step(scenario_path):
    """Test the basic pose data recoreded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.load_scenario(scenario_path)
    (ego, hazard) = gym.state.scenario.entities[:2]

    poses = {ego: np.random.randn(6)}
    current = gym.state.poses.copy()

    gym.state.next_t = 1.0
    gym.state.step(poses)

    assert np.allclose(gym.state.poses[ego], poses[ego])
    assert np.allclose(gym.state.poses[hazard], current[hazard])
    assert gym.state.t == 1.0


def test_reset(scenario_path):
    """Test the basic pose data recoreded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.load_scenario(scenario_path)

    ego = gym.state.scenario.entities[0]
    poses = gym.state.recorded_poses(ego).copy()
    assert poses.shape[0] == 2

    gym.step()
    assert gym.state.t == 0.1

    gym.reset_scenario()
    assert np.allclose(poses, gym.state.recorded_poses(ego))
    assert gym.state.t == 0.0


def test_to_scenario(all_scenarios) -> None:
    """
    Rollout a single scenario and write to a new scenario.

    Output the xosc then load it again and rollout the
    recorded version.

    """
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    # rollout
    gym = ScenarioGym()
    gym.load_scenario(scenario_path)
    gym.rollout()
    old_scenario = gym.state.scenario

    poses = gym.state.recorded_poses()[old_scenario.entities[0]]
    assert np.unique(poses, axis=0).shape[0] == poses.shape[0]
    new_scenario = gym.state.to_scenario()

    ego = new_scenario.entities[0]
    assert len(ego.trajectory.t) == ego.trajectory.data.shape[0]

    # reload and test
    traj1 = old_scenario.entities[0].trajectory
    n_entities = len(old_scenario.entities)
    n_stationary = sum(1 for t in old_scenario.trajectories.values() if len(t) == 1)

    traj2 = new_scenario.entities[0].trajectory
    assert (
        len(new_scenario.entities) == n_entities
    ), "New scenario has a different number of entities."
    assert all(
        (
            isinstance(entity, type(old_entity))
            for entity, old_entity in zip(
                old_scenario.entities, new_scenario.entities
            )
        )
    ), "Entities are not the same type."
    assert n_stationary == sum(
        1 for t in new_scenario.trajectories.values() if len(t) == 1
    ), "New scenario has a different number of stationary entities."
    assert all(
        [
            np.allclose(traj1.position_at_t(0.0), traj2.position_at_t(0.0)),
            np.allclose(traj1.position_at_t(5.0), traj2.position_at_t(5.0)),
            np.allclose(traj1.position_at_t(10.0), traj2.position_at_t(10.0)),
        ]
    ), "Recorded and true trajectories differ."
