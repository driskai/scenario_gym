import numpy as np
import pytest as pt

from scenario_gym.scenario.actions import UpdateStateVariableAction
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def scenario_path(all_scenarios):
    """Get a path for a scenario to test."""
    return all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]


@pt.fixture
def scenario(scenario_path):
    """Get a scenario to test."""
    s = import_scenario(scenario_path)
    action = UpdateStateVariableAction(2.0, "TestAction", "ego", {"var": 1.0})
    s.add_action(action, inplace=True)
    return s


def test_poses(scenario):
    """Test the basic pose data recoreded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)

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


def test_state_info(scenario):
    """Test running a scenario and getting data from the state."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)

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


def test_step(scenario):
    """Test the basic pose data recoreded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)
    (ego, hazard) = gym.state.scenario.entities[:2]

    poses = {ego: np.random.randn(6)}
    current = gym.state.poses.copy()

    gym.state.next_t = 1.0
    gym.state.step(poses)

    assert np.allclose(gym.state.poses[ego], poses[ego])
    assert np.allclose(gym.state.poses[hazard], current[hazard])
    assert gym.state.t == 1.0


def test_state_actions(scenario):
    """Test simulation a scenario with actions."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)
    assert not gym.state.entity_state[
        scenario.entities[0]
    ], "No actions should be applied."
    gym.rollout()
    assert (
        gym.state.entity_state[scenario.entities[0]]["var"] == 1.0
    ), "Action not applied."


def test_reset(scenario):
    """Test the basic pose data recoreded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)

    ego = gym.state.scenario.entities[0]
    poses = gym.state.recorded_poses(ego).copy()
    assert poses.shape[0] == 2

    gym.step()
    assert gym.state.t == 0.1

    gym.reset_scenario()
    assert np.allclose(poses, gym.state.recorded_poses(ego))
    assert gym.state.t == 0.0


def test_to_scenario(scenario) -> None:
    """
    Rollout a single scenario and write to a new scenario.

    Output the xosc then load it again and rollout the
    recorded version.
    """
    # rollout
    gym = ScenarioGym()
    gym.set_scenario(scenario)
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
