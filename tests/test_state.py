import numpy as np
import pytest as pt

from scenario_gym.scenario.actions import UpdateStateVariableAction
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.state import State
from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def scenario_path(all_scenarios):
    """Get a path for a scenario to test."""
    return all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]


@pt.fixture
def t0_scenario(all_scenarios):
    """Get a path for a scenario to test."""
    pth = all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]
    return import_scenario(pth).reset_start()


@pt.fixture
def scenario(scenario_path):
    """Get a scenario to test."""
    s = import_scenario(scenario_path)
    action = UpdateStateVariableAction(3.0, "TestAction", "ego", {"var": 1.0})
    s.add_action(action, inplace=True)
    return s


def test_poses(scenario):
    """Test the basic pose data recoreded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)

    assert gym.state.t == gym.state.scenario.ego.trajectory.min_t
    assert gym.state.poses
    assert all(
        (
            np.allclose(
                v, (gym.state.poses[e] - gym.state.prev_poses[e]) / gym.state.dt
            )
            for e, v in gym.state.velocities.items()
        )
    ), "Velocities not correct."
    print(gym.state.recorded_poses())
    assert all(
        len(poses) == 2
        for e, poses in gym.state.recorded_poses().items()
        if e in gym.state.poses
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
        len(poses) == 3
        for e, poses in gym.state.recorded_poses().items()
        if e in gym.state.poses
    ), "Wrong number of recorded poses."


def test_state_info(scenario):
    """Test running a scenario and getting data from the state."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)

    for _ in range(50):
        gym.step()

    assert len(gym.state.poses) >= 2, "Not enough entities in the state."
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


def test_step(t0_scenario):
    """Test the basic pose data recorded in the gym state."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(t0_scenario)
    (ego, hazard) = gym.state.scenario.entities[:2]

    current = gym.state.poses.copy()
    next_poses = current.copy()
    next_poses[ego] = np.random.randn(6)
    next_poses[hazard] = np.random.randn(6)

    gym.state.next_t = 1.0
    gym.state.step(next_poses)

    assert np.allclose(gym.state.poses[ego], next_poses[ego])
    assert np.allclose(gym.state.poses[hazard], next_poses[hazard])
    assert gym.state.t == 1.0


def test_step_with_vanishing(t0_scenario):
    """Test updating the state with vanishing entities."""
    scenario = t0_scenario.reset_start(t0_scenario.entities[1])

    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(scenario)
    (ego, hazard) = gym.state.scenario.entities[:2]

    current = gym.state.poses.copy()
    current.pop(hazard)

    gym.state.next_t = 1.0
    gym.state.step(current)

    assert np.allclose(gym.state.poses[ego], current[ego])
    assert hazard not in gym.state.poses, "Vanishing entity not removed."
    assert hazard not in gym.state.prev_poses, "Vanishing entity not removed."
    assert hazard not in gym.state.velocities, "Vanishing entity not removed."
    assert hazard in gym.state.distances, "Vanishing entity should stay here."
    assert (
        hazard in gym.state.recorded_poses()
    ), "Vanishing entity should stay here."
    assert gym.state.t == 1.0

    gym.state.next_t = 2.0
    current[hazard] = hazard.trajectory.position_at_t(2.0)
    haz_prev = hazard.trajectory.position_at_t(1.0)
    haz_v = (current[hazard] - haz_prev) / gym.state.dt
    gym.state.step(current)

    assert hazard in gym.state.poses, "Vanishing entity not returned."
    assert hazard in gym.state.prev_poses, "Vanishing entity not returned."
    assert hazard in gym.state.velocities, "Vanishing entity not returned."
    assert np.allclose(
        gym.state.poses[hazard], current[hazard]
    ), "Vanishing entity has wrong pose."
    assert np.allclose(
        gym.state.prev_poses[hazard], haz_prev
    ), "Vanishing entity has wrong prev pose."
    assert np.allclose(
        gym.state.velocities[hazard], haz_v
    ), "Vanishing entity has wrong velocity."


def test_reset(scenario_path):
    """Test resetting the scenario with vanishing entities."""
    scenario = import_scenario(scenario_path)
    state = State(scenario)

    n = sum(1 for e in scenario.entities if e.trajectory.min_t <= 0.0)
    state.reset(-1.0, 0.0)
    assert len(state.poses) == n, "Wrong number of entities."
    assert len(state.prev_poses) == n, "Wrong number of entities."

    n = sum(1 for e in scenario.entities if e.trajectory.max_t >= 100.0)
    state.reset(-1.0, 100.0)
    assert len(state.poses) == n, "Wrong number of entities."
    assert len(state.poses) == n, "Wrong number of entities."


def test_reset_persist(scenario_path):
    """Test resetting the scenario with vanishing entities."""
    scenario = import_scenario(scenario_path)
    state = State(scenario, persist=True)

    n = len(scenario.entities)
    state.reset(-1.0, 0.0)
    assert len(state.poses) == n, "Wrong number of entities."
    assert len(state.prev_poses) == n, "Wrong number of entities."

    state.reset(-1.0, 100.0)
    assert len(state.poses) == n, "Wrong number of entities."
    assert len(state.poses) == n, "Wrong number of entities."


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


def test_to_scenario(all_scenarios) -> None:
    """
    Rollout a single scenario and write to a new scenario.

    Output the xosc then load it again and rollout the
    recorded version.
    """
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    # note we must reset start so that the recorded version will match the input
    scenario = import_scenario(scenario_path).reset_start()

    # rollout
    gym = ScenarioGym()
    gym.set_scenario(scenario)
    gym.rollout()

    poses = gym.state.recorded_poses()[scenario.entities[0]]
    assert np.unique(poses, axis=0).shape[0] == poses.shape[0]
    new_scenario = gym.state.to_scenario()

    ego = new_scenario.entities[0]
    assert len(ego.trajectory.t) == ego.trajectory.data.shape[0]

    # reload and test
    traj1 = scenario.entities[0].trajectory
    n_entities = len(scenario.entities)
    n_stationary = sum(1 for t in scenario.trajectories.values() if len(t) == 1)

    traj2 = new_scenario.entities[0].trajectory
    assert (
        len(new_scenario.entities) == n_entities
    ), "New scenario has a different number of entities."
    assert all(
        (
            isinstance(entity, type(old_entity))
            for entity, old_entity in zip(scenario.entities, new_scenario.entities)
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
