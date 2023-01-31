import multiprocessing as mp

import numpy as np
import pytest as pt

from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.trajectory import Trajectory
from scenario_gym.xosc_interface import import_scenario


@pt.fixture(scope="module")
def scenario(all_scenarios):
    """Load a single scenario."""
    return import_scenario(all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"])


@pt.fixture(scope="module")
def vanishing_scenario(scenario):
    """Create a scenario that vanishes."""
    scenario = scenario.copy()
    new_data = scenario.entities[1].trajectory.data.copy()
    subset = new_data[np.logical_and(new_data[:, 0] < 16.5, new_data[:, 0] > 2.0)]
    traj = Trajectory(subset)
    scenario.entities[1].trajectory = traj
    return scenario


def test_gym(scenario):
    """Rollout a single scenario."""
    gym = ScenarioGym(timestep=0.5, terminal_conditions=["max_length", "collision"])
    gym.set_scenario(scenario)
    gym.rollout()

    gym.reset_scenario()
    gym.step()
    assert np.allclose(gym.state.dt, 0.5)

    gym.timestep = 0.2
    gym.step()
    assert np.allclose(gym.state.t, gym.state.prev_t + 0.2)

    gym.rollout()
    v = gym.state.velocities[gym.state.scenario.entities[0]]
    assert (v[:2] == np.zeros(2)).all()


def test_reset_scenario(scenario):
    """Test loading and reseting a scenario."""
    gym = ScenarioGym()
    gym.set_scenario(scenario)
    assert [gym.state.poses[e] is not None for e in gym.state.poses]

    gym.set_scenario(scenario)
    assert (gym.state.scenario.entities[0].ref == "ego") and (
        gym.state.scenario.entities[1].ref == "vehicle_0"
    ), "Should be relabeled"


def test_reset_scenario_with_vanishing(vanishing_scenario):
    """Test loading and reseting a scenario."""
    gym = ScenarioGym()
    gym.set_scenario(vanishing_scenario)
    for e in vanishing_scenario.entities:
        if e.trajectory.min_t <= gym.state.t:
            assert e in gym.state.poses, f"{e.ref} should be in poses"


def test_rollout(vanishing_scenario):
    """Test rollout."""
    gym = ScenarioGym(timestep=0.1)
    gym.set_scenario(vanishing_scenario)
    gym.rollout()
    assert gym.state.scenario.entities[1] not in gym.state.poses


def test_persist(vanishing_scenario):
    """Test rollout."""
    gym = ScenarioGym(timestep=0.1, persist=True)
    gym.set_scenario(vanishing_scenario)
    assert len(gym.state.poses) == len(vanishing_scenario.entities), (
        "Entities not found in poses: ",
        str(
            [e.ref for e in vanishing_scenario.entities if e not in gym.state.poses]
        ),
    )
    while not gym.state.is_done:
        gym.step()
        assert len(gym.state.poses) == len(vanishing_scenario.entities), (
            "Entities not found in poses: ",
            str(
                [
                    e.ref
                    for e in vanishing_scenario.entities
                    if e not in gym.state.poses
                ]
            ),
        )


def test_render(scenario, vanishing_scenario):
    """Test the rendering of the gym."""
    gym = ScenarioGym()
    gym.set_scenario(scenario)
    gym.rollout(render=True)
    gym.set_scenario(vanishing_scenario)
    gym.rollout(
        render=True,
        video_path=scenario.path.replace("Scenarios", "Recordings").replace(
            ".xosc", "_vanishing.mp4"
        ),
    )


def test_manual_rollout(all_scenarios):
    """Test rollout manually."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym(timestep=0.2)
    gym.load_scenario(scenario_path)
    gym.render()
    while not gym.state.is_done:
        gym.step()
        gym.render()
    gym.close()


def test_run_scenarios(all_scenarios):
    """Test using the run scenarios class method."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    ScenarioGym.run_scenarios(
        [scenario_path, scenario_path],
        timestep=0.075,
    )


def _render_scenario(scenario):
    """Render one scenario."""
    gym = ScenarioGym(timestep=0.075)
    gym.set_scenario(scenario)
    gym.rollout(render=True)


def test_multi_process_scenarios(all_scenarios):
    """Test running scenarios in multiple processes."""
    scenarios = []
    num_processes = 4
    for _, path in zip(range(num_processes), all_scenarios.values()):
        scenarios.append(import_scenario(path))

    with mp.Pool(num_processes) as p:
        p.map(_render_scenario, scenarios)
