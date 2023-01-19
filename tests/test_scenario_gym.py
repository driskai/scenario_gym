import multiprocessing as mp

import numpy as np

from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.xosc_interface import import_scenario


def test_gym(all_scenarios):
    """Rollout a single scenario."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym(timestep=0.5, terminal_conditions=["max_length", "collision"])
    gym.load_scenario(scenario_path)
    gym.rollout()

    gym.reset_scenario()
    gym.step()
    assert np.allclose(gym.state.dt, 0.5)

    gym.timestep = 0.2
    gym.step()
    assert np.allclose(gym.state.t, 0.7)

    gym.rollout()
    v = gym.state.velocities[gym.state.scenario.entities[0]]
    assert (v[:2] == np.zeros(2)).all()


def test_reset_scenario(all_scenarios):
    """Test loading and reseting a scenario."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym()
    gym.load_scenario(scenario_path)
    assert [gym.state.poses[e] is not None for e in gym.state.scenario.entities]

    gym.load_scenario(scenario_path, relabel=True)
    assert (gym.state.scenario.entities[0].ref == "ego") and (
        gym.state.scenario.entities[1].ref == "vehicle_0"
    ), "Should be relabeled"


def test_render(all_scenarios):
    """Test the rendering of the gym."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym(timestep=0.1)
    gym.load_scenario(scenario_path)
    gym.rollout(render=True)
    gym.rollout(
        render=True,
        video_path=scenario_path.replace("Scenarios", "Recordings").replace(
            ".xosc", ".mp4"
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
