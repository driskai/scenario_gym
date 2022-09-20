import numpy as np

from scenario_gym.scenario_gym import ScenarioGym


def test_scenario_gym(all_scenarios):
    """Rollout a single scenario."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym(timestep=0.5, terminal_conditions=["max_length", "collision"])
    gym.load_scenario(scenario_path)
    gym.rollout()

    gym.reset_scenario()
    gym.step()
    assert all([np.allclose(e.dt, 0.5) for e in gym.state.scenario.entities])

    gym.timestep = 0.2
    gym.step()
    assert all([np.allclose(e.dt, 0.2) for e in gym.state.scenario.entities])

    gym.rollout()
    v = gym.state.scenario.entities[0].velocity
    assert (v[:2] == np.zeros(2)).all()


def test_reset_scenario(all_scenarios):
    """Test loading and reseting a scenario."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym()
    gym.load_scenario(scenario_path)
    assert [e.pose is not None for e in gym.state.scenario.entities]

    gym.load_scenario(scenario_path, relabel=True)
    assert (gym.state.scenario.entities[0].ref == "ego") and (
        gym.state.scenario.entities[1].ref == "vehicle_0"
    ), "Should be relabeled"


def test_render(all_scenarios):
    """Test the rendering of the gym."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym(timestep=0.1)
    gym.load_scenario(scenario_path)
    gym.reset_scenario()
    gym.rollout(render=True)

    gym = ScenarioGym(timestep=0.1, render_layers=["driveable_surface", "centers"])
    gym.load_scenario(scenario_path)
    gym.reset_scenario()
    gym.rollout(render=True)
