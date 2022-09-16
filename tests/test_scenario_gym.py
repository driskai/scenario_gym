import os

import numpy as np

from scenario_gym.scenario_gym import ScenarioGym


def test_scenario_gym():
    """Rollout a single scenario."""
    base = os.path.dirname(__file__)
    scenario_path = os.path.join(
        base,
        "input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc",
    )
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


def test_reset_scenario():
    """Test loading and reseting a scenario."""
    base = os.path.dirname(__file__)
    scenario_path = os.path.join(
        base,
        "input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc",
    )
    gym = ScenarioGym()
    gym.load_scenario(scenario_path)
    assert [e.pose is not None for e in gym.state.scenario.entities]

    gym.load_scenario(scenario_path, relabel=True)
    assert (gym.state.scenario.entities[0].ref == "ego") and (
        gym.state.scenario.entities[1].ref == "vehicle_0"
    ), "Should be relabeled"


def test_render():
    """Test the rendering of the gym."""
    base = os.path.dirname(__file__)
    scenario_path = os.path.join(
        base,
        "input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc",
    )
    gym = ScenarioGym()
    gym.load_scenario(scenario_path)
    gym.reset_scenario()
    gym.rollout(render=True)

    gym = ScenarioGym(render_layers=["driveable_surface", "centers"])
    gym.load_scenario(scenario_path)
    gym.reset_scenario()
    gym.rollout(render=True)


def test_scenarios():
    """Rollout every scenario."""
    base = os.path.join(os.path.dirname(__file__), "input_files/Scenarios/")
    files = [
        "3fee6507-fd24-432f-b781-ca5676c834ef.xosc",
        "41dac6fa-6f83-461e-a145-08692da5f3c7.xosc",
        "9c324146-be03-4d4e-8112-eaf36af15c17.xosc",
        "a2281876-e0b4-4048-a08a-1ce69f94c085.xosc",
        "a5e43fe4-646a-49ba-82ce-5f0063776566.xosc",
        "a98d5c7d-76aa-49bf-b88c-97db5d5c7433.xosc",
        "d9726503-e04a-4e8b-b487-8805ef790c92.xosc",
        "d9726503-e04a-4e8b-b487-8805ef790c93.xosc",
        "e1bdb607-206b-4f40-9bc4-59ded182ecc8.xosc",
        "e56ae853-4266-4c30-865f-96737d87b601.xosc",
        "fbb6b5ca-3fcb-4a7b-9757-b8554a753e69.xosc",
    ]
    gym = ScenarioGym()
    for f in files:
        gym.load_scenario(base + f)
        gym.rollout()
