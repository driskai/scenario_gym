import time

import pytest as pt

import scenario_gym
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.xosc_interface import import_scenario

speed_test = pt.mark.skipif("not config.getoption('speed_tests')")


@pt.fixture
def paths(all_scenarios):
    """Get filepaths for the scenarios to be used."""
    scenarios = [
        "41dac6fa-6f83-461e-a145-08692da5f3c7",
        "9c324146-be03-4d4e-8112-eaf36af15c17",
        "a5e43fe4-646a-49ba-82ce-5f0063776566",
        "a98d5c7d-76aa-49bf-b88c-97db5d5c7433",
        "d9726503-e04a-4e8b-b487-8805ef790c92",
        "e1bdb607-206b-4f40-9bc4-59ded182ecc8",
        "e56ae853-4266-4c30-865f-96737d87b601",
    ]
    paths = [all_scenarios[s] for s in scenarios]
    return paths


@speed_test
def test_gym_speed(paths):
    """
    Loads and runs scenarios with different config.

    Use -s with pytest to see the result.
    """
    n = 10
    gym = ScenarioGym(timestep=1.0 / 30.0)
    start = time.time()
    for _ in range(n):
        for s in paths:
            gym.load_scenario(s)
            gym.rollout()
    t = (time.time() - start) / (n * len(paths))
    total_length = sum(import_scenario(p).length for p in paths)
    num_steps = [int(import_scenario(p).length * 30) for p in paths]
    print(
        "Completed in {:.4}s per scenario, {:.4}\u03BCs per step.".format(
            t, 1e6 * len(paths) * t / sum(num_steps)
        )
    )
    print("Running at {}x real time.".format(int(total_length / t)))


@speed_test
def test_render_speed(paths):
    """Loads scenarios with rendering."""
    n = 2

    gym = ScenarioGym(timestep=1.0 / 30.0)
    start = time.time()
    for _ in range(n):
        for s in paths:
            gym.load_scenario(s)
            gym.rollout(render=True)
    t = (time.time() - start) / (n * len(paths))
    total_length = sum(import_scenario(p).length for p in paths)
    num_steps = [int(import_scenario(p).length * 30) for p in paths]
    print(
        "Completed in {:.4}s per scenario, {:.4}\u03BCs per step.".format(
            t, 1e6 * len(paths) * t / sum(num_steps)
        )
    )
    print("Running at {}x real time.".format(int(total_length / t)))


@speed_test
def test_sensor_speed(paths):
    """Measures speed when running with RasterizedMapSensor."""
    n = 10

    def create_agent(ref, entity):
        if ref == "ego":
            controller = scenario_gym.controller.ReplayTrajectoryController(entity)
            sensor = scenario_gym.sensor.RasterizedMapSensor(entity, freq=1)
            return scenario_gym.agent.ReplayTrajectoryAgent(
                entity, controller, sensor
            )

    gym = ScenarioGym(timestep=1.0 / 30.0)
    start = time.time()
    for _ in range(n):
        for s in paths:
            gym.load_scenario(s, create_agent=create_agent)
            gym.rollout()
    t = (time.time() - start) / (n * len(paths))
    num_steps = [int(import_scenario(p).length * 30) for p in paths]
    print(
        "Completed in {:.4}s per scenario, {:.4}s per step.".format(
            t, len(paths) * t / sum(num_steps)
        )
    )
