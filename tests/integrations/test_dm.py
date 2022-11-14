import os
import random

import numpy as np
from absl.testing import absltest
from dm_env.specs import Array
from dm_env.test_utils import EnvironmentTestMixin

from scenario_gym import Agent, State
from scenario_gym.controller import VehicleController
from scenario_gym.integrations.deepmind_env import ScenarioGym
from scenario_gym.sensor.map import MapObservation, RasterizedMapSensor


class MapOnlySensor(RasterizedMapSensor):
    """Sensor returning only the rasterized map."""

    observation_type = np.ndarray

    def _step(self, state: State, obs: MapObservation) -> np.ndarray:
        """Get the map from the base sensor's observation."""
        return super()._step(state, obs).map


class ExampleAgent(Agent):
    """Agent with map sensor, vehicle controller and zero reward."""

    def __init__(self, entity):
        super().__init__(
            entity,
            VehicleController(entity),
            MapOnlySensor(
                entity,
                channels_first=False,
                height=30,
                width=30,
                n=5,
                layers=["entity"],
            ),
        )

    def reward(self, state):
        """Return zero reward."""
        return 0.0


class ExampleEnv(ScenarioGym):
    """Implementation to test dm_env ScenarioGym."""

    def observation_spec(self):
        """Return a map sensor observation spec."""
        return Array((5, 5, 1), bool)

    def action_spec(self):
        """Return a vehicle action observation spec."""
        return Array((2,), np.float32)

    @staticmethod
    def _create_agent(s, e):
        """Create an agent for the ego."""
        if e.ref == "ego":
            return ExampleAgent(e)


class EnvTest(EnvironmentTestMixin, absltest.TestCase):
    """Test class mixin."""

    def make_object_under_test(self):
        """Make the gym instance to be tested."""
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "input_files",
            "Scenarios",
            "3071b41f-903f-4465-a5bb-77262f2aa08a.xosc",
        )

        env = ExampleEnv()
        env.load_scenario(
            path,
            create_agent=ExampleEnv._create_agent,
        )
        return env


def test_dm_env(all_scenarios):
    """Test implementing an env with random scenario sampling."""
    scenarios = list(all_scenarios.values())

    def update_scenario(self):
        """Sample a scenario to use."""
        scenario = random.choice(scenarios)
        self.load_scenario(scenario, create_agent=ExampleEnv._create_agent)

    gym = ExampleEnv(update_scenario=update_scenario)

    for _ in range(10):
        timestep = gym.reset()
        while not timestep.last:
            timestep = gym.step(np.zeros(2))
