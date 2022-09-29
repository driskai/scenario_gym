import os

import numpy as np
from absl.testing import absltest
from dm_env.specs import Array
from dm_env.test_utils import EnvironmentTestMixin

from scenario_gym import Agent
from scenario_gym.controller import VehicleController
from scenario_gym.integrations.deepmind_env import ScenarioGym
from scenario_gym.sensor import RasterizedMapSensor


class ExampleAgent(Agent):
    """Agent with map sensor, vehicle controller and zero reward."""

    def __init__(self, e):
        super().__init__(
            e, VehicleController(e), RasterizedMapSensor(e, layers=["entity"], n=5)
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
