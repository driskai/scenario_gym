"""
Agent that uses keyboard actions controlled by the user for movement.
"""
import math

import cv2

import scenario_gym
from scenario_gym.action import VehicleAction
from scenario_gym.agent import Agent
from scenario_gym.controller import VehicleController
from scenario_gym.entity import Entity
from scenario_gym.manager import ScenarioManager
from scenario_gym.scenario import Scenario
from scenario_gym.sensor.common import KeyboardInputDetector, KeyboardObservation
from scenario_gym.state import State


class KeyboardAgent(Agent):
    """An agent controlled by the user with keyboard actions."""

    def __init__(
        self,
        entity: Entity,
        controller: VehicleController,
        sensor: KeyboardInputDetector,
        max_throttle: float = 5.0,
        max_steer: float = 0.3,
    ):
        """
        Init the agent.

        Parameters
        ----------
        entity : Entity
            The entity.

        controller : BasicVehicleController
            The controller for the agent. A vehicle controller.

        sensor: KeyboardInputDetector
            The sensor which return keyboard inputs.

        max_throttle : float
            Maximum acceleration of the agent.

        max_steer : float
            Maximum steering angle of the agent.
        """
        super().__init__(entity, controller, sensor)
        self.max_throttle = max_throttle
        self.max_steer = max_steer

    def _keyboard_action(self, key: int):
        # Gets key presses and moves accordingly 8 and 27 are delete and escape keys
        # Arrow keys are bad in cv2. So we use keys 'w', 'a','s','d' for movement.
        # w = top, a = left, s = down, d = right

        throttle, steer = 0, 0
        if key == ord("d"):
            steer = -1
        elif key == ord("s"):
            throttle = -1
        elif key == ord("a"):
            steer = 1
        elif key == ord("w"):
            throttle = 1
        else:
            pass

        return throttle, steer

    def _reset(self):
        pass

    def _step(self, observation: KeyboardObservation) -> VehicleAction:
        # convert the key-stroke from the observation to a speed and steer

        throttle, steer = self._keyboard_action(observation.last_keystroke)

        throttle *= self.max_throttle
        steer *= self.max_steer

        return VehicleAction(throttle, steer)


class KeyboardConfig(ScenarioManager):

    PARAMETERS = {
        "timestep": 0.0333,
        "headless_rendering": False,
        "terminal_conditions": ["ego_off_road", "ego_collision"],
        "max_throttle": 10,
        "max_steer": 0.9,
    }

    def __init__(
        self,
        config_path: str = None,
        **kwargs,
    ):
        super().__init__(config_path=config_path, **kwargs)

    def create_agent(self, scenario: Scenario, entity: Entity) -> Agent:
        if entity.ref == "ego":
            controller = VehicleController(
                entity,
                max_steer=self.max_steer,
                max_accel=self.max_throttle,
            )
            sensor = KeyboardInputDetector(entity)
            return KeyboardAgent(
                entity,
                controller,
                sensor,
                max_throttle=self.max_throttle,
                max_steer=self.max_steer,
            )


if __name__ == "__main__":

    config = KeyboardConfig()
    gym = scenario_gym.ScenarioGym(
        timestep=config.timestep,
        headless_rendering=config.headless_rendering,
        terminal_conditions=config.terminal_conditions,
    )
    scenario_path = (
        "../tests/input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc"
    )
    gym.load_scenario(scenario_path, create_agent=config.create_agent)
    gym.rollout(render=True)
