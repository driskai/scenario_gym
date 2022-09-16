from typing import List

import numpy as np

from scenario_gym.agent import Agent
from scenario_gym.controller import Controller
from scenario_gym.entity import Entity
from scenario_gym.pedestrian.action import PedestrianAction
from scenario_gym.pedestrian.behaviour import PedestrianBehaviour
from scenario_gym.pedestrian.observation import PedestrianObservation
from scenario_gym.sensor import Sensor
from scenario_gym.state import State


class PedestrianAgent(Agent):
    """A pedestrian agent with a behaviour model."""

    def __init__(
        self,
        entity: Entity,
        controller: Controller,
        sensor: Sensor,
        route: List[np.array],
        speed_desired: float,
        behaviour: PedestrianBehaviour,
    ):
        """Init the agent."""
        super().__init__(entity, controller, sensor)
        self.route = route
        self.goal_idx = 0
        self.speed_desired = speed_desired
        self.behaviour = behaviour
        self.force = np.array([0.0, 0.0])

    def _reset(self):
        """Reset the agent."""
        pass

    def _step(
        self, state: State, observation: PedestrianObservation
    ) -> PedestrianAction:
        """
        Produce the next action.

        Parameters
        ----------
        state : State
            Driving gym state. All info from environment.
        observation : PedestrianObservation
            All info from environment within perception radius.

        """
        # Change goal point to next one in path of close enough
        if self.goal_idx < len(self.route) - 1:
            while (
                np.linalg.norm(self.entity.pose[[0, 1]] - self.route[self.goal_idx])
                < 2
                and self.goal_idx < len(self.route) - 1
            ):
                self.goal_idx += 1
            speed, heading = self.behaviour.step(state, observation, self)
        else:  # reached goal
            speed = 0
            heading = 0
            self.force[:] = 0

        return PedestrianAction(speed, heading)
