from typing import Tuple

import numpy as np

from scenario_gym.agent import Agent
from scenario_gym.pedestrian.behaviour import (
    BehaviourParameters,
    PedestrianBehaviour,
)
from scenario_gym.pedestrian.observation import PedestrianObservation


class RandomWalkParameters(BehaviourParameters):
    """Parameters for the random walk model."""

    bias_lon = 0.0
    bias_lat = 0.0
    std_lon = 0.000002
    std_lat = 0.0000001


class RandomWalk(PedestrianBehaviour):
    """Random walk model."""

    def __init__(self, params: RandomWalkParameters):
        super().__init__(params)
        self.bias_lon = params.bias_lon
        self.bias_lat = params.bias_lat
        self.std_lon = params.std_lon
        self.std_lat = params.std_lat

    def _step(
        self,
        observation: PedestrianObservation,
        agent: Agent,
    ) -> Tuple[float, float]:
        """Return the new speed and heading according to the random walk model."""
        speed_rand = np.random.normal(
            agent.speed_desired + self.bias_lon, self.std_lon
        )
        goal = agent.route[agent.goal_idx] - observation.pose[[0, 1]]
        heading = np.arctan2(goal[1], goal[0])  # angle to goal point
        heading_rand = np.random.normal(heading + self.bias_lat, self.std_lat)
        return speed_rand, heading_rand
