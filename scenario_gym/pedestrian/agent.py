from typing import List

import numpy as np
from shapely.geometry import LineString, Point

from scenario_gym.agent import Agent
from scenario_gym.entity import Entity
from scenario_gym.pedestrian.action import PedestrianAction
from scenario_gym.pedestrian.behaviour import PedestrianBehaviour
from scenario_gym.pedestrian.controller import PedestrianController
from scenario_gym.pedestrian.observation import PedestrianObservation
from scenario_gym.pedestrian.sensor import PedestrianSensor


class PedestrianAgent(Agent):
    """A pedestrian agent with a behaviour model."""

    def __init__(
        self,
        entity: Entity,
        route: List[np.array],
        speed_desired: float,
        behaviour: PedestrianBehaviour,
        max_speed: float = 5.0,
        head_rot_angle: float = 0.0,
        distance_threshold: float = 1.0,
    ):
        """Init the agent."""
        super().__init__(
            entity,
            PedestrianController(entity, max_speed=max_speed),
            PedestrianSensor(
                entity,
                head_rot_angle=head_rot_angle,
                distance_threshold=distance_threshold,
            ),
        )
        self.goal_idx = 0
        self.speed_desired = speed_desired
        self.behaviour = behaviour
        self.force = np.array([0.0, 0.0])

        self.route = route
        self.route_geom = LineString(route)
        self.route_arcs = np.hstack(
            [[0.0], np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
        )

    def _step(self, observation: PedestrianObservation) -> PedestrianAction:
        """
        Produce the next action.

        Parameters
        ----------
        observation : PedestrianObservation
            All info from environment within perception radius.

        """
        # Change goal point to next one in path of close enough
        if self.goal_idx <= len(self.route) - 1:
            s = self.route_geom.project(Point(*observation.pose[:2]))
            self.goal_idx = np.argwhere(self.route_arcs <= s).max() + 1
        if self.goal_idx <= len(self.route) - 1:
            speed, heading = self.behaviour.step(observation, self)
        else:  # reached goal
            speed = 0
            heading = 0
            self.force[:] = 0
        return PedestrianAction(speed, heading)
