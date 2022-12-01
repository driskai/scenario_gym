import numpy as np

from scenario_gym.controller import Controller
from scenario_gym.entity import Entity
from scenario_gym.pedestrian.action import PedestrianAction
from scenario_gym.state import State


class PedestrianController(Controller):
    """Pedestrian controller that applies a PedestrianAction."""

    def __init__(
        self,
        entity: Entity,
        max_speed: float = 5.0,
    ):
        """Init the pedestrian controller."""
        super().__init__(entity)
        self.max_speed = max_speed

    def _reset(self, state: State) -> None:
        """Reset the controller."""
        self.speed = 0.0

    def _step(self, state: State, action: PedestrianAction) -> np.ndarray:
        """
        Produce the new pose.

        Parameters
        ----------
        state : State
            Global state.

        action : PedestrianAction
            The action. Contains speed and heading angle.

        """
        pose = state.poses[self.entity].copy()
        self.speed = np.clip(action.speed, -self.max_speed, self.max_speed)
        pose[[0, 1]] += (
            self.speed
            * state.dt
            * np.array([np.cos(action.heading), np.sin(action.heading)])
        )
        pose[3] = action.heading
        return pose
