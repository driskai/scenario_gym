import numpy as np

from scenario_gym.state import State

from .base import Metric


class EgoAvgSpeed(Metric):
    """Measure the average speed of the ego."""

    name = "ego_avg_speed"

    def _reset(self, state: State) -> None:
        """Reset the average speed."""
        self.ego = state.scenario.entities[0]
        self.ego_avg_speed = np.linalg.norm(state.velocities[self.ego][:3])
        self.t = 0.0

    def _step(self, state: State) -> None:
        """Update the average speed."""
        speed = np.linalg.norm(state.velocities[self.ego][:3])
        w = self.t / state.t
        self.ego_avg_speed += (1.0 - w) * (speed - self.ego_avg_speed)
        self.t = state.t

    def get_state(self) -> float:
        """Return the current average speed."""
        return self.ego_avg_speed


class EgoMaxSpeed(Metric):
    """Measure the maximum speed of the ego."""

    name = "ego_max_speed"

    def _reset(self, state: State) -> None:
        """Reset the maximum speed."""
        self.ego = state.scenario.entities[0]
        self.ego_max_speed = np.linalg.norm(state.velocities[self.ego][:3])

    def _step(self, state: State) -> None:
        """Update the maximum speed."""
        speed = np.linalg.norm(state.velocities[self.ego][:3])
        self.ego_max_speed = np.maximum(speed, self.ego_max_speed)

    def get_state(self) -> float:
        """Return the current max speed."""
        return self.ego_max_speed


class EgoDistanceTravelled(Metric):
    """Measure the distance travelled by the ego."""

    name = "ego_distance_travelled"

    def _reset(self, state: State) -> None:
        """Find the ego."""
        self.ego = state.scenario.entities[0]

    def _step(self, state: State) -> None:
        """Pass as entity will update its distance."""
        self.dist = state.distances[self.ego]

    def get_state(self) -> float:
        """Return the current distance travelled."""
        return self.dist
