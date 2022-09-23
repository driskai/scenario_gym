"""Using the Metric class to record data."""
import os

import numpy as np

from scenario_gym.metrics import Metric
from scenario_gym.scenario_gym import ScenarioGym


class EgoMaxSpeedMetric(Metric):
    """Measures the max speed achieved by the ego in the scenario."""

    def __init__(self):
        self.max_speed = 0.0

    def _reset(self, state):
        """Reset the metric."""
        self.ego = state.scenario.agents["ego"]
        self.max_speed = 0.0

    def _step(self, state):
        """Update the max speed."""
        self.max_speed = np.maximum(
            np.linalg.norm(self.ego.entity.velocity[:2]),
            self.max_speed,
        )

    def get_state(self) -> float:
        """Return current max speed."""
        return self.max_speed


class EgoCollisionObserver(Metric):
    """Check if the ego has collided with another entity."""

    def __init__(self):
        self.ego_collision = False
        self.ego = None

    def _reset(self, state):
        """Reset the metric."""
        self.ego = state.scenario.entities[0]
        self.ego_collision = False

    def _step(self, state):
        """Check for collisions."""
        self.ego_collision = len(state.collisions[self.ego]) > 0

    def get_state(self) -> float:
        """Return current max speed."""
        return self.ego_collision


def run_metrics():
    """Add metrics to the gym and run on a scenario."""

    base = os.path.join(
        os.path.dirname(__file__),
        "../tests/input_files/Scenarios/",
    )
    s = "3fee6507-fd24-432f-b781-ca5676c834ef.xosc"

    gym = ScenarioGym(
        metrics=[
            EgoMaxSpeedMetric(),
            EgoCollisionObserver(),
        ],
    )
    gym.load_scenario(os.path.join(base, s))
    gym.rollout()

    max_speed = gym.metrics[0].get_state()
    ego_collision = gym.metrics[1].get_state()
    print(f"Ego max speed: {max_speed:.4}m/s.")
    print(f"Ego collided: {ego_collision}.")


if __name__ == "__main__":
    run_metrics()
