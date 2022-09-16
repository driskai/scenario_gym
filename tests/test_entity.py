import os

import numpy as np

from scenario_gym import Metric, ScenarioGym, Trajectory


def test_recorded_poses():
    """Test that all poses get recorded by the gym."""

    class CheckPose(Metric):
        def _reset(self, state):
            """Get the ego."""
            self.ego = state.scenario.entities[0]
            assert self.ego.record_trajectory, "Ego not being recorded."

        def _step(self, state):
            """Check the recorded pose."""
            assert np.allclose(
                np.hstack([[state.t], self.ego.pose]), self.ego.recorded_poses[-1]
            ), "Ego pose is not the same as the last recorded pose."

        def get_state(self):
            """Return None."""
            return None

    filepath = os.path.join(
        os.path.dirname(__file__),
        "input_files",
        "Scenarios",
        "3fee6507-fd24-432f-b781-ca5676c834ef.xosc",
    )
    gym = ScenarioGym(
        timestep=0.10001,
        terminal_conditions=[lambda s: s.t >= 1],
        metrics=[CheckPose()],
    )
    gym.record()
    gym.load_scenario(filepath)

    trajectory = Trajectory(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        fields=["t", "x", "y"],
    )
    ego = gym.state.scenario.entities[0]
    ego.trajectory = trajectory

    gym.reset_scenario()
    gym.rollout()

    assert (
        ego.recorded_poses[-1, [1, 2]] == np.ones(2)
    ).all(), "Final pose is not the same as the trajectory."
