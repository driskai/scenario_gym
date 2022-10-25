import numpy as np

from scenario_gym.manager import ScenarioManager
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


def test_manager():
    """Test that custom arguments can be supplied to the config."""
    manager = ScenarioManager()

    # change the timestep
    timestep = manager.timestep
    new_manager = ScenarioManager(timestep=timestep + 1)
    assert (
        new_manager.timestep == timestep + 1
    ), f"Should be {timestep+1} but got {new_manager.timestep}."

    # try adding a random argument
    new_manager = ScenarioManager(test_argument_12345="test")
    assert hasattr(new_manager, "test_argument_12345"), "Custom arg not assigned."
    assert new_manager.test_argument_12345 == "test", "Arg incorrectly assigned."


def test_add_metric(all_scenarios):
    """Test adding a metric to the scenario."""
    files = [
        all_scenarios[s]
        for s in (
            "3fee6507-fd24-432f-b781-ca5676c834ef",
            "41dac6fa-6f83-461e-a145-08692da5f3c7",
            "9c324146-be03-4d4e-8112-eaf36af15c17",
        )
    ]
    manager = ScenarioManager()
    manager.add_metric(EgoMaxSpeedMetric())
    output = manager.run_scenarios(files)
    assert len(output) == 3, "Should have returned results for 3 scenarios."

    gym = ScenarioGym()
    gym.load_scenario(files[0])
    gym.rollout()


def test_viewer_params(all_scenarios):
    """Test adding a metric to the scenario."""
    s = all_scenarios["3fee6507-fd24-432f-b781-ca5676c834ef"]
    manager = ScenarioManager(
        viewer_params={
            "render_layers": ["driveable_surface", "text"],
            "codec": "mp4v",
        }
    )
    gym = manager.make_gym()
    gym.load_scenario(s)
    gym.render()
    assert gym.viewer.render_layers == [
        "driveable_surface",
        "text",
    ], "Parameter not passed."
    assert gym.viewer.codec == "mp4v", "Parameter not passed."

    # repeat with incorrect argument
    manager = ScenarioManager(magnification=2)
    gym = manager.make_gym()
    gym.load_scenario(s)
    gym.render()
    assert gym.viewer.mag != 2, "Parameter should not be passed."
