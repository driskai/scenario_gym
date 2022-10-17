import pytest as pt

from scenario_gym.callback import StateCallback
from scenario_gym.metrics import Metric
from scenario_gym.scenario_gym import ScenarioGym


@pt.fixture
def scenario_path(all_scenarios):
    """Get a path for a scenario to test."""
    return all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]


class TestCallback(StateCallback):
    """A simple callback to add a feature to the state."""

    def __call__(self, state):
        """Set the feature to whether the time is <= 1."""
        state.my_feature = state.t <= 1


class TestDependentCallback(StateCallback):
    """A simple callback that is dependented on the TestCallback."""

    required_callbacks = [TestCallback]

    def __call__(self, state):
        """Set the feature to the opposite of the existing features."""
        state.dependent_feature = not state.my_feature


def test_callback(scenario_path):
    """Test that we can add a callback to the state."""
    gym = ScenarioGym(state_callbacks=[TestCallback()])
    gym.load_scenario(scenario_path)
    assert gym.state.my_feature, "Feature should be added at reset."
    gym.rollout()
    assert gym.state.t > 1 and not gym.state.my_feature, "Feature should be False."


def test_metric_requires(scenario_path):
    """Test that metrics can be forced to require callbacks."""

    class TestMetric(Metric):
        required_callbacks = [TestCallback]

        def _reset(self, state):
            pass

        def _step(self, state):
            pass

        def get_state(self):
            pass

    gym = ScenarioGym(state_callbacks=[TestCallback()], metrics=[TestMetric()])
    try:
        gym.load_scenario(scenario_path)
    except ValueError as e:
        raise "Should not raise any value error:" from e

    gym = ScenarioGym(state_callbacks=[], metrics=[TestMetric()])
    with pt.raises(ValueError):
        gym.load_scenario(scenario_path)


def test_callback_requires(scenario_path):
    """Test that the callbacks can be forced to require callbacks."""
    gym = ScenarioGym(state_callbacks=[TestCallback(), TestDependentCallback()])
    try:
        gym.load_scenario(scenario_path)
    except ValueError as e:
        raise "Should not raise any value error:" from e

    gym = ScenarioGym(state_callbacks=[TestDependentCallback()])
    with pt.raises(ValueError):
        gym.load_scenario(scenario_path)

    gym = ScenarioGym(state_callbacks=[TestDependentCallback(), TestCallback()])
    with pt.raises(ValueError):
        gym.load_scenario(scenario_path)
