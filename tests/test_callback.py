import pytest as pt

from scenario_gym.callback import StateCallback
from scenario_gym.metrics import Metric
from scenario_gym.scenario_gym import ScenarioGym


@pt.fixture
def scenario_path(all_scenarios):
    """Get a path for a scenario to test."""
    return all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]


class ExampleCallback(StateCallback):
    """A simple callback to add a feature to the state."""

    my_feature: bool

    def __call__(self, state):
        """Set the feature to whether the time is <= 1."""
        self.my_feature = state.t <= 1


class ExampleDependentCallback(StateCallback):
    """A simple callback that is dependented on the TestCallback."""

    required_callbacks = [ExampleCallback]

    dependent_feature: bool

    def __call__(self, state):
        """Set the feature to the opposite of the existing features."""
        self.dependent_feature = not self.callbacks[0].my_feature


def test_get_callback(scenario_path):
    """Test getting a callback instance from the state."""
    cb1, cb2 = ExampleCallback(), ExampleDependentCallback()
    gym = ScenarioGym(state_callbacks=[cb1, cb2])
    gym.load_scenario(scenario_path)
    assert gym.state.get_callback(ExampleCallback) is cb1
    assert gym.state.get_callback(ExampleDependentCallback) is cb2


def test_callback(scenario_path):
    """Test that we can add a callback to the state."""
    gym = ScenarioGym(state_callbacks=[ExampleCallback()])
    gym.load_scenario(scenario_path)
    assert gym.state_callbacks[0].my_feature, "Feature should be added at reset."
    gym.rollout()
    assert (
        gym.state.t > 1 and not gym.state_callbacks[0].my_feature
    ), "Feature should be False."


def test_metric_requires(scenario_path):
    """Test that metrics can be forced to require callbacks."""

    class ExampleMetric(Metric):
        required_callbacks = [ExampleCallback]

        def _reset(self, state):
            pass

        def _step(self, state):
            pass

        def get_state(self):
            pass

    gym = ScenarioGym(
        state_callbacks=[ExampleCallback()], metrics=[ExampleMetric()]
    )
    try:
        gym.load_scenario(scenario_path)
    except ValueError as e:
        raise "Should not raise any value error:" from e

    gym = ScenarioGym(state_callbacks=[], metrics=[ExampleMetric()])
    with pt.raises(ValueError):
        gym.load_scenario(scenario_path)


def test_callback_requires(scenario_path):
    """Test that the callbacks can be forced to require callbacks."""
    gym = ScenarioGym(
        state_callbacks=[ExampleCallback(), ExampleDependentCallback()]
    )
    try:
        gym.load_scenario(scenario_path)
    except ValueError as e:
        raise "Should not raise any value error:" from e

    gym = ScenarioGym(state_callbacks=[ExampleDependentCallback()])
    with pt.raises(ValueError):
        gym.load_scenario(scenario_path)

    gym = ScenarioGym(
        state_callbacks=[ExampleDependentCallback(), ExampleCallback()]
    )
    with pt.raises(ValueError):
        gym.load_scenario(scenario_path)
