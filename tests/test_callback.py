import pytest as pt

from scenario_gym.callback import StateCallback
from scenario_gym.metrics import Metric
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.trajectory import Trajectory
from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def scenario(all_scenarios):
    """Get a scenario to test."""
    s = import_scenario(all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"])
    data = s.entities[0].trajectory.data.copy()
    data[0, 0] = 0.0
    s.entities[0].trajectory = Trajectory(data)
    return s


class ExampleCallback(StateCallback):
    """A simple callback to add a feature to the state."""

    my_feature: bool

    def _reset(self, state):
        """Reset the feature to True."""
        self.my_feature = True

    def __call__(self, state):
        """Set the feature to whether the time is <= 1."""
        self.my_feature = state.t <= 1


class ExampleSubclassCallback(ExampleCallback):
    """A simple callback to add a feature to the state."""

    def __call__(self, state):
        """Set the feature to whether the time is <= 1."""
        self.my_feature = state.t <= 1000


class ExampleDependentCallback(StateCallback):
    """A simple callback that is dependented on the TestCallback."""

    required_callbacks = [ExampleCallback]

    dependent_feature: bool

    def _reset(self, state):
        """Reset the feature to True."""
        self.dependent_feature = True

    def __call__(self, state):
        """Set the feature to the opposite of the existing features."""
        self.dependent_feature = not self.callbacks[0].my_feature


def test_get_callback(scenario):
    """Test getting a callback instance from the state."""
    cb1, cb2 = ExampleCallback(), ExampleDependentCallback()
    gym = ScenarioGym(state_callbacks=[cb1, cb2])
    gym.set_scenario(scenario)
    assert gym.state.get_callback(ExampleCallback) is cb1
    assert gym.state.get_callback(ExampleDependentCallback) is cb2


def test_callback(scenario):
    """Test that we can add a callback to the state."""
    gym = ScenarioGym(state_callbacks=[ExampleCallback()])
    gym.set_scenario(scenario)
    assert gym.state_callbacks[0].my_feature, "Feature should be added at reset."
    gym.rollout()
    assert (
        gym.state.t > 1 and not gym.state_callbacks[0].my_feature
    ), "Feature should be False."


def test_subclass_callback(scenario):
    """Test that we can add a callback to the state."""
    cb1, cb2 = ExampleSubclassCallback(), ExampleDependentCallback()
    gym = ScenarioGym(state_callbacks=[cb1, cb2])
    gym.set_scenario(scenario)
    assert gym.state_callbacks[0].my_feature, "Feature should be added at reset."
    gym.rollout()
    assert gym.state.get_callback(ExampleCallback) is cb1
    assert (
        gym.state.t > 1 and gym.state_callbacks[0].my_feature
    ), "Feature should be False."


def test_metric_requires(scenario):
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
        gym.set_scenario(scenario)
    except ValueError as e:
        raise "Should not raise any value error:" from e

    gym = ScenarioGym(state_callbacks=[], metrics=[ExampleMetric()])
    with pt.raises(ValueError):
        gym.set_scenario(scenario)


def test_callback_requires(scenario):
    """Test that the callbacks can be forced to require callbacks."""
    gym = ScenarioGym(
        state_callbacks=[ExampleCallback(), ExampleDependentCallback()]
    )
    try:
        gym.set_scenario(scenario)
    except ValueError as e:
        raise "Should not raise any value error:" from e

    gym = ScenarioGym(state_callbacks=[ExampleDependentCallback()])
    with pt.raises(ValueError):
        gym.set_scenario(scenario)
