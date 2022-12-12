from scenario_gym.callback import StateCallback
from scenario_gym.metrics import (
    CollisionMetric,
    EgoAvgSpeed,
    EgoDistanceTravelled,
    EgoMaxSpeed,
)
from scenario_gym.metrics.base import Metric, cache_mean
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.state import State


def test_add_metric(all_scenarios):
    """Test adding a metric to the scenario."""
    s = all_scenarios["3fee6507-fd24-432f-b781-ca5676c834ef"]
    gym = ScenarioGym(
        metrics=[
            EgoAvgSpeed(),
            EgoMaxSpeed(),
            EgoDistanceTravelled(name="test_metric"),
            CollisionMetric(),
        ]
    )
    gym.load_scenario(s)
    gym.rollout()
    mets = [m.get_state() for m in gym.metrics]
    assert 4.0 <= mets[0] <= 5.0, "Avg speed incorrect"
    assert 10.0 <= mets[1] <= 12.0, "Max speed incorrect"
    assert 90.0 <= mets[2] <= 110.0, "Distance incorrect"
    assert not mets[3], "Collisions incorrect"

    data = gym.get_metrics()
    assert "test_metric" in data, "Name did not get applied."
    assert len(data) == 4, "Incrorect number of metrics returned."


def test_cache_mean(all_scenarios):
    """Test the cache_mean decorator."""
    s_ids = [
        "3fee6507-fd24-432f-b781-ca5676c834ef",
        "41dac6fa-6f83-461e-a145-08692da5f3c7",
    ]

    # get mean
    gym = ScenarioGym(
        metrics=[
            EgoAvgSpeed(),
        ]
    )

    gym.load_scenario(all_scenarios[s_ids[0]])
    gym.rollout()
    m1 = gym.metrics[0].get_state()

    gym.load_scenario(all_scenarios[s_ids[1]])
    gym.rollout()
    m2 = gym.metrics[0].get_state()

    avg = 0.5 * (m1 + m2)

    gym = ScenarioGym(
        metrics=[
            cache_mean(EgoAvgSpeed)(),
        ]
    )

    assert gym.metrics[0].previous_value == 0.0
    assert gym.metrics[0]._prev_count == 0.0

    gym.load_scenario(all_scenarios[s_ids[0]])
    gym.rollout()

    gym.load_scenario(all_scenarios[s_ids[1]])
    gym.rollout()

    assert gym.metrics[0].previous_value == avg
    assert gym.metrics[0].previous_value == 0.0


class ExampleCallback(StateCallback):
    """Callback for testing."""

    def _reset(self, state) -> None:
        """Set the value to 0."""
        self.value = 0.0

    def __call__(self, state: State) -> None:
        """Get the current time * 10."""
        self.value = state.t * 10.0


class DepMetric(Metric):
    """Example metric depending on a callback."""

    required_callbacks = [ExampleCallback]

    def _reset(self, state):
        """Set the value to the value from the callback."""
        self.value = self.callbacks[0].value

    def _step(self, state):
        """Set the value to the value from the callback / current time."""
        self.value = self.callbacks[0].value / state.t

    def get_state(self):
        """Return the value."""
        return self.value


def test_dependent_callback(all_scenarios):
    """Test using a metric that requires a callback."""
    s = all_scenarios["3fee6507-fd24-432f-b781-ca5676c834ef"]
    cb = ExampleCallback()
    met = DepMetric()
    gym = ScenarioGym(state_callbacks=[cb], metrics=[met])
    gym.load_scenario(s)
    assert met.callbacks == [cb]
    gym.rollout()
    assert cb.value == gym.state.t * 10.0
    assert met.get_state() == 10.0
