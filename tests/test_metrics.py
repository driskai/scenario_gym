import os

from scenario_gym.metrics import (
    CollisionMetric,
    EgoAvgSpeed,
    EgoDistanceTravelled,
    EgoMaxSpeed,
)
from scenario_gym.metrics.base import cache_mean
from scenario_gym.scenario_gym import ScenarioGym


def test_add_metric():
    """Test adding a metric to the scenario."""
    # try adding a metric
    base = "./tests/input_files/Scenarios/"
    s = "3fee6507-fd24-432f-b781-ca5676c834ef.xosc"
    gym = ScenarioGym(
        metrics=[
            EgoAvgSpeed(),
            EgoMaxSpeed(),
            EgoDistanceTravelled(name="test_metric"),
            CollisionMetric(),
        ]
    )
    gym.load_scenario(os.path.join(base, s))
    gym.rollout()
    mets = [m.get_state() for m in gym.metrics]
    assert 4.0 <= mets[0] <= 5.0, "Avg speed incorrect"
    assert 10.0 <= mets[1] <= 12.0, "Max speed incorrect"
    assert 90.0 <= mets[2] <= 110.0, "Distance incorrect"
    assert not mets[3], "Collisions incorrect"

    data = gym.get_metrics()
    assert "test_metric" in data, "Name did not get applied."
    assert len(data) == 4, "Incrorect number of metrics returned."


def test_cache_mean():
    """Test the cache_mean decorator."""
    # try adding a metric
    base = "./tests/input_files/Scenarios/"
    s_ids = [
        "3fee6507-fd24-432f-b781-ca5676c834ef.xosc",
        "41dac6fa-6f83-461e-a145-08692da5f3c7.xosc",
    ]

    # get mean
    gym = ScenarioGym(
        metrics=[
            EgoAvgSpeed(),
        ]
    )

    gym.load_scenario(os.path.join(base, s_ids[0]))
    gym.rollout()
    m1 = gym.metrics[0].get_state()

    gym.load_scenario(os.path.join(base, s_ids[1]))
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

    gym.load_scenario(os.path.join(base, s_ids[0]))
    gym.rollout()

    gym.load_scenario(os.path.join(base, s_ids[1]))
    gym.rollout()

    assert gym.metrics[0].previous_value == avg
    assert gym.metrics[0].previous_value == 0.0
