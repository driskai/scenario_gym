from scenario_gym.metrics.rss import RSS, RSSDistances
from scenario_gym.scenario_gym import ScenarioGym


def test_add_rss(all_scenarios):
    """Test adding the RSS metric to the scenario."""
    s = all_scenarios["3fee6507-fd24-432f-b781-ca5676c834ef"]
    gym = ScenarioGym(
        state_callbacks=[RSSDistances()],
        metrics=[RSS()],
    )
    gym.load_scenario(s)
    gym.rollout()

    data = gym.get_metrics()
    assert (
        "RSS_safe_longitudinal" in data and "RSS_safe_lateral" in data
    ), "Name did not get applied."
    assert len(data) == 2, "Incorrect number of metrics returned."
    assert (
        type(data["RSS_safe_longitudinal"]) is bool
    ), "Non-boolean RSS metric output for safe longitudinal distance."
    assert (
        type(data["RSS_safe_lateral"]) is bool
    ), "Non-boolean RSS metric output for safe lateral distance."
