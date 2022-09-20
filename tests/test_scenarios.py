import scenario_gym


def test_scenarios(all_scenarios):
    """Test all available scenarios."""
    gym = scenario_gym.ScenarioGym()
    for s in all_scenarios.values():
        gym.load_scenario(s)
        gym.rollout()
