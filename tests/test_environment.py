from scenario_gym.environment import ScenarioGym


def test_env(all_scenarios) -> None:
    """Test the gym environment."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    env = ScenarioGym()
    env.load_scenario(scenario_path)

    obs, done = env.reset(seed=123), False
    while not done:
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        assert obs in env.observation_space
