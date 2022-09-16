import os

from scenario_gym.environment import ScenarioGym


def test_env() -> None:
    """Test the gym environment."""
    base = os.path.dirname(__file__)
    scenario_path = os.path.join(
        base,
        "input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc",
    )

    env = ScenarioGym()
    env.load_scenario(scenario_path)

    obs, done = env.reset(seed=123), False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        assert obs in env.observation_space
