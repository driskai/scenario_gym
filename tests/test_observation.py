import numpy as np

from scenario_gym.observation import SingleEntityObservation
from scenario_gym.scenario_gym import ScenarioGym


def test_observation(all_scenarios) -> None:
    """Test the standard observation."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    gym = ScenarioGym()
    gym.load_scenario(scenario_path)

    ego = gym.state.scenario.entities[0]
    obs = SingleEntityObservation(
        ego,
        *gym.state.get_entity_data(ego),
    )

    assert np.allclose(obs.pose, gym.state.poses[ego])
    assert np.allclose(obs.velocity, gym.state.velocities[ego])
