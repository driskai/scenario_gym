import numpy as np

from scenario_gym.observation import Observation
from scenario_gym.scenario_gym import ScenarioGym


def test_observation(all_scenarios) -> None:
    """Test the standard observation."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    gym = ScenarioGym()
    gym.load_scenario(scenario_path)

    ego = gym.state.scenario.entities[0]
    obs = Observation.from_entity(gym.state, ego)

    assert obs.entity == ego
    assert np.allclose(obs.pose, gym.state.poses[ego])


class ExampleObservation1(Observation):
    """Example observation with a custom method."""

    def get_z_data(self, state):
        """Get the current z value for the entity."""
        return state.poses[self.entity][2]


class ExampleObservation2(Observation):
    """Example observation with a custom method."""

    def get_h_data(self, state):
        """Get the current heading value for the entity."""
        return state.poses[self.entity][3]


class ExampleObservation3(ExampleObservation2):
    """Example observation subclass."""

    pass


def test_combine_observations(all_scenarios) -> None:
    """Test combining observations."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    gym = ScenarioGym()
    gym.load_scenario(scenario_path)

    class ExampleObservation(
        ExampleObservation1,
        ExampleObservation3,
    ):
        """Example combined observation.."""

        pass

    ego = gym.state.scenario.entities[0]
    obs = ExampleObservation.from_entity(gym.state, ego)
    z = obs.get_z_data(gym.state)
    h = obs.get_h_data(gym.state)
    assert obs.pose[2] == z
    assert obs.pose[3] == h
