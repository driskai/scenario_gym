from dataclasses import dataclass

import numpy as np

from scenario_gym.observation import (
    Observation,
    SingleEntityObservation,
    combine_observations,
)
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


def test_merge_observations():
    """Test merging observations from different dataclasses."""

    @dataclass
    class X:
        a: int

    @dataclass
    class Y:
        b: int

    Z = combine_observations(X, Y)

    x = X(0)
    y = Y(1)
    z = Z.from_obs(x, y)
    z2 = Z(0, 1)
    assert (z.a, z.b) == (0, 1)
    assert z == z2
    assert isinstance(z, Observation)
