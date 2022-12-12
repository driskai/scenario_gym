import numpy as np

from scenario_gym import ScenarioGym
from scenario_gym.pedestrian.agent import PedestrianAgent
from scenario_gym.pedestrian.random_walk import RandomWalk, RandomWalkParameters


def test_random_walk(pedestrian_scenario):
    """Test the random walk model."""
    gym = ScenarioGym()
    gym.set_scenario(pedestrian_scenario)
    state = gym.state

    params = RandomWalkParameters()
    agent = PedestrianAgent(
        pedestrian_scenario.entities[1],
        route=np.array([[0.0, 0.0], [20.0, 0.0]]),
        speed_desired=2.0,
        behaviour=RandomWalk(params),
    )
    agent.reset(state)
    agent.step(state)
    assert agent.last_action.speed > 0
