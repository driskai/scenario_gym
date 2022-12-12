import numpy as np

from scenario_gym import ScenarioGym
from scenario_gym.pedestrian.agent import PedestrianAgent
from scenario_gym.pedestrian.social_force import SocialForce, SocialForceParameters


def test_social_force(pedestrian_scenario):
    """Test the social force model."""
    gym = ScenarioGym()
    gym.set_scenario(pedestrian_scenario)
    state = gym.state

    params = SocialForceParameters()
    agent = PedestrianAgent(
        pedestrian_scenario.entities[1],
        route=np.array([[0.0, 0.0], [20.0, 0.0]]),
        speed_desired=2.0,
        behaviour=SocialForce(params),
    )
    agent.reset(state)
    agent.step(state)
    assert agent.last_action.speed > 0
