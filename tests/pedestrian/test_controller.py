from scenario_gym import ScenarioGym
from scenario_gym.pedestrian.action import PedestrianAction
from scenario_gym.pedestrian.controller import PedestrianController


def test_controller(pedestrian_scenario):
    """Test the pedestrian controller."""
    gym = ScenarioGym()
    gym.set_scenario(pedestrian_scenario)
    state = gym.state

    entity = pedestrian_scenario.entities[1]
    pos = state.poses[entity]

    controller = PedestrianController(entity)
    action = PedestrianAction(5.0, 0.0)

    controller.reset(state)
    new_pose = controller.step(state, action)

    assert new_pose[0] > pos[0]
    assert new_pose[3] == 0.0
