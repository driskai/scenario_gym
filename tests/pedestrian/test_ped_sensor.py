from scenario_gym import ScenarioGym
from scenario_gym.pedestrian.sensor import PedestrianSensor


def test_ped_sensor(pedestrian_scenario):
    """Test the pedestrian sensor."""
    ped = pedestrian_scenario.entities[1]
    sensor = PedestrianSensor(ped, distance_threshold=10)

    gym = ScenarioGym()
    gym.set_scenario(pedestrian_scenario)
    state = gym.state
    obs = sensor.reset(state)
    assert len(obs.near_peds) == 1, "Should have found one nearby pedestrian."
