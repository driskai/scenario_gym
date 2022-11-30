from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.sensor.common import (
    CombinedSensor,
    EgoLocalizationSensor,
    FutureCollisionDetector,
    GlobalCollisionDetector,
    KeyboardInputDetector,
)
from scenario_gym.sensor.map import RasterizedMapSensor


def test_combined_sensor(all_scenarios):
    """Test combining a sensor."""
    s = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    gym = ScenarioGym()
    gym.load_scenario(s)
    ego = gym.state.scenario.entities[0]
    state = gym.state

    sensor = CombinedSensor(
        ego,
        EgoLocalizationSensor(ego),
        FutureCollisionDetector(ego),
        GlobalCollisionDetector(ego),
        KeyboardInputDetector(ego),
    )
    initial_obs = sensor.step(state)
    obs = sensor.step(state)
    assert len(initial_obs) == 4 == len(obs), "Should have two observations."

    # check attributes exist
    obs[1].future_collision
    obs[2].collisions
    obs[3].last_keystroke


def test_map_sensor(all_scenarios):
    """Test the rasterized sensor module."""
    # load a test scenario
    s = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    gym = ScenarioGym()
    gym.load_scenario(s)
    e = gym.state.scenario.entities[0]

    # test with default layers
    sensor = RasterizedMapSensor(e, height=30, width=30, n=61)
    sensor._reset(gym.state)
    out = sensor._step(gym.state).map

    assert out.shape == (
        61,
        61,
        len(sensor.layers),
    ), f"Invalid shape: {out.shape}."
    assert out[
        ..., sensor.layers.index("driveable_surface")
    ].any(), "The ego starts on the road so the road should be in the map."
    assert out[
        30, 30, sensor.layers.index("entity")
    ], "The ego is at (0, 0) so this should be True."

    # test with all layers
    sensor = RasterizedMapSensor(
        e, layers=RasterizedMapSensor._all_layers, height=30, width=30, n=61
    )
    sensor._reset(gym.state)
    out = sensor._step(gym.state).map

    assert out.shape == (
        61,
        61,
        len(sensor.layers),
    ), f"Invalid shape: {out.shape}."
    assert out[
        ..., sensor.layers.index("driveable_surface")
    ].any(), "The ego starts on the road so the road should be in the map."
    assert out[
        30, 30, sensor.layers.index("entity")
    ], "The ego is at (0, 0) so this should be True."
