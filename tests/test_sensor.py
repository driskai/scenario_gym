from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.sensor import RasterizedMapSensor


def test_sensor():
    """Test the rasterized sensor module."""
    # load a test scenario
    s = "./tests/input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc"
    gym = ScenarioGym()
    gym.load_scenario(s)
    e = gym.state.scenario.entities[0]

    # test with default layers
    sensor = RasterizedMapSensor(e, height=30, width=30, n=61)
    sensor._reset()
    out = sensor._step(gym.state)

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
    sensor._reset()
    out = sensor._step(gym.state)

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
