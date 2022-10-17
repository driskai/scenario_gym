import numpy as np
import pytest as pt

from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def scenario_path(all_scenarios):
    """Get a path for a scenario to test."""
    return all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]


def test_scenario(scenario_path):
    """Test running a scenario."""
    gym = ScenarioGym(timestep=0.1)
    gym.load_scenario(scenario_path)
    s = gym.state.scenario

    for _ in range(10):
        gym.step()

    assert np.allclose(s.t, 1.0), "Incorrect time."
    assert np.allclose(s.dt, 0.1), "Incorrect timestep."

    e = s.entities[0]
    distances = [np.linalg.norm(e_.pose[:3] - e.pose[:3]) for e_ in s.entities[1:]]

    assert (
        len(s.get_entities_in_radius(e.pose[0], e.pose[1], np.min(distances) - 0.1))
        == 1
    ) and (
        len(s.get_entities_in_radius(e.pose[0], e.pose[1], np.max(distances) + 1))
        == 1 + len(distances)
    ), "Incorrect entities returned."

    names, geoms = s.get_road_info_at_entity(e)
    assert "Road" in names, "Entity is on the road."


def test_import_scenario(scenario_path):
    """Test importing a scenario from xosc."""
    s = import_scenario(scenario_path)
    assert all((e.catalog_entry.catalog_category is not None for e in s.entities))
    assert s.road_network is not None

    s = import_scenario(scenario_path)
    e1, e2 = s.entities[1], s.entities[2]
    assert e1.ref[:-2] == e1.catalog_entry.catalog_category.lower()
    assert e2.ref[:-2] == e2.catalog_entry.catalog_category.lower()
