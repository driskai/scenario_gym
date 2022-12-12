from copy import deepcopy

import pytest as pt

from scenario_gym.road_network import RoadNetwork
from scenario_gym.scenario import Scenario
from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def example_scenario(all_scenarios):
    """Get a scenario to test."""
    return import_scenario(all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"])


@pt.fixture
def example_entity(example_scenario):
    """Get an entity to use as an example."""
    return deepcopy(example_scenario.entities[0])


def test_length(example_scenario):
    """Test the length of the scenario is computed correctly."""
    l = max((e.trajectory.max_t for e in example_scenario.entities))
    assert example_scenario.length == l, "Incorrect length."


def test_deepcopy_scenario(example_scenario):
    """Test deepcopying a scenario."""
    s_new = deepcopy(example_scenario)
    assert len(s_new.entities) == len(example_scenario.entities)
    assert id(s_new.road_network) != id(
        example_scenario.road_network
    ), "Road networks should be different."


def test_copy_scenarios(example_entity):
    """Test copying scenario."""
    s = Scenario(
        [example_entity],
        road_network=RoadNetwork(),
        properties={"x": 1, "y": 2},
    )
    s_new = s.copy()
    assert id(s_new.road_network) == id(
        s.road_network
    ), "Road networks should be the same."
    assert s_new.properties == s.properties, "Properties should be the same."
