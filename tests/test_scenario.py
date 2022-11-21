from copy import deepcopy

import pytest as pt

from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def example_scenario(all_scenarios):
    """Get a scenario to test."""
    return import_scenario(all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"])


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
