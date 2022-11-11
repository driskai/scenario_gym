import pytest as pt

from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def scenario_path(all_scenarios):
    """Get a path for a scenario to test."""
    return all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]


def test_import_scenario(scenario_path):
    """Test importing a scenario from xosc."""
    s = import_scenario(scenario_path)
    assert all((e.catalog_entry.catalog_category is not None for e in s.entities))
    assert s.road_network is not None

    s = import_scenario(scenario_path)
    e1, e2 = s.entities[1], s.entities[2]
    assert e1.ref[:-2] == e1.catalog_entry.catalog_category.lower()
    assert e2.ref[:-2] == e2.catalog_entry.catalog_category.lower()
