import os
from copy import deepcopy
from tempfile import TemporaryDirectory

import numpy as np
from lxml import etree

from scenario_gym.catalog_entry import Catalog
from scenario_gym.entity import Pedestrian, Vehicle
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.xosc_interface import (
    import_scenario,
    read_catalog,
    write_catalogs,
    write_scenario,
)


def test_read_catalog(all_catalogs):
    """Test reading catalogs."""

    def test_entry(e):
        """Check the attributes of a loaded entity."""
        if isinstance(e, Vehicle):
            assert e.catalog_entry.rear_axle.wheel_diameter > 0
            assert e.catalog_entry.mass > 0
        elif isinstance(e, Pedestrian):
            assert e.catalog_entry.mass > 0
        else:
            raise TypeError(f"No catalog type for {e}.")

    for catalog in all_catalogs.values():
        if "Scenario_Gym" not in catalog:
            continue
        _, entries = read_catalog(catalog)
        for e in entries.values():
            try:
                test_entry(e)
            except TypeError as err:
                print(catalog)
                raise err


def test_import_scenario(all_scenarios):
    """Test importing a scenario."""
    path = all_scenarios["511c3b32-1bd5-4c16-9649-4c92da443301"]
    scenario = import_scenario(path)

    assert len(scenario.entities) == 3
    assert "Greenwich_Road_Network_002" in scenario.road_network.name
    assert scenario.entities[0].ref == "ego"
    assert isinstance(scenario.entities[0], Vehicle)
    assert isinstance(scenario.entities[2], Pedestrian)
    assert scenario.entities[0].catalog_entry.rear_axle.wheel_diameter > 0
    assert scenario.entities[2].catalog_entry.mass > 0


def test_write_scenario(all_scenarios) -> None:
    """
    Rollout a single scenario and write to a new scenario.

    Output the xosc then load it again and rollout the
    recorded version.

    """
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    out_path = scenario_path.replace("Scenarios", "Recordings").replace(
        ".xosc", "_test.xosc"
    )
    scenario = import_scenario(scenario_path).reset_start()

    # rollout
    gym = ScenarioGym()
    gym.set_scenario(scenario)
    gym.rollout()
    old_scenario = gym.state.scenario
    traj1 = old_scenario.entities[0].trajectory

    # output to OpenSCENARIO
    new_scenario = gym.state.to_scenario()
    new_scenario.properties["a"] = 1
    new_scenario.properties["files"] = ["a.txt"]
    write_scenario(new_scenario, out_path)

    # reload and test
    n_entities = len(gym.state.scenario.entities)
    n_stationary = sum(
        1 for t in gym.state.scenario.trajectories.values() if len(t) == 1
    )
    gym.load_scenario(out_path)
    assert gym.state.scenario.properties["a"] == 1, "Properties not copied."
    assert gym.state.scenario.properties["files"] == ["a.txt"]
    traj2 = gym.state.scenario.entities[0].trajectory
    assert (
        len(gym.state.scenario.entities) == n_entities
    ), "New scenario has a different number of entities."
    assert all(
        (
            isinstance(entity, type(old_entity))
            for entity, old_entity in zip(
                old_scenario.entities, gym.state.scenario.entities
            )
        )
    ), "Entities are not the same type."
    assert n_stationary == sum(
        1 for t in gym.state.scenario.trajectories.values() if len(t) == 1
    ), "New scenario has a different number of stationary entities."
    assert all(
        [
            np.allclose(traj1.position_at_t(0.0), traj2.position_at_t(0.0)),
            np.allclose(traj1.position_at_t(5.0), traj2.position_at_t(5.0)),
            np.allclose(traj1.position_at_t(10.0), traj2.position_at_t(10.0)),
        ]
    ), "Recorded and true trajectories differ."


def test_properties(all_scenarios):
    """Test loading properties from xosc."""
    s = all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"]
    s = import_scenario(s)
    assert s.properties["prop"] == 64, "Property not loaded correctly."
    assert s.properties["prop2"] == "64a", "Property not loaded correctly."
    assert s.properties["files"] == ["test.txt"], "Property not loaded correctly."


def test_write_scenario_without_references(all_scenarios) -> None:
    """
    Rollout a single scenario and write to a new scenario.

    Output the xosc then load it again and rollout the
    recorded version.

    """
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    out_path = scenario_path.replace("Scenarios", "Recordings").replace(
        ".xosc", "_test_no_refs.xosc"
    )
    scenario = import_scenario(scenario_path).reset_start()

    # rollout
    gym = ScenarioGym()
    gym.set_scenario(scenario)
    gym.rollout()

    # output to OpenSCENARIO
    new_scenario = gym.state.to_scenario()
    write_scenario(new_scenario, out_path, use_catalog_references=False)

    et = etree.parse(out_path)
    assert len(et.getroot().findall("Entities/ScenarioObject/Vehicle")) == len(
        new_scenario.vehicles
    ), "Not all vehicles were written."
    assert (
        len(et.getroot().findall("Entities/ScenarioObject/CatalogReference")) == 0
    ), "Catalog references were written."


def test_write_catalogs(all_scenarios):
    """Test creating catalogs."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]
    scenario = import_scenario(scenario_path)
    entries = [deepcopy(scenario.entities[0].catalog_entry)]

    new_catalog = Catalog(
        entries[0].catalog.name + "_test",
        "Test_Scenario_Gym",
    )
    for ce in entries:
        ce.catalog = new_catalog

    with TemporaryDirectory() as tmpdir:
        write_catalogs(tmpdir, entries)

        assert os.path.isdir(
            os.path.join(tmpdir, "Test_Scenario_Gym")
        ), "Catalog directory not created."
        assert os.path.isdir(
            os.path.join(tmpdir, "Test_Scenario_Gym", "VehicleCatalogs")
        ), "Vehicle catalog directory not created."

        cat_path = os.path.join(
            tmpdir,
            "Test_Scenario_Gym",
            "VehicleCatalogs",
            f"{entries[0].catalog.name}.xosc",
        )
        assert os.path.isfile(cat_path), "Catalog not written."

        read_cat, entries2 = read_catalog(cat_path)

    assert read_cat.group_name == "Test_Scenario_Gym", "Catalog not written."
    assert (
        entries2["car1"].catalog_entry.mass == entries[0].mass
    ), "Catalog entry not written."
