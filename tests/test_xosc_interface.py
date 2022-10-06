from scenario_gym.entity import Pedestrian, Vehicle
from scenario_gym.xosc_interface import import_scenario, read_catalog


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
    assert "Greenwich_Road_Network_002" in scenario.road_network.path
    assert scenario.entities[0].ref == "ego"
    assert isinstance(scenario.entities[0], Vehicle)
    assert isinstance(scenario.entities[2], Pedestrian)
    assert scenario.entities[0].catalog_entry.rear_axle.wheel_diameter > 0
    assert scenario.entities[2].catalog_entry.mass > 0
