import numpy as np
import pytest as pt

from scenario_gym import RoadNetwork, Scenario, Trajectory
from scenario_gym.entity import Pedestrian, Vehicle
from scenario_gym.xosc_interface.catalogs import read_catalog


@pt.fixture(scope="session")
def ped_catalog(all_catalogs):
    """Get a pedestrian catalog entry."""
    for k, v in all_catalogs.items():
        if "ScenarioGymPedestrianCatalog" in k:
            ents = read_catalog(v)
            return ents[1]["pedestrian1"].catalog_entry
    raise ValueError("Pedestrian catalog not found.")


@pt.fixture(scope="session")
def veh_catalog(all_catalogs):
    """Get a pedestrian catalog entry."""
    for k, v in all_catalogs.items():
        if "ScenarioGymVehicleCatalog" in k:
            ents = read_catalog(v)
            return ents[1]["car1"].catalog_entry
    raise ValueError("Vehicle catalog not found.")


@pt.fixture(scope="session")
def six_way_rn(all_road_networks):
    """Get the six-way road network."""
    road_network = all_road_networks["dRisk Unity 6-lane Intersection"]
    return RoadNetwork.create_from_json(road_network)


@pt.fixture(scope="session")
def pedestrian_scenario(six_way_rn, veh_catalog, ped_catalog):
    """Create a scenario with pedestrians."""
    ego = Vehicle(
        veh_catalog,
        Trajectory(
            np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]]),
            fields=["t", "x", "y"],
        ),
        ref="ego",
    )
    entities = [ego]
    for i in range(1, 3):
        ped = Pedestrian(
            ped_catalog,
            Trajectory(
                np.array([[0.0, 0.0, i * 2.0], [10.0, 10.0, 0.0]]),
                fields=["t", "x", "y"],
            ),
            ref=f"ped_{i}",
        )
        entities.append(ped)
    return Scenario(
        entities,
        name="test_ped_scenario",
        road_network=six_way_rn,
    )
