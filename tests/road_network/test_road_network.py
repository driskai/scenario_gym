import os

import numpy as np
import pytest as pt
from shapely.geometry import LineString, Point, Polygon

from scenario_gym.road_network import (
    Building,
    Intersection,
    Pavement,
    Road,
    RoadGeometry,
    RoadNetwork,
    RoadObject,
)


@pt.fixture
def empty_road_network():
    """Create an empty road network."""
    return RoadNetwork()


@pt.fixture
def road_network(all_road_networks):
    """Load the 6-way road network."""
    return RoadNetwork.create_from_json(
        all_road_networks["dRisk Unity 6-lane Intersection"]
    )


@pt.fixture
def z_road_network():
    """Create a basic road network with elevation."""
    center = LineString(
        [
            [0.0, 0.0],
            [10.0, 0.0],
        ]
    )
    boundary = Polygon(
        [
            [0.0, -3.0],
            [0.0, 3.0],
            [10.0, 3.0],
            [10.0, -3.0],
        ]
    )
    elevation = np.array(
        [
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 2.0],
            [0.0, -3.0, 0.0],
            [0.0, 3.0, 0.0],
            [10.0, 3.0, 0.5],
            [10.0, -3.0, 0.5],
        ]
    )
    r = Road(
        "123",
        boundary,
        center,
        [],
        elevation=elevation,
    )
    return RoadNetwork(
        roads=[r],
        intersections=[],
    )


def test_empty(empty_road_network):
    """Test that the empty road network is empty."""
    assert (
        not empty_road_network.roads and not empty_road_network.intersections
    ), "Should all be empty."


def test_roads(road_network):
    """Check that roads have been loaded."""
    assert len(road_network.roads) > 0, "No roads."


def test_add_objects(empty_road_network, road_network):
    """Test adding a new road to the road network."""
    road = road_network.roads[0]
    empty_road_network.add_new_road_object([road], "new_objects")
    try:
        empty_road_network.new_objects
        empty_road_network._new_objects
        empty_road_network.add_new_objects
    except AttributeError as e:
        raise e

    assert empty_road_network.new_objects, "Object not added."


def test_intersections(road_network):
    """Check that intersections have been loaded."""
    assert len(road_network.intersections) > 0, "No intersections."


def test_lanes(road_network):
    """Check that lanes have been loaded."""
    assert len(road_network.lanes) > 0, "No lanes."


def test_road_lanes(road_network):
    """Check that the road has its lanes loaded."""
    assert len(road_network.roads[0].lanes) == 6, "No lanes."


def test_lane_center(road_network):
    """Test that the lane center exists and has positive length."""
    assert (
        road_network.roads[-1].lanes[0].center.length > 0
    ), "Incorrect lane center."


def test_pavements(road_network):
    """Check that there are no pavements/crossings but the properties exist."""
    assert (not road_network.pavements) and (
        not road_network.crossings
    ), "Road network does not have crossings or pavements."


def test_connectivity(road_network):
    """Test the connectivity of a road and intersection."""
    r = road_network.roads[0]
    i = road_network.intersections[0]
    assert (road_network.get_successor_lanes(r.lanes[0]) in i.lanes,) or (
        road_network.get_predecessor_lanes(r.lanes[0]) in i.lanes,
    ), "Intersection should follow road."
    assert i in road_network.get_intersections(
        r
    ), "Road should be connected to intersection"
    assert r in road_network.get_connecting_roads(
        i
    ), "Road should be connected to intersection"


def test_road_network_objects(road_network):
    """Check that the road_network_objects property works."""
    assert len(road_network.road_network_objects) > 0, "No road objects."


def test_road_network_geometries(road_network):
    """Check that the road_network_geometries property works."""
    assert len(road_network.road_network_geometries) > 0, "No road geoms."


def test_driveable_surface(road_network):
    """Test that the driveable surface has positive area."""
    assert road_network.driveable_surface.area > 0, "Driveable surface wrong."


def test_walkable_surface(road_network):
    """Test the walkable surface exists for a road network with pavements."""
    roads = road_network.roads.copy()
    intersections = road_network.intersections.copy()
    pavement = Pavement(
        "1234", Point(0.0, 0.0).buffer(2.0), LineString([[-1.0, 0.0], [1.0, 0.0]])
    )
    new_road_network = RoadNetwork(
        roads=roads,
        intersections=intersections,
        pavements=[pavement],
    )
    assert (
        "pavements" in new_road_network.object_names
    ), "pavements has not been added to the dict."
    assert (
        pavement in new_road_network.road_network_geometries
    ), "pavement missing from geoms."
    assert new_road_network.walkable_surface.area > 0, "Walkable surface wrong."


def test_impenetrable_surface(road_network):
    """Test the impenetrable surface exists for a road network with buildings."""
    roads = road_network.roads.copy()
    intersections = road_network.intersections.copy()
    building = Building("1234", Point(0.0, 0.0).buffer(2.0))
    new_road_network = RoadNetwork(
        roads=roads,
        intersections=intersections,
        buildings=[building],
    )
    assert (
        "buildings" in new_road_network.object_names
    ), "buildings has not been added to the dict."
    assert (
        building in new_road_network.road_network_geometries
    ), "building missing from geoms."
    assert (
        new_road_network.impenetrable_surface.area > 0
    ), "Impenetrable surface wrong."


def test_object_by_id(road_network):
    """Check that we can retrive a road by its id."""
    r = road_network.roads[0]
    assert r.id in road_network._object_by_id, "Could not find the road id."
    assert r == road_network.object_by_id(r.id), "Object found but not returned."


def get_get_lane_parent(road_network):
    """Test that we get the correct parent object of lane."""
    r = road_network.roads[0]
    l = r.lane_centers[0]
    assert road_network.get_lane_parent(l) == r, "Incorrect object returned."


def test_get_geometries_at_point(road_network):
    """Test the geometry getter method."""
    names, objs = road_network.get_geometries_at_point(0.0, 0.0)
    assert "Intersection" in names and "Lane" in names, "Returned wrong names."
    assert isinstance(objs[0], Intersection), "Should return an intersection."
    assert not road_network.get_geometries_at_point(999999.0, 0.0)[
        0
    ], "Should return no geometries at this point."


def test_clear_cache(road_network):
    """Test clearing the cache."""
    road_network._object_by_id
    road_network.driveable_surface
    road_network.roads
    road_network.clear_cache()
    assert all(
        (
            "_object_by_id" not in road_network.__dict__,
            "driveable_surface" not in road_network.__dict__,
            "roads" not in road_network.__dict__,
        )
    ), "Cached objects found."
    assert (
        road_network.get_lane_parent.__func__.cache_info().currsize == 0
    ), "Lru caches not cleared."
    assert (
        RoadNetwork.create_from_json.__func__.cache_info().currsize != 0
    ), "Class method caches cleared."


def test_elevation(road_network, z_road_network):
    """Test the elevation interpolation."""
    z = road_network.elevation_at_point(0.0, 0.0)
    assert z == 0, "No z-values provided so z should be 0."

    z0 = z_road_network.elevation_at_point(0.0, 0.0)
    assert z0 == 1.0, "z-value does not equal input value."

    z1 = z_road_network.elevation_at_point(3.2, -1.1)
    assert 0.0 <= z1 <= 2.0, "z-value outside of total range."

    z2 = z_road_network.elevation_at_point(-332, 1000.1)
    assert 0.0 <= z2 <= 2.0, "z-value outside of total range."

    z3 = z_road_network.elevation_at_point(np.zeros(3), np.zeros(3))
    assert np.allclose(z3, np.ones(3)), "Incorrect broadcasting"


def test_new_object(road_network):
    """Test creating a new geometry."""
    traffic_light = RoadObject("1234")
    road_network2 = RoadNetwork(
        roads=road_network.roads,
        intersections=road_network.intersections,
        traffic_lights=[traffic_light],
    )
    road_network2.traffic_lights
    assert len(road_network2.traffic_lights) == 1, "Road markings not found."
    assert traffic_light == road_network2.object_by_id(
        "1234"
    ), "Traffic light not found."


def test_new_geometry(road_network):
    """Test creating a new geometry."""
    boundary = Polygon(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    road_marking = RoadGeometry(
        "12345",
        boundary,
    )

    road_network2 = RoadNetwork(
        roads=road_network.roads,
        intersections=road_network.intersections,
        road_markings=[road_marking],
    )
    road_network2.road_markings
    assert len(road_network2.road_markings) == 1, "Road markings not found."
    assert road_marking == road_network2.object_by_id(
        "12345"
    ), "Road marking not found."


def test_all_road_networks(all_road_networks):
    """Test all road networks in the tests directory."""
    failed = []
    for r, path in all_road_networks.items():
        if not os.path.exists(path):
            failed.append((r, FileNotFoundError(path)))
        try:
            RoadNetwork.create_from_json(path)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            failed.append((r, e))
    assert len(failed) == 0, f"Road networks failed: {failed}."


def test_io(z_road_network):
    """Test that a we can save and load a road network from json."""
    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "input_files",
        "Road_Networks",
        "tmp_z_road_network.json",
    )
    z_road_network.to_json(output_path)

    loaded_road_network = RoadNetwork.create_from_json(output_path)
    assert all(
        (
            len(loaded_road_network.roads) == len(z_road_network.roads),
            len(loaded_road_network.intersections)
            == len(z_road_network.intersections),
        )
    ), "Incorrect roads and intersections in new road network."

    new_r = loaded_road_network.roads[0]
    old_r = z_road_network.roads[0]

    assert np.allclose(
        new_r.boundary.difference(old_r.boundary).area, 0.0
    ), "Different road boundaries."
    assert np.allclose(
        new_r.center.difference(old_r.center).area, 0.0
    ), "Different road centers."
    assert new_r.elevation is not None, "Elevation missing."
    assert np.allclose(
        new_r.elevation, old_r.elevation
    ), "Different elevation profiles."
