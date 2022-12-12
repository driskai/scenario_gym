"""
Import scenarios from the Argoverse 2 Motion Forecasting dataset.

https://www.argoverse.org/about.html#terms-of-use
"""
import json
from contextlib import suppress
from pathlib import Path
from typing import Dict

import numpy as np
from shapely.geometry import LineString, Polygon

from scenario_gym.catalog_entry import BoundingBox, Catalog, CatalogEntry
from scenario_gym.entity import Entity
from scenario_gym.road_network import (
    Lane,
    LaneType,
    Road,
    RoadGeometry,
    RoadNetwork,
)
from scenario_gym.scenario import Scenario
from scenario_gym.trajectory import Trajectory

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        """
\tPandas is required for this integration.
\tInstall by `pip install pandas`.
"""
    )


class Lane(Lane):
    """Add the argoverse information to the lane object."""

    def __init__(
        self,
        is_intersection: bool,
        left_neighbour_id: str,
        right_neighbour_id: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_intersection = is_intersection
        self.left_neighbour_id = left_neighbour_id
        self.right_neighbour_id = right_neighbour_id


track_types = [
    "VEHICLE",
    "PEDESTRIAN",
    "MOTORCYCLIST",
    "CYCLIST",
    "BUS",
    "STATIC",
    "BACKGROUND",
    "CONSTRUCTION",
    "RIDERLESS_BICYCLE",
    "UNKNOWN",
]


class Catalogs:
    """
    Catalogs used in the argoverse dataset.

    Using the dimensions defined in their scenario_visualization.py
    """

    vehicle_box = BoundingBox(1.8, 3.8, 0.0, 0.0)
    argoverse_catalog = Catalog("ArgoverseCatalog", None)
    vehicle = CatalogEntry(
        argoverse_catalog,
        "vehicle",
        "car",
        "Vehicle",
        vehicle_box,
    )

    pedestrian_box = BoundingBox(0.4, 0.4, 0.0, 0.0)
    pedestrian = CatalogEntry(
        argoverse_catalog,
        "pedestrian",
        "pedestrian",
        "Pedestrian",
        pedestrian_box,
    )

    motorbike_box = BoundingBox(0.2, 0.8, 0.0, 0.0)
    motorcyclist = CatalogEntry(
        argoverse_catalog,
        "motorcyclist",
        "motorbike",
        "Vehicle",
        motorbike_box,
    )

    cyclist_box = BoundingBox(0.7, 2.0, 0.0, 0.0)
    cyclist = CatalogEntry(
        "ArgoverseCatalog",
        "cyclist",
        "bicycle",
        "Vehicle",
        cyclist_box,
    )

    bus_box = BoundingBox(2.8, 11.0, 0.0, 0.0)
    bus = CatalogEntry(
        argoverse_catalog,
        "bus",
        "bus",
        "Vehicle",
        bus_box,
    )

    riderless_bicycle_box = BoundingBox(0.3, 1.5, 0.0, 0.0)
    riderless_bicycle = CatalogEntry(
        argoverse_catalog,
        "riderless_bicycle",
        "obstacle",
        "Vehicle",
        riderless_bicycle_box,
    )


def import_argoverse_scenario(path: str) -> Scenario:
    """
    Import a recorded scenario from the argoverse data.

    This assumes fixed bounding box sizes for each entity. For now
    ignoring object types: background, construction,
    static and unkown.
    """
    path = Path(path)
    scenario_id = path.parts[-1]

    pq_path = Path(path, f"scenario_{scenario_id}.parquet")
    main_df = pd.read_parquet(pq_path).sort_values("timestep")
    dfs = list(main_df.groupby("track_id"))
    all_ids = sorted(main_df["track_id"].unique())
    assert "AV" in all_ids, "No AV found to use as ego."
    all_ids.remove("AV")

    entities = []
    for track_id, df in dfs:

        if track_id != "AV" and not df["observed"].any():
            continue

        # get catalog
        object_type = df["object_type"].iloc[0]
        with suppress(AttributeError):
            catalog_entry = getattr(Catalogs, object_type)

        # get start and end in seconds
        start = df["start_timestamp"].iloc[0] / 1e9
        end = df["end_timestamp"].iloc[0] / 1e9
        num = df["num_timestamps"].iloc[0] - 1
        t_scale = (end - start) / num

        # build trajectory
        traj_data = df[
            [
                "timestep",
                "position_x",
                "position_y",
                "heading",
            ]
        ].to_numpy()
        traj_data[:, 0] = t_scale * traj_data[:, 0]

        v0 = df[["velocity_x", "velocity_y"]].iloc[0].to_numpy()
        t_pre = np.array(
            [
                -0.1,
                *(traj_data[0, [1, 2]] - 0.1 * v0),
                traj_data[0, 3],
            ]
        )

        traj_data = np.concatenate(
            [
                t_pre[None],
                traj_data,
            ],
            axis=0,
        )
        trajectory = Trajectory(traj_data, fields=["t", "x", "y", "h"])

        entity_ref = (
            f"entity_{1+all_ids.index(track_id)}" if track_id != "AV" else "ego"
        )
        entity = Entity(catalog_entry, ref=entity_ref)
        entity.trajectory = trajectory
        entities.append(entity)

    ego = None
    for e in entities:
        if e.ref == "ego":
            ego = e
            break
    if ego is not None:
        entities.remove(ego)
        entities.insert(0, ego)

    road_network_data = json.load(
        open(Path(path, f"log_map_archive_{scenario_id}.json"), "r")
    )
    road_network = create_argoverse_road_network(road_network_data)

    scenario = Scenario(
        entities,
        name=scenario_id,
        path=str(path.absolute()),
        road_network=road_network,
    )
    return scenario


def create_argoverse_road_network(data: Dict) -> RoadNetwork:
    """Create a road network from the argoverse log map."""
    driveable_areas = []
    for area in data["drivable_areas"].values():
        poly = Polygon([[v["x"], v["y"]] for v in area["area_boundary"]])
        driveable_areas.append(
            RoadGeometry(
                area["id"],
                poly,
            )
        )

    roads = []
    for l_data in data["lane_segments"].values():
        center = LineString([[d["x"], d["y"]] for d in l_data["centerline"]])
        boundary = center.buffer(1.75, cap_style=2)
        lane = Lane(
            l_data["is_intersection"],
            l_data["left_neighbor_id"],
            l_data["right_neighbor_id"],
            l_data["id"],
            boundary,
            center,
            l_data["successors"],
            l_data["predecessors"],
            LaneType.driving,  # data["lane_type"],
        )
        roads.append(
            Road(
                f"road_{l_data['id']}",
                boundary,
                center,
                [lane],
            )
        )

    # TODO pedestrian crossings
    return RoadNetwork(
        roads=roads,
        intersections=[],
        driveable_areas=driveable_areas,
    )
