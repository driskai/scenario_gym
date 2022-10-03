from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from odr_importer.road_objects.lane import Lane as xodrLane
from odr_importer.road_objects.lane import LaneOrientation
from odr_importer.road_objects.network import RoadNetwork as xodrRoadNetwork
from shapely.geometry import LineString, Polygon

from scenario_gym.road_network import Lane, LaneType, Road


def xodr_lane_to_sg(
    lane: xodrLane,
    right_hand_drive: bool,
    simplify_tolerance: float,
) -> Optional[Lane]:
    """Convert an OpenDRIVE lane to a scenario_gym lane."""
    if lane.type is None:
        return None

    lane_center_coords = lane.centre_line
    if (right_hand_drive and lane.orientation is LaneOrientation.LEFT) or (
        not right_hand_drive and lane.orientation is LaneOrientation.RIGHT
    ):
        lane_center_coords = np.flip(lane_center_coords, axis=0)

    lane_center = LineString(lane_center_coords[:, :2])
    if simplify_tolerance is not None:
        lane_center = lane_center.simplify(simplify_tolerance)

    elevation = lane_center_coords
    boundary = np.vstack(
        [
            lane.boundary_line,
            np.flip(lane.lane_offset_line, axis=0),
        ]
    )
    lane_boundary = Polygon(boundary)
    lane_type = LaneType[lane.type]

    return Lane(
        str(uuid4()),
        lane_boundary,
        lane_center,
        [],
        [],
        lane_type,
        elevation=elevation,
    )


def road_to_sg(
    xodr_road,
    right_hand_drive: bool,
    simplify_tolerance: float,
) -> List[Road]:
    """Convert an OpenDRIVE road into a list of roads."""
    roads, old_to_new_lanes = [], {}
    for _, xodr_lane_section in enumerate(xodr_road.lane_sections):
        if simplify_tolerance is not None:
            (
                x_boundary,
                y_boundary,
            ) = xodr_lane_section.boundary.exterior.simplify(simplify_tolerance).xy
        else:
            x_boundary, y_boundary = xodr_lane_section.boundary.exterior.xy

        road_boundary = Polygon(list(zip(x_boundary, y_boundary)))

        xyz_centre = xodr_lane_section.get_centre_line()
        road_center = LineString(xyz_centre[:, :2])
        if simplify_tolerance is not None:
            road_center = road_center.simplify(simplify_tolerance)
        road_elevation = xyz_centre

        lanes = []
        for lane in xodr_lane_section.lanes:
            sg_lane = xodr_lane_to_sg(lane, right_hand_drive, simplify_tolerance)
            if sg_lane is not None:
                lanes.append(sg_lane)
                old_to_new_lanes[lane] = sg_lane

        road = Road(
            str(uuid4()),
            road_boundary,
            road_center,
            lanes=lanes,
            elevation=road_elevation,
        )
        roads.append(road)

    return roads, old_to_new_lanes


def add_connection(conn: Tuple[Lane, Lane]) -> None:
    """Connect a pair of successive lanes."""
    pre, suc = conn
    if pre not in suc.predecessors:
        suc.predecessors.append(pre)
    if suc not in pre.successors:
        pre.successors.append(suc)


def xodr_to_sg_roads(
    road_network: xodrRoadNetwork,
    right_hand_drive: bool,
    simplify_tolerance: float,
) -> List[Road]:
    """
    Convert an odr_importer road network to a list of roads.

    Parameters
    ----------
    road_network : xodrRoadNetwork
        Imported OpenDRIVE file.

    simplify_tolerance : float
        Points per m for simplifying center and boundary lines.

    right_hand_drive : bool
        Whether the roads are right hand drive.

    """
    xodr_id_to_sg_road_objects: Dict[int, List[Road]] = {}
    xodr_lane_to_sg_lane: Dict[xodrLane, Lane] = {}

    for road in road_network.get_roads():
        roads, old_to_new_lanes = road_to_sg(
            road, right_hand_drive, simplify_tolerance
        )
        xodr_id_to_sg_road_objects[road.id] = roads
        xodr_lane_to_sg_lane.update(old_to_new_lanes)

    added_connections = set()
    for xodr_lane, sg_lane in xodr_lane_to_sg_lane.items():

        for xodr_predecessor_lane, _ in xodr_lane.predecessor_data:
            try:
                sg_pred_lane = xodr_lane_to_sg_lane[xodr_predecessor_lane]
            except KeyError:
                raise KeyError(
                    f"Failed to find {xodr_predecessor_lane} in xodr dict"
                )
            if xodr_lane.id > 0:
                connection_tuple = (sg_lane, sg_pred_lane)
            else:
                connection_tuple = (sg_pred_lane, sg_lane)

            if connection_tuple not in added_connections:
                add_connection(connection_tuple)
                added_connections.add(connection_tuple)

        for xodr_successor_lane, _ in xodr_lane.successor_data:
            try:
                sg_successor_lane = xodr_lane_to_sg_lane[xodr_successor_lane]
            except KeyError:
                raise KeyError(f"Failed to find {xodr_successor_lane} in xodr dict")

            if xodr_lane.id > 0:
                connection_tuple = (sg_successor_lane, sg_lane)
            else:
                connection_tuple = (sg_lane, sg_successor_lane)

            if connection_tuple not in added_connections:
                add_connection(connection_tuple)
                added_connections.add(connection_tuple)

    return list(set().union(*xodr_id_to_sg_road_objects.values()))
