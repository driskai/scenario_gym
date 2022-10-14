from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from pyxodr.road_objects.lane import Lane as xodrLane
from pyxodr.road_objects.network import RoadNetwork as xodrRoadNetwork
from shapely.geometry import LineString, Polygon

from scenario_gym.road_network import Lane, LaneType, Road


def xodr_lane_to_sg(
    lane: xodrLane,
    simplify_tolerance: float,
) -> Optional[Lane]:
    """Convert an OpenDRIVE lane to a scenario_gym lane."""
    if lane.type is None:
        return None

    lane_traffic_flow_line = LineString(lane.traffic_flow_line[:, :2])
    if simplify_tolerance is not None:
        lane_traffic_flow_line = lane_traffic_flow_line.simplify(simplify_tolerance)

    elevation = lane.traffic_flow_line
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
        lane_traffic_flow_line,
        [],
        [],
        lane_type,
        elevation=elevation,
    )


def road_to_sg(
    xodr_road,
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

        xyz_centre = xodr_lane_section.get_offset_line()
        road_center = LineString(xyz_centre[:, :2])
        if simplify_tolerance is not None:
            road_center = road_center.simplify(simplify_tolerance)
        road_elevation = xyz_centre

        lanes = []
        for lane in xodr_lane_section.lanes:
            sg_lane = xodr_lane_to_sg(lane, simplify_tolerance)
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
    simplify_tolerance: float,
) -> List[Road]:
    """
    Convert a pyxodr road network to a list of roads.

    Parameters
    ----------
    road_network : xodrRoadNetwork
        Imported OpenDRIVE file.

    simplify_tolerance : float
        Points per m for simplifying center and boundary lines.

    """
    xodr_id_to_sg_road_objects: Dict[int, List[Road]] = {}
    xodr_lane_to_sg_lane: Dict[xodrLane, Lane] = {}

    for road in road_network.get_roads():
        roads, old_to_new_lanes = road_to_sg(road, simplify_tolerance)
        xodr_id_to_sg_road_objects[road.id] = roads
        xodr_lane_to_sg_lane.update(old_to_new_lanes)

    for xodr_lane, sg_lane in xodr_lane_to_sg_lane.items():
        successor_lanes = xodr_lane.traffic_flow_successors
        for successor_lane in successor_lanes:
            try:
                successor_sg_lane = xodr_lane_to_sg_lane[successor_lane]
            except KeyError:
                raise KeyError(
                    f"Could not find successor lane {successor_lane} in "
                    + "OpenDRIVE to Scenario Gym dict; one of the successors of "
                    + f"{xodr_lane}."
                )
            add_connection((sg_lane, successor_sg_lane))

    return list(set().union(*xodr_id_to_sg_road_objects.values()))
