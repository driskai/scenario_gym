import os
import warnings
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from lxml import etree
from scenariogeneration import xosc

from scenario_gym.catalog_entry import BoundingBox, CatalogEntry
from scenario_gym.entity import Entity
from scenario_gym.road_network import RoadNetwork
from scenario_gym.scenario import Scenario
from scenario_gym.trajectory import Trajectory


def import_scenario(osc_file: str, relabel: bool = True) -> Scenario:
    """
    Import a scenario from an OpenScenario file.

    Parameters
    ----------
    osc_file : str
        The filepath to the OpenScenario file.

    relabel : bool
        Whether to relabel entities after loading.

    """
    if not os.path.exists(osc_file):
        raise FileNotFoundError(f"Could not find file: {osc_file}.")

    cwd = os.path.dirname(osc_file)
    et = etree.parse(osc_file)
    osc_root = et.getroot()
    scenario = Scenario(name=os.path.basename(osc_file))
    scenario.scenario_path = osc_file
    entities = {}

    # Read catalogs:
    catalogs = {}
    for catalog_location in osc_root.iterfind("CatalogLocations/"):
        catalog_path = catalog_location.find("Directory").attrib["path"]
        catalog_path = os.path.join(cwd, catalog_path)
        for catalog_file in os.listdir(catalog_path):
            if catalog_file.endswith(".xosc"):
                name, entries = read_catalog(
                    os.path.join(catalog_path, catalog_file)
                )
                catalogs[name] = entries

    # Import road network:
    scene_graph_file = osc_root.find("RoadNetwork/SceneGraphFile")
    if scene_graph_file is not None:
        filepath = os.path.join(cwd, scene_graph_file.attrib["filepath"])
        filepath = filepath if filepath.endswith(".json") else f"{filepath}.json"
        if os.path.exists(filepath):
            scenario.road_network = RoadNetwork.create_from_json(filepath)
        else:
            warnings.warn(f"Could not find road network file: {filepath}.")

    # add the entities to the scenario
    for scenario_object in osc_root.iterfind("Entities/ScenarioObject"):
        entity_ref = scenario_object.attrib["name"]
        cat_ref = scenario_object.find("CatalogReference")
        if cat_ref is None:
            raise NotImplementedError(
                "Scenario objects can only be loaded from catalog references."
            )
        catalog_name = cat_ref.attrib["catalogName"]
        entry_name = cat_ref.attrib["entryName"]
        try:
            entity = deepcopy(catalogs[catalog_name][entry_name])
            entity.ref = entity_ref
            entities[entity_ref] = entity
        except KeyError as e:
            if catalog_name not in catalogs:
                warnings.warn(f"Could not find catalog: {catalog_name}")
            elif entry_name not in catalogs[catalog_name]:
                warnings.warn(
                    f"Could not find entry {entry_name} in catalog {catalog_name}."
                )
            else:
                raise e

    # Read init actions:
    for private in osc_root.iterfind("Storyboard/Init/Actions/Private"):
        entity_ref = private.attrib["entityRef"]
        for wp in private.iterfind(
            "PrivateAction/TeleportAction/Position/WorldPosition"
        ):
            tp = traj_point_from_time_and_position(0, wp)
            # Add a single-waypoint trajectory:
            if entity_ref in entities:
                entities[entity_ref].trajectory = Trajectory(np.stack([tp], axis=0))

    # Read maneuver actions:
    for maneuver_group in osc_root.iterfind("Storyboard/Story/Act/ManeuverGroup"):
        entity_ref = maneuver_group.find("Actors/EntityRef")
        assert (
            entity_ref is not None
        ), "Could not find entity reference in maneuver group."
        entity_ref = entity_ref.attrib["entityRef"]
        trajectory_points = []

        vertices = maneuver_group.findall(
            "Maneuver/Event/Action/PrivateAction/RoutingAction/"
            + "FollowTrajectoryAction/TrajectoryRef/Trajectory/Shape/"
            + "Polyline/Vertex"
        )
        # Also check the path without "TrajectoryRef" for backwards
        # compatibility with OpenSCENARIO 1.0:
        vertices.extend(
            maneuver_group.findall(
                "Maneuver/Event/Action/PrivateAction/RoutingAction/"
                + "FollowTrajectoryAction/Trajectory/Shape/Polyline/Vertex"
            )
        )

        for vertex in vertices:
            t = float(vertex.attrib["time"])
            wp = vertex.find("Position/WorldPosition")
            trajectory_points.append(traj_point_from_time_and_position(t, wp))
        if entity_ref in entities:
            traj_data = np.stack(trajectory_points, axis=0)
            entities[entity_ref].trajectory = Trajectory(traj_data)
            if (np.isnan(traj_data[:, 3]).sum() > 0) and (
                scenario.road_network is not None
            ):
                entities[entity_ref].trajectory.update_z_from_road_network(
                    scenario.road_network
                )

    for e in entities.values():
        if e.trajectory is not None:
            scenario.add_entity(e)
        elif e.ref == "ego":
            raise ValueError("Ego does not have a trajectory.")
        else:
            warnings.warn(f"Entity {e.ref} does not have a trajectory.")

    if relabel:
        scenario = relabel_scenario(scenario)

    return scenario


def relabel_scenario(scenario: Scenario) -> Scenario:
    """
    Relabel the entities of the scenario.

    Will be relabelled to ego, vehicle_1, vehicle_2,
    ..., pedestrian_1, ..., other_1, ...

    """
    vehicles, pedestrians, others = 0, 0, 0
    scenario.entities[0].ref = "ego"
    for e in scenario.entities[1:]:
        scenario._ref_to_entity.pop(e.ref)
        if e.catalog_entry.catalog_type == "Vehicle":
            e.ref = f"vehicle_{vehicles}"
            vehicles += 1
        elif e.catalog_entry.catalog_type == "Pedestrian":
            e.ref = f"pedestrian_{pedestrians}"
            pedestrians += 1
        else:
            e.ref = f"other_{others}"
            others += 1
        scenario._ref_to_entity[e.ref] = e
    return scenario


@lru_cache(maxsize=None)
def read_catalog(catalog_file: str) -> Tuple[str, Dict[str, Entity]]:
    """Read a catalog and return it's name and a dictionary of entities."""
    et = etree.parse(catalog_file)
    osc_root = et.getroot()
    catalog = osc_root.find("Catalog")
    # catalog_type = os.path.dirname(catalog_file).split("/")[-1].rstrip("s")
    name = catalog.attrib["name"]
    entries = {}
    for ce in catalog.getchildren():
        entry_name = ce.attrib["name"]
        cname = ce.tag.lower() + "Category"
        category = ce.attrib[cname] if cname in ce.attrib else None
        bb_center = ce.find("BoundingBox/Center")
        bb_dimensions = ce.find("BoundingBox/Dimensions")
        bb = BoundingBox(
            float(bb_dimensions.attrib["width"]),
            float(bb_dimensions.attrib["length"]),
            float(bb_center.attrib["x"]),
            float(bb_center.attrib["y"]),
        )
        catalog_entry = CatalogEntry(name, entry_name, category, ce.tag, bb)
        entity = Entity(catalog_entry)
        entries[entry_name] = entity

    return name, entries


def traj_point_from_time_and_position(t, world_position) -> np.ndarray:
    """Return the trajectory point as an array [t, x, y, z, h, p, r]."""
    return np.array(
        [
            t,
            float(world_position.attrib["x"]),
            float(world_position.attrib["y"]),
            float(world_position.attrib.get("z", np.NaN)),
            float(world_position.attrib.get("h", np.NaN)),
            float(world_position.attrib.get("p", np.NaN)),
            float(world_position.attrib.get("r", np.NaN)),
        ],
    )


def write_scenario(
    scenario: Scenario,
    filepath: str,
    base_catalog_path: str = "../Catalogs",
    base_road_network_path: str = "../Road_Networks",
    osc_minor_version: int = 0,
) -> None:
    """
    Write a recorded gym scenario to an OpenScenario file.

    Parameters
    ----------
    scenario : Scenario
        The scenario object.

    filepath : str
        The desired filepath.

    base_catalog_path : str
        Base path to the catalogs.

    base_road_network_path : str
        Base path to the road networks.

    osc_minor_version : int
        The OpenScenario minor version.

    """
    rn_name = scenario.road_network.path.split("/")[-1].split(".")[0]
    scenegraph = os.path.join(base_road_network_path, f"{rn_name}.json")
    rn = xosc.RoadNetwork("", scenegraph)

    entities = xosc.Entities()
    catalog = xosc.Catalog()
    for e in scenario.entities:
        ce = e.catalog_entry
        if ce.catalog_type not in catalog.catalogs:
            catalog_dir = os.path.join(
                base_catalog_path,
                f"{ce.catalog_type}Catalogs",
            )
            catalog.add_catalog(f"{ce.catalog_type}Catalog", catalog_dir)
        catalog_ref = xosc.CatalogReference(ce.catalog_name, ce.catalog_entry)
        entities.add_scenario_object(e.ref, catalog_ref)

    init = xosc.Init()
    for e in scenario.entities:
        if is_stationary(e):
            pose = e.recorded_poses[0][1:]
            assert np.isfinite(
                pose[2]
            ), f"Heading should be finite but is {pose[2]}"
            action = xosc.TeleportAction(
                xosc.WorldPosition(
                    *(float(p) if np.isfinite(p) else None for p in pose)
                )
            )
            init.add_init_action(e.ref, action)

    act = xosc.Act(
        scenario.name.replace(".xosc", ""),
        get_simulation_time_trigger(0),
    )
    maneuver_groups = []
    for idx, e in enumerate(scenario.entities):
        m_group = get_maneuver_group(
            e, osc_minor_version=osc_minor_version, check_stationary=(idx > 0)
        )
        if m_group:
            maneuver_groups.append(m_group)
            act.add_maneuver_group(m_group)

    story = xosc.Story(scenario.name.replace(".xosc", ""))
    story.add_act(act)
    sb = xosc.StoryBoard(init)
    sb.add_story(story)

    desc = f"\
Scenario {scenario.name.replace('.xosc', '')} recorded \
in the dRISK Scenario Gym subject to the dRISK License \
Agreement (https://drisk.ai/license/).\
"
    s = xosc.Scenario(
        desc,
        "∂RISK",
        xosc.ParameterDeclarations(),
        entities=entities,
        storyboard=sb,
        roadnetwork=rn,
        catalog=catalog,
        osc_minor_version=osc_minor_version,
    )
    element = ET.Element("OpenSCENARIO")
    element.append(s.header.get_element())
    element.append(s.parameters.get_element())
    element.append(s.catalog.get_element())
    element.append(s.roadnetwork.get_element())
    element.append(s.entities.get_element())
    element.append(s.storyboard.get_element())
    s.write_xml(filepath)


def get_simulation_time_trigger(
    t: float,
    delay: float = 0.0,
) -> xosc.ValueTrigger:
    """Get a simulation time trigger."""
    return xosc.ValueTrigger(
        "startSimTrigger",
        delay=delay,
        conditionedge=xosc.ConditionEdge.rising,
        valuecondition=xosc.SimulationTimeCondition(
            value=t, rule=xosc.Rule.greaterThan
        ),
    )


def is_stationary(
    e: Entity,
) -> bool:
    """
    Check if an entity is stationary for the entire scenario.

    Any nan values are replaced with 0s.
    """
    poses = e.recorded_poses
    return (
        len(
            np.unique(
                np.where(
                    np.isnan(poses[:, 1:]),
                    0.0,
                    poses[:, 1:],
                ),
                axis=0,
            )
        )
        <= 1
    )


def get_follow_trajectory_event(
    e: Entity,
    osc_minor_version: int = 0,
    check_stationary: bool = True,
) -> Optional[xosc.Event]:
    """Get a follow trajectory event for an entity."""
    if check_stationary and is_stationary(e):
        return None

    ts, poses = e.recorded_poses[:, 0], e.recorded_poses[:, 1:]
    positions = [
        xosc.WorldPosition(*(float(p) if np.isfinite(p) else None for p in pose))
        for pose in poses
    ]
    polyline = xosc.Polyline(ts.tolist(), positions)
    traj = xosc.Trajectory(f"{e.ref}_trajectory", False)
    traj.add_shape(polyline)
    follow_trajectory_action = xosc.FollowTrajectoryAction(
        traj,
        following_mode=xosc.FollowMode.position,
        reference_domain=xosc.ReferenceContext.absolute,
        scale=1,
        offset=0,
    )
    follow_trajectory_action.version_minor = osc_minor_version
    follow_trajectory_event = xosc.Event(
        f"{e.ref}_follow_trajectory_event",
        xosc.Priority.overwrite,
    )
    follow_trajectory_event.add_action(
        "follow_trajectory_action",
        follow_trajectory_action,
    )
    follow_trajectory_event.add_trigger(get_simulation_time_trigger(0))
    return follow_trajectory_event


def get_events(
    e: Entity, osc_minor_version: int = 0, check_stationary: bool = True
) -> List[xosc.Event]:
    """Get events for the given entity."""
    events = []
    follow_trajectory_event = get_follow_trajectory_event(
        e,
        osc_minor_version=osc_minor_version,
        check_stationary=check_stationary,
    )
    if follow_trajectory_event:
        events.append(follow_trajectory_event)
    # NOTE: ignoring non-trajectory actions for now
    return events


def get_maneuver(
    e: Entity, osc_minor_version: int = 0, check_stationary: bool = True
) -> Optional[xosc.Maneuver]:
    """Get maneuvers for the given entity."""
    events = get_events(
        e, osc_minor_version=osc_minor_version, check_stationary=check_stationary
    )
    if events:
        maneuver = xosc.Maneuver(f"{e.ref}_maneuver")
        for event in events:
            maneuver.add_event(event)
        return maneuver
    else:
        return None


def get_maneuver_group(
    e: Entity, osc_minor_version: int = 0, check_stationary: bool = True
) -> Optional[xosc.ManeuverGroup]:
    """Get the maneuver group for the given entity."""
    maneuver = get_maneuver(
        e, osc_minor_version=osc_minor_version, check_stationary=check_stationary
    )
    if maneuver:
        mangrp = xosc.ManeuverGroup(f"{e.ref}_maneuver_group")
        mangrp.add_actor(e.ref)
        mangrp.add_maneuver(maneuver)
        return mangrp
    else:
        return None
