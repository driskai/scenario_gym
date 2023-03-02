import os
from typing import List, Optional
from xml.etree import ElementTree as ET

import numpy as np
from scenariogeneration import xosc

from scenario_gym.entity import Entity
from scenario_gym.scenario import Scenario
from scenario_gym.trajectory import is_stationary


def write_scenario(
    scenario: Scenario,
    filepath: str,
    base_road_network_path: str = "../Road_Networks",
    road_network_extenstion: str = "json",
    base_catalog_path: str = "../Catalogs",
    use_catalog_references: bool = True,
    osc_minor_version: int = 2,
) -> None:
    """
    Write a scenario to an OpenScenario file.

    Parameters
    ----------
    scenario : Scenario
        The scenario object.

    filepath : str
        The desired filepath.

    base_road_network_path : str
        Base path to the road networks.

    road_network_extenstion : str
        The extension of the road network file.

    base_catalog_path : str
        Base relative path to the catalogs.

    use_catalog_references : bool
        Whether to use catalog references for entities that have catalogs.

    osc_minor_version : int
        The OpenScenario minor version.

    """
    name = (
        scenario.name
        if scenario.name is not None
        else (os.path.splitext(os.path.basename(filepath)[-1])[0])
    )

    rn_name = (
        scenario.road_network.name
        if scenario.road_network.name is not None
        else None
    )
    scenegraph = os.path.join(
        base_road_network_path,
        f"{rn_name}.{road_network_extenstion}",
    )
    rn = xosc.RoadNetwork("", scenegraph)

    entities = xosc.Entities()
    catalog = xosc.Catalog()
    for e in scenario.entities:
        ce = e.catalog_entry
        if use_catalog_references and ce.catalog is not None:
            if ce.catalog_type not in catalog.catalogs:
                catalog_dir = os.path.join(
                    base_catalog_path,
                    ce.catalog.group_name,
                    f"{ce.catalog_type}Catalogs",
                )
                catalog.add_catalog(f"{ce.catalog_type}Catalog", catalog_dir)
            obj = xosc.CatalogReference(ce.catalog.name, ce.catalog_entry)
        else:
            obj = ce.to_xosc()
        entities.add_scenario_object(e.ref, obj)

    init = xosc.Init()
    for e in scenario.entities:
        if is_stationary(e.trajectory.data[:, 1:]):
            pose = e.trajectory.data[0, 1:]
            if not np.isfinite(pose[3]):
                raise ValueError(f"Heading must be finite but is {pose[2]}.")
            action = xosc.TeleportAction(
                xosc.WorldPosition(
                    *(float(p) if np.isfinite(p) else None for p in pose)
                )
            )
            init.add_init_action(e.ref, action)

    act = xosc.Act(
        filepath.replace(".xosc", ""),
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

    story = xosc.Story(name)
    story.add_act(act)
    sb = xosc.StoryBoard(init)
    sb.add_story(story)

    properties = xosc.Properties()
    for k, v in scenario.properties.items():
        if k == "files" and isinstance(v, list):
            for f in v:
                properties.add_file(f)
        else:
            properties.add_property(k, str(v))

    desc = (
        f"Scenario {name} recorded in the dRISK Scenario Gym subject to the dRISK "
        "License Agreement (https://drisk.ai/license/)."
    )
    s = xosc.Scenario(
        desc,
        "∂RISK",
        xosc.ParameterDeclarations(),
        entities=entities,
        storyboard=sb,
        roadnetwork=rn,
        catalog=catalog,
        osc_minor_version=osc_minor_version,
        header_properties=properties,
    )
    element = ET.Element("OpenSCENARIO")
    element.extend(
        (
            s.header.get_element(),
            s.parameters.get_element(),
            s.catalog.get_element(),
            s.roadnetwork.get_element(),
            s.entities.get_element(),
            s.storyboard.get_element(),
        )
    )
    s.write_xml(filepath)


def get_simulation_time_trigger(t: float, delay: float = 0.0) -> xosc.ValueTrigger:
    """Get a simulation time trigger."""
    return xosc.ValueTrigger(
        "startSimTrigger",
        delay=delay,
        conditionedge=xosc.ConditionEdge.rising,
        valuecondition=xosc.SimulationTimeCondition(
            value=t, rule=xosc.Rule.greaterThan
        ),
    )


def get_follow_trajectory_event(
    e: Entity,
    osc_minor_version: int = 0,
    check_stationary: bool = True,
) -> Optional[xosc.Event]:
    """Get a follow trajectory event for an entity."""
    if check_stationary and is_stationary(e.trajectory.data):
        return None
    ts, poses = e.trajectory.t, e.trajectory.data[:, 1:]
    positions = [
        xosc.WorldPosition(*(float(p) if np.isfinite(p) else None for p in pose))
        for pose in poses
    ]
    polyline = xosc.Polyline(ts.tolist(), positions)
    traj = xosc.Trajectory(f"{e.ref}_trajectory", False)
    traj.add_shape(polyline)
    follow_trajectory_action = xosc.FollowTrajectoryAction(
        traj,
        following_mode=xosc.FollowingMode.position,
        reference_domain=xosc.ReferenceContext.absolute,
        scale=1,
        offset=0,
    )
    follow_trajectory_action.version_minor = osc_minor_version
    follow_trajectory_event = xosc.Event(
        f"{e.ref}_follow_trajectory_event",
        xosc.Priority.override,
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
