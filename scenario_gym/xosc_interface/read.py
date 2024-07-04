import os
import warnings
from contextlib import suppress
from typing import Dict, List, Optional, Type

import numpy as np
from lxml import etree
from lxml.etree import Element

from scenario_gym.entity import Entity, Pedestrian, Vehicle
from scenario_gym.road_network import RoadNetwork
from scenario_gym.scenario import Scenario, ScenarioAction
from scenario_gym.scenario.actions import UserDefinedAction
from scenario_gym.trajectory import Trajectory
from scenario_gym.utils import load_properties_from_xml

from .catalogs import load_object, read_catalog


def import_scenario(
    osc_file: str,
    relabel: bool = True,
    entity_types: Optional[List[Type[Entity]]] = None,
) -> Scenario:
    """
    Import a scenario from an OpenScenario file.

    Parameters
    ----------
    osc_file : str
        The filepath to the OpenScenario file.

    relabel : bool
        Whether to relabel entities after loading.

    entity_types : Optional[List[Type[Entity]]]
        Additional entity types to use when loading the scenario. Can be used to
        allow custom entities to be loaded from OpenSCENARIO.

    """
    if not os.path.exists(osc_file):
        raise FileNotFoundError

    cwd = os.path.dirname(osc_file)
    et = etree.parse(osc_file)
    osc_root = et.getroot()
    entities = {}

    # Read catalogs:
    catalogs: Dict[str, Dict[str, Entity]] = {}
    for catalog_location in osc_root.iterfind("CatalogLocations/"):
        rel_catalog_path = catalog_location.find("Directory").attrib["path"]
        if not os.path.isabs(rel_catalog_path):
            catalog_path = os.path.join(cwd, rel_catalog_path)
        else:
            catalog_path = rel_catalog_path
        for catalog_file in os.listdir(catalog_path):
            if catalog_file.endswith(".xosc"):
                catalog, entries = read_catalog(
                    os.path.join(catalog_path, catalog_file),
                    entity_types=entity_types,
                )
                catalogs[catalog.name] = entries

    # Import road network:
    rn_path = None
    scene_graph_file = osc_root.find("RoadNetwork/SceneGraphFile")
    if scene_graph_file is not None:
        rn_path = scene_graph_file.attrib["filepath"]
    else:
        logic_file = osc_root.find("RoadNetwork/LogicFile")
        if logic_file is not None:
            rn_path = logic_file.attrib["filepath"]

    road_network = None
    if rn_path is not None:
        if not os.path.isabs(rn_path):
            filepath = os.path.join(cwd, rn_path)
        else:
            filepath = rn_path
        extension = os.path.splitext(filepath)[1]
        if extension == "":
            filepath = f"{filepath}.json"
        with suppress(FileNotFoundError):
            road_network = RoadNetwork.create_from_file(filepath)

    # add the entities to the scenario
    for scenario_object in osc_root.iterfind("Entities/ScenarioObject"):
        entity_ref = scenario_object.attrib["name"]
        cat_ref = scenario_object.find("CatalogReference")
        if cat_ref is None:
            ent = None
            for element in scenario_object.getchildren():
                ent = load_object(element)
            if ent is None:
                warnings.warn(
                    "Could not find a catalog reference or entry for entity "
                    f"{ent.tag}.Perhaps you need to add an entity type to "
                    "`entity_types`."
                )
            else:
                ent.ref = entity_ref
                entities[entity_ref] = ent
        else:
            catalog_name = cat_ref.attrib["catalogName"]
            entry_name = cat_ref.attrib["entryName"]
            try:
                entity = catalogs[catalog_name][entry_name].copy()
                entity.ref = entity_ref
                entities[entity_ref] = entity
            except KeyError as e:
                if catalog_name not in catalogs:
                    warnings.warn(f"Could not find catalog: {catalog_name}")
                elif entry_name not in catalogs[catalog_name]:
                    warnings.warn(
                        f"Could not find entry {entry_name} in catalog "
                        f"{catalog_name}."
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
    actions = []
    for man_group in osc_root.iterfind("Storyboard/Story/Act/ManeuverGroup"):
        entity_ref = man_group.find("Actors/EntityRef")
        assert (
            entity_ref is not None
        ), "Could not find entity reference in maneuver group."
        entity_ref = entity_ref.attrib["entityRef"]
        entity = entities.get(entity_ref)

        if entity is None:
            continue

        for event in man_group.findall("Maneuver/Event"):
            traj_action = event.find(
                "Action/PrivateAction/RoutingAction/FollowTrajectoryAction"
            )
            if traj_action is not None:
                trajectory = read_trajectory_event(
                    traj_action,
                    road_network=road_network,
                )
                if trajectory is not None:
                    entity.trajectory = trajectory
                    continue

            user_action = event.find("Action/UserDefinedAction")
            start_trigger = event.find("StartTrigger")
            if user_action is not None:
                actions.extend(
                    load_user_defined_action(
                        entity,
                        user_action,
                        start_trigger=start_trigger,
                    )
                )

    header = osc_root.find("FileHeader")
    if header is not None:
        properties, files = load_properties_from_xml(header)
        if files and "files" not in properties:
            properties["files"] = files
    else:
        properties = {}

    scenario = Scenario(
        list(entities.values()),
        name=os.path.splitext(os.path.basename(osc_file))[0],
        road_network=road_network,
        properties=properties,
        actions=actions,
    )

    if relabel:
        scenario = relabel_scenario(scenario)

    return scenario


def read_trajectory_event(
    trajectory_action: Element,
    road_network: Optional[RoadNetwork] = None,
) -> Optional[Trajectory]:
    """Read a trajectory event from a ManeuverGroup."""
    # trajectory points
    trajectory_points = []
    vertices = trajectory_action.findall(
        "TrajectoryRef/Trajectory/Shape/Polyline/Vertex"
    )
    vertices.extend(trajectory_action.findall("Trajectory/Shape/Polyline/Vertex"))
    if not vertices:
        return None

    for vertex in vertices:
        t = float(vertex.attrib["time"])
        wp = vertex.find("Position/WorldPosition")
        trajectory_points.append(traj_point_from_time_and_position(t, wp))

    traj_data = np.stack(trajectory_points, axis=0)
    if (np.isnan(traj_data[:, 3]).sum() > 0) and (road_network is not None):
        traj_data[:, 3] = road_network.elevation_at_point(
            traj_data[:, 1], traj_data[:, 2]
        )

    return Trajectory(traj_data)


def load_user_defined_action(
    entity: Entity,
    user_action: Element,
    start_trigger: Optional[Element] = None,
) -> List[ScenarioAction]:
    """Load a user-defined action from an OpenSCENARIO file."""
    cond = start_trigger.find(
        "ConditionGroup/Condition/ByValueCondition/SimulationTimeCondition"
    )
    t = float(cond.attrib.get("value"))

    acts = []
    for child in user_action.getchildren():
        acts.append(
            UserDefinedAction(
                t,
                child.tag,
                entity.ref,
                {k: v for k, v in child.attrib.items()},
            )
        )
    return acts


def relabel_scenario(scenario: Scenario) -> Scenario:
    """
    Relabel the entities of the scenario.

    Will be relabelled to ego, vehicle_1, vehicle_2,
    ..., pedestrian_1, ..., other_1, ...

    """
    vehicles, pedestrians, others = 0, 0, 0
    scenario.entities[0].ref = "ego"
    old_to_new = {}
    for e in scenario.entities[1:]:
        cur = e.ref
        with suppress(KeyError):
            scenario._ref_to_entity.pop(cur)
        if isinstance(e, Vehicle):
            e.ref = f"vehicle_{vehicles}"
            vehicles += 1
        elif isinstance(e, Pedestrian):
            e.ref = f"pedestrian_{pedestrians}"
            pedestrians += 1
        else:
            e.ref = f"other_{others}"
            others += 1
        scenario._ref_to_entity[e.ref] = e
        old_to_new[cur] = e.ref
    for action in scenario.actions:
        if action.entity_ref in old_to_new:
            action.entity_ref = old_to_new[action.entity_ref]
    return scenario


def traj_point_from_time_and_position(t, world_position) -> np.ndarray:
    """Return the trajectory point as an array [t, x, y, z, h, p, r]."""
    return np.array(
        [
            t,
            float(world_position.attrib["x"]),
            float(world_position.attrib["y"]),
            float(world_position.attrib.get("z", np.nan)),
            float(world_position.attrib.get("h", np.nan)),
            float(world_position.attrib.get("p", np.nan)),
            float(world_position.attrib.get("r", np.nan)),
        ],
    )
