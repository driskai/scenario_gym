import os
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Type

import numpy as np
from lxml import etree

from scenario_gym.catalog_entry import CatalogEntry
from scenario_gym.entity import Entity
from scenario_gym.road_network import RoadNetwork
from scenario_gym.scenario import Scenario
from scenario_gym.trajectory import Trajectory

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
    osc_root = resolve_parameters(osc_root)
    scenario = Scenario(name=os.path.basename(osc_file))
    scenario.scenario_path = osc_file
    entities = {}

    # Read catalogs:
    catalogs: Dict[str, Dict[str, Entity]] = {}
    for catalog_location in osc_root.iterfind("CatalogLocations/"):
        rel_catalog_path = catalog_location.find("Directory").attrib["path"]
        catalog_path = os.path.join(cwd, rel_catalog_path)
        for catalog_file in os.listdir(catalog_path):
            if catalog_file.endswith(".xosc"):
                name, entries = read_catalog(
                    os.path.join(catalog_path, catalog_file),
                    entity_types=entity_types,
                )
                catalogs[name] = entries
                scenario.add_catalog_location(name, rel_catalog_path)

    # Import road network:
    logic_file = osc_root.find("RoadNetwork/LogicFile")
    if logic_file is not None:
        rn_path = logic_file.attrib["filepath"]
    else:
        scene_graph_file = osc_root.find("RoadNetwork/SceneGraphFile")
        rn_path = scene_graph_file.attrib["filepath"]

    filepath = os.path.join(cwd, rn_path)
    extension = os.path.splitext(filepath)[1]
    if extension == "":
        filepath = f"{filepath}.json"

    if not os.path.exists(filepath):
        warnings.warn(f"Could not find road network file: {filepath}.")

    if extension == ".json" or extension == "":
        scenario.road_network = RoadNetwork.create_from_json(filepath)
    elif extension == ".xodr":
        scenario.road_network = RoadNetwork.create_from_xodr(filepath)

    # add the entities to the scenario
    for scenario_object in osc_root.iterfind("Entities/ScenarioObject"):
        entity_ref = scenario_object.attrib["name"]
        cat_ref = scenario_object.find("CatalogReference")
        if cat_ref is not None:
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
                        f"Could not find entry {entry_name} in "
                        + f"catalog {catalog_name}."
                    )
                else:
                    raise e
        else:
            # No catalog entry, assume the EntityObjects are defined as child
            # elements
            entity_object_allowed_tags = {
                "CatalogReference",
                "Vehicle",
                "Pedestrian",
                "MiscObject",
                "ExternalObjectReference",
            }
            for sub_element in scenario_object.getchildren():
                if sub_element.tag in entity_object_allowed_tags:
                    entity = load_object(
                        None, sub_element, entity_types=entity_types
                    )
                    if entity is None:
                        entity = Entity(
                            CatalogEntry.from_xml(catalog_name, sub_element)
                        )
                    entity.ref = entity_ref
                    entities[entity_ref] = entity

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


def resolve_parameters(xosc_root: etree._Element) -> etree._Element:
    """
    Resolve all parameter instances in an xosc file to their values.

    Parameters
    ----------
    xosc_root : etree._Element
        etree root Element of the xosc file.

    Returns
    -------
    etree._Element
        Modified etree Element with all parameter instances replaced with their
        values.

    """
    resolved_xosc_root = deepcopy(xosc_root)

    def _recursive_modify_keys(
        original_element: etree._Element,
        resolved_element: etree._Element,
        parameter_to_values: Dict[str, str],
    ):
        parameter_to_values = deepcopy(parameter_to_values)
        for parameter_declarations in original_element.findall(
            "ParameterDeclarations"
        ):
            for parameter_declaration in parameter_declarations.findall(
                "ParameterDeclaration"
            ):
                value = parameter_declaration.attrib["value"]
                parameter_to_values[parameter_declaration.attrib["name"]] = value

        for key, value in original_element.attrib.items():
            if (
                value != ""
                and str(value)[0] == "$"
                and original_element.tag != "ParameterDeclaration"
            ):
                resolved_value = parameter_to_values[str(value)[1:]]
                resolved_element.attrib[key] = resolved_value

        for child_element, child_resolved_element in zip(
            original_element.getchildren(), resolved_element.getchildren()
        ):
            _recursive_modify_keys(
                child_element, child_resolved_element, parameter_to_values
            )

    _recursive_modify_keys(xosc_root, resolved_xosc_root, {})

    return resolved_xosc_root


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
        scenario._ref_to_entity.pop(cur)
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
        old_to_new[cur] = e.ref
    for action in scenario.actions:
        action.entity_ref = old_to_new[action.entity_ref]
    return scenario


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
