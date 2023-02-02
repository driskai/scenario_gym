from __future__ import annotations

import json
import warnings
from contextlib import suppress
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np

from scenario_gym.entity import Entity
from scenario_gym.entity.pedestrian import Pedestrian
from scenario_gym.entity.vehicle import Vehicle
from scenario_gym.road_network import RoadNetwork
from scenario_gym.scenario.actions import ScenarioAction, UpdateStateVariableAction
from scenario_gym.trajectory import Trajectory
from scenario_gym.utils import cached_property


class Scenario:
    """
    The scenario_gym representation of a scenario.

    A scenario consists of a set of entities and a road network. The entities have
    trajectories and catalog entries and may have additional entity specific
    properties. The scenario also may have a list of actions which can specify
    events that occur. E.g. a traffic light changing color.
    """

    def __init__(
        self,
        entities: List[Entity],
        name: Optional[str] = None,
        path: Optional[str] = None,
        road_network: Optional[RoadNetwork] = None,
        actions: Optional[List[ScenarioAction]] = None,
        properties: Optional[Dict[Any, Any]] = None,
    ):
        self._entities = entities
        self._ref_to_entity: Dict[str, Entity] = {e.ref: e for e in entities}

        self.name = name
        self.path = path
        self.road_network = road_network
        self.actions = actions if actions is not None else []
        self.properties = properties if properties is not None else {}

        self._vehicles: Optional[List[Entity]] = None
        self._pedestrians: Optional[List[Entity]] = None
        self._catalog_locations: Optional[Dict[str, str]] = None

    @property
    def entities(self) -> List[Entity]:
        """Get the entities in the scenario."""
        return self._entities

    @property
    def vehicles(self) -> List[Entity]:
        """Get the entities that have vehicle catalogs."""
        if self._vehicles is None:
            self._vehicles = [e for e in self.entities if isinstance(e, Vehicle)]
        return self._vehicles

    @property
    def catalog_locations(self) -> Dict[str, str]:
        """Get the locations of each catalog."""
        if self._catalog_locations is None:
            catalogs = set((e.catalog_entry.catalog for e in self.entities))
            self._catalog_locations = {
                catalog.catalog_name: catalog.rel_path for catalog in catalogs
            }
        return self._catalog_locations

    @property
    def pedestrians(self) -> List[Entity]:
        """Get the entities that have pedestrian catalogs."""
        if self._pedestrians is None:
            self._pedestrians = [
                e for e in self.entities if isinstance(e, Pedestrian)
            ]
        return self._pedestrians

    @property
    def trajectories(self) -> Dict[str, Trajectory]:
        """Return a dictionary mapping entity references to the trajectory."""
        return {e.ref: e.trajectory for e in self.entities}

    @cached_property
    def length(self) -> float:
        """Return the length of the scenario in seconds."""
        return max([t.max_t for t in self.trajectories.values()])

    def entity_by_name(self, e_ref: str) -> Optional[Entity]:
        """Return an entity given a unique reference."""
        with suppress(KeyError):
            return self._ref_to_entity[e_ref]

    def __copy__(self) -> Scenario:
        """Create a copy of a scenario without copying the road network."""
        return self.__class__(
            name=f"Copy of {self.name}" if self.name is not None else None,
            road_network=self.road_network,
            actions=deepcopy(self.actions),
            entities=[e.copy() for e in self.entities],
            properties=self.properties,
        )

    def copy(self) -> Scenario:
        """Create a copy of the scenario."""
        return copy(self)

    def add_entity(self, e: Entity, inplace: bool = False) -> Scenario:
        """Create a new scenario with the entity added."""
        if e.ref in self._ref_to_entity:
            i = 0
            while True:
                new_ref = f"{e.ref}_{i}"
                if new_ref not in self._ref_to_entity:
                    break
                i += 1
            old_ref = e.ref
            e.ref = new_ref
            warnings.warn(
                f"An entity with ref {old_ref} exists. Adding with ref {new_ref}."
            )
        scenario = self.copy() if not inplace else self
        scenario._entities.append(e)
        scenario._ref_to_entity[e.ref] = e
        scenario._vehicles = None
        scenario._pedestrians = None
        return scenario

    def remove_entity(self, e: Entity, inplace: bool = False) -> Scenario:
        """Create a new scenario with the entity added."""
        scenario = self.copy() if not inplace else self
        scenario._entities.remove(e)
        scenario._ref_to_entity.pop(e.ref)
        scenario._vehicles = None
        scenario._pedestrians = None
        return scenario

    def make_ego(self, e: Entity, inplace: bool = False) -> Scenario:
        """Set e to the ego entity."""
        try:
            idx = self._entities.index(e)
        except ValueError:
            idx = None
        scenario = self.copy() if not inplace else self

        if idx is not None:
            e = scenario._entities.pop(idx)
        else:
            scenario._ref_to_entitiy[e.ref] = e
        scenario._entities.insert(0, e)
        scenario.vehicles = None
        scenario.pedestrians = None
        return scenario

    def add_action(self, action: ScenarioAction, inplace: bool = False) -> Scenario:
        """Add an action to the scenario."""
        scenario = self.copy() if not inplace else self
        scenario.actions.append(action)
        return scenario

    def translate(self, x: np.ndarray, inplace: bool = False) -> Scenario:
        """Return a new scenario with all entities translated."""
        scenario = self.copy() if not inplace else self
        for e in scenario.entities:
            e.trajectory = e.trajectory.translate(x)
        return scenario

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        e_classes: Tuple[Type[Entity]] = (Vehicle, Pedestrian, Entity),
        a_classes: Tuple[Type[ScenarioAction]] = (UpdateStateVariableAction,),
    ):
        """Load the scenario from a dictionary."""
        entities = []
        for e_data in data["entities"]:
            for Ent in e_classes:
                if Ent.__name__ == e_data.get("entity_class", Entity):
                    break
            entities.append(Ent.from_dict(e_data))

        road_network = data.get("road_network")
        if road_network is not None:
            road_network = RoadNetwork.create_from_json(road_network["path"])

        actions = []
        for a_data in data.get("actions", ()):
            for Act in a_classes:
                if Act.__name__ == a_data.get(
                    "action_class", "UpdateStateVariableAction"
                ):
                    break
            actions.append(Act.from_dict(a_data))

        return cls(
            entities,
            name=data.get("name"),
            path=data.get("path"),
            road_network=road_network,
            actions=actions,
            properties=data.get("properties", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Write the scenario to a dictionary."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "name": self.name,
            "path": self.path,
            "road_network": {
                "name": self.road_network.name,
                "path": self.road_network.path,
            }
            if self.road_network is not None
            else None,
            "actions": [act.to_dict() for act in self.actions],
            "properties": self.properties,
        }

    @classmethod
    def from_json(
        cls,
        path: str,
        e_classes: Tuple[Type[Entity]] = (Vehicle, Pedestrian, Entity),
        a_classes: Tuple[Type[ScenarioAction]] = (UpdateStateVariableAction,),
    ):
        """Load the scenario from a json file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, e_classes=e_classes, a_classes=a_classes)

    def to_json(self, path) -> None:
        """Write the scenario to a json file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def describe(self) -> None:
        """Generate a text overview of the scenario."""
        rn = self.road_network.path.split("/")[-1].split(".")[0]
        name = (
            self.name.replace(".xosc", "")
            if self.name is not None
            else self.path.split("/")[-1].split(".")[0]
        )
        title = f"Scenario: {name}"
        under_header = "=" * len(title)

        entity_header = "Entity".ljust(10) + "Type".ljust(10) + "Cateogry".ljust(10)
        entities = ""
        for e in self.entities:
            entities += (
                f"{e.ref}".ljust(10)
                + f"{e.type}".ljust(10)
                + f"{e.catalog_entry.catalog_category}".ljust(10)
                + "\n"
            )

        print(
            f"""
{title}
{under_header}
Filepath: {self.path}
Road network: {rn}
Number of entities: {len(self.entities)}
Total duration: {self.length:.4}s

Entities
--------
{entity_header}
{entities}
"""
        )

    def plot(self, figsize: Tuple[int, int] = (10, 10), show: bool = True) -> None:
        """
        Visualise the scenario.

        Parameters
        ----------
        figsize : Tuple[int, int]
            The figure size.

        show : bool
            If set to False will not call `plt.show` so the figure can be modified
            or saved.

        """
        name = (
            self.name.replace(".xosc", "")
            if self.name
            else self.scenario_path.split("/")[-1].split(".")[0]
        )
        plt.figure(figsize=figsize)
        for geom in self.road_network.driveable_surface.geoms:
            plt.fill(*geom.exterior.xy, c="gray", alpha=0.25)
            for i in geom.interiors:
                plt.fill(*i.xy, c="white")
        for r in self.road_network.roads:
            plt.plot(*r.center.xy, c="white")
        for i, e in enumerate(self.entities):
            c = "r" if i == 0 else "b"
            plt.plot(*e.trajectory.data[:, [1, 2]].T, c=c, label=e.ref)
            plt.plot(*e.trajectory.data[0, [1, 2]].T, c=c, marker="o")
        data = np.vstack([e.trajectory.data[:, [1, 2]] for e in self.entities])
        b_min, b_max = data.min(0), data.max(0)
        plt.axis("equal")
        plt.xlim(b_min[0] - 10.0, b_max[0] + 10.0)
        plt.ylim(b_min[1] - 10.0, b_max[1] + 10.0)
        plt.legend()
        plt.title(name)
        if show:
            plt.show()
