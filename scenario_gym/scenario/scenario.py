from __future__ import annotations

import json
import warnings
from contextlib import suppress
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np

from scenario_gym.entity import Entity, MiscObject, Pedestrian, Vehicle
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
        road_network: Optional[RoadNetwork] = None,
        actions: Optional[List[ScenarioAction]] = None,
        properties: Optional[Dict[Any, Any]] = None,
    ):
        self._entities = entities
        self._ref_to_entity: Dict[str, Entity] = {e.ref: e for e in entities}

        self.name = name
        self.road_network = road_network
        self.actions = actions if actions is not None else []
        self.properties = properties if properties is not None else {}

        self._vehicles: Optional[List[Entity]] = None
        self._pedestrians: Optional[List[Entity]] = None

    @property
    def entities(self) -> List[Entity]:
        """Get the entities in the scenario."""
        return self._entities

    @property
    def ego(self) -> Entity:
        """
        Get the ego entity.

        The ego entity is defined as the entity with the ref "ego" or the first
        entity if no entity has the ref "ego".
        """
        ego = self.entity_by_name("ego")
        if ego is not None:
            return ego
        return self.entities[0]

    @property
    def vehicles(self) -> List[Entity]:
        """Get the entities that have vehicle catalogs."""
        if self._vehicles is None:
            self._vehicles = [e for e in self.entities if isinstance(e, Vehicle)]
        return self._vehicles

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
            actions=[a.copy() for a in self.actions],
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
        idx = self._entities.index(e)
        scenario = self.copy() if not inplace else self
        scenario._entities.pop(idx)
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

        actions = []
        for a in scenario.actions:
            actions.append(a.translate(x, inplace=inplace))
        scenario.actions = actions

        return scenario

    def reset_start(self, entity: Optional[Entity] = None) -> Scenario:
        """Reset the start time to the start of an entity's trajectory."""
        if entity is None:
            entity = self.ego
        start_time = entity.trajectory.min_t
        return self.translate(np.array([-start_time, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

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
            if road_network.get("path") is not None:
                path = Path(road_network["path"])
                road_network = RoadNetwork.create_from_file(path)
            else:
                road_network = RoadNetwork.create_from_dict(road_network)

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
            road_network=road_network,
            actions=actions,
            properties=data.get("properties", {}),
        )

    def to_dict(
        self,
        road_network_path: Optional[str] = "../Road_Networks",
    ) -> Dict[str, Any]:
        """Write the scenario to a dictionary."""
        if self.road_network is None:
            road_network = None
        elif road_network_path is not None:
            if not Path(road_network_path).is_file():
                road_network_path = str(
                    Path(
                        road_network_path,
                        f"{self.road_network.name}.json",
                    )
                )
            road_network = {
                "path": road_network_path,
                "name": self.road_network.name,
            }
        else:
            road_network = self.road_network.to_dict()
        return {
            "entities": [e.to_dict() for e in self.entities],
            "name": self.name,
            "actions": [act.to_dict() for act in self.actions],
            "road_network": road_network,
            "properties": self.properties,
        }

    @classmethod
    def from_json(
        cls,
        path: str,
        road_network_dir: Optional[str] = None,
        e_classes: Tuple[Type[Entity]] = (Vehicle, Pedestrian, Entity),
        a_classes: Tuple[Type[ScenarioAction]] = (UpdateStateVariableAction,),
    ):
        """
        Load the scenario from a json file.

        Parameters
        ----------
        path : str
            The path to the json file.

        road_network_dir : str, optional
            The directory to search for the road network file. If the road network
            path in the json is absolute then this will be ignored. Otherwise the
            path used will be (road_network_dir / road network path) if the
            road_network_dir is relative otherwise it will be the `directory of
            path / road_network_dir / road network path`.

        e_classes : Tuple[Type[Entity]], optional
            The classes to use for loading entities.

        a_classes : Tuple[Type[ScenarioAction]], optional
            The classes to use for loading actions.

        """
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("road_network", {}).get("path") is not None:
            rn_path = Path(data["road_network"]["path"])
            if not rn_path.is_absolute():
                if road_network_dir is None:
                    rn_path = Path(path).parent / rn_path
                elif Path(road_network_dir).is_absolute():
                    rn_path = Path(road_network_dir) / rn_path
                else:
                    rn_path = Path(
                        Path(path).parent,
                        road_network_dir,
                        rn_path,
                    )
                print(rn_path)
                data["road_network"]["path"] = str(rn_path)
        return cls.from_dict(
            data,
            e_classes=e_classes,
            a_classes=a_classes,
        )

    def to_json(
        self, path, road_network_path: Optional[str] = "../Road_Networks"
    ) -> None:
        """Write the scenario to a json file."""
        data = self.to_dict(road_network_path=road_network_path)
        with open(path, "w") as f:
            json.dump(data, f)

    def describe(self) -> None:
        """Generate a text overview of the scenario."""
        rn = self.road_network.name if self.road_network is not None else "None"
        name = (
            self.name.replace(".xosc", "") if self.name is not None else "scenario"
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
        name = self.name if self.name is not None else "Scenario"
        plt.figure(figsize=figsize)
        if self.road_network is not None:
            for geom in self.road_network.driveable_surface.geoms:
                plt.fill(*geom.exterior.xy, c="gray", alpha=0.25)
                for i in geom.interiors:
                    plt.fill(*i.xy, c="white")
            for r in self.road_network.roads:
                plt.plot(*r.center.xy, c="white")
        for i, e in enumerate(self.entities):
            if i == 0:
                c = "r"
            elif isinstance(e, Pedestrian):
                c = "g"
            elif isinstance(e, MiscObject):
                c = "gray"
            else:
                c = "b"
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
