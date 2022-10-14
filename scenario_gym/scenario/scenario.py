import warnings
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.vectorized import contains

from scenario_gym.entity import BatchReplayEntity, Entity
from scenario_gym.entity.pedestrian import Pedestrian
from scenario_gym.entity.vehicle import Vehicle
from scenario_gym.road_network import RoadNetwork, RoadObject
from scenario_gym.scenario.actions import ScenarioAction
from scenario_gym.scenario.utils import detect_collisions
from scenario_gym.trajectory import Trajectory


class Scenario:
    """
    The scenario_gym representation of a scenario.

    A scenario consists of a set of entities and a road network. The entities have
    trajectories and catalog entries and may have additional entity specific
    properties. The scenario also may have a list of actions which can specify
    events that occur. E.g. a traffic light changing color.

    """

    def __init__(self, name: Optional[str] = None):
        """Init the scenario."""
        self.name = name

        self.scenario_path: Optional[str] = None
        self.road_network: Optional[RoadNetwork] = None
        self.agents: Dict[str, "Agent"] = {}  # noqa F821
        self.non_agents = BatchReplayEntity()

        self._actions: List[ScenarioAction] = []
        self._catalog_locations: Dict[str, str] = {}
        self._entities: List[Entity] = []
        self._vehicles: Optional[List[Entity]] = None
        self._pedestrians: Optional[List[Entity]] = None
        self._t: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._collisions: Optional[Dict[Entity, List[Entity]]] = None
        self._ref_to_entity: Dict[str, Entity] = {}

    @property
    def entities(self) -> List[Entity]:
        """Get the entities in the scenario."""
        return self._entities

    def add_entity(self, e: Entity, catalog_location: Optional[str] = None) -> None:
        """Add a new entity to the scenario."""
        if e.catalog_entry.catalog_name not in self.catalog_locations:
            if catalog_location is None:
                warnings.warn(
                    f"Catalog file {e.catalog_entry.catalog_name} does not exist "
                    "and has not been provided. This will mean the scenario cannot "
                    "be written to OpenSCENARIO. Add the catalog location by "
                    "providing `catalog_location=[filepath to xosc]` or via "
                    "`scenario.add_catalog_location`."
                )
            else:
                self.add_catalog_location(
                    e.catalog_entry.catalog_name,
                    catalog_location,
                )
        elif catalog_location is not None and (
            self.catalog_locations[e.catalog_entry.catalog_name] == catalog_location
        ):
            raise ValueError(
                "Different `catalog_location` already exists for "
                f"{e.catalog_entry.catalog_name}."
            )

        e.t = self.t
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
        self._entities.append(e)
        self._ref_to_entity[e.ref] = e
        self._vehicles = None
        self._pedestrians = None

    def entity_by_name(self, e_ref: str) -> Optional[Entity]:
        """Return an entity given a unique reference."""
        try:
            e = self._ref_to_entity[e_ref]
            return e
        except KeyError:
            return None

    def make_ego(self, e: Entity) -> None:
        """Set e to the ego entity."""
        if e not in self.entities:
            self.add_entity(e)
        self._entities.remove(e)
        self._entities.insert(0, e)

    @property
    def trajectories(self) -> Dict[str, Trajectory]:
        """Return a dictionary mapping entity references to the trajectory."""
        return {e.ref: e.trajectory for e in self.entities}

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
    def catalog_locations(self) -> Dict[str, str]:
        """
        Get the filepaths for each catalog.

        These are indexed by the name.
        """
        return self._catalog_locations

    def add_catalog_location(self, catalog_name: str, filepath: str) -> None:
        """Add a catalog location to the scenario."""
        self._catalog_locations[catalog_name] = filepath

    @property
    def actions(self) -> List[ScenarioAction]:
        """Return the actions attached to the scenario."""
        return self._actions

    def add_action(self, action: ScenarioAction) -> None:
        """Add an action to the scenario."""
        self._actions.append(action)

    @property
    def t(self) -> float:
        """Get the  current time in seconds (s)."""
        return self._t

    @t.setter
    def t(self, t: float) -> None:
        self.prev_t = self._t
        self._t = t
        for e in self.entities:
            e.t = t
        self._collisions: Optional[Dict[Entity, List[Entity]]] = None
        return self._t

    @property
    def prev_t(self) -> float:
        """Get the current previous time (s)."""
        return self._prev_t

    @prev_t.setter
    def prev_t(self, prev_t: float) -> None:
        self._prev_t = prev_t

    @property
    def dt(self) -> float:
        """Return the previous timestep."""
        return self.t - self.prev_t

    @cached_property
    def length(self) -> float:
        """Return the length of the scenario in seconds."""
        return max([t.max_t for t in self.trajectories.values()])

    def describe(self) -> None:
        """Generate a text overview of the scenario."""
        rn = self.road_network.path.split("/")[-1].split(".")[0]
        name = (
            self.name.replace(".xosc", "")
            if self.name
            else self.scenario_path.split("/")[-1].split(".")[0]
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
Filepath: {self.scenario_path}
Road network: {rn}
Number of entities: {len(self.entities)}
Total duration: {self.length:.4}s

Entities
--------
{entity_header}
{entities}
"""
        )

    def plot(self, figsize: Tuple[int, int] = (10, 10)) -> None:
        """
        Visualise the scenario.

        Parameters
        ----------
        figsize : Tuple[int, int]
            The figure size.

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
        plt.show()

    def collisions(self) -> Dict[Entity, List[Entity]]:
        """Return current collisions occuring in the scenario."""
        if self._collisions is None:
            self._collisions = detect_collisions(self.entities)
        return self._collisions

    def get_road_info_at_entity(
        self, e: Entity
    ) -> Tuple[List[str], List[RoadObject]]:
        """Return the road network information at the entities location."""
        if not self.road_network:
            return [], []
        return self.road_network.get_geometries_at_point(*e.pose[:2])

    def get_entities_in_area(
        self, area: Union[MultiPolygon, Polygon]
    ) -> List[Entity]:
        """
        Return all entities who's center point is within an area.

        Parameters
        ----------
        area : Union[MultiPolygon, Polygon]
            A shapely geometry covering the chosen area.

        """
        pos = np.array([e.pose[:2] for e in self.entities])
        in_area = contains(area, pos[:, 0], pos[:, 1])
        return [e for e, b in zip(self.entities, in_area) if b]

    def get_entities_in_radius(self, x: float, y: float, r: float) -> List[Entity]:
        """
        Get entities with center point within a circle.

        Parameters
        ----------
        x : float
            The x-coordinate of the center.

        y : float
            The x-coordinate of the center.

        r : float
            The radius of the center.

        """
        return self.get_entities_in_area(Point(x, y).buffer(r))

    def translate(self, x: np.ndarray) -> "Scenario":
        """Return a new scenario with all entities translated."""
        for e in self.entities:
            e.trajectory = e.trajectory.translate(x)
