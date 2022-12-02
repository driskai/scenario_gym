from __future__ import annotations

import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.vectorized import contains

from scenario_gym.callback import StateCallback
from scenario_gym.entity import BatchReplayEntity, Entity
from scenario_gym.road_network import RoadObject
from scenario_gym.scenario import Scenario, ScenarioAction
from scenario_gym.state.utils import detect_collisions
from scenario_gym.trajectory import Trajectory, is_stationary

Agent = TypeVar("Agent")


class State:
    """
    The global state of the gym.

    Holds the current time, the terminal state and the positions and velocities of
    all the entities.

    Can also be parameterised with different end conditions for the scenario.
    E.g. to end when the recorded scenario ends or if a collision occurs.
    Additional information may be provided through custom methods passed
    as state_callbacks.
    """

    def __init__(
        self,
        scenario: Scenario,
        conditions: Optional[List[Union[str, Callable[[State], bool]]]] = None,
        state_callbacks: Optional[Dict[str, StateCallback]] = None,
    ):
        """
        Init the state.

        Parameters
        ----------
        scenario : Scenario
            The scenario to be simulated.

        conditions : Optional[List[Union[str, Callable[[State], bool]]]]
            Terminal conditions that will end the scenario if any is met. May be a
            string referencing an entry of the TERMINAL_CONDITIONS dictionary.

        state_callbacks : Optional[List[StateCallback]]
            Methods to be called on the state when the timestep is updated.
            Can be used to add additional information to the state that can is then
            accessible by all agents.

        """
        self._scenario = scenario
        if conditions is None:
            self.terminal_conditions = [TERMINAL_CONDITIONS["max_length"]]
        else:
            self.terminal_conditions = [
                cond if callable(cond) else TERMINAL_CONDITIONS[cond]
                for cond in conditions
            ]
        self.state_callbacks = [] if state_callbacks is None else state_callbacks

        self.next_t: Optional[float] = None
        self._t: Optional[float] = None
        self._prev_t: Optional[float] = None
        self.is_done = False
        self.last_keystroke: Optional[int] = None

        self._collisions: Optional[Dict[Entity, List[Entity]]] = None
        self._callbacks: Dict[Type[StateCallback], StateCallback] = {}

        self.unapplied_actions: List[ScenarioAction]
        self.poses: Dict[Entity, np.ndarray]
        self.prev_poses: Dict[Entity, np.ndarray]
        self.velocities: Dict[Entity, np.ndarray]
        self.distances: Dict[Entity, float]
        self.entity_state: Dict[Entity, Any]
        self._recorded_poses: Dict[Entity, List[Tuple[float, np.ndarray]]]

        self.agents: Dict[Entity, Agent] = {}
        self.non_agents = BatchReplayEntity()

    @property
    def scenario(self) -> Scenario:
        """Get the current scenario."""
        return self._scenario

    def reset(self, t_minus1: float, t_0: float):
        """
        Reset the state to the initial timestep.

        Parameters
        ----------
        t_minus1 : float
            Time before the initial timestep to use for initial velocities.

        t_0 : float
            Initial timestep.

        """
        self._reset_data()
        self.is_done = False

        # set initial poses
        prev_poses, poses = {}, {}
        for entity in self.scenario.entities:
            prev_poses[entity] = entity.trajectory.position_at_t(t_minus1)
            poses[entity] = entity.trajectory.position_at_t(t_0)

        self.update_poses(t_minus1, prev_poses)
        self.update_poses(t_0, poses)

        for action in self.scenario.actions:
            action.reset()
        self.update_actions()

        for cb in self.state_callbacks:
            cb.reset(self)
        self.update_callbacks()

        for agent in self.agents.values():
            agent.reset(self)

    def _reset_data(self) -> None:
        """Reset stored simulation data."""
        self.next_t: Optional[float] = None
        self._t: Optional[float] = None
        self._prev_t: Optional[float] = None
        self.unapplied_actions = self.scenario.actions.copy()

        entities = self.scenario.entities
        self.poses: Dict[Entity, np.ndarray] = OrderedDict.fromkeys(entities)
        self.prev_poses: Dict[Entity, np.ndarray] = OrderedDict.fromkeys(entities)
        self.velocities: Dict[Entity, np.ndarray] = OrderedDict.fromkeys(entities)
        self.distances: Dict[Entity, float] = OrderedDict.fromkeys(
            entities,
            value=0.0,
        )
        self.entity_state: Dict[Entity, Any] = OrderedDict.fromkeys(entities, None)
        self._recorded_poses: Dict[
            Entity, List[Tuple[float, np.ndarray]]
        ] = OrderedDict()
        for entity in self.poses:
            self._recorded_poses[entity] = []

    def step(self, new_poses: Dict[Entity, np.ndarray]) -> None:
        """Update by one timestep."""
        self._clear_cache()
        self.update_poses(self.next_t, new_poses)
        self.update_actions()
        self.update_callbacks()
        self.is_done = self.check_terminal()

    def _clear_cache(self) -> None:
        """Clear cached data on step."""
        self._collisions = None
        self._callbacks = {}

    @property
    def t(self):
        """Get the time in seconds (s)."""
        return self._t

    @t.setter
    def t(self, t: float) -> None:
        self.prev_t = self._t
        self._t = t
        return self._t

    @property
    def prev_t(self) -> float:
        """Get the previous time (s)."""
        return self._prev_t

    @prev_t.setter
    def prev_t(self, prev_t: float) -> None:
        self._prev_t = prev_t

    @property
    def dt(self):
        """Get the previous timestep."""
        return self.t - self.prev_t

    def update_poses(self, t: float, new_poses: Dict[Entity, np.ndarray]):
        """Update poses of all entities."""
        self.t = t
        self.prev_poses.update(self.poses)
        self.poses.update(new_poses)
        if self.prev_t is not None:
            self.update_statistics()
        for entity, pose in self.poses.items():
            self._recorded_poses[entity].append((self.t, pose))

    def update_statistics(self) -> None:
        """Update entity velocities and distance travelled."""
        for entity in self.scenario.entities:
            self.velocities[entity] = (
                self.poses[entity] - self.prev_poses[entity]
            ) / self.dt
            self.distances[entity] += np.linalg.norm(
                (self.poses[entity] - self.prev_poses[entity])[:3]
            )

    def update_actions(self) -> None:
        """Update state actions."""
        unapplied: List[ScenarioAction] = []
        for act in self.unapplied_actions:
            if self.t >= act.t:
                self.apply_action(act)
            else:
                unapplied.append(act)
        self.unapplied_actions = unapplied

    def apply_action(self, action: ScenarioAction) -> None:
        """Apply an action to the state."""
        entity = self.scenario.entity_by_name(action.entity_ref)
        if entity is None:
            warnings.warn(
                f"No entity with name {entity.ref} was found for action "
                f"{action.__class__.__name__}."
            )
        else:
            action.apply(self, entity)

    def update_callbacks(self) -> None:
        """Update all state callbacks."""
        for m in self.state_callbacks:
            m(self)

    def check_terminal(self) -> bool:
        """Check if the state is terminal."""
        return any(cond(self) for cond in self.terminal_conditions)

    def recorded_poses(
        self,
        entity: Optional[Entity] = None,
    ) -> Dict[Entity, np.ndarray]:
        """Get recorded poses for each or a given entity."""
        if entity is not None:
            poses = self._recorded_poses.get(entity, None)
            if not poses:
                return np.empty((0, 7))
            ts, poses = map(np.array, zip(*poses))
            return np.concatenate([ts[:, None], poses], axis=1)
        data: Dict[Entity, np.ndarray] = {}
        for ent, poses in self._recorded_poses.items():
            if not poses:
                data[ent] = np.empty((0, 7))
            else:
                ts, poses = map(np.array, zip(*poses))
                data[ent] = np.concatenate([ts[:, None], poses], axis=1)
        return data

    def get_entity_data(
        self, entity: Entity
    ) -> Tuple[float, float, np.ndarray, np.ndarray, float, np.ndarray, Any]:
        """Get state data for a specific entity."""
        return (
            self.t,
            self.next_t,
            self.poses.get(entity, None),
            self.velocities.get(entity, None),
            self.distances.get(entity, None),
            self.recorded_poses(entity=entity),
            self.entity_state.get(entity, None),
        )

    def collisions(self) -> Dict[Entity, List[Entity]]:
        """Return collisions between entities at the current time."""
        if self._collisions is None:
            self._collisions = detect_collisions(self.poses)
        return self._collisions

    def get_callback(
        self, Callback: Type[StateCallback]
    ) -> Optional[StateCallback]:
        """Get a particular type of callback."""
        if Callback not in self._callbacks:
            for callback in self.state_callbacks:
                if isinstance(callback, Callback):
                    self._callbacks[Callback] = callback
        return self._callbacks.get(Callback)

    def get_entity_box_points(self, e: Entity) -> np.ndarray:
        """Get the coordinates of the bounding box of an entity."""
        return e.get_bounding_box_points(self.poses[e])

    def get_entity_box_geom(self, e: Entity) -> Polygon:
        """Get the geometry of the bounding box of an entity."""
        return e.get_bounding_box_geom(self.poses[e])

    def get_road_info_at_entity(
        self, e: Entity
    ) -> Tuple[List[str], List[RoadObject]]:
        """Return the road network information at the entities location."""
        if not self.scenario.road_network:
            return [], []
        return self.scenario.road_network.get_geometries_at_point(
            *self.poses[e][:2]
        )

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
        pos = np.array([pose[:2] for pose in self.poses.values()])
        in_area = contains(area, pos[:, 0], pos[:, 1])
        return [e for e, b in zip(self.poses, in_area) if b]

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

    def to_scenario(self, name: Optional[str] = None) -> Scenario:
        """Create a scenario from the historical data in the state."""
        if name is None:
            name = (
                f"Simulation of {self.scenario.name}"
                if self.scenario.name is None
                else None
            )
        entities = []
        for entity, poses in self.recorded_poses().items():
            new_entity = deepcopy(entity)
            if is_stationary(poses):
                poses = poses[None, 0]
            new_entity.trajectory = Trajectory(poses)
            entities.append(new_entity)
        return Scenario(
            entities,
            name=name,
            path=self.scenario.path,
            road_network=self.scenario.road_network,
            actions=self.scenario.actions,
        )


TERMINAL_CONDITIONS = {
    "max_length": lambda s: s.t > s.scenario.length,
    "collision": lambda s: any(len(l) > 0 for l in s.collisions().values()),
    "ego_collision": lambda s: len(s.collisions()[s.scenario.entities[0]]) > 0,
    "ego_off_road": lambda s: not (
        s.scenario.road_network.driveable_surface.contains(
            Point(*s.poses[s.scenario.entities[0]][:2])
        )
    ),
}
