import math
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

from scenario_gym.entity import Entity
from scenario_gym.metrics.base import Metric
from scenario_gym.state import State


def angle_between(x: float, a_low: float, a_high: float) -> bool:
    """Return True if angle x is between a_low and a_high."""
    x = x % (math.pi * 2)
    a_low = a_low % (math.pi * 2)
    a_high = a_high % (math.pi * 2)
    return (
        ((a_low < x) or (x <= a_high))
        if (a_low >= a_high)
        else (a_low <= x < a_high)
    )


class CollisionTypes(Enum):
    """Enumerates possible collision types."""

    other = 0
    t_bone = 1
    head_on = 2
    rear_end = 3
    side_swipe = 4
    non_vehicle = 5


class CollisionPoints(Enum):
    """Enumerates possible collision points around a bounding box."""

    front = 0
    front_corner = 1
    side = 2
    back = 3
    back_corner = 4


class CollisionMetric(Metric):
    """
    Detects and classifies collisions with the ego.

    Records all collisions between entities and the ego. If the
    hazard entity is a vehicle then the collision is classified
    into t_bone, head_on, rear_end or side_swipe. If not then it
    is recorded as other.
    """

    name = "collisions"

    def __init__(self, c_tol: float = 0.4, name: Optional[str] = None):
        self.ego: Optional[Entity] = None
        self.collisions: List[Tuple[float, str, CollisionTypes]] = []
        self.c_tol = c_tol
        super().__init__(name=name)

    def _reset(self, state: State) -> None:
        """Reset the ego and recorded collisions."""
        ego = state.scenario.entity_by_name("ego")
        if ego is None:
            ego = state.scenario.entities[0]
        self.ego = ego
        self.collisions: List[Tuple[float, str, CollisionTypes]] = []
        self.last_timestep: List[Entity] = []

    def _step(self, state: State) -> None:
        """Update recorded collisions."""
        for e_other in state.collisions()[self.ego]:
            if e_other not in self.last_timestep:
                self.collisions.append(self.record_collision(state, e_other))
        self.last_timestep = state.collisions()[self.ego].copy()

    def get_state(self) -> List[Tuple[float, str, str]]:
        """Return the recorded collisions."""
        return [(t, ref, c.name) for t, ref, c in self.collisions]

    def record_collision(
        self, state: State, hazard: Entity
    ) -> Tuple[float, str, CollisionTypes]:
        """Classify the collision and record it."""
        if hazard.catalog_entry.catalog_type != "Vehicle":
            return (state.t, hazard.ref, CollisionTypes.non_vehicle)

        ego_box = self.ego.get_bounding_box_geom(state.poses[self.ego])
        hazard_box = hazard.get_bounding_box_geom(state.poses[hazard])

        collision_point = np.array(
            ego_box.intersection(hazard_box).centroid.xy
        ).squeeze()
        collision_angle = (hazard.pose[3] - self.ego.pose[3]) % (math.pi * 2)

        ego_angle = (
            np.arctan2(*np.flip(collision_point - self.ego.pose[:2]))
            - self.ego.pose[3]
        ) % (math.pi * 2)
        hazard_angle = (
            np.arctan2(*np.flip(collision_point - hazard.pose[:2])) - hazard.pose[3]
        ) % (math.pi * 2)

        ego_point = self.get_collision_point(ego_box, ego_angle, self.ego.pose[3])
        hazard_point = self.get_collision_point(
            hazard_box, hazard_angle, hazard.pose[3]
        )

        ego_front = ego_point in (
            CollisionPoints.front,
            CollisionPoints.front_corner,
        )
        ego_back = ego_point in (CollisionPoints.back, CollisionPoints.back_corner)
        hazard_front = hazard_point in (
            CollisionPoints.front,
            CollisionPoints.front_corner,
        )
        hazard_back = hazard_point in (
            CollisionPoints.back,
            CollisionPoints.back_corner,
        )

        if ego_front and hazard_front:
            if angle_between(
                collision_angle,
                math.pi / 4,
                3 * math.pi / 4,
            ) or angle_between(
                collision_angle,
                5 * math.pi / 4,
                7 * math.pi / 4,
            ):
                ctype = CollisionTypes.t_bone
            elif angle_between(
                collision_angle,
                7 * math.pi / 4,
                math.pi / 4,
            ):
                ctype = CollisionTypes.side_swipe
            else:
                ctype = CollisionTypes.head_on
        elif (ego_front or ego_back) and (hazard_front or hazard_back):
            if angle_between(
                collision_angle,
                math.pi / 4,
                3 * math.pi / 4,
            ) or angle_between(
                collision_angle,
                5 * math.pi / 4,
                7 * math.pi / 4,
            ):
                ctype = CollisionTypes.t_bone
            else:
                ctype = CollisionTypes.rear_end
        elif any([ego_front, ego_back, hazard_front, hazard_back]):
            if angle_between(
                collision_angle,
                math.pi / 4,
                3 * math.pi / 4,
            ) or angle_between(
                collision_angle,
                5 * math.pi / 4,
                7 * math.pi / 4,
            ):
                ctype = CollisionTypes.t_bone
            else:
                ctype = CollisionTypes.side_swipe
        else:
            ctype = CollisionTypes.side_swipe

        return state.t, hazard.ref, ctype

    def get_collision_point(
        self,
        box: Polygon,
        angle: float,
        heading: float,
    ) -> CollisionPoints:
        """Classify the collision point of the angle given the bounding box."""
        c_tol = self.c_tol
        corners = (
            np.arctan2(  # corners are BL, FL, FR, BR
                *np.flip(
                    np.array(box.exterior.coords[:-1]).T
                    - np.array(box.centroid.xy),
                    axis=0,
                ),
            )
            - heading
        )
        if (angle_between(angle, corners[1] - c_tol, corners[1] + c_tol)) or (
            angle_between(angle, corners[2] - c_tol, corners[2] + c_tol)
        ):
            return CollisionPoints.front_corner
        elif (angle_between(angle, corners[0] - c_tol, corners[0] + c_tol)) or (
            angle_between(angle, corners[3] - c_tol, corners[3] + c_tol)
        ):
            return CollisionPoints.back_corner
        elif angle_between(angle, corners[0] + c_tol, corners[3] - c_tol):
            return CollisionPoints.back
        elif angle_between(angle, corners[2] - c_tol, corners[1] + c_tol):
            return CollisionPoints.front
        return CollisionPoints.side


class CollisionPointMetric(Metric):
    """
    Finds the co-ordinate, and relative agles of collision.

    Records the position of any collisions that
    occur between entities and the ego. Collects
    the (x, y) coordinate and relative angle [rad]
    between the entities at the collision point.
    """

    name = "collision_points"

    def __init__(self, name: Optional[str] = None):
        self.ego: Optional[Entity] = None
        self.collisions: List[Tuple[str, np.ndarray, float]] = []
        super().__init__(name=name)

    def _reset(self, state: State) -> None:
        """Reset the ego and recorded collisions."""
        ego = state.scenario.entity_by_name("ego")
        if ego is None:
            ego = state.scenario.entities[0]
        self.ego = ego
        self.collisions: List[Tuple[str, np.ndarray, float]] = []
        self.last_timestep: List[Entity] = []

    def _step(self, state: State) -> None:
        """Update recorded collision angle and position."""
        for e_other in state.collisions()[self.ego]:
            if e_other not in self.last_timestep:
                self.collisions.append(
                    self.record_collision_position(state, e_other)
                )
        self.last_timestep = state.collisions()[self.ego].copy()

    def get_state(self) -> List[Tuple[str, np.ndarray, float]]:
        """Return the entity reference, coordinates and angle of collisions."""
        return self.collisions

    def record_collision_position(
        self, state: State, hazard: Entity
    ) -> Tuple[str, np.ndarray, float]:
        """Calculate the coordinate and relative angle of entities at collision."""
        ego_box = self.ego.get_bounding_box_geom(state.poses[self.ego])
        hazard_box = hazard.get_bounding_box_geom(state.poses[hazard])

        collision_point = np.array(
            ego_box.intersection(hazard_box).centroid.xy
        ).squeeze()
        collision_angle = (hazard.pose[3] - self.ego.pose[3]) % (math.pi * 2)
        return hazard.ref, collision_point, collision_angle
