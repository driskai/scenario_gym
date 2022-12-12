import warnings
from typing import Dict, List, Tuple

import numpy as np
from numpy.linalg import norm
from shapely.geometry import LineString, Polygon

from scenario_gym.callback import StateCallback
from scenario_gym.entity import Entity
from scenario_gym.metrics.rss.rss_utils import (
    acceleration,
    ahead,
    coord_change,
    direction,
    inverse_direction,
)
from scenario_gym.state import State


class RSSParameters:
    """RSS parameters."""

    RESPONSE_TIME = 0.6  # SECONDS
    MIN_LONG_ACCEL = 1.2 * 9.81  # METRES PER SECONDS SQUARED
    MAX_LONG_ACCEL = 1.2 * 9.81  # METRES PER SECONDS SQUARED
    MIN_SAFE_CLEARANCE = 0.1  # METRES
    SHADOW_LENGTH = 100  # METRES
    VISIBLE_RADIUS = 50  # METRES
    LANE_ANGLE_VARIATION = 0.985  # COS(ANGLE)
    TIME_HORIZON = 3  # SECONDS


class RSSDistances(StateCallback):
    """
    Determine if the ego-entity distance has become unsafe.

    This is the per-timestep pre-computation for the first two rules of the
    RSS metric. Raw values are calculated and attached to the global state or
    entity attributes. Unsafe distances are flagged per entity in the global
    state, which the metric uses to return a boolean value per rule.
    """

    def _reset(self, state: State) -> None:
        """Reset callback and declares variables."""
        self.ego = state.scenario.entities[0]
        self.entities = state.scenario.entities
        # Initialise default callback parameters
        self.ego_params = {}
        self.entity_params = [{} for _ in self.entities[1:]]
        self.safe_distances = [[0.0, 0.0] for _ in self.entities[1:]]
        self.intersect = [["safe"] for _ in self.entities[1:]]
        self.entity_safe_ratios = {
            entity: [float("inf"), float("inf")] for entity in self.entities
        }

    def __call__(self, state: State) -> None:
        """
        Label each entity to specify if its distances to the ego are unsafe.

        Collates current timestep data for each entity, and determines the
        safety of the ego with respect to its longitudinal and lateral distance
        to each entity.

        --  Creates entity parameter dictionary from current pose and bounding box
        --  Calculates safe lateral and longitudinal distance to each entity
        --  Calculates safe distance ratios for rss-based colour-coding
        --  Determines, for each entity, if its position intersects the
            corresponding ego safe buffer, and if so, which direction should be
            flagged as unsafe
        """
        if state.t == 0.0:
            # Require at least two poses to calculate velocity
            return

        # Establish Ego parameters
        ego = self.ego
        entities = self.entities
        ego_heading = direction(state.poses[ego][3])
        ego_inverse_heading = inverse_direction(list(ego_heading))
        entity_params = []
        ego_position = state.poses[ego][0:2]
        # Create a dictionary for each entity of form:
        # {position, heading, velocity, acceleration, box_points, length, width}
        # With directional and positional parameters defined with respect to ego
        # frame.
        for entity in entities:
            entity_dictionary = self.get_entity_parameters(
                state,
                entity,
                ego_heading,
                ego_inverse_heading,
                ego_position,
                state.dt,
            )
            if entity_dictionary is None:
                warnings.warn(
                    "RSSDistances _step: Error in handling of entity {0} at time "
                    "{1}. Invalid pose: {2}. RSS metric skips entity at this "
                    "timestep.".format(entity, state.t, state.poses[entity])
                )
                continue
            else:
                entity_params.append(entity_dictionary)
        if not entity_params:
            warnings.warn(
                "Zero entity parameters generated at timestep: {0}".format(state.t)
            )
            return

        # Calculate safe distances between ego and all other entities
        ego_params = entity_params.pop(0)
        safe_distances = []
        for entity in entity_params:
            safe_long = abs(self.safe_longitudinal_distance(ego_params, entity))
            safe_lat = abs(self.safe_lateral_distance(ego_params, entity))
            safe_distances.append([safe_lat, safe_long])

        # Assign evaluated parameters to the state
        assert len(entity_params) == len(safe_distances)
        self.ego_params = ego_params
        self.entity_params = entity_params
        self.safe_distances = safe_distances

        # Check for each entity if there is an intersection of safe distance buffer
        # and assign safe distance ratios to entity as an attribute
        for i in range(len(entity_params)):
            self.entity_safe_ratios[entities[i + 1]] = self.safe_ratios(
                ego_params, entity_params[i], safe_distances[i]
            )
            self.intersect[i].append(
                self.unsafe_distance(
                    ego_params,
                    entity_params[i],
                    self.intersect[i],
                    safe_distances[i],
                )
            )
            if self.intersect[i] is None:
                # Entity trimmed poses results in IndexError, go to next entity
                warnings.warn(
                    "safe_longitudinal: IndexError in handling of entity number: "
                    "{0} at timestep: {1} seconds. Continue to next "
                    "entity.".format(i, state.t)
                )

    @staticmethod
    def safe_ratios(
        ego: Dict,
        haz: Dict,
        safe_distances: List[float],
    ) -> List[float]:
        """
        Attach safe_distance ratios to entity.

        safe ratio defined as actual_distance / safe_distance -> larger ratio safer
        """
        try:
            safe_lat = safe_distances[0] + 0.5 * abs(
                np.dot(
                    [haz["width"], haz["length"]], inverse_direction(haz["heading"])
                )
            )
            safe_long = safe_distances[1] + 0.5 * abs(
                np.dot([haz["width"], haz["length"]], haz["heading"])
            )
        except IndexError:
            # safe_distances not calculated
            warnings.warn(
                "RSSDistances safe_ratios: Safe distances not calculated. Default "
                "safe distances as ego dimensions"
            )
            safe_lat = 0.5 * ego["width"]
            safe_long = 0.5 * ego["length"]

        actual_lat = max(
            1e-6,
            abs(haz["position"][0])
            - 0.5 * ego["width"]
            - 0.5
            * abs(
                np.dot(
                    [haz["width"], haz["length"]], inverse_direction(haz["heading"])
                )
            ),
        )
        actual_long = max(
            1e-6,
            abs(haz["position"][1])
            - 0.5 * ego["length"]
            - 0.5 * abs(np.dot([haz["width"], haz["length"]], haz["heading"])),
        )
        return [abs(actual_lat / safe_lat), abs(actual_long / safe_long)]

    @staticmethod
    def unsafe_distance(
        ego: Dict, haz: Dict, intersect: List[str], safe_distances: List[float]
    ) -> str:
        """
        Determine if longitudinal or lateral distance is hazardous to ego.

        This method is called per timestep per entity, and returns a flag to be
        attached to the global state.

        --  Laterally unsafe if both longitudinal and lateral distances are unsafe,
            and the lateral distance became unsafe after the longitudinal distance.
        --  Longitudinally unsafe if both longitudinal and lateral distances are
            unsafe, and the longitudinal distance became unsafe after the lateral
            distance.
        """
        if "unsafe_lateral" in intersect or "unsafe_longitudinal" in intersect:
            # Already found
            intersect.append("found")
            return intersect

        buffer, lengths, widths = RSSDistances.generate_buffer(ego, safe_distances)

        assert (
            buffer.area > 0.0
        ), "safe_longitudinal: buffer constructed as a 'Z' rather than as a '[]'"

        hazard_area = Polygon(haz["box_points"])
        if hazard_area.intersects(buffer):
            # Find which direction became unsafe last: this is the unsafe direction
            for j in range(len(intersect), 0, -1):
                try:
                    if intersect[j - 1] == "lateral":
                        # longitudinal intersection last: longitudinally unsafe
                        return "unsafe_longitudinal"
                    elif intersect[j - 1] == "longitudinal":
                        # lateral intersection last: laterally unsafe
                        return "unsafe_lateral"
                except IndexError:
                    break
                # default, if no previous intersection found.
                if j == 1:
                    ego_dim = [ego["width"], ego["length"]]
                    if abs(
                        abs(haz["position"][0])
                        - abs(np.dot(haz["position"], ego_dim))
                    ) / safe_distances[0] > abs(
                        abs(
                            haz["position"][1]
                            - np.dot(
                                haz["position"],
                                inverse_direction(ego_dim),
                            )
                        )
                        / safe_distances[1]
                    ):
                        return "unsafe_longitudinal"
                    else:
                        return "unsafe_lateral"
        # This entity does not intersect buffer, so check if a dimension intersects
        return RSSDistances.write_intersections(lengths, widths, haz)

    @staticmethod
    def safe_longitudinal_distance(ego: Dict, haz: Dict) -> float:
        """Determine if the longitudinal distance is safe."""
        MAX_LONG_ACCEL = RSSParameters.MAX_LONG_ACCEL
        MIN_LONG_ACCEL = RSSParameters.MIN_LONG_ACCEL
        MIN_SAFE_CLEARANCE = RSSParameters.MIN_SAFE_CLEARANCE
        RESPONSE_TIME = RSSParameters.RESPONSE_TIME
        ego_direction = ego["heading"]
        hazard_direction = haz["heading"]
        ego_velocity = ego["velocity"]
        hazard_velocity = haz["velocity"]
        max_long_accel = abs(
            MAX_LONG_ACCEL * np.dot(ego_direction, hazard_direction)
        )
        if np.dot(ego_direction, hazard_direction) > 0:
            # Moving in the same direction (longitudinal component)
            if ahead(ego, haz):
                vf = norm(ego_velocity)
                vr = np.dot(hazard_velocity, ego_direction)
            else:
                vf = np.dot(hazard_velocity, ego_direction)
                vr = norm(ego_velocity)
            if vr == 0.0:
                # If rear car is stationary, safe
                return MIN_SAFE_CLEARANCE + 0.5 * ego["length"]
            d0 = RSSDistances.long_dist_same_direction(
                vf, vr, max_long_accel, RESPONSE_TIME, MIN_LONG_ACCEL
            )
        else:
            # Moving in the opposite direction (longitudinal component)
            v1 = abs(np.dot(ego_velocity, ego_direction))
            v2 = -abs(np.dot(hazard_velocity, ego_direction))
            # Makes no distinction between driver in correct or incorrect lane,
            # where RSS is specific
            if np.sign(haz["position"][1]) == np.sign(haz["velocity"][1]):
                return MIN_SAFE_CLEARANCE + 0.5 * ego["length"]
            d0 = RSSDistances.long_dist_opp_direction(
                v1, v2, max_long_accel, RESPONSE_TIME, MIN_LONG_ACCEL
            )
        return d0 + MIN_SAFE_CLEARANCE + 0.5 * ego["length"]

    @staticmethod
    def safe_lateral_distance(ego: Dict, haz: Dict) -> float:
        """Determine if the lateral distance is safe."""
        MAX_LONG_ACCEL = RSSParameters.MAX_LONG_ACCEL
        MIN_LONG_ACCEL = RSSParameters.MIN_LONG_ACCEL
        MIN_SAFE_CLEARANCE = RSSParameters.MIN_SAFE_CLEARANCE
        RESPONSE_TIME = RSSParameters.RESPONSE_TIME
        haz_position = np.array(haz["position"])
        # Resolve into velocity components para and perp to hazard's velocity
        # Velocity takes form [longitudinal, lateral]
        # Define lateral velocity as ego's velocity component perp to hazard
        v = haz["velocity"][0]  # component perpendicular to ego's heading
        max_lat_accel = MAX_LONG_ACCEL * abs(
            np.dot(inverse_direction(ego["heading"]), haz["heading"])
        )
        min_lat_accel = MIN_LONG_ACCEL * abs(
            np.dot(inverse_direction(ego["heading"]), haz["heading"])
        )
        if np.sign(-haz_position[0]) == np.sign(v):
            # lateral convergence
            v = abs(v)
            if v == 0.0:
                # Driving parallel, safe for sufficient constant distance.
                return MIN_SAFE_CLEARANCE + 0.5 * ego["width"]
            d0 = RSSDistances.lat_dist(
                v, max_lat_accel, min_lat_accel, RESPONSE_TIME
            )
        else:
            # lateral divergence: safe distance is the avg of car widths
            # plus min_safe_distance
            d0 = 0
        return d0 + MIN_SAFE_CLEARANCE + 0.5 * ego["width"]

    @staticmethod
    def write_intersections(
        buffer_lengths: List[LineString],
        buffer_widths: List[LineString],
        haz_dict: Dict,
    ) -> str:
        """
        Flag buffer intersection direction.

        For ego and another entity, check if the entity has a smaller distance in
        the longitudinal or lateral direction than the corresponding safe distance,
        and appends this to the recorded list.
        """
        haz_area = Polygon(haz_dict["box_points"])
        lat_inter = False
        long_inter = False

        if haz_area.intersects(buffer_lengths[0]) or haz_area.intersects(
            buffer_lengths[1]
        ):
            lat_inter = True
        if haz_area.intersects(buffer_widths[0]) or haz_area.intersects(
            buffer_widths[1]
        ):
            long_inter = True

        if lat_inter and long_inter:
            new_record = "both"
        elif lat_inter:
            new_record = "lateral"
        elif long_inter:
            new_record = "longitudinal"
        else:
            new_record = "safe"
        return new_record

    @staticmethod
    def get_entity_parameters(
        state: State,
        entity: Entity,
        ego_heading: List[float],
        ego_inverse_heading: List[float],
        ego_position: List[float],
        dt: float,
    ) -> Dict:
        """Calculate entity parameters and returns these as a dictionary."""
        entity_pose = state.poses[entity]
        entity_velocity = state.velocities[entity]
        if len(entity_pose) != 6:
            warnings.warn(
                "Entity pose should have six elements, [x, y, z, h, r, p]. "
                "Received {0} elements.".format(len(entity_pose))
            )
            return
        ego_position = np.array(ego_position)
        entity_heading = direction(entity_pose[3])
        entity_acceleration = acceleration(state.recorded_poses(entity), dt)
        # All vectors take form [lateral, longitudinal] / [x, y]
        entity_dictionary = {
            "position": coord_change(entity_pose[0:2], ego_heading, ego_position),
            "heading": [
                np.dot(entity_heading, ego_inverse_heading),
                np.dot(entity_heading, ego_heading),
            ],
            "velocity": [
                np.dot(entity_velocity[:2], ego_inverse_heading),
                np.dot(entity_velocity[:2], ego_heading),
            ],
            "accel": [
                np.dot(
                    entity_acceleration,
                    ego_inverse_heading,
                ),
                np.dot(entity_acceleration, ego_heading),
            ],
            "box_points": [
                coord_change(point, ego_heading, ego_position)
                for point in entity.get_bounding_box_points(entity_pose)
            ],
            "length": entity.catalog_entry.bounding_box.length,
            "width": entity.catalog_entry.bounding_box.width,
        }
        return entity_dictionary

    @staticmethod
    def generate_buffer(
        ego: Dict, safe_distances: List
    ) -> Tuple[Polygon, List[LineString]]:
        """
        Generate ego safe buffer corresponding to entity's safe distances.

        Generates a rectangular buffer around the entity with length and width
        corresponding to safe longitudinal and lateral distances with respect to
        the other entity. Returns the Polygon of this buffer along with a list of
        linestrings of its lengths and widths.
        """
        assert ego["position"] == [0.0, 0.0], ego["position"]
        try:
            safe_longitudinal_distance = safe_distances[1]
            safe_lateral_distance = safe_distances[0]
        except IndexError:
            # Safe distances not calculated
            warnings.warn(
                "RSSDistances generate_buffer: Safe distances not calculated: "
                "Buffer cannot be instantiated. Default safe distances as 3 "
                "metres lateral, 5 metres safe longitudinal"
            )
            safe_longitudinal_distance = 5
            safe_lateral_distance = 3

        buffer_vector = [
            np.array([0, safe_longitudinal_distance]),
            np.array([safe_lateral_distance, 0]),
        ]
        buffer = [
            np.array(buffer_vector[0] + buffer_vector[1]),
            np.array(buffer_vector[0] - buffer_vector[1]),
            np.array(-buffer_vector[0] - buffer_vector[1]),
            np.array(-buffer_vector[0] + buffer_vector[1]),
        ]
        widths = [
            LineString(
                [
                    [100 * buffer[0][0], buffer[0][1]],
                    [100 * buffer[1][0], buffer[1][1]],
                ]
            ),
            LineString(
                [
                    [100 * buffer[2][0], buffer[2][1]],
                    [100 * buffer[3][0], buffer[3][1]],
                ]
            ),
        ]
        lengths = [
            LineString(
                [
                    [buffer[0][0], 100 * buffer[0][1]],
                    [buffer[2][0], 100 * buffer[2][1]],
                ]
            ),
            LineString(
                [
                    [buffer[1][0], 100 * buffer[1][1]],
                    [buffer[3][0], 100 * buffer[3][1]],
                ]
            ),
        ]
        return Polygon(buffer), lengths, widths

    @staticmethod
    def long_dist_same_direction(
        vf: float,
        vr: float,
        max_long_accel: float,
        RESPONSE_TIME: float,
        MIN_LONG_ACCEL: float,
    ) -> float:
        """Return the minimum safe longitudinal distance for same directions."""
        return max(
            0,
            vr * RESPONSE_TIME
            + min(
                vf**2 / (2 * max_long_accel),
                0.5 * max_long_accel * RESPONSE_TIME**2,
            )
            + (vr + RESPONSE_TIME * max_long_accel) ** 2 / (2 * MIN_LONG_ACCEL)
            - vf**2 / (2 * max_long_accel),
        )

    @staticmethod
    def long_dist_opp_direction(
        v1: float,
        v2: float,
        max_long_accel: float,
        RESPONSE_TIME: float,
        MIN_LONG_ACCEL: float,
    ) -> float:
        """Return the minimum safe longitudinal distance for opposing directions."""
        return max(
            0,
            (
                (2 * v1 + RESPONSE_TIME * max_long_accel) * RESPONSE_TIME / 2
                + (v1 + RESPONSE_TIME * max_long_accel) ** 2 / (2 * MIN_LONG_ACCEL)
                + (2 * abs(v2) + RESPONSE_TIME * max_long_accel) * RESPONSE_TIME / 2
                + (abs(v2) + RESPONSE_TIME * max_long_accel) ** 2
                / (2 * MIN_LONG_ACCEL)
            ),
        )

    @staticmethod
    def lat_dist(
        v: float, max_lat_accel: float, min_lat_accel: float, RESPONSE_TIME: float
    ):
        """Return the minimum safe lateral distance between the entity and ego."""
        return max(
            0,
            0.5 * RESPONSE_TIME * (2 * v + RESPONSE_TIME * max_lat_accel)
            + (v + RESPONSE_TIME * max_lat_accel) ** 2 / (2 * min_lat_accel)
            - 0.5 * RESPONSE_TIME**2 * max_lat_accel
            - (RESPONSE_TIME * max_lat_accel) ** 2 / (2 * min_lat_accel),
        )
