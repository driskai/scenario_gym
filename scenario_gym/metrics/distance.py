import numpy as np

from scenario_gym.entity.pedestrian import Pedestrian
from scenario_gym.state import State

from .base import Metric


# ---------- Distance Metrics ----------#
class EgoDistanceTravelled(Metric):
    """Measure the distance travelled by the ego."""

    name = "ego_distance_travelled"

    def _reset(self, state: State) -> None:
        """Find the ego."""
        self.ego = state.scenario.ego

    def _step(self, state: State) -> None:
        """Pass as entity will update its distance."""
        self.dist = state.distances[self.ego]

    def get_state(self) -> float:
        """Return the current distance travelled."""
        return self.dist


class DistanceToEgo(Metric):
    """Measure distance to ego vehicle."""

    name = "distance_to_ego"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.distance_to_ego_dict = {}

    def _step(self, state: State) -> None:
        for entity in state.get_entities_in_radius(*state.poses[self.ego][:2], 10):
            if entity == self.ego:
                continue

            # Get ego vehicle position.
            pos_ego = state.poses[self.ego][:2]
            # Get non_ego vehicle position.
            pos_non_ego = state.poses[entity][:2]

            difference_vector = pos_ego - pos_non_ego
            distance = np.linalg.norm(difference_vector)

            # state.t inclusion in the dictionary (line below). To track/timestamp
            # TTC in simulation.
            if state.t not in self.distance_to_ego_dict:
                self.distance_to_ego_dict[state.t] = []
            self.distance_to_ego_dict[state.t].append(
                {"distance_to_ego": {f"{entity}": distance}}
            )

    def get_state(self) -> dict:
        """Return distance to ego vehicle."""
        return self.distance_to_ego_dict


class SafeLongDistance(Metric):
    """
    Determines safe longitudinal distance.

    All speed and acceleration inputs must be for the longitudinal axis
    responseTime: time it takes rear car to react and begin braking
    speedFront: current velocity of front car
    accFront: current acceleration of front car
    accFrontMaxBrake: max braking of front car
    speedRear: current velocity of rear car
    accRear: current acceleration of rear car
    accRearMaxResp: max acceleration of rear car during response time
    accRearMinBrake: min braking of rear car
    """

    name = "safe_longitudinal_distance"

    # Time it takes rear car to begin braking (longitudinal and lateral).
    # https://trl.co.uk/sites/default/files/PPR313_new.pdf
    # Unit: seconds (s)
    responseTime = 1.5
    # Max braking of front car (longitudinal)
    # Bokare, P. S., & Maurya, A. K. (2017).
    # Acceleration-deceleration behaviour of various vehicle types.
    # Transportation research procedia, 25, 4733-4749.
    # Unit: m/s^2
    accFrontMaxBrake = 2.5
    # Max acceleration of rear car during response time (longitudinal)
    # Bokare, P. S., & Maurya, A. K. (2017).
    # Acceleration-deceleration behaviour of various vehicle types.
    # Transportation research procedia, 25, 4733-4749.
    # Unit: m/s^2
    accRearMaxResp = 2.5
    # Min braking of rear car (longitudinal).
    # Unit: m/s^2
    accRearMinBrake = 1
    # Max braking capability of rear car (longitudinal).
    # Bokare, P. S., & Maurya, A. K. (2017).
    # Acceleration-deceleration behaviour of various vehicle types.
    # Transportation research procedia, 25, 4733-4749.
    # Unit: m/s^2
    accRearMaxBrake = 4

    # Class attributes for function _is_ego_leading_following_or_side -> str output.
    LEADING = "leading"
    FOLLOWING = "following"
    SIDEBYSIDE = "side-by-side"
    OUT_OF_BOUNDS = "out-of-bounds"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.safe_long_dist_dict = {}
        self.safe_long_dist_brake_dict = {}
        self.long_dist_dict = {}
        self.long_risk_dict = {}

    def _pose_velocity_lane_and_lane_center(self, entity, state) -> tuple:
        # Initialize variables to deduce the closest lane center coordinates.
        min_distance = float("inf")
        final_lane_center_index = None
        final_lane_index = None

        # Get entity vehicle pose, velocity, and road network information.
        entity_position = state.poses[entity][:2]
        entity_velocity = state.velocities[entity][:2]
        entity_road_network_information = state.get_road_info_at_entity(entity)

        # Get lane type and lane ID.
        road_network_types = entity_road_network_information[0]
        road_network_ids = entity_road_network_information[1]

        if "Lane" not in road_network_types:
            return entity_position, entity_velocity, 0, 0

        for lane_index, road_network_type in enumerate(road_network_types):
            if road_network_type == "Lane":
                lane = road_network_ids[lane_index]
                # Loop through lane center points to find the closest one to entity
                for lane_center_index, coords in enumerate(lane.center.coords):
                    # Compute the Euclidean distance between ego and lane center
                    distance = abs(np.linalg.norm(entity_position - coords))
                    # Update the closest lane center index
                    if distance < min_distance:
                        min_distance = distance
                        final_lane_center_index = lane_center_index
                        final_lane_index = lane_index

        current_lane = road_network_ids[final_lane_index]
        final_lane_center_index = final_lane_center_index

        return (
            entity_position,
            entity_velocity,
            current_lane,
            final_lane_center_index,
        )

    def _is_ego_leading_following_or_side(
        self,
        pos1,
        current_lane_vehicle_1,
        final_lane_center_index_vehicle_1,
        pos2,
        current_lane_vehicle_2,
    ) -> str:
        """
        Dev Note: Safe longitudinal distance is currently limited.

        To instances where both vehicles are in the same lane.
        """
        if current_lane_vehicle_1 == 0 or current_lane_vehicle_2 == 0:
            return self.OUT_OF_BOUNDS

        elif current_lane_vehicle_1 == current_lane_vehicle_2:
            current_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1
                ]
            )
            next_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1 - 1
                ]
            )
            longitudinal_direction = -(
                next_lane_center_point - current_lane_center_point
            )
            # Project relative position vectors onto the direction vector
            proj1 = np.dot(pos1, longitudinal_direction)
            proj2 = np.dot(pos2, longitudinal_direction)
            # Determine which position vector is ahead
            if proj1 > proj2:
                return self.LEADING
            else:
                return self.FOLLOWING

        else:
            return self.SIDEBYSIDE

    def _longitudinal_speed(
        self, entity_vel, entity_current_lane, entity_final_lane_center_index
    ) -> float:
        # Convert closest lane center point and
        # next-nearest lane center to array.
        # Does not matter if the next lane center point is 'ahead' or 'behind'
        # direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # Longitudinal direction (no sign needed for speed).
        current_lane_center_point = np.array(
            entity_current_lane.center.coords[entity_final_lane_center_index]
        )
        next_lane_center_point = np.array(
            entity_current_lane.center.coords[entity_final_lane_center_index - 1]
        )

        longitudinal_direction = next_lane_center_point - current_lane_center_point
        normalised_longitudinal_direction = longitudinal_direction / np.linalg.norm(
            longitudinal_direction
        )

        # Calculate longitudinal speed
        longitudinal_speed = np.dot(entity_vel, normalised_longitudinal_direction)
        # Take the absolute value
        longitudinal_speed = np.abs(longitudinal_speed)

        return longitudinal_speed

    def _longitudinal_distance(
        self, pos_ego, pos_non_ego, ego_current_lane, ego_final_lane_center_index
    ) -> float:
        # Convert closest lane center point and
        # next-nearest lane center to array.
        # Does not matter if the next lane
        # center point is 'ahead' or 'behind' direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # Longitudinal direction (no sign needed for speed)
        current_lane_center_point = np.array(
            ego_current_lane.center.coords[ego_final_lane_center_index]
        )
        next_lane_center_point = np.array(
            ego_current_lane.center.coords[ego_final_lane_center_index - 1]
        )

        longitudinal_direction = next_lane_center_point - current_lane_center_point
        normalised_longitudinal_direction = longitudinal_direction / np.linalg.norm(
            longitudinal_direction
        )

        # Calculate the vector between ego and non-ego vehicle positions
        relative_position = pos_non_ego - pos_ego
        # Project the relative position onto the longitudinal direction
        longitudinal_distance = np.dot(
            relative_position, normalised_longitudinal_direction
        )
        # Take the absolute value of the longitudinal distance
        longitudinal_distance = np.abs(longitudinal_distance)

        return longitudinal_distance

    def _calculate_safe_long_dist(self, speed_rear, speed_front) -> tuple:
        sign = [1, -1][speed_rear + self.responseTime * self.accRearMaxResp < 0]
        safeLonDis = (
            0.5 * np.power(speed_front, 2) / self.accFrontMaxBrake
            - speed_rear * self.responseTime
            - 0.5 * np.power(self.responseTime, 2) * self.accRearMaxResp
            - 0.5
            * sign
            * np.power(speed_rear + self.responseTime * self.accRearMaxResp, 2)
            / self.accRearMinBrake
        )
        safeLonDisBrake = (
            0.5 * np.power(speed_front, 2) / self.accFrontMaxBrake
            - speed_rear * self.responseTime
            - 0.5 * np.power(self.responseTime, 2) * self.accRearMaxResp
            - 0.5
            * sign
            * np.power(speed_rear + self.responseTime * self.accRearMaxResp, 2)
            / self.accRearMaxBrake
        )

        return safeLonDis, safeLonDisBrake

    def _long_risk_index(
        self, safe_distance, safe_distance_brake, distance
    ) -> float:
        """
        Calculate the longitudinal risk index [0,1].

        safeDistance: safe longitudinal distance (use function SafeLonDistance).
        safeDistanceBrake: safe longitudinal distance under max braking capacity
        (use function SafeLonDistance with max braking acceleration)
        distance: current longitudinal/lateral distance between cars
        """
        if safe_distance + distance > 0:
            r = 0
        elif safe_distance_brake + distance <= 0:
            r = 1
        else:
            r = 1 - (safe_distance_brake + distance) / (
                safe_distance_brake - safe_distance
            )
        return r

    def _step(self, state: State) -> None:
        """All speed and acceleration inputs must be longitudinal axis."""
        # Get entities within 10m of the ego, if entities exist...
        if len(state.get_entities_in_radius(*state.poses[self.ego][:2], 10)) > 1:
            for entity in state.get_entities_in_radius(
                *state.poses[self.ego][:2], 10
            ):
                if entity == self.ego:
                    continue
                if isinstance(entity, Pedestrian):
                    continue

                # Get ego vehicle pose, velocity, and lane.
                (
                    pos_ego,
                    vel_ego,
                    ego_current_lane,
                    ego_lane_center_index,
                ) = self._pose_velocity_lane_and_lane_center(self.ego, state)
                # Get non_ego vehicle pose, velocity, and lane.
                (
                    pos_non_ego,
                    vel_non_ego,
                    non_ego_current_lane,
                    non_ego_lane_center_index,
                ) = self._pose_velocity_lane_and_lane_center(entity, state)

                ego_relative_position_string = (
                    self._is_ego_leading_following_or_side(
                        pos_ego,
                        ego_current_lane,
                        ego_lane_center_index,
                        pos_non_ego,
                        non_ego_current_lane,
                    )
                )

                if ego_relative_position_string == "following":
                    speed_rear = self._longitudinal_speed(
                        vel_ego, ego_current_lane, ego_lane_center_index
                    )
                    speed_front = self._longitudinal_speed(
                        vel_non_ego, non_ego_current_lane, non_ego_lane_center_index
                    )
                elif ego_relative_position_string == "leading":
                    speed_rear = self._longitudinal_speed(
                        vel_non_ego, non_ego_current_lane, non_ego_lane_center_index
                    )
                    speed_front = self._longitudinal_speed(
                        vel_ego, ego_current_lane, ego_lane_center_index
                    )
                else:
                    continue

                safeLonDis, safeLonDisBrake = self._calculate_safe_long_dist(
                    speed_rear, speed_front
                )
                long_dist = self._longitudinal_distance(
                    pos_ego, pos_non_ego, ego_current_lane, ego_lane_center_index
                )

                long_risk = self._long_risk_index(
                    safeLonDis, safeLonDisBrake, long_dist
                )

                # state.t inclusion in the dictionary (line below). To
                # track/timestamp TTC in simulation.
                if state.t not in self.safe_long_dist_dict:
                    self.safe_long_dist_dict[state.t] = []
                self.safe_long_dist_dict[state.t].append(
                    {"safe_longitudinal_distance": {f"{entity}": safeLonDis}}
                )

                if state.t not in self.safe_long_dist_brake_dict:
                    self.safe_long_dist_brake_dict[state.t] = []
                self.safe_long_dist_brake_dict[state.t].append(
                    {
                        "safe_longitudinal_distance_brake": {
                            f"{entity}": safeLonDisBrake
                        }
                    }
                )

                if state.t not in self.long_dist_dict:
                    self.long_dist_dict[state.t] = []
                self.long_dist_dict[state.t].append(
                    {"longitudinal_distance": {f"{entity}": long_dist}}
                )

                if state.t not in self.long_risk_dict:
                    self.long_risk_dict[state.t] = []
                self.long_risk_dict[state.t].append(
                    {"longitudinal_risk": {f"{entity}": long_risk}}
                )

    def get_state(self) -> tuple:
        """Return the current distance travelled."""
        return (
            self.safe_long_dist_dict,
            self.safe_long_dist_brake_dict,
            self.long_dist_dict,
            self.long_risk_dict,
        )


class SafeLatDistance(Metric):
    """
    Determines safe latitudinal distance.

    Speed and acceleration inputs must be for the lateral axis
    responseTime: time it takes rear car to react and begin braking
    speedLeft: current velocity of left car
    speedRight: current velocity of right car
    accMaxResp: max acceleration of both cars towards each other during response
    accMinBrake: min braking of both cars
    """

    name = "safe_latitudinal_distance"

    # Time it takes rear car to begin braking (longitudinal and lateral).
    # https://trl.co.uk/sites/default/files/PPR313_new.pdf
    # Unit: seconds (s)
    responseTime = 1.5
    # Max acceleration of both cars towards
    # each other during response time (lateral).
    # Unit: m/s^2
    accMaxResp = 1
    # Min braking of both cars (lateral).
    # https://doi.org/10.1007/s12544-013-0120-2
    # Unit: m/s^2
    accMinBrake = 1.5
    # Max braking capability of both cars (lateral).
    # Replace for ACC_MIN_BRAKE when calculating 'safeDistanceBrake'.
    # https://doi.org/10.1007/s12544-013-0120-2
    # Unit: m/s^2
    accMaxBrake = 4

    # Class attributes for function _is_ego_left_or_right -> str output.
    RIGHT = "right"
    LEFT = "left"
    INLINE = "inline"
    OUT_OF_BOUNDS = "out-of-bounds"

    def _reset(self, state: State) -> None:
        """Find the ego."""
        self.ego = state.scenario.ego
        self.safe_lat_dist_dict = {}
        self.safe_lat_dist_brake_dict = {}
        self.lat_dist_dict = {}
        self.lat_risk_dict = {}

    def _pose_velocity_lane_and_lane_center(self, entity, state) -> tuple:
        # Initialize variables to deduce the closest lane center coordinates.
        min_distance = float("inf")
        final_lane_center_index = None
        final_lane_index = None

        # Get entity vehicle pose, velocity, and road network information.
        entity_position = state.poses[entity][:2]
        entity_velocity = state.velocities[entity][:2]
        entity_road_network_information = state.get_road_info_at_entity(entity)

        # Get lane type and lane ID.
        road_network_types = entity_road_network_information[0]
        road_network_ids = entity_road_network_information[1]

        if "Lane" not in road_network_types:
            return entity_position, entity_velocity, 0, 0

        for lane_index, road_network_type in enumerate(road_network_types):
            if road_network_type == "Lane":
                lane = road_network_ids[lane_index]
                # Loop through lane center points to find the closest one to entity
                for lane_center_index, coords in enumerate(lane.center.coords):
                    # Compute the Euclidean distance between ego and lane center
                    distance = abs(np.linalg.norm(entity_position - coords))
                    # Update the closest lane center index
                    if distance < min_distance:
                        min_distance = distance
                        final_lane_center_index = lane_center_index
                        final_lane_index = lane_index

        current_lane = road_network_ids[final_lane_index]
        final_lane_center_index = final_lane_center_index

        return (
            entity_position,
            entity_velocity,
            current_lane,
            final_lane_center_index,
        )

    def _is_ego_left_or_right(
        self,
        pos1,
        current_lane_vehicle_1,
        final_lane_center_index_vehicle_1,
        pos2,
        current_lane_vehicle_2,
    ) -> str:
        """
        Dev Note: Safe latitudinal distance is currently limited.

        To instances where both vehicles are in separate lanes.
        """
        if current_lane_vehicle_1 == 0 or current_lane_vehicle_2 == 0:
            return self.OUT_OF_BOUNDS

        elif current_lane_vehicle_1 != current_lane_vehicle_2:
            current_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1
                ]
            )
            next_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1 - 1
                ]
            )
            longitudinal_direction = -(
                next_lane_center_point - current_lane_center_point
            )
            relative_position = pos2 - pos1
            cross_product = np.cross(longitudinal_direction, relative_position)
            if cross_product < 0:
                return self.RIGHT
            else:
                return self.LEFT

        else:
            return self.INLINE

    def _latitudinal_speed(
        self, entity_vel, entity_current_lane, entity_final_lane_center_index
    ) -> float:
        # Convert closest lane center point and
        # next-nearest lane center to array.
        # Does not matter if the next lane center point is 'ahead' or 'behind'
        # direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # latitudinal direction (no sign needed for speed)
        current_lane_center_point = np.array(
            entity_current_lane.center.coords[entity_final_lane_center_index]
        )
        next_lane_center_point = np.array(
            entity_current_lane.center.coords[entity_final_lane_center_index - 1]
        )

        longitudinal_direction = -(
            next_lane_center_point - current_lane_center_point
        )

        latitudinal_direction = np.array(
            [-longitudinal_direction[1], longitudinal_direction[0]]
        )
        normalised_latitudinal_direction = latitudinal_direction / np.linalg.norm(
            latitudinal_direction
        )

        # Calculate longitudinal speed
        latitudinal_speed = np.dot(entity_vel, normalised_latitudinal_direction)
        # Take the absolute value
        latitudinal_speed = np.abs(latitudinal_speed)

        return latitudinal_speed

    def _latitudinal_distance(
        self, pos_ego, pos_non_ego, ego_current_lane, ego_final_lane_center_index
    ) -> float:
        # Convert closest lane center point and next-nearest lane center to array
        # Does not matter if the next lane center point is 'ahead' or 'behind'
        # direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # latitudinal direction (no sign needed for speed)
        current_lane_center_point = np.array(
            ego_current_lane.center.coords[ego_final_lane_center_index]
        )
        next_lane_center_point = np.array(
            ego_current_lane.center.coords[ego_final_lane_center_index - 1]
        )

        longitudinal_direction = -(
            next_lane_center_point - current_lane_center_point
        )

        latitudinal_direction = np.array(
            [-longitudinal_direction[1], longitudinal_direction[0]]
        )
        normalised_latitudinal_direction = latitudinal_direction / np.linalg.norm(
            latitudinal_direction
        )

        # Calculate the vector between ego and non-ego vehicle positions
        relative_position = pos_non_ego - pos_ego
        # Project the relative position onto the latitudinal direction
        latitudinal_distance = np.dot(
            relative_position, normalised_latitudinal_direction
        )
        # Take the absolute value of the latitudinal distance
        latitudinal_distance = np.abs(latitudinal_distance)

        return latitudinal_distance

    def _calculate_safe_lat_dist(self, speed_left, speed_right) -> tuple:
        # assuming vehicles travel towards each other during response time for
        # boundary condition
        vLeft = speed_left - self.responseTime * self.accMaxResp
        vRight = speed_right + self.responseTime * self.accMaxResp
        # left vehicle move to left
        if vLeft >= 0:
            # right vehicle move to left
            if vRight >= 0:
                acc_left_min_gap = acc_right_max_gap = self.accMaxBrake
                acc_left_max_gap = acc_right_min_gap = self.accMinBrake
            # right vehicle move to right
            else:
                acc_left_min_gap = acc_right_min_gap = self.accMaxBrake
                acc_left_max_gap = acc_right_max_gap = self.accMinBrake

        # left vehicle move to right
        else:
            # right vehicle move to left
            if vRight >= 0:
                acc_left_min_gap = acc_right_min_gap = self.accMinBrake
                acc_left_max_gap = acc_right_max_gap = self.accMaxBrake
            # right vehicle move to right
            else:
                acc_left_min_gap = acc_right_max_gap = self.accMinBrake
                acc_left_max_gap = acc_right_min_gap = self.accMaxBrake

        sign_left = [1, -1][vLeft < 0]
        sign_right = [1, -1][vRight < 0]

        safeLatDis = (
            0.5 * (speed_left + vLeft) * self.responseTime
            + 0.5 * sign_left * np.power(vLeft, 2) / acc_left_min_gap
            - (
                0.5 * (speed_right + vRight) * self.responseTime
                + 0.5 * sign_right * np.power(vRight, 2) / acc_right_min_gap
            )
        )
        safeLatDisBrake = (
            0.5 * (speed_left + vLeft) * self.responseTime
            + 0.5 * sign_left * np.power(vLeft, 2) / acc_left_max_gap
            - (
                0.5 * (speed_right + vRight) * self.responseTime
                + 0.5 * sign_right * np.power(vRight, 2) / acc_right_max_gap
            )
        )

        return safeLatDis, safeLatDisBrake

    def _lat_risk_index(
        self, safe_distance, safe_distance_brake, distance
    ) -> float:
        """
        Calculate the latitudinal risk index [0,1].

        safeDistance: safe lateral distance (use function SafeLatDistance)
        safeDistanceBrake: safe lateral distance under max braking capacity
        (use function SafeLatDistance with max braking acceleration)
        distance: current lateral distance between cars
        """
        if safe_distance + distance > 0:
            r = 0
        elif safe_distance_brake + distance <= 0:
            r = 1
        else:
            r = 1 - (safe_distance_brake + distance) / (
                safe_distance_brake - safe_distance
            )
        return r

    def _step(self, state: State) -> None:
        """All speed and acceleration inputs must be latitudinal axis."""
        # Get entities within 10m of the ego, if entities exist...
        if len(state.get_entities_in_radius(*state.poses[self.ego][:2], 10)) > 1:
            for entity in state.get_entities_in_radius(
                *state.poses[self.ego][:2], 10
            ):
                if entity == self.ego:
                    continue
                if isinstance(entity, Pedestrian):
                    continue

                # Get ego vehicle pose, velocity, and lane.
                (
                    pos_ego,
                    vel_ego,
                    ego_current_lane,
                    ego_lane_center_index,
                ) = self._pose_velocity_lane_and_lane_center(self.ego, state)
                # Get non_ego vehicle pose, velocity, and lane.
                (
                    pos_non_ego,
                    vel_non_ego,
                    non_ego_current_lane,
                    non_ego_lane_center_index,
                ) = self._pose_velocity_lane_and_lane_center(entity, state)

                check = self._is_ego_left_or_right(
                    pos_ego,
                    ego_current_lane,
                    ego_lane_center_index,
                    pos_non_ego,
                    non_ego_current_lane,
                )

                if check is None or check == "inline":
                    continue
                elif check == "right":
                    speed_right = self._latitudinal_speed(
                        vel_ego, ego_current_lane, ego_lane_center_index
                    )
                    speed_left = self._latitudinal_speed(
                        vel_non_ego, non_ego_current_lane, non_ego_lane_center_index
                    )
                elif check == "left":
                    speed_right = self._latitudinal_speed(
                        vel_non_ego, non_ego_current_lane, non_ego_lane_center_index
                    )
                    speed_left = self._latitudinal_speed(
                        vel_ego, ego_current_lane, ego_lane_center_index
                    )
                else:
                    continue

                safeLatDis, safeLatDisBrake = self._calculate_safe_lat_dist(
                    speed_left, speed_right
                )
                lat_dist = self._latitudinal_distance(
                    pos_ego, pos_non_ego, ego_current_lane, ego_lane_center_index
                )

                lat_risk = self._lat_risk_index(
                    safeLatDis, safeLatDisBrake, lat_dist
                )

                # state.t inclusion in the dictionary (line below). To
                # track/timestamp TTC in simulation.
                if state.t not in self.safe_lat_dist_dict:
                    self.safe_lat_dist_dict[state.t] = []
                self.safe_lat_dist_dict[state.t].append(
                    {"safe_latitudinal_distance": {f"{entity}": safeLatDis}}
                )

                if state.t not in self.safe_lat_dist_brake_dict:
                    self.safe_lat_dist_brake_dict[state.t] = []
                self.safe_lat_dist_brake_dict[state.t].append(
                    {
                        "safe_latitudinal_distance_brake": {
                            f"{entity}": safeLatDisBrake
                        }
                    }
                )

                if state.t not in self.lat_dist_dict:
                    self.lat_dist_dict[state.t] = []
                self.lat_dist_dict[state.t].append(
                    {"latitudinal_distance": {f"{entity}": lat_dist}}
                )

                if state.t not in self.lat_dist_dict:
                    self.lat_dist_dict[state.t] = []
                self.lat_dist_dict[state.t].append(
                    {"latitudinal_risk": {f"{entity}": lat_risk}}
                )

    def get_state(self) -> tuple:
        """Return the current distance travelled."""
        return (
            self.safe_lat_dist_dict,
            self.safe_lat_dist_brake_dict,
            self.lat_dist_dict,
            self.lat_risk_dict,
        )
