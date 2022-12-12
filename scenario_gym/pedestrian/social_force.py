from typing import Tuple, Union

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import nearest_points

from scenario_gym.agent import Agent
from scenario_gym.entity import Entity
from scenario_gym.pedestrian.behaviour import PedestrianBehaviour
from scenario_gym.pedestrian.observation import PedestrianObservation
from scenario_gym.pedestrian.random_walk import RandomWalkParameters
from scenario_gym.utils import NDArray
from scenario_gym.viewer import rotate_coords


class SocialForceParameters(RandomWalkParameters):
    """Parameters for the social force model."""

    distance_threshold = 3
    sight_weight = 0.5
    sight_weight_use = True
    sight_angle = 200
    relaxation_time = 1.5
    ped_repulse_V = 1.0
    ped_repulse_sigma = 1.0
    ped_attract_C = 0.0
    boundary_repulse_U = 10.0
    boundary_repulse_R = 0.2
    imp_boundary_repulse_U = 2.0
    imp_boundary_repulse_R = 0.1


class SocialForce(PedestrianBehaviour):
    """Social force model."""

    def __init__(self, params: SocialForceParameters):
        """Init the behaviour model."""
        super().__init__(params)
        self.bias_lon = params.bias_lon
        self.bias_lat = params.bias_lat
        self.std_lon = params.std_lon
        self.std_lat = params.std_lat

    def _step(self, observation: PedestrianObservation, agent: Agent) -> Tuple:
        """Return the new speed and heading using the social force model."""
        # Start with attraction force to goal point
        force_sum = self._force_to_goal(
            observation,
            agent.route[agent.goal_idx],
            agent.speed_desired,
        )

        for (
            pedestrian,
            pose,
            vel,
        ) in observation.near_peds:  # Forces from other pedestrians
            # Vector of agent's sight (velocity angle + head angle)
            view_dir_vector = rotate_coords(vel[[0, 1]], observation.head_rot_angle)
            view_dir_unit_vector = view_dir_vector / (
                np.linalg.norm(view_dir_vector) + 0.0000000001
            )

            force_repulsion = self._force_pedestrian_repulsion(
                observation, (pedestrian, pose, vel)
            )
            force_attraction = self._force_pedestrian_attraction(
                observation, (pedestrian, pose, vel)
            )
            if self.params.sight_weight_use:
                force_sum += (
                    self._sight_weight(force_repulsion, view_dir_unit_vector)
                    * force_repulsion
                )
                force_sum += (
                    self._sight_weight(force_attraction, view_dir_unit_vector)
                    * force_attraction
                )
            else:
                force_sum += force_attraction
                force_sum += force_repulsion

        # get current position
        point = Point(*observation.pose[:2])

        # Force from closest walkable boundary
        if observation.walkable_surface.area > 0:
            if observation.walkable_surface.contains(point):
                force_sum += self._force_boundary(
                    observation,
                    observation.walkable_surface,
                    self.params.boundary_repulse_R,
                    self.params.boundary_repulse_U,
                )

        # Force from immovable boundary
        if observation.impenetrable_surface.area > 0:
            sign = 1 - 2 * observation.impenetrable_surface.contains(point)
            force_sum += sign * self._force_boundary(
                observation,
                observation.impenetrable_surface,
                self.params.imp_boundary_repulse_R,
                self.params.imp_boundary_repulse_U,
            )

        # Random fluctuations
        speed_rand = np.random.normal(self.bias_lon, self.std_lon)
        heading_rand = np.random.normal(self.bias_lat, self.std_lat)

        speed = min(
            np.linalg.norm(force_sum) + speed_rand,
            agent.speed_desired * self.max_speed_factor,
        )  # Limit to max speed
        heading = np.arctan2(force_sum[1], force_sum[0]) + heading_rand
        agent.force = force_sum

        return speed, heading

    def _force_to_goal(
        self,
        obs: PedestrianObservation,
        goal_point: NDArray,
        speed_desired: float,
    ) -> np.ndarray:
        """Compute the attraction force from the goal."""
        agent_pos = obs.pose[[0, 1]]
        agent_vel = obs.velocity[[0, 1]]
        dir_vector = goal_point - agent_pos
        dir_vector_norm = np.linalg.norm(dir_vector)
        if dir_vector_norm == 0:
            dir_vector_norm += 0.000000001
        unit_dir_vector = dir_vector / dir_vector_norm
        force_vector = (
            1
            / self.params.relaxation_time
            * (speed_desired * unit_dir_vector - agent_vel)
        )
        return force_vector

    def _force_pedestrian_repulsion(
        self,
        obs: PedestrianObservation,
        other_pedestrian: Tuple[Entity, NDArray, NDArray],
    ) -> NDArray:
        """Compute the repulsion force from other pedestrians."""
        agent_pos = obs.pose[[0, 1]]
        other_ped, other_pose, other_v = other_pedestrian
        other_pos = other_pose[[0, 1]]
        other_dir = other_v[[0, 1]]

        # Vector to other agent
        r_ao = agent_pos - other_pos
        r_ao_norm = np.linalg.norm(r_ao)

        # Auxiliary calculations
        v_vel_magnitude = np.linalg.norm(other_dir) + 0.0000000001
        unit_other_dir = other_dir / v_vel_magnitude
        other_step = v_vel_magnitude * (obs.next_t - obs.t)
        r_ao_other = r_ao - other_step * unit_other_dir
        r_ao_other_norm = np.linalg.norm(r_ao_other) + 0.0000000001

        # Ellipse semi-minor axis b
        b = (1 / 2) * np.sqrt((r_ao_norm + r_ao_other_norm) ** 2 - other_step**2)
        db = (
            (1 / 4)
            * (1 / b)
            * (r_ao_norm + r_ao_other_norm)
            * (r_ao / r_ao_norm + r_ao_other / r_ao_other_norm)
        )
        force_vector = (
            self.params.ped_repulse_V
            / self.params.ped_repulse_sigma
            * np.exp(-b / self.params.ped_repulse_sigma)
            * db
        )  # gradient
        return force_vector

    def _force_pedestrian_attraction(
        self,
        obs: PedestrianObservation,
        other_pedestrian: Tuple[Entity, NDArray, NDArray],
    ) -> NDArray:
        """Compute the attraction force from other pedestrians."""
        agent_pos = obs.pose[[0, 1]]
        other_pos = other_pedestrian[1][[0, 1]]
        # Vector to other agent
        r_ao = agent_pos - other_pos
        return 2 * self.params.ped_attract_C * r_ao  # gradient

    def _force_boundary(
        self,
        obs: PedestrianObservation,
        object: Union[Polygon, MultiPolygon],
        param_r: float,
        param_u: float,
    ) -> NDArray:
        """
        Compute the force from the boundary of an object.

        Can be an attractive or a repulsive force.
        """
        agent_pos = obs.pose[[0, 1]]
        agent_point = Point(*agent_pos)
        closest_point, _ = nearest_points(object, agent_point)
        closest_pos = np.array(closest_point.xy).squeeze()
        r_aB = agent_pos - closest_pos
        r_aB_norm = np.linalg.norm(r_aB)
        r_aB_unit = r_aB / (r_aB_norm + 0.0000000001)
        return (
            param_u / param_r * r_aB_unit * np.exp(-r_aB_norm / param_r)
        )  # gradient

    def _sight_weight(
        self, force_vector: np.ndarray, view_dir_unit_vector: np.ndarray
    ) -> float:
        """Compute the weight force depending on angle of sight."""
        dot_dir = np.dot(view_dir_unit_vector, force_vector) / (
            np.linalg.norm(force_vector) + 0.0000000001
        )
        if dot_dir >= np.cos(self.params.sight_angle / 2 * np.pi / 180):
            return 1.0
        return self.params.sight_weight
