"""Test pedestrian agent creation."""
import os
from copy import deepcopy
from typing import Tuple

import numpy as np
from shapely.geometry import Point, Polygon

import scenario_gym
from scenario_gym.agent import Agent, ReplayTrajectoryAgent
from scenario_gym.controller import ReplayTrajectoryController
from scenario_gym.entity import Entity
from scenario_gym.manager import ScenarioManager
from scenario_gym.pedestrian.agent import PedestrianAgent
from scenario_gym.pedestrian.controller import PedestrianController
from scenario_gym.pedestrian.route import RouteFinder
from scenario_gym.pedestrian.sensor import PedestrianSensor
from scenario_gym.pedestrian.social_force import SocialForce, SocialForceParameters
from scenario_gym.road_network import RoadNetwork
from scenario_gym.scenario import Scenario
from scenario_gym.sensor import EgoLocalizationSensor
from scenario_gym.trajectory import Trajectory
from scenario_gym.xosc_interface import import_scenario, read_catalog


class PedestrianConfig(ScenarioManager):

    PARAMETERS = {
        "timestep": 0.0333,
        "headless_rendering": False,
        "num_pedestrians": 20,
        "speed": 5.0,
        "max_speed_factor": 1.3,
        "bias_lon": 0.0,
        "bias_lat": 0.0,
        "std_lon": 0.000002,
        "std_lat": 0.0000001,
        "distance_threshold": 3,
        "sight_weight": 0.5,
        "sight_weight_use": True,
        "sight_angle": 200,
        "relaxation_time": 1.5,
        "ped_repulse_V": 5.0,
        "ped_repulse_sigma": 0.5,
        "ped_attract_C": 0.0,
        "boundary_repulse_U": 10.0,
        "boundary_repulse_R": 0.2,
        "imp_boundary_repulse_U": 10.0,
        "imp_boundary_repulse_R": 0.2,
        "use_raw_traj": False,
    }

    @property
    def sf_params(self) -> SocialForceParameters:
        """Get the parmeters of the social force model."""
        return SocialForceParameters(
            max_speed_factor=self.max_speed_factor,
            bias_lon=self.bias_lon,
            bias_lat=self.bias_lat,
            std_lon=self.std_lon,
            std_lat=self.std_lat,
            distance_threshold=self.distance_threshold,
            sight_weight=self.sight_weight,
            sight_weight_use=self.sight_weight_use,
            sight_angle=self.sight_angle,
            relaxation_time=self.relaxation_time,
            ped_repulse_V=self.ped_repulse_V,
            ped_repulse_sigma=self.ped_repulse_sigma,
            ped_attract_C=self.ped_attract_C,
            boundary_repulse_U=self.boundary_repulse_U,
            boundary_repulse_R=self.boundary_repulse_R,
        )

    def create_agent(self, sc: Scenario, entity: Entity) -> Agent:
        if entity.ref == "ego":
            sensor = EgoLocalizationSensor(entity)
            controller = ReplayTrajectoryController(entity)
            return ReplayTrajectoryAgent(entity, controller, sensor)
        elif entity.type == "Pedestrian":
            speed_desired = np.random.uniform(
                0.5 * self.speed, 1.5 * self.speed
            )  # random desired speed
            behaviour = SocialForce(self.sf_params)

            # Find route for pedestrian along walkable surface
            route_finder = RouteFinder(sc.road_network)
            start = entity.trajectory[0][[1, 2]]
            finish = entity.trajectory[-1][[1, 2]]
            if self.use_raw_traj:
                route = entity.trajectory.data[:, [1, 2]]
            else:
                route = route_finder.find_route(start, finish)
                if route is None:
                    route = entity.trajectory.data[:, [1, 2]]
            return PedestrianAgent(entity, route, speed_desired, behaviour)

    def add_random_pedestrians(self, sc: Scenario):
        _, catalog = read_catalog(
            os.path.join(
                os.path.dirname(sc.path),
                (
                    "../Catalogs/Scenario_Gym/PedestrianCatalogs/"
                    "ScenarioGymPedestrianCatalog.xosc"
                ),
            )
        )
        base_entity = catalog["pedestrian1"]

        for i in range(self.num_pedestrians):
            e = deepcopy(base_entity)
            e.trajectory = self.sample_pedestrian_trajectory(sc.road_network)
            e.ref = f"new_pedestrian_{i}"
            sc.add_entity(e)

    def polygon_random(self, poly: Polygon) -> Tuple[float, float]:
        """Sample a point in the polygon uniformly with respect to its area."""
        x_min, y_min, x_max, y_max = poly.bounds
        while True:
            r_x, r_y = np.random.rand(2)
            x = (x_max - x_min) * r_x + x_min
            y = (y_max - y_min) * r_y + y_min
            if poly.contains(Point(x, y)):
                return x, y

    def sample_pedestrian_point(
        self, road_network: RoadNetwork
    ) -> Tuple[float, float]:
        if road_network.walkable_surface.area > 0.0:
            return self.polygon_random(road_network.walkable_surface)
        else:
            poly = road_network.driveable_surface.buffer(5.0).difference(
                road_network.driveable_surface
            )
            return self.polygon_random(poly)

    def sample_pedestrian_trajectory(self, road_network: RoadNetwork) -> Trajectory:
        return Trajectory(
            np.array(
                [
                    [0.0, *self.sample_pedestrian_point(road_network)],
                    [10.0, *self.sample_pedestrian_point(road_network)],
                ]
            ),
            fields=["t", "x", "y"],
        )


if __name__ == "__main__":

    config = PedestrianConfig()
    gym = scenario_gym.ScenarioGym(
        timestep=config.timestep,
        headless_rendering=config.headless_rendering,
        terminal_conditions=config.terminal_conditions,
    )

    scenario_path = os.path.join(
        os.path.dirname(__file__),
        "../tests/input_files/Scenarios/1518e754-318f-4847-8a30-2dce552b4504.xosc",
    )

    scenario = import_scenario(scenario_path)
    config.add_random_pedestrians(scenario)
    gym.set_scenario(scenario, create_agent=config.create_agent)
    gym.reset_scenario()
    gym.rollout(render=True)
