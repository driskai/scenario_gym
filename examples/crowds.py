"""
Crowd modelling with a social force model for pedestrian dynamics.

Simulate the dynamics of a crowd of pedestrians moving along a
road. Pedestrians aim to walk along a pavement without
stepping into the road. A building will restrict the pavement
width halfway along forcing the crowd to spill into the road.
"""
import os
from copy import deepcopy

import numpy as np
from shapely.geometry import LineString, Polygon
from social_force import PedestrianConfig

from scenario_gym import Scenario, ScenarioGym, Trajectory
from scenario_gym.road_network import Building, Lane, Pavement, Road, RoadNetwork
from scenario_gym.xosc_interface import read_catalog


def main():

    # define the gym and rendering config
    gym = ScenarioGym(
        timestep=1 / 15,
        headless_rendering=True,
        render_layers=[
            "driveable_surface",
            "walkable_surface",
            "buildings",
            "road_centers",
        ],
    )

    # define the scenario config using the Social Force model
    config = PedestrianConfig(
        **{
            "relaxation_time": 1.5,
            "ped_repulse_V": 5.0,
            "ped_repulse_sigma": 0.5,
            "ped_attract_C": 0.0,
            "boundary_repulse_U": 10.0,
            "boundary_repulse_R": 0.2,
            "imp_boundary_repulse_U": 10.0,
            "imp_boundary_repulse_R": 0.2,
        }
    )

    # setup the scenario
    scenario = make_scenario()
    gym.set_scenario(scenario, create_agent=config.create_agent)

    # rollout and render
    gym.rollout(render=True, video_path="crowd_model1.mp4")


def make_scenario():
    """Setup a scenario with a crowd of pedestrians."""
    ents = []
    _, veh_catalog = read_catalog(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "tests",
            "input_files",
            "Catalogs",
            "Scenario_Gym",
            "VehicleCatalogs",
            "ScenarioGymVehicleCatalog.xosc",
        )
    )
    e = deepcopy(veh_catalog["car1"])
    e.ref = "ego"
    e.trajectory = Trajectory(
        np.array(
            [
                [0.0, 2.0, 0.0, np.pi / 2],
                [30.0, 2.0, 0.0, np.pi / 2],
            ]
        ),
        fields=["t", "x", "y", "h"],
    )
    ents.append(e)

    _, ped_catalog = read_catalog(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "tests",
            "input_files",
            "Catalogs",
            "Scenario_Gym",
            "PedestrianCatalogs",
            "ScenarioGymPedestrianCatalog.xosc",
        )
    )
    e_ped = deepcopy(ped_catalog["pedestrian1"])
    for idx in range(30):
        e = deepcopy(e_ped)
        e.ref = f"entity_{idx+1}"
        e.trajectory = Trajectory(
            np.array(
                [
                    [
                        0.0,
                        -4.2 - np.random.rand() * 5,
                        -10.0 + np.random.randn() * 2,
                    ],
                    [
                        30.0,
                        -4.2 - np.random.rand() * 1,
                        23.0 + np.random.randn() * 4,
                    ],
                ]
            ),
            fields=["t", "x", "y"],
        )
        ents.append(e)
    s = Scenario(ents, road_network=construct_road_network(), name="crowd_scenario")
    return s


def make_boundary(w_low, w_high, l_low, l_high):
    """Helper to make a rectangular road boundary."""
    return Polygon(
        np.array(
            [
                [w_low, l_low],
                [w_high, l_low],
                [w_high, l_high],
                [w_low, l_high],
            ]
        )
    )


def make_center(w, l_low, l_high):
    """Helper to make a vertical road center."""
    return LineString(
        np.array(
            [
                [w, l_low],
                [w, l_high],
            ]
        )
    )


def construct_road_network(w=4.2, l=25):
    """Define a road network with a single road and buildings."""
    lanes = [
        Lane(
            "lane_0",
            make_boundary(-w, 0, -l, l),
            make_center(-w / 2, -l, l),
            [],
            [],
            "driving",
        ),
        Lane(
            "lane_1",
            make_boundary(0, w, -l, l),
            make_center(w / 2, -l, l),
            [],
            [],
            "driving",
        ),
    ]
    road = Road(
        "road_0",
        make_boundary(-w, w, -l, l),
        make_center(0, -l, l),
        lanes=lanes,
    )
    pavements = [
        Pavement(
            "pave_0",
            make_boundary(-w - 2, -w, -l, l),
            make_center(-w - 1, -l, l),
        ),
        Pavement(
            "pave_1",
            make_boundary(w, w + 1, -l, l),
            make_center(w + 1 / 2, -l, l),
        ),
    ]
    building = Building(
        "building_0",
        Polygon(
            np.array(
                [
                    [-10.0, 0.0],
                    [-10.0, 10.0],
                    [-5.2, 10.0],
                    [-5.2, 0.0],
                ]
            )
        ),
    )
    return RoadNetwork(
        roads=[road],
        intersections=[],
        pavements=pavements,
        buildings=[building],
    )


if __name__ == "__main__":
    main()
