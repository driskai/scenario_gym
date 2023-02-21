from dataclasses import dataclass, field
from random import choice
from typing import Dict, Optional

import numpy as np
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import load_all_maps
from scipy.spatial.transform import Rotation
from shapely.geometry import LineString

from scenario_gym.catalog_entry import BoundingBox, Catalog, CatalogEntry
from scenario_gym.entity import Entity
from scenario_gym.road_network import Lane, LaneType, Road, RoadNetwork
from scenario_gym.scenario import Scenario
from scenario_gym.trajectory import Trajectory


@dataclass
class NuScenesInstanceData:
    """Class for storing data for NuScenes "instances" (entities)."""

    category_name: str
    trajectory: list = field(default_factory=list)
    times: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    rotations: list = field(default_factory=list)


class Catalogs:
    """
    Catalogs used in the nuScenes dataset.

    Using the dimensions defined "at runtime" from the bounding box data.
    """

    nuScenes_catalog = Catalog("nuScenesCatalog", None)


class NuScenesImporter:
    """
    Class for importing nuScenes scenes into scenario gym scenarios.

    Parameters
    ----------
    data_root : str
        Dataset root path to be passed into the NuScenes constructor.
    dataset : str, optional
        Dataset name to be passed into the NuScenes constructor, by default
        "v1.0-mini"
    map_radius_multiplier : float, optional
        When including a map around a scene, a centrepoint will be computed as the
        average position of all trajectories; a radius will be computed as the max
        x and y range of coordinates. All lanes within this radius multiplied by
        the map_radius_multiplier of the centrepoint will be included. By default
        1.5
    pre_loaded_data : NuScenes, optional
        Pre-loaded (indexed) data object, for much faster init if the NuScenes
        dataset has been pre-loaded already, by default None.

    """

    def __init__(
        self,
        data_root: str,
        dataset: str = "v1.0-mini",
        map_radius_multiplier: float = 1.5,
        pre_loaded_data: Optional[NuScenes] = None,
    ):
        self.data_root = data_root
        self.dataset = dataset

        if pre_loaded_data is not None:
            self.data = pre_loaded_data
        else:
            self.data = NuScenes(self.dataset, dataroot=self.data_root)
        self.predict_helper = PredictHelper(self.data)

        self.maps = load_all_maps(self.predict_helper)

        self.map_radius_multiplier = map_radius_multiplier

    def _convert_nuScenes_map_to_road_network(
        self, map_name, centre_coordinate: np.ndarray, map_radius: float
    ) -> RoadNetwork:
        map = self.maps[map_name]

        lane_records = map.get_records_in_radius(
            *centre_coordinate, map_radius, ["lane", "lane_connector"]
        )

        roads = []
        lane_centres = map.discretize_lanes(
            lane_records["lane"], 0.1
        ) | map.discretize_lanes(lane_records["lane_connector"], 0.1)

        lane_keys_and_records = [("lane", l) for l in lane_records["lane"]] + [
            ("lane_connector", l) for l in lane_records["lane_connector"]
        ]
        for lane_key, lane_record in lane_keys_and_records:
            lane = map.get(lane_key, lane_record)
            bounding_poly = map.extract_polygon(lane["polygon_token"])
            lane_centre = LineString(np.array(lane_centres[lane_record])[:, :2])

            sg_lane = Lane(
                lane_record,
                bounding_poly,
                lane_centre,
                map.get_outgoing_lane_ids(lane_record),
                map.get_incoming_lane_ids(lane_record),
                LaneType.driving,
                elevation=np.array(lane_centres[lane_record]),
            )

            roads.append(
                Road(
                    f"road_{lane_key}_{lane_record}",
                    bounding_poly,
                    lane_centre,
                    [sg_lane],
                )
            )

        road_network = RoadNetwork(
            roads=roads,
            intersections=[],
        )

        return road_network

    def convert_to_gym(
        self, scene_token: str, ego_instance_token: Optional[str] = None
    ) -> Scenario:
        """
        Convert a nuScenes scene to a scenario gym scenario.

        Where ego_instance_token is provided, the instance (agent) which corresponds
        to that token will be treated as the ego. Otherwise, a random car will be
        chosen.

        See https://www.nuscenes.org/nuscenes#data-format for nuScenes schema.

        Parameters
        ----------
        scene_token : str
            Unique identifier for a nuScenes scene
        ego_instance_token : Optional[str], optional
            Identifier for the instance to be used as the scenario gym ego, by
            default None

        Returns
        -------
        Scenario
            Converted scenario gym scenario corresponding to this nuScenes scene.

        Raises
        ------
        KeyError
            If the provided ego_instance_token is not found in the scene.
        ValueError
            If no ego_instance_token is provided and there are no cars in the scene
            to be chosen as the ego.

        """
        scene_data = self.data.get("scene", scene_token)

        sample_annotations: list[list[dict]] = []

        first_sample_token = scene_data["first_sample_token"]
        last_sample_token = scene_data["last_sample_token"]
        current_sample_token = first_sample_token

        while current_sample_token != last_sample_token:
            if current_sample_token is None or current_sample_token == "":
                print(
                    "WARNING: Got an unexpected sample token of "
                    + str(current_sample_token)
                )
                break
            sample_annotations.append(
                self.predict_helper.get_annotations_for_sample(current_sample_token)
            )
            current_sample_token = self.data.get("sample", current_sample_token)[
                "next"
            ]

        # Since nuScenes sampled at 2Hz
        times = np.linspace(
            0.0,
            0.5 * (len(sample_annotations) - 1),
            len(sample_annotations),
        )

        # Now link from instance IDs to relevant instance (entity) data
        instance_token_to_data: Dict[str, NuScenesInstanceData] = {}
        for sample, time in zip(sample_annotations, times):
            for annotation in sample:
                instance_token = annotation["instance_token"]
                if instance_token not in instance_token_to_data.keys():
                    instance_token_to_data[instance_token] = NuScenesInstanceData(
                        annotation["category_name"]
                    )
                instance_token_to_data[instance_token].trajectory.append(
                    annotation["translation"]
                )
                instance_token_to_data[instance_token].times.append(time)
                instance_token_to_data[instance_token].sizes.append(
                    annotation["size"]
                )
                instance_token_to_data[instance_token].rotations.append(
                    annotation["rotation"]
                )

        if ego_instance_token is not None:
            if ego_instance_token not in instance_token_to_data.keys():
                raise KeyError("Ego instance token not found in scene.")
        else:
            potential_ego_instance_tokens = [
                i
                for i, d in instance_token_to_data.items()
                if "vehicle.car" in d.category_name
            ]
            if len(potential_ego_instance_tokens) == 0:
                raise ValueError("No potential ego vehicles in scene (no cars).")
            ego_instance_token = choice(potential_ego_instance_tokens)
            print(f"Chose ego instance token {ego_instance_token}")

        entities: Entity = []

        instance_tokens = set(instance_token_to_data.keys()) - set(
            [ego_instance_token]
        )
        instance_tokens = [ego_instance_token] + list(instance_tokens)
        for instance_token in instance_tokens:
            instance_data = instance_token_to_data[instance_token]
            entity_type = (
                "Pedestrian"
                if instance_data.category_name.split(".")[0]
                in {"human", "pedestrian"}
                else "Vehicle"
            )
            entity_category = "_".join(instance_data.category_name.split(".")[1:])
            # The scenario gym works in a slightly different way from the nuScenes
            # dataset. Bounding boxes are set at the catalog level and are fixed in
            # time.
            bounding_box = BoundingBox(
                *np.array(instance_data.sizes).mean(axis=0)[:2], 0.0, 0.0
            )
            setattr(
                Catalogs,
                instance_token,
                CatalogEntry(
                    Catalogs.nuScenes_catalog,
                    instance_data.category_name,
                    entity_category,
                    entity_type,
                    bounding_box,
                    {},
                    [],
                ),
            )

            rotations = Rotation.from_quat(instance_data.rotations).as_euler("xyz")[
                :, 0
            ]

            sg_rotations = np.pi * np.ones_like(rotations) - rotations

            traj_data = np.vstack(
                [
                    np.array(instance_data.times),
                    np.array(instance_data.trajectory).T[:2],
                    sg_rotations,
                ]
            ).T

            trajectory = Trajectory(traj_data, fields=["t", "x", "y", "h"])

            entity_ref = (
                f"entity_{instance_token}"
                if instance_token != ego_instance_token
                else "ego"
            )
            entity = Entity(getattr(Catalogs, instance_token), ref=entity_ref)
            entity.trajectory = trajectory
            entities.append(entity)

        all_trajectory_data = np.vstack(
            [np.array(d.trajectory) for d in instance_token_to_data.values()]
        )

        x_range, y_range, _ = np.ptp(all_trajectory_data, axis=0)
        radius = max([x_range, y_range]) * self.map_radius_multiplier

        centre_coordinate = np.mean(all_trajectory_data, axis=0)[:2]

        map_name = self.predict_helper.get_map_name_from_sample_token(
            first_sample_token
        )

        road_network = self._convert_nuScenes_map_to_road_network(
            map_name, centre_coordinate, radius
        )

        scenario = Scenario(
            entities,
            name=scene_token,
            road_network=road_network,
        )
        return scenario
