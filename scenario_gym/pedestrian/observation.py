from dataclasses import dataclass
from typing import List, Tuple

from shapely.geometry import MultiPolygon

from scenario_gym.entity import Entity
from scenario_gym.observation import SingleEntityObservation
from scenario_gym.utils import NDArray


@dataclass
class PedestrianObservation(SingleEntityObservation):
    """
    An observation for a pedestrian agent.

    Contains the objects and road elements detected within a given radius.

    Todo: Instead of returning complete state, return observation with only entities
    and pedestrian road info within a given radius by adapting RasterizedMapSensor
    or ObjectDetectionSensor.
    """

    head_rot_angle: float
    near_peds: List[Tuple[Entity, NDArray, NDArray]]
    walkable_surface: MultiPolygon
    impenetrable_surface: MultiPolygon
