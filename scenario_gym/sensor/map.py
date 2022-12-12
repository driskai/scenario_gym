import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.vectorized import contains

from scenario_gym.entity import Entity
from scenario_gym.observation import SingleEntityObservation
from scenario_gym.road_network import RoadNetwork
from scenario_gym.state import State
from scenario_gym.utils import ArrayLike, NDArray

from .base import Sensor


@dataclass
class MapObservation(SingleEntityObservation):
    """Observation with a raster map."""

    map: np.ndarray


class RasterizedMapSensor(Sensor):
    """
    Returns a rasterized semantic map as a 2d grid of vectors.

    Additional custom layers may be implemented by subclassing this sensor
    and implementing a perpare and a get method for the new layer.

    The prepare method should be called `_prepare_{}_layer` where {} is replaced
    with the string layer name. This method is called once when the road network
    is first seen and can be used to prepare any required data e.g. using prep on
    any shapely objects to improve performance.

    The get method should be called `_{}_layer` where {} is replaced with the
    string layer name. This method is called each step with the state and the map
    coordiantes to return the value of the map at each coordinate.
    """

    _all_layers: List[str] = [
        "entity",
        "driveable_surface",
        "road",
        "intersection",
        "lane",
        "walkable_surface",
        "pavement",
        "crossing",
    ]

    def __init__(
        self,
        entity: Entity,
        layers: Optional[List[str]] = None,
        height: float = 20.0,
        width: float = 20.0,
        freq: Optional[float] = 1.0,
        n: Optional[int] = None,
        channels_first: bool = False,
    ):
        """
        Init the sensor.

        Parameters
        ----------
        entity : Entity
            The entity.

        layers : Optional[List[str]]
            The layers to be observed as string names. The available layers can be
            found in `RasterizedMapSensor._all_layers`. The order in which these
            are returned will be used for the output array.

        height : float
            The length of the box around the entity in the y-direction in the
            local frame.

        width : float
            The length of the box around the entity in the x-direction in the
            local frame.

        freq : int
            The frequency of sampling points for the rasterized map. Only one of
            `freq` and `n` should be passed.

        n : Optional[int]
            The number of sampling points for the rasterized map. Only one of
            `freq` and `n` should be passed.

        channels_first : bool
            If given returns (C, W, H) rather than (W, H, C)

        """
        super().__init__(entity)
        self.layers = (
            layers if layers is not None else ["entity", "driveable_surface"]
        )
        self.check_layers()

        self.height = height
        self.width = width
        self.channels_first = channels_first
        if n is None:
            assert freq is not None, "At least one of n and freq must be provided."
            self.nw, self.nh = int(freq * width), int(freq * height)
        else:
            self.nw = self.nh = n

        self.X = np.array(
            np.meshgrid(
                np.linspace(-self.width / 2, self.width / 2, self.nw),
                np.linspace(-self.height / 2, self.height / 2, self.nh),
            )
        ).transpose(1, 2, 0)

    def check_layers(self) -> None:
        """Check that all layers are implemented correctly."""
        for layer in self.layers:
            try:
                getattr(self, f"_{layer}_layer")
                getattr(self, f"_prepare_{layer}_layer")
            except AttributeError:
                raise NotImplementedError(
                    f"Layer {layer} does not have a get and/or a prepare method."
                )

    def _reset(self, state: State) -> MapObservation:
        """Reset the sensor at the start of the scenario."""
        self._road_network: Optional[RoadNetwork] = None
        return self._step(state)

    def _step(self, state: State) -> MapObservation:
        """Return the rasterized map around the entity."""
        if self._road_network is None:
            self._prepare_layers(state)

        pose = state.poses[self.entity]
        coords = self._get_coords(pose).reshape(-1, 2)
        layers = [getattr(self, f"_{l}_layer")(state, coords) for l in self.layers]
        obs_map = np.array(layers).reshape(len(layers), self.nw, self.nw)
        return MapObservation(
            self.entity,
            *state.get_entity_data(self.entity),
            obs_map if self.channels_first else obs_map.transpose(1, 2, 0),
        )

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        """Return the output shape of the rasterized map."""
        if self.channels_first:
            return (len(self.layers), self.nw, self.nh)
        return (self.nw, self.nh, len(self.layers))

    def _get_coords(self, pose: ArrayLike) -> NDArray:
        """Get the coordinates at which the map should be constructed."""
        X = self.X  # (nw, nh, 2)

        xy, theta = pose[[0, 1]], pose[3] + math.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        return (X @ R.T) + xy[None, None, :]

    def _prepare_layers(self, state: State) -> None:
        """Get all data needed to compute the map at future timesteps."""
        self._road_network = state.scenario.road_network
        for layer in self.layers:
            getattr(self, f"_prepare_{layer}_layer")(state)

    def _prepare_entity_layer(self, state: State) -> None:
        """Prepare the entity layer."""
        pass

    def _entity_layer(self, state: State, coords: ArrayLike) -> NDArray:
        """
        Check which points are occupied by the bounding box of an entity.

        Note: this includes the sensor's own entity.
        """
        entities = prep(
            MultiPolygon(
                [
                    e.get_bounding_box_geom(state.poses[e])
                    for e in state.scenario.entities
                ]
            )
        )
        return contains(entities, coords[:, 0], coords[:, 1])

    def _prepare_driveable_surface_layer(self, state: State) -> None:
        """Prepare the driveable surface layer."""
        self._driveable_surface = prep(self._road_network.driveable_surface)

    def _driveable_surface_layer(self, state: State, coords: ArrayLike) -> NDArray:
        """Check which of the given points lie in the driveable surface."""
        return contains(self._driveable_surface, coords[:, 0], coords[:, 1])

    def _prepare_road_layer(self, state: State) -> None:
        """Prepare the road layer."""
        self._roads = prep(
            unary_union(
                [r.boundary for r in self._road_network.roads],
            )
        )

    def _road_layer(self, state: State, coords: ArrayLike) -> ArrayLike:
        """Check which points lie in a road."""
        return contains(self._roads, coords[:, 0], coords[:, 1])

    def _prepare_intersection_layer(self, state: State) -> None:
        """Prepare the intersection layer."""
        self._intersections = prep(
            unary_union(
                [i.boundary for i in self._road_network.intersections],
            )
        )

    def _intersection_layer(self, state: State, coords: ArrayLike) -> NDArray:
        """Check which points lie in an intersection."""
        return contains(self._intersections, coords[:, 0], coords[:, 1])

    def _prepare_lane_layer(self, state: State) -> None:
        """Prepare the lane layer."""
        self._lanes = prep(
            unary_union(
                [l.boundary for r in self._road_network.roads for l in r.lanes],
            )
        )

    def _lane_layer(self, state: State, coords: ArrayLike) -> NDArray:
        """Check which points lie in a lane."""
        return contains(self._lanes, coords[:, 0], coords[:, 1])

    def _prepare_walkable_surface_layer(self, state: State) -> None:
        """Prepare the walkable surface layer."""
        self._walkable_surface = prep(self._road_network.walkable_surface)

    def _walkable_surface_layer(self, state: State, coords: ArrayLike) -> NDArray:
        """Check which points lie in a walkable surface."""
        return contains(self._walkable_surface, coords[:, 0], coords[:, 1])

    def _prepare_pavement_layer(self, state: State) -> None:
        """Prepare the pavement layer."""
        self._pavements = prep(
            unary_union([p.boundary for p in self._road_network.pavements])
        )

    def _pavement_layer(self, state: State, coords: ArrayLike) -> NDArray:
        """Check which points lie in a pavement."""
        return contains(self._pavements, coords[:, 0], coords[:, 1])

    def _prepare_crossing_layer(self, state: State) -> None:
        """Prepare the crossing layer."""
        self._crossings = prep(
            unary_union([c.boundary for c in self._road_network.crossings])
        )

    def _crossing_layer(self, state: State, coords: ArrayLike) -> NDArray:
        """Check which points lie in a pedestrian crossing."""
        return contains(self._crossings, coords[:, 0], coords[:, 1])
