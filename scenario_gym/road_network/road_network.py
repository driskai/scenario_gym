import json
from functools import _lru_cache_wrapper, cached_property, lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp2d
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .base import RoadGeometry, RoadObject
from .objects import (
    Building,
    Crossing,
    Intersection,
    Lane,
    LaneType,
    Pavement,
    Road,
)


class RoadNetwork:
    """
    A collection of roads, intersections, etc that form a road network.

    The road network implements layers that give different objects in
    the network. Default layers can be seein the object_names attribute.

    Any list objects that subclasses RoadObject or RoadGeometry can be passed
    as keywords to add custom objects to the road network.
    """

    _default_object_names: Dict[str, Type[RoadObject]] = {
        "roads": Road,
        "intersections": Intersection,
        "lanes": Lane,
        "pavements": Pavement,
        "crossings": Crossing,
        "buildings": Building,
    }

    @classmethod
    @lru_cache(maxsize=15)
    def create_from_json(cls, filepath: str):
        """
        Create the road network from a json file.

        Parameters
        ----------
        filepath : str
            The path to the json file.

        """
        with open(filepath) as f:
            data = json.load(f)
        return cls.create_from_dict(data, path=filepath)

    @classmethod
    def create_from_dict(cls, data: Dict, **kwargs):
        """
        Create a road network from a dictoinary of road data.

        The dictionary must have keys 'Roads' and 'Intersections' and
        optionally with keys for other road objects. Each of these must hold
        a list of dicts with the data for each object. These should hold their
        required fields e.g. Center, Boundary, successors, predecessors.
        """
        assert (
            "Roads" in data or "roads" in data
        ), "Json data must contain road information."
        assert (
            "Intersections" in data or "intersections" in data
        ), "Json data must contain intersection information."

        objects = {}
        for obj, obj_cls in cls._default_object_names.items():
            if obj in data:
                key = obj
            elif obj.capitalize() in data:
                key = obj.capitalize()
            else:
                continue
            objects[obj] = [obj_cls.from_dict(obj_data) for obj_data in data[key]]

        return cls(**kwargs, **objects)

    def __init__(
        self,
        path: Optional[str] = None,
        **road_objects: Dict[str, List[RoadObject]],
    ):
        """
        Construct the road network.

        This takes lists of road objects as keywords. The keyword used determines
        how the objects will be stored. E.g. `roads=[...]` will define the `roads`
        attribute. This way custom road objects can be passed e.g. passing
        `road_markings=[...]` will mean a `road_markings` attribute is created.
        Every object in the list must be a subclass of `RoadObject`. `roads` and
        `intersections` must be passed even if they are empty lists.

        Parameters
        ----------
        path: Optional[str]
            The filepath of the road network data.

        road_objects : Dict[str, List[RoadObject]]
            The road objects as keywords. `roads` and `intersections` must be
            passed.

        """
        assert ("roads" in road_objects and road_objects["roads"]) or (
            ("intersections" in road_objects and road_objects["intersections"])
        ), "At least one road or intersection is required."
        self._elevation_func: Optional[Callable[[float, float], float]] = None
        self.path = path
        self.object_names = self._default_object_names.copy()

        all_object_names = list(
            set(self.object_names.keys())
            .union(road_objects.keys())
            .difference(["roads", "intersections"])
        )
        for object_name in ["roads", "intersections"] + all_object_names:
            if object_name in road_objects:
                objects = road_objects[object_name]
                assert all(
                    (isinstance(obj, RoadObject) for obj in objects)
                ), "Only lists of RoadObject subclasses should be provided"
            else:
                objects = []
            if object_name not in self.object_names:
                if len(objects) == 0:
                    continue
                else:
                    self.object_names[object_name] = objects[0].__class__
            setattr(self, f"_{object_name}", objects)
            try:
                getattr(self, object_name)
            except AttributeError:
                setattr(self, object_name, objects)

    @cached_property
    def roads(self) -> List[Road]:
        """Get all roads in the road network."""
        return self._roads

    @cached_property
    def intersections(self) -> List[Intersection]:
        """Get all intersections in the road network."""
        return self._intersections

    @cached_property
    def lanes(self) -> List[Lane]:
        """Get all lanes in the road network."""
        return list(
            set(sum([x.lanes for x in self.roads + self.intersections], [])).union(
                self._lanes
            )
        )

    @cached_property
    def road_network_objects(self) -> List[RoadObject]:
        """Get all the road objects in the network."""
        return [
            obj for obj_name in self.object_names for obj in getattr(self, obj_name)
        ]

    @cached_property
    def road_network_geometries(self) -> List[RoadGeometry]:
        """Get all road geometries in the network."""
        geoms = []
        for obj_name, obj_class in self.object_names.items():
            if issubclass(obj_class, RoadGeometry):
                geoms.extend(getattr(self, obj_name))
        return geoms

    @cached_property
    def driveable_surface(self) -> MultiPolygon:
        """Get the union of boundaries of driveable geometries."""
        merged = unary_union(
            [g.boundary for g in self.road_network_geometries if g.driveable]
        )
        return MultiPolygon([merged]) if isinstance(merged, Polygon) else merged

    @cached_property
    def walkable_surface(self) -> MultiPolygon:
        """Get the union of boundaries of non-driveable geometries."""
        merged = unary_union(
            [g.boundary for g in self.road_network_geometries if g.walkable]
        )
        return MultiPolygon([merged]) if isinstance(merged, Polygon) else merged

    @cached_property
    def impenetrable_surface(self) -> MultiPolygon:
        """Get the union of all impenetrable geometries."""
        merged = unary_union(
            [g.boundary for g in self.road_network_geometries if g.impenetrable]
        )
        return MultiPolygon([merged]) if isinstance(merged, Polygon) else merged

    def object_by_id(self, i: str) -> RoadObject:
        """Get the object with the given id."""
        return self._object_by_id[i]

    @cached_property
    def _object_by_id(self) -> Dict[str, RoadObject]:
        """Return a dict indexing all objects by id."""
        return {x.id: x for x in self.road_network_objects}

    @cached_property
    def driveable_lanes(self) -> List[Lane]:
        """Get all driveable lanes in the network."""
        return [l for l in self.lanes if l.type is LaneType["driving"]]

    @cached_property
    def _lanes_by_id(self) -> Dict[str, Lane]:
        """Return a dict indexing all lanes by id."""
        return {l.id: l for l in self.lanes}

    def get_successor_lanes(self, l: Lane) -> List[Lane]:
        """Get lanes that succeed the given lane."""
        return [self._lanes_by_id[l_] for l_ in l.successors]

    def get_predecessor_lanes(self, l: Lane) -> List[Lane]:
        """Get lanes that predecess the given lane."""
        return [self._lanes_by_id[l_] for l_ in l.predecessors]

    def get_connecting_roads(self, i: Intersection) -> List[Road]:
        """Get roads that connect to the given intersection."""
        return [r for r in self.roads if r in i.connecting_roads]

    def get_intersections(self, r: Road) -> List[Intersection]:
        """Get intersections that connect to the given road."""
        return [i for i in self.intersections if r in i.connecting_roads]

    @lru_cache(maxsize=200)
    def get_lane_parent(self, l: Lane) -> Union[Road, Intersection]:
        """Get the object that the lane belongs to."""
        for x in self.roads + self.intersections:
            if l in x.lanes:
                return x

    def get_geometries_at_point(
        self,
        x: float,
        y: float,
    ) -> Tuple[List[str], List[RoadGeometry]]:
        """
        Get all geometries at a given xy point.

        TODO: Move to a spatial index for speed.

        Parameters
        ----------
        x : float
            The x-coordinate at the point.

        y : float
            The y-coordinate at the point.

        Returns
        -------
        Tuple[List[str], List[RoadObject]]
            A list of string identifiers of the geometry (e.g. Road, Lane)
            and the actual objects.

        """
        p = Point(x, y)

        names, geoms = [], []
        for x in self.road_network_geometries:
            if x.boundary.contains(p):
                names.append(x.__class__.__name__)
                geoms.append(x)
        return names, geoms

    def to_json(self, filepath: str) -> None:
        """Save the road network to json file."""
        data = {}
        for obj_name in self.object_names:
            data[obj_name] = [obj.to_dict() for obj in getattr(self, obj_name)]
        json.dump(data, open(filepath, "w"))

    def clear_cache(self) -> None:
        """Clear the cached properties and lru cache methods."""
        for method in dir(self.__class__):
            obj = getattr(self.__class__, method)
            if isinstance(obj, cached_property) and (method in self.__dict__):
                del self.__dict__[method]
            elif isinstance(obj, _lru_cache_wrapper):
                getattr(self, method).__func__.cache_clear()
            else:
                try:
                    func = getattr(obj, "__func__")
                    if isinstance(func, _lru_cache_wrapper):
                        func.cache_clear()
                except AttributeError:
                    continue

    def elevation_at_point(self, x: ArrayLike, y: ArrayLike) -> NDArray:
        """Estimate the elevation at (x, y) by interpolating."""
        if self._elevation_func is None:
            self._interpolate_elevation()
        return self._elevation_func(x, y)

    def _interpolate_elevation(self) -> None:
        """Interpolate the elevation values of the geometries."""
        elevs = [
            geom.elevation
            for geom in self.road_network_geometries
            if geom.elevation is not None
        ]
        if len(elevs) == 0:
            self._elevation_func = lambda x, y: np.zeros_like(x)
        else:
            elevation_values = np.concatenate(elevs, axis=0)
            x, y, z = elevation_values.T
            self._elevation_func = interp2d(
                x,
                y,
                z,
                kind="linear",
                bounds_error=False,
            )
