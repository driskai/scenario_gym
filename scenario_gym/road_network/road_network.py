import json
from contextlib import suppress
from functools import _lru_cache_wrapper, lru_cache
from pathlib import Path
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from pyxodr.road_objects.network import RoadNetwork as xodrRoadNetwork
from scipy.interpolate import interp2d
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from scenario_gym.utils import ArrayLike, NDArray, cached_property

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
from .xodr import xodr_to_sg_roads


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
        return cls.create_from_dict(data, name=Path(filepath).stem, path=filepath)

    @classmethod
    @lru_cache(maxsize=15)
    def create_from_xodr(
        cls,
        filepath: str,
        resolution: float = 0.1,
        simplify_tolerance: float = 0.2,
    ):
        """
        Import a road network from an OpenDRIVE file.

        Will first parse the road network and then convert it to
        a scenario_gym road network. Every lane section of the file
        is converted to a road and each lane within the section is
        converted to a lane. Connectivity information is stored in
        the lanes. Any lane of type None is ignored.

        Parameters
        ----------
        filepath : str
            The filepath to the xodr file.

        resolution : float
            Resolution for importing the base OpenDRIVE file.

        simplify_tolerance : float
            Points per m for simplifying center and boundary lines.

        """
        path = Path(filepath).absolute()
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}.")

        # parse OpenDRIVE file
        xodr_network = xodrRoadNetwork(
            str(path),
            resolution=resolution,
        )

        roads = xodr_to_sg_roads(
            xodr_network,
            simplify_tolerance,
        )

        return cls(roads=roads, name=path.stem, path=str(path))

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
        name: Optional[str] = None,
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
        name: Optional[str]
            Optional name for the road network.

        path: Optional[str]
            The filepath of the road network data.

        road_objects : Dict[str, List[RoadObject]]
            The road objects as keywords. `roads` and `intersections` must be
            passed.

        """
        self.name = name
        self.path = path

        self._elevation_func: Optional[Callable[[float, float], float]] = None
        self._lane_parents: Dict[Lane, Optional[Union[Road, Intersection]]] = {}

        self.object_names = self._default_object_names.copy()
        self.object_classes = {v: k for k, v in self.object_names.items()}
        all_object_names = list(
            set(self.object_names.keys())
            .union(road_objects.keys())
            .difference(["roads", "intersections"])
        )
        for object_name in ["roads", "intersections"] + all_object_names:
            objects = (
                road_objects[object_name] if object_name in road_objects else []
            )
            assert all(
                (isinstance(obj, RoadObject) for obj in objects)
            ), "Only lists of RoadObject subclasses should be provided"

            if object_name not in self.object_names:
                self.object_names[object_name] = (
                    objects[0].__class__ if objects else RoadObject
                )
            self.add_new_road_object(objects, object_name)

    def add_new_road_object(
        self, objs: Union[RoadObject, List[RoadObject]], obj_name: str
    ) -> None:
        """
        Add a new object type to the road network.

        This will add an attribute for the raw objects as a list as well
        as a public attribute if it does not already exist. It will also add
        an add_{obj_name} method to add new objects to the list.
        """
        if hasattr(self, f"_{obj_name}"):
            raise ValueError(
                f"Road network already has {obj_name}. Use self.add_{obj_name}."
            )
        setattr(self, f"_{obj_name}", objs)
        try:
            getattr(self, obj_name)
        except AttributeError:
            setattr(self, obj_name, objs)
        try:
            getattr(self, f"add_{obj_name}")
        except AttributeError:

            def add_obj(self, objs):
                getattr(self, f"_{obj_name}").extend(
                    objs if isinstance(objs, list) else [objs]
                )
                self.clear_cache()

            setattr(self, f"add_{obj_name}", MethodType(add_obj, self))

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

    def get_lane_parent(self, l: Lane) -> Optional[Union[Road, Intersection]]:
        """Get the object that the lane belongs to."""
        if l not in self._lane_parents:
            for x in self.roads + self.intersections:
                if l in x.lanes:
                    self._lane_parents[l] = x
                    return x
            self._lane_parents[l] = None
        return self._lane_parents[l]

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
        with open(filepath, "w") as f:
            json.dump(data, f)

    def clear_cache(self) -> None:
        """Clear the cached properties and lru cache methods."""
        self._lane_parents.clear()
        self._elevation_func = None
        for method in dir(self.__class__):
            obj = getattr(self.__class__, method)
            if isinstance(obj, _lru_cache_wrapper):
                getattr(self, method).__func__.cache_clear()
            elif (
                isinstance(cached_property, type)
                and isinstance(obj, cached_property)
                and (method in self.__dict__)
            ):
                del self.__dict__[method]
            else:
                with suppress(AttributeError):
                    func = obj.__func__
                    if isinstance(func, _lru_cache_wrapper) and (
                        obj.__self__ is self
                    ):
                        func.cache_clear()

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
        if not elevs:
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
