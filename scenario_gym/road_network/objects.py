from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from shapely.geometry import LineString, Polygon

from scenario_gym.utils import ArgsKwargs

from .base import RoadGeometry, RoadLike


class LaneType(Enum):
    """Enumerates OpenDrive standard lane types."""

    none = 0
    driving = 1
    HOV = 2
    bidirectional = 3
    biking = 4
    border = 5
    bus = 6
    connectingRamp = 7
    curb = 8
    entry = 9
    exit = 10
    median = 11
    mwyEntry = 12
    mwyExit = 13
    offRamp = 14
    onRamp = 15
    parking = 16
    rail = 17
    restricted = 18
    roadWorks = 19
    shoulder = 20
    sidewalk = 21
    special1 = 22
    special2 = 23
    special3 = 24
    stop = 25
    taxi = 26
    tram = 27


class Lane(RoadLike):
    """
    A lane object.

    The lane has a center and a boundary and holds information of its successor
    and predeceessor lanes.
    """

    walkable = False

    @classmethod
    def load_data_from_dict(cls, l: Dict[str, Any]) -> ArgsKwargs:
        """Create from dictionary."""
        args, kwargs = super().load_data_from_dict(l)
        typ = l["type"] if "type" in l else "driving"
        lane_type = LaneType[typ if typ in LaneType.__members__ else "driving"]
        return (
            *args,
            l["successors"] if "successors" in l else [],
            l["predecessors"] if "predecessors" in l else [],
            lane_type,
        ), kwargs

    def __init__(
        self,
        id: str,
        boundary: Polygon,
        center: LineString,
        successors: List[str],
        predecessors: List[str],
        _type: Union[str, LaneType],
        elevation: Optional[np.ndarray] = None,
    ):
        super().__init__(id, boundary, center, elevation=elevation)
        self.successors = successors
        self.predecessors = predecessors
        if isinstance(_type, str):
            if _type not in LaneType.__members__:
                raise ValueError(
                    f"{self.type} is not a valid lane type. "
                    "Check objects.py to see all valid lane types.",
                )
            else:
                _type = LaneType[_type]
        self._type = _type

    @property
    def type(self) -> LaneType:
        """Get the type of the lane."""
        return self._type

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of the objects data."""
        data = super().to_dict()
        data.update(
            {
                "successors": self.successors,
                "predecessors": self.predecessors,
            }
        )
        if self.type is not None:
            data["type"] = self.type.name
        return data


class Road(RoadLike):
    """
    A road object.

    The road has a center and a boundary and lanes.
    """

    walkable = False

    @classmethod
    def load_data_from_dict(cls, r: Dict[str, Any]) -> ArgsKwargs:
        """Create from dictionary."""
        args, kwargs = super().load_data_from_dict(r)
        lanes = [Lane.from_dict(l) for l in r["lanes" if "lanes" in r else "Lanes"]]
        return (*args, lanes), kwargs

    def __init__(
        self,
        id: str,
        boundary: Polygon,
        center: LineString,
        lanes: List[Lane],
        elevation: Optional[np.ndarray] = None,
    ):
        super().__init__(id, boundary, center, elevation=elevation)
        self.lanes = lanes

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of the objects data."""
        data = super().to_dict()
        data["lanes"] = [l.to_dict() for l in self.lanes]
        return data


class Intersection(RoadGeometry):
    """
    An intersection object.

    The intersection has a boundary, connecting lanes and the ids of
    the roads it connects.
    """

    driveable = True
    walkable = False

    @classmethod
    def load_data_from_dict(cls, i: Dict[str, Any]) -> ArgsKwargs:
        """Create from dictionary."""
        args, kwargs = super().load_data_from_dict(i)
        lanes = [Lane.from_dict(l) for l in i["lanes" if "lanes" in i else "Lanes"]]
        return (*args, lanes, i["connecting_roads"]), kwargs

    def __init__(
        self,
        id: str,
        boundary: Polygon,
        lanes: List[Lane],
        connecting_roads: List[str],
        elevation: Optional[np.ndarray] = None,
    ):
        super().__init__(id, boundary, elevation=elevation)
        self.lanes = lanes
        self.connecting_roads = connecting_roads

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of the objects data."""
        data = super().to_dict()
        data.update(
            {
                "lanes": [l.to_dict() for l in self.lanes],
                "connecting_roads": self.connecting_roads,
            }
        )
        return data


class Pavement(RoadLike):
    """
    A pavement object.

    The pavement has a boundary and a center.
    """

    driveable = False


class Crossing(RoadLike):
    """
    A crossing object.

    The crossing has a boundary and center and ids of pavements it connects.
    """

    driveable = False

    @classmethod
    def load_data_from_dict(cls, c: Dict[str, Any]) -> ArgsKwargs:
        """Create from dictionary."""
        args, kwargs = super().load_data_from_dict(c)
        return (
            *args,
            c["pavements" if "pavements" in c else "Pavements"],
        ), kwargs

    def __init__(
        self,
        id: str,
        boundary: Polygon,
        center: LineString,
        pavements: List[str],
        elevation: Optional[np.ndarray] = None,
    ):
        super().__init__(id, boundary, center, elevation=elevation)
        self.pavements = pavements

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of the objects data."""
        data = super().to_dict()
        data["pavements"] = self.pavements
        return data


class Building(RoadGeometry):
    """
    A geometry describing the area of a building.

    These are modelled as solid blocks that cannot be
    entered by vehicles or pedestrians.
    """

    driveable = False
    impenetrable = True
