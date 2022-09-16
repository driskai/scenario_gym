from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from shapely.geometry import LineString, Polygon

from .base import RoadGeometry, RoadLike
from .utils import load_road_geometry_from_json


class LaneType(Enum):
    """Enumerates OpenDrive standard lane types."""

    none = 0
    driving = 1
    biking = 3
    border = 4
    connectingRamp = 5
    curb = 6
    entry = 7
    exit = 8
    median = 9
    offRamp = 10
    onramp = 11
    parking = 12
    restricted = 13
    sidewalk = 14
    shoulder = 15
    stop = 16


class Lane(RoadLike):
    """
    A lane object.

    The lane has a center and a boundary and holds information of its successor
    and predeceessor lanes.
    """

    walkable = False

    @classmethod
    def from_dict(cls, l: Dict[str, Any]):
        """Create from dictionary."""
        typ = l["type"] if "type" in l else "driving"
        lane_type = LaneType[typ if typ in LaneType.__members__ else "driving"]
        return cls(
            l["id"],
            *load_road_geometry_from_json(l),
            l["successors"] if "successors" in l else [],
            l["predecessors"] if "predecessors" in l else [],
            lane_type,
        )

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
        data = super().to_json()
        data.update(
            {
                "center": [
                    {"x": float(x), "y": float(y)} for x, y in self.center.coords
                ],
                "successors": self.successors,
                "predecessors": self.predecessors,
                "type": self.type.name,
            }
        )
        return data


class Road(RoadLike):
    """
    A road object.

    The road has a center and a boundary and lanes.
    """

    walkable = False

    @classmethod
    def from_dict(cls, r: Dict[str, Any]):
        """Create from dictionary."""
        lanes = [Lane.from_dict(l) for l in r["lanes" if "lanes" in r else "Lanes"]]
        return cls(r["id"], *load_road_geometry_from_json(r), lanes)

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
        data = super().to_json()
        data.update(
            {
                "center": [
                    {"x": float(x), "y": float(y)} for x, y in self.center.coords
                ],
                "lanes": [l.to_dict() for l in self.lanes],
            }
        )
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
    def from_dict(cls, i: Dict[str, Any]):
        """Create from dictionary."""
        boundary, _ = load_road_geometry_from_json(i)
        lanes = [Lane.from_dict(l) for l in i["lanes" if "lanes" in i else "Lanes"]]
        return cls(i["id"], boundary, lanes, i["connecting_roads"])

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
        data = super().to_json()
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

    @classmethod
    def from_dict(cls, p: Dict[str, Any]):
        """Create from dictionary."""
        boundary, center = load_road_geometry_from_json(p)
        return cls(p["id"], boundary, center)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of the objects data."""
        data = super().to_json()
        data["center"] = [
            {"x": float(x), "y": float(y)} for x, y in self.center.coords
        ]
        return data


class Crossing(RoadLike):
    """
    A crossing object.

    The crossing has a boundary and center and ids of pavements it connects.
    """

    driveable = False

    @classmethod
    def from_dict(cls, c: Dict[str, Any]):
        """Create from dictionary."""
        boundary, center = load_road_geometry_from_json(c)
        return cls(
            c["id"],
            boundary,
            center,
            c["pavements" if "pavements" in c else "Pavements"],
        )

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
        data = super().to_json()
        data["center"] = [
            {"x": float(x), "y": float(y)} for x, y in self.center.coords
        ]
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
