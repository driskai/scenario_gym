from typing import Any, Dict, Optional

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

from scenario_gym.utils import ArgsKwargs

from .utils import load_road_geometry_from_json, polygon_to_data


class RoadObject:
    """
    Base class for an object in the road network.

    All objects have an id attribute and implement __eq__ and
    __hash__ methods.
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary."""
        args, kwargs = cls.load_data_from_dict(data)
        return cls(*args, **kwargs)

    @classmethod
    def load_data_from_dict(cls, data: Dict[str, Any]) -> ArgsKwargs:
        """Load raw data from dictionary."""
        return (data["Id" if "Id" in data else "id"],), {}

    def __init__(self, id: str):
        self.id = id

    def __eq__(self, other: Any) -> bool:
        """Check if another road object is the same as the current object."""
        if isinstance(other, str):
            return self.id == other
        return hasattr(other, "id") and (other.id == self.id)

    def __hash__(self) -> int:
        """Return a hash of the id."""
        return hash(self.id)

    def __repr__(self) -> str:
        """Return a repr string with the object type and its id."""
        return f"{self.__class__.__name__}(id={self.id})"

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary with id."""
        return {"id": self.id}


class RoadGeometry(RoadObject):
    """
    A geometric object in the road.

    These objects have a boundary given by a shapely polygon.

    The driveable variable indicates if vehicles should use
    the geometry. Similarly the walkable surface variable affects
    whether the geometry is included in the walkable surface. The
    impenetrable variable indicates if a geometry may not be entered
    by any entity. An instance or subclass my overwrite these variables.
    """

    driveable = True
    walkable = True
    impenetrable = False

    @classmethod
    def load_data_from_dict(cls, data: Dict[str, Any]) -> ArgsKwargs:
        """Load raw data from dictionary."""
        (obj_id,), _ = super().load_data_from_dict(data)
        boundary, _ = load_road_geometry_from_json(data)
        if "Elevation" in data and data["Elevation"] is not None:
            elevation = np.array(data["Elevation"])
        else:
            elevation = None
        return (obj_id, boundary), {"elevation": elevation}

    def __init__(
        self, id: str, boundary: Polygon, elevation: Optional[np.ndarray] = None
    ):
        super().__init__(id)

        if not boundary.is_valid:
            boundary = make_valid(boundary)
        self.boundary = boundary

        if elevation is not None:
            assert (
                elevation.ndim == 2 and elevation.shape[1] == 3
            ), "Invalid shape for elevation profile."
        self.elevation = elevation

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary with id and boundary."""
        data = super().to_dict()
        data["Boundary"] = polygon_to_data(self.boundary)
        data["Elevation"] = (
            self.elevation.tolist() if self.elevation is not None else None
        )
        return data


class RoadLike(RoadGeometry):
    """
    A geometry with a center line.

    Used for roads, lanes, pavements, etc.
    """

    @classmethod
    def load_data_from_dict(cls, data: Dict[str, Any]) -> ArgsKwargs:
        """Load raw data from dictionary."""
        boundary, center = load_road_geometry_from_json(data)
        if "Elevation" in data and data["Elevation"] is not None:
            elevation = np.array(data["Elevation"])
        else:
            elevation = None
        return (data["Id" if "Id" in data else "id"], boundary, center), {
            "elevation": elevation
        }

    def __init__(
        self,
        id: str,
        boundary: Polygon,
        center: LineString,
        elevation: Optional[np.ndarray] = None,
    ):
        super().__init__(id, boundary, elevation=elevation)
        self.center = center

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary with id, boundary and center."""
        data = super().to_dict()
        data["Center"] = [
            {"x": float(x), "y": float(y)} for x, y in self.center.coords
        ]
        return data
