from typing import Dict, List, Optional, Tuple, Union

from shapely.geometry import LinearRing, LineString, Polygon


def load_road_geometry_from_json(
    data: Dict,
) -> Tuple[Optional[Polygon], Optional[LineString]]:
    """Load the boundary and center from json data."""
    if "Boundary" in data:
        boundary = data["Boundary"]
        if isinstance(boundary, list):
            boundary = Polygon([[v["x"], v["y"]] for v in boundary])
        elif isinstance(boundary, dict):
            boundary = Polygon(
                [[v["x"], v["y"]] for v in boundary["exterior"]],
                holes=[
                    LinearRing([[v["x"], v["y"]] for v in i])
                    for i in boundary["interiors"]
                ],
            )
        else:
            raise ValueError(
                f"Type {type(boundary)} is not supported for boundary."
            )
    else:
        boundary = None
    if "Center" in data:
        center = data["Center"]
        if isinstance(center, list):
            center = LineString([[v["x"], v["y"]] for v in center])
        else:
            raise ValueError(f"Type {type(center)} is not supported for center.")
    else:
        center = None
    return boundary, center


def polygon_to_data(
    poly: Polygon,
) -> Union[List[Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    """Return a dict or list representing the polygon data."""
    exterior = [{"x": float(x), "y": float(y)} for x, y in poly.exterior.coords]
    if not poly.interiors:
        return exterior
    return {
        "exterior": exterior,
        "interiors": [
            [{"x": float(x), "y": float(y)} for x, y in i.coords]
            for i in poly.interiors
        ],
    }
