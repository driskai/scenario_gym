from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon
from shapely.strtree import STRtree

ArgsKwargs = Tuple[Tuple[Any, ...], Dict[str, Any]]


def detect_geom_collisions(
    geoms: List[Polygon],
    others: Optional[List[Polygon]] = None,
) -> Dict[str, List[Polygon]]:
    """
    Detect collisions between polygons.

    Parameters
    ----------
    geoms : List[Polygon]
        The geometries to use.

    others : Optional[List[Polygon]]
        Additional geometries to include when checking for collisions
        with each geometry in geoms.

    Returns
    -------
    Dict[str, List[Polygon]]
        A dictionary that maps the id of each polygon in geoms to the
        polygons that it intersects.

    """
    all_geoms = geoms if others is None else geoms + others
    tree = STRtree(all_geoms)
    return {
        g: [
            g_prime
            for g_prime in tree.geometries.take(
                tree.query(g, predicate="intersects").tolist()
            )
            if g != g_prime
        ]
        for g in geoms
    }
