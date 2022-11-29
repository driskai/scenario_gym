import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

try:
    from functools import cached_property
except ImportError:

    def cached_property(fn):
        """Replace cached_property with a size 1 lru cache."""
        return property(lru_cache(maxsize=1)(fn))


try:
    from numpy.typing import ArrayLike, NDArray
except ImportError:
    ArrayLike = NDArray = np.ndarray


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
    with warnings.catch_warnings():  # STRTree not included in Shapely 2.0
        warnings.simplefilter("ignore")
        tree = STRtree(all_geoms)
    return {
        id(g): [
            g_prime
            for g_prime in tree.query(g)
            if ((id(g) != id(g_prime)) and g.intersects(g_prime))
        ]
        for g in geoms
    }
