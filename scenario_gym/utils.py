from contextlib import suppress
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from lxml.etree import Element
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
) -> Dict[Polygon, List[Polygon]]:
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
    Dict[Polygon, List[Polygon]]
        A dictionary that maps each polygon in geoms to the polygons which it
        intersects.

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


def load_properties_from_xml(
    element: Element,
) -> Tuple[Dict[str, Union[str, float]], List[str]]:
    """
    Load properties from the xml element.

    These can be either `Property` or `File` elements. `Property` elements are
    given with `name` and `value` attributes (`name` must be unique for the
    entry) and are returned as a dict of values indexed by `name`. The value
    will be converted to a float if it can otherwise the string will be
    returned. `File` elements must have a `filepath` attribute which will
    be parsed. Multiple files can be stored with one entity.

    Returns
    -------
    properties : Dict[str, Union[str, float]]
        A dictionary of properties indexed by name.

    files : List[str]
        A list of filepaths for external files.

    """
    files = []
    properties = {}
    prop = element.find("Properties")
    if prop is not None:
        for child in prop.findall("Property"):
            try:
                v = child.attrib["value"]
                with suppress(ValueError):
                    v = float(v)
                properties[child.attrib["name"]] = v
            except KeyError as e:
                raise RuntimeError(
                    "Property could not be loaded without `value` key."
                ) from e
        for file in prop.findall("File"):
            files.append(file.attrib["filepath"])
    return properties, files
