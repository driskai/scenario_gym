from typing import Dict, List, Optional

from shapely.geometry import Polygon
from shapely.strtree import STRtree

from scenario_gym.entity import Entity


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
        id(g): [
            g_prime
            for g_prime in tree.query(g)
            if ((id(g) != id(g_prime)) and g.intersects(g_prime))
        ]
        for g in geoms
    }


def detect_collisions(
    entities: List[Entity],
    others: Optional[List[Entity]] = None,
) -> Dict[Entity, List[Entity]]:
    """
    Return collisions between entities.

    Checks the bounding boxes of all entities at the current time
    and return any that overlap. Returns a dict of entities in
    the scenario with a list of all other entities that they are
    colliding with.

    Parameters
    ----------
    entities : List[Entity]
        The entities.

    others : Optional[List[Entity]]
        Additional entities to include when checking for collisions
        with each geometry in geoms.

    """
    geom_to_ent = {}
    for e in entities + others if others is not None else entities:
        g = e.get_bounding_box_geom()
        geom_to_ent[id(g)] = e
        geom_to_ent[e] = g

    geoms = [geom_to_ent[e] for e in entities]
    other_geoms = [geom_to_ent[e] for e in others] if others is not None else None

    collisions = detect_geom_collisions(geoms, others=other_geoms)
    return {
        e: [geom_to_ent[id(g_prime)] for g_prime in collisions[id(g)]]
        for e, g in zip(entities, geoms)
    }
