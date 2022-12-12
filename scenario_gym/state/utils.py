from itertools import chain
from typing import Dict, List, Optional

import numpy as np

from scenario_gym.entity import Entity
from scenario_gym.utils import detect_geom_collisions


def detect_collisions(
    entities: Dict[Entity, np.ndarray],
    others: Optional[Dict[Entity, np.ndarray]] = None,
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
    for e, pose in (
        entities.items()
        if others is None
        else chain(entities.items(), others.items())
    ):
        g = e.get_bounding_box_geom(pose)
        geom_to_ent[g] = e
        geom_to_ent[e] = g

    geoms = [geom_to_ent[e] for e in entities]
    other_geoms = [geom_to_ent[e] for e in others] if others is not None else None

    collisions = detect_geom_collisions(geoms, others=other_geoms)
    return {
        e: [geom_to_ent[g_prime] for g_prime in collisions[g]]
        for e, g in zip(entities, geoms)
    }
