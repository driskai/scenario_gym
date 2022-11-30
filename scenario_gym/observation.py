from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type

import numpy as np

from scenario_gym.entity import Entity


@dataclass
class Observation:
    """Base class for an observation."""

    pass


@dataclass
class SingleEntityObservation(Observation):
    """Data for a single entity."""

    entity: Entity
    t: float
    next_t: float
    pose: np.ndarray
    velocity: np.ndarray
    distance_travelled: float
    recorded_poses: np.ndarray
    entity_state: Any


def combine_observations(
    *obs: Tuple[Type[Observation], ...],
    prefixes: Optional[Tuple[Optional[str], ...]] = None,
) -> Type[Observation]:
    """
    Create a class to combine multiple observations.

    The class will inherit the type annotations from the input classes and create
    a dataclass from them. If duplicate field names are found the `prefxies`
    argument can be used to create unique names by specifying a prexfix used
    for arguments from each observation. The created class will accept inputs
    from

    """
    if prefixes is not None and len(prefixes) != len(obs):
        raise ValueError

    annots = OrderedDict()
    maps = OrderedDict()
    for idx, ob in enumerate(obs):
        try:
            fields = ob.__dataclass_fields__
        except AttributeError as e:
            raise f"Obsevation {ob} not a dataclass." from e
        for f in fields.values():
            name = f.name
            if name in annots:
                if prefixes is None:
                    continue
                else:
                    pre = prefixes[idx]
                    name = f"{pre}_{name}"
                    if name in annots:
                        raise ValueError(
                            f"Prefix {pre} still leads to duplicate name for "
                            f"{name}."
                        )
            annots[name] = f.type
            maps[name] = (idx, name)

    @classmethod
    def from_obs(cls, *obs):
        """Create the class from observation instances."""
        args = []
        for (i, name) in maps.values():
            val = getattr(obs[i], name)
            args.append(val)
        return cls(*args)

    return dataclass(
        type(
            "CombinedObservation",
            (Observation,),
            {
                "__annotations__": annots,
                "from_obs": from_obs,
            },
        )
    )
