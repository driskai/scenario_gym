from dataclasses import dataclass
from typing import Any

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
