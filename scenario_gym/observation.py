from abc import ABC

import numpy as np


class Observation(ABC):
    """The observation for a given entity at a given time."""

    pass


class SingleEntityPoseObservation(Observation):
    """An observation which holds the current position of an entity."""

    def __init__(self, position: np.ndarray):
        """Construct the position observation."""
        self.position = position
