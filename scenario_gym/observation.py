from abc import ABC
from inspect import getfullargspec

import numpy as np

from scenario_gym.entity import Entity
from scenario_gym.state import State


class BaseObservation(ABC):
    """The observation for a given agent at a given time."""

    pass


class Observation(BaseObservation):
    """
    Standard observation for entities that all other observations should inherit.

    All observations contain the entity's information and then may add additional
    information that a sensor may provide. These are implemented by providing
    annotations for the data to be available and then fixing these values during
    the sensor's `_step` method.

    For example a map sensor would add a map attribute as:

    ```
    Class MapObservation(Observation):

        map: np.ndarray
    ```

    This would then be updated in a sensor as:

    ```
    Class MapSensor(Sensor):

        ...

        def _step(self, observation):
            rasterized_map = ...
            observation.map = rasterized_map
            return observation
    ```
    """

    t: float
    next_t: float
    pose: np.ndarray
    velocity: np.ndarray
    distance_travelled: float
    recorded_poses: np.ndarray

    def __init__(
        self,
        entity: Entity,
        t: float,
        next_t: float,
        pose: np.ndarray,
        velocity: np.ndarray,
        distance_travelled: float,
        recorded_poses: np.ndarray,
    ):
        self.entity = entity
        self.t = t
        self.next_t = next_t
        self.pose = pose
        self.velocity = velocity
        self.distance_travelled = distance_travelled
        self.recorded_poses = recorded_poses

    def __init_subclass__(cls) -> None:
        subclass_spec = getfullargspec(cls.__init__).args
        base_spec = getfullargspec(Observation.__init__).args
        if len(subclass_spec) != len(base_spec):
            raise RuntimeError(
                "Subclassed observation must take the same positional arguments as "
                "the base `Observation`. If overwriting __init__ do not add any "
                "additional arguments."
            )

    @classmethod
    def from_entity(cls, state: State, entity: Entity):
        """Create the observation from a base entity."""
        return cls(
            entity,
            state.t,
            state.next_t,
            state.poses[entity],
            state.velocities[entity],
            state.distances[entity],
            state.recorded_poses(entity),
        )
