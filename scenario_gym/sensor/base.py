from abc import ABC, abstractmethod
from typing import Optional

from scenario_gym.entity import Entity
from scenario_gym.observation import Observation
from scenario_gym.state import State


class Sensor(ABC):
    """Produce an observation for a given entity from the global state."""

    def __init__(self, entity: Entity):
        """Init the sensor."""
        self.entity = entity

    def reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        self._last_observation: Optional[Observation] = None
        self._reset()

    def step(self, state: State) -> Observation:
        """Produce an observation from the global state."""
        self.last_observation = self._step(state)
        return self.last_observation

    @abstractmethod
    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        raise NotImplementedError

    @abstractmethod
    def _step(self, state: State) -> Observation:
        """Produce an observation from the global state."""
        raise NotImplementedError

    @property
    def last_observation(self) -> Optional[Observation]:
        """Get the previous observation."""
        return self._last_observation

    @last_observation.setter
    def last_observation(self, obs: Observation) -> None:
        self._last_observation = obs
