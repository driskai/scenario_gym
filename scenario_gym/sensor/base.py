from abc import ABC, abstractmethod
from typing import Optional

from scenario_gym.entity import Entity
from scenario_gym.observation import Observation
from scenario_gym.state import State


class Sensor(ABC):
    """
    Produce an observation for a given entity from the global state.

    All sensors should implement the `_reset` and `_step` method to produce
    observations which should contain all information that an agent may use to
    select actions. The `_reset` method should produce an initial observation for
    the agent. The `_step` method should produce the observation to be used at each
    timestep.
    """

    def __init__(self, entity: Entity):
        """Init the sensor."""
        self.entity = entity
        self.initial_observation: Optional[Observation] = None
        self._last_observation: Optional[Observation] = None

    def reset(self, state: State) -> Observation:
        """Reset the sensor at the start of the scenario."""
        self._last_observation = None
        self.initial_observation = self._reset(state)
        return self.initial_observation

    def step(self, state: State) -> Observation:
        """Produce the observation from the global state."""
        self.last_observation = self._step(state)
        return self.last_observation

    @abstractmethod
    def _reset(self, state: State) -> Observation:
        """Reset the sensor and return an initial observation."""
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
    def last_observation(self, obs: Optional[Observation]) -> None:
        self._last_observation = obs
