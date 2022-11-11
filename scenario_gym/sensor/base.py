from abc import ABC, abstractmethod
from typing import Optional, Type

from scenario_gym.entity import Entity
from scenario_gym.observation import Observation
from scenario_gym.state import State


class Sensor(ABC):
    """
    Produce an observation for a given entity from the global state.

    All sensors should implement the `_reset` and `_step` method to produce
    observations which should contain all information that an agent may use to
    select actions. This includes the agent's entity's position, velocity and
    distance travelled as well as the simulation time. These bits of base
    information are provided as input to the `_step` method via the `Observation`
    base class. The `_step` method may then add data to this class and return it
    or may create a new data structure to return.

    The `_reset` method takes no parameters but has access to the
    """

    observation_type: Type = Observation

    def __init__(self, entity: Entity):
        """Init the sensor."""
        self.entity = entity
        self.initial_observation: Optional[Observation] = None
        self._last_observation: Optional[self.observation_type] = None

    def reset(self, state: State) -> None:
        """Reset the sensor at the start of the scenario."""
        self._last_observation = None
        self.initial_observation = Observation.from_entity(state, self.entity)
        self._reset()

    def step(self, state: State) -> observation_type:
        """Produce the observation from the global state."""
        if issubclass(self.observation_type, Observation):
            obs = self.observation_type.from_entity(state, self.entity)
        else:
            obs = Observation.from_entity(state, self.entity)
        self.last_observation = self._step(state, obs)
        return self.last_observation

    @abstractmethod
    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        raise NotImplementedError

    @abstractmethod
    def _step(self, state: State, obs: Observation) -> observation_type:
        """Produce an observation from the global state."""
        raise NotImplementedError

    @property
    def last_observation(self) -> Optional[observation_type]:
        """Get the previous observation."""
        return self._last_observation

    @last_observation.setter
    def last_observation(self, obs: observation_type) -> None:
        self._last_observation = obs
