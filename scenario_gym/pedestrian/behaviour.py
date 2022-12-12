from abc import abstractmethod
from typing import Tuple

from scenario_gym.agent import Agent
from scenario_gym.pedestrian.observation import PedestrianObservation


class BehaviourParameters:
    """Parameters for the behaviour model."""

    max_speed_factor = 1.3

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PedestrianBehaviour:
    """Base class for pedestrian behavior models."""

    def __init__(self, params: BehaviourParameters):
        self.params = params
        self.max_speed_factor = params.max_speed_factor

    def step(
        self,
        observation: PedestrianObservation,
        agent: Agent,
    ) -> Tuple[float, float]:
        """
        Return the new speed and heading according to behavior model.

        Parameters
        ----------
        observation : PedestrianObservation
            The observation for the pedestrian.

        agent : Agent
            The pedestrian agent.

        Returns
        -------
        speed : float

        heading : float

        """
        return self._step(observation, agent)

    @abstractmethod
    def _step(self, observation: PedestrianObservation, agent: Agent) -> Tuple:
        """Return the new speed and heading according to behavior model."""
        pass
