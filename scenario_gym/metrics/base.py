from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from scenario_gym.callback import StateCallback
from scenario_gym.state import State


class Metric(ABC):
    """
    Base class for a metric in scenario_gym.

    All metrics implement reset and step methods to update internal states during
    scenario rollout and the get_state method to return the metric value.

    The `required_callbacks` attribute can be set to a list of StateCallback
    subclasses. At reset the state will be checked to make sure each is present and
    the instance of each will be stored in `self.callbacks` in the same order as
    required callbacks.
    """

    name: Optional[str] = None
    required_callbacks: List[Type[StateCallback]] = []

    def __init__(self, name: Optional[str] = None):
        """
        Construct metric and set the name.

        Parameters
        ----------
        name : Optional[str]
            If not given then the name attribute will be used or
            the class name if that is None also.

        """
        if name is not None:
            self.name = name
        elif self.name is None:
            self.name = self.__class__.__name__
        self.callbacks: List[StateCallback] = []

    def reset(self, state: State) -> None:
        """Reset the metric at the start of a new scenario."""
        self.callbacks.clear()
        for CB in self.required_callbacks:
            cb = state.get_callback(CB)
            if cb is None:
                raise ValueError(
                    "Cannot run metric {} without callback {}.".format(
                        self.__class__.__name__,
                        CB.__name__,
                    )
                )
            self.callbacks.append(cb)
        self._reset(state)

    def step(self, state: State) -> None:
        """Update the metric after one timestep."""
        self._step(state)

    @abstractmethod
    def _reset(self, state: State) -> None:
        """Reset the metric at the start of a new scenario."""
        raise NotImplementedError

    @abstractmethod
    def _step(self, state: State) -> None:
        """Update the metric after one timestep."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Any:
        """Return the current value of the metric."""
        raise NotImplementedError


def cache_metric(Met: Type[Metric]) -> Type[Metric]:
    """Wrap _step to cache the value whenver the state is terminal."""
    prev_step = Met._step
    Met.previous_value = None

    def new_step(self, state):
        prev_step(self, state)
        if state.is_done:
            self.previous_value = self.get_state()

    Met._step = new_step
    return Met


def cache_mean(Met: Type[Metric]) -> Type[Metric]:
    """Wrap _step to keep a running mean of the value."""

    def previous_value(self):
        val = self._previous_value
        self._previous_value = 0.0
        self._prev_count = 0
        return val

    prev_step = Met._step
    Met._previous_value = 0.0
    Met._prev_count = 0
    Met.previous_value = property(previous_value)

    def new_step(self, state):
        prev_step(self, state)
        if state.is_done:
            self._prev_count += 1
            self._previous_value += (
                self.get_state() - self._previous_value
            ) / self._prev_count

    Met._step = new_step
    return Met
