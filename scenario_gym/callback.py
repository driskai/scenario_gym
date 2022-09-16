from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Type, TypeVar

State = TypeVar("State")


class StateCallback(ABC):
    """
    A callback to provide additional information to the state.

    Callbacks are updated during each timestep and should modify
    the state inplace via the `__call__` method.
    """

    required_callbacks: List[Type[StateCallback]] = []

    @abstractmethod
    def __call__(self, state: State) -> None:
        """Modify the state inplace with new information."""
        raise NotImplementedError

    def reset(self, state: State) -> None:
        """Reset the callback at the start of a new scenario."""
        position = state.state_callbacks.index(self)
        previous_callbacks_ = state.state_callbacks[:position]

        for CB in self.required_callbacks:
            if not any(isinstance(cb, CB) for cb in previous_callbacks_):
                raise ValueError(
                    "Cannot run callback {} without callback {}.".format(
                        self.__class__.__name__,
                        CB.__name__,
                    )
                )
