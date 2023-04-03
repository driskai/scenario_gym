from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Type, TypeVar

State = TypeVar("State")


class StateCallback(ABC):
    """
    A callback to provide additional information to the state.

    Callbacks are updated during each timestep and should modify their own internal
    state in the `__call__` method.
    """

    required_callbacks: List[Type[StateCallback]] = []

    def __init__(self):
        self.callbacks: List[StateCallback] = []

    def reset(self, state: State) -> None:
        """Reset the callback check dependents are there."""
        self.callbacks.clear()
        for req in self.required_callbacks:
            cb = state.get_callback(req)
            if cb is None:
                raise ValueError(
                    f"Callback {req.__name__} is required for {self.__class__}."
                )
            self.callbacks.append(cb)
        self._reset(state)

    def _reset(self, state: State) -> None:
        """Reset the callback's parameters."""
        pass

    @abstractmethod
    def __call__(self, state: State) -> None:
        """Update callback with new information."""
        raise NotImplementedError
