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
        pre_callbacks = []
        for cb in state.state_callbacks:
            if cb is self:
                break
            if cb.__class__ in self.required_callbacks:
                pre_callbacks.append(cb)

        missing = set(self.required_callbacks).difference(
            (cb.__class__ for cb in pre_callbacks)
        )
        if missing:
            raise ValueError(
                f"Cannot run callback {self.__class__.__name__} without callbacks"
                f" {missing}."
            )

        self.callbacks.extend(pre_callbacks)
        self._reset(state)

    def _reset(self, state: State) -> None:
        """Reset the callback's parameters."""
        pass

    @abstractmethod
    def __call__(self, state: State) -> None:
        """Update callback with new information."""
        raise NotImplementedError
