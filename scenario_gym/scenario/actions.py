from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, TypeVar

import numpy as np

from scenario_gym.entity import Entity

State = TypeVar("State")


class ScenarioAction(ABC):
    """
    Base class for scenario actions.

    Actions are applied at the first timestamp with time greater than or equal to
    the action time. They are applied with the _apply method which must be
    implemented.

    """

    def __init__(
        self,
        action_class: str,
        entity_ref: str,
        action_variables: Dict[str, Any],
    ):
        """
        Create the action.

        action_class : str
            The name of the action class.

        entity_ref : str
            Reference of the entity to which the action applies.

        action_variables :  Dict[str, Any]
            Dictionary of action variables.

        """
        self.action_class = action_class
        self.entity_ref = entity_ref
        self.action_variables = action_variables

    def apply(self, state: State, entity: Optional[Entity]) -> None:
        """Apply the action to the environment state."""
        self._apply(state, entity)

    @abstractmethod
    def _apply(self, state: State, entity: Optional[Entity]) -> None:
        """Apply the action to the environment state."""
        raise NotImplementedError

    @abstractmethod
    def trigger_condition(self, state: State) -> bool:
        """Condition for when to apply the action."""
        raise NotImplementedError

    def copy(self):
        """Return a copy of the action."""
        return deepcopy(self)

    def translate(self, x: np.ndarray, inplace: bool = False):
        """Translate the action."""
        return self.copy() if not inplace else self

    def to_dict(self) -> Dict[str, Any]:
        """Write the action to a dictionary."""
        return {
            "action_class": self.action_class,
            "entity_ref": self.entity_ref,
            "action_variables": self.action_variables,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the action from a dictionary."""
        return cls(
            data["action_class"],
            data["entity_ref"],
            data["action_variables"],
        )


class FixedTAction(ScenarioAction):
    """Action that is applied at a fixed time."""

    def __init__(self, t: float, *args, **kwargs):
        """
        Create the action.

        Parameters
        ----------
        t : float
            The time at which the action should be applied.

        """
        super().__init__(*args, **kwargs)
        self.t = t

    def trigger_condition(self, state: State) -> bool:
        """Update when the state time is greater than action time."""
        return state.t >= self.t

    def translate(self, x: np.ndarray, inplace: bool = False):
        """Translate the action."""
        act = self.copy() if not inplace else self
        act.t += x[0]
        return act

    def to_dict(self) -> Dict[str, Any]:
        """Write the action to a dictionary."""
        data = super().to_dict()
        data["t"] = self.t
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the action from a dictionary."""
        return cls(
            data["t"],
            data["action_class"],
            data["entity_ref"],
            data["action_variables"],
        )


class UserDefinedAction(FixedTAction):
    """Custom action provided by the user."""

    def _apply(self, state: State, entity: Optional[Entity]) -> None:
        """Apply the user-defined action."""
        pass


class UpdateStateVariableAction(FixedTAction):
    """Action that sets state variables for the entity."""

    def _apply(self, state: State, entity: Optional[Entity]) -> None:
        """Update the entity with action variables."""
        if entity is not None:
            if state.entity_state[entity] is None:
                state.entity_state[entity] = {}
            for k, v in self.action_variables.items():
                state.entity_state[entity][k] = v

    def trigger_condition(self, state: State) -> bool:
        """Update when the state time is greater than action time."""
        return state.t > self.t

    def to_dict(self) -> Dict[str, Any]:
        """Write the action to a dictionary."""
        return {
            "t": self.t,
            "action_class": self.action_class,
            "entity_ref": self.entity_ref,
            "action_variables": self.action_variables,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the action from a dictionary."""
        return cls(
            data["t"],
            data["action_class"],
            data["entity_ref"],
            data["action_variables"],
        )
