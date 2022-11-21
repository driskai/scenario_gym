from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

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
        t: float,
        action_class: str,
        entity_ref: str,
        action_variables: Dict[str, Any],
    ):
        """
        Create the action.

        Parameters
        ----------
        t : float
            The time at which the action should be applied.

        action_class : str
            The name of the action class.

        entity_ref : str
            Reference of the entity to which the action applies.

        action_variables :  Dict[str, Any]
            Dictionary of action variables.

        """
        self.t = t
        self.action_class = action_class
        self.entity_ref = entity_ref
        self.action_variables = action_variables

    def apply(self, state: State, entity: Optional[Entity]) -> None:
        """Apply the action to the environment state."""
        self._apply(state, entity)
        self._applied = True

    @abstractmethod
    def _apply(self, state: State, entity: Optional[Entity]) -> None:
        """Apply the action to the environment state."""
        raise NotImplementedError


class UpdateStateVariableAction(ScenarioAction):
    """Action that sets state variables for the entity."""

    def _apply(self, state: State, entity: Optional[Entity]) -> None:
        """Update the entity with action variables."""
        if entity is not None:
            for k, v in self.action_variables.items():
                state.entity_state[entity] = k
