from typing import Callable, Dict, List, Optional, Union

from shapely.geometry import Point

from scenario_gym.callback import StateCallback
from scenario_gym.entity import Entity
from scenario_gym.scenario import Scenario


class State:
    """
    The global state of the gym.

    Holds the current time, the terminal state and the scenario data.

    Can also be parameterised with different end conditions for the scenario.
    E.g. to end when the recorded scenario ends or if a collision occurs.
    Additional information may be provided through custom methods passed
    as state_callbacks.
    """

    def __init__(
        self,
        conditions: Optional[List[Union[str, Callable[["State"], bool]]]] = None,
        state_callbacks: Optional[List[StateCallback]] = None,
    ):
        """
        Init the state.

        Parameters
        ----------
        dt : float
            The timestep used in the gym.

        conditions : Optional[List[Union[str, Callable[["State"], bool]]]]
            Terminal conditions that will end the scenario if any is met. May be a
            string referencing an entry of the TERMINAL_CONDITIONS dictionary.

        state_callbacks : Optional[List[StateCallback]]
            Methods to be called on the state when the timestep is updated.
            Can be used to add additional information to the state that can is then
            accessible by all agents.

        """
        if conditions is None:
            self.terminal_conditions = [TERMINAL_CONDITIONS["max_length"]]
        else:
            self.terminal_conditions = [
                cond if callable(cond) else TERMINAL_CONDITIONS[cond]
                for cond in conditions
            ]
        self.state_callbacks = [] if state_callbacks is None else state_callbacks
        self._scenario: Optional[Scenario] = None
        self.next_t: Optional[float] = None
        self._t: Optional[float] = None
        self._prev_t: Optional[float] = None
        self.is_done = False

    @property
    def scenario(self) -> Scenario:
        """Get the current scenario."""
        return self._scenario

    @scenario.setter
    def scenario(self, s: Scenario) -> None:
        self._scenario = s
        self._scenario.t = self.t

    @property
    def t(self):
        """Get the time in seconds (s)."""
        return self._t

    @t.setter
    def t(self, t: float) -> None:
        self.prev_t = self._t
        self.scenario.t = t
        self._t = t
        return self._t

    @property
    def prev_t(self) -> float:
        """Get the previous time (s)."""
        return self._prev_t

    @prev_t.setter
    def prev_t(self, prev_t: float) -> None:
        self._prev_t = prev_t

    @property
    def dt(self):
        """Get the previous timestep."""
        return self.t - self.prev_t

    def update_callbacks(self) -> None:
        """Update all state callbacks."""
        for m in self.state_callbacks:
            m(self)

    def check_terminal(self) -> bool:
        """Check if the state is terminal."""
        return any(cond(self) for cond in self.terminal_conditions)

    @property
    def collisions(self) -> Dict[Entity, List[Entity]]:
        """Return collisions between entities at the current time."""
        return self.scenario.collisions()


TERMINAL_CONDITIONS = {
    "max_length": lambda s: s.t > s.scenario.length,
    "collision": lambda s: any(len(l) > 0 for l in s.collisions.values()),
    "ego_collision": lambda s: len(s.collisions[s.scenario.entities[0]]) > 0,
    "ego_off_road": lambda s: not (
        s.scenario.road_network.driveable_surface.contains(
            Point(*s.scenario.agents["ego"].entity.pose[:2])
        )
    ),
}
