"""Provides a selection of commonly used sensors."""
from copy import deepcopy
from typing import Dict, List

import numpy as np

from scenario_gym.entity import Entity
from scenario_gym.observation import Observation
from scenario_gym.state import State, detect_collisions

from .base import Sensor


class CombinedSensor(Sensor):
    """Combines different observations into one."""

    def __init__(self, entity: Entity, *sensors: Sensor):
        """Init the sensor."""
        assert [s.entity == entity for s in sensors]
        self.sensors = sensors

    def _reset(self) -> None:
        """Reset all sensors."""
        for s in self.sensors:
            s.reset()

    def _step(self, state) -> List[Observation]:
        """Get observations from all sensors."""
        return [s.step(state) for s in self.sensors]


class EgoLocalizationSensor(Sensor):
    """Observation containing just the base entity information."""

    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        pass

    def _step(self, state: State, obs: Observation) -> Observation:
        """Return the entity observation."""
        return obs


class FutureCollisionObservation(Observation):
    """Contains a bool variable indicating if a future collision is predicted."""

    future_collision: bool


class FutureCollisionDetector(Sensor):
    """
    Detects any future collisions in the scenario.

    Entity trajectories are used to obtain their future position
    and compare t to the sensor's entity.
    """

    observation_type = FutureCollisionObservation

    def __init__(self, entity: Entity, horizon: float = 5.0):
        """
        Init the sensor.

        Parameters
        ----------
        entity : Entity
            The entity.

        horizon : float
            The time horizon over which to look for collisions.

        """
        super().__init__(entity)
        self.horizon = horizon

    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        pass

    def _step(self, state: State, obs: FutureCollisionObservation) -> bool:
        """Produce an observation from the global state."""
        # make copeies of each entity to avoid modifying
        others = {
            a.entity: (deepcopy(a.entity), a)
            for a in state.scenario.agents.values()
        }

        # check for collisions over the horizon
        obs.future_collision = False
        for t in np.linspace(state.t, state.t + self.horizon, 10):
            for e, (e_ref, a) in others.items():
                e_ref.pose = a.trajectory.position_at_t(t)
            other_ents, _ = map(list, zip(*others.values()))
            collisions = detect_collisions(other_ents)
            if len(collisions[others[self.entity][0]]) > 0:
                obs.future_collision = True
        return obs


class CollisionObservation(Observation):
    """Observation with currently occuring collisions."""

    collisions: Dict[Entity, List[Entity]]


class GlobalCollisionDetector(Sensor):
    """Returns collisions observed in the scene."""

    observation_type = CollisionObservation

    def _step(
        self, state: State, obs: CollisionObservation
    ) -> CollisionObservation:
        """Produce an observation from the global state."""
        obs.collisions = state.collisions()
        return obs


class KeyboardObservation(Observation):
    """Observation with current key pressed by user."""

    last_keystroke: int


class KeyboardInputDetector(Sensor):
    """Detects keyboard input."""

    observation_type = KeyboardObservation

    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        pass

    def _step(self, state: State, obs: KeyboardObservation) -> int:
        """Produce an observation from the global state."""
        obs.last_keystroke = state.last_keystroke
        return obs
