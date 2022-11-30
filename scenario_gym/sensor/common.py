"""Provides a selection of commonly used sensors."""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from scenario_gym.entity import Entity
from scenario_gym.observation import Observation, SingleEntityObservation
from scenario_gym.state import State, detect_collisions

from .base import Sensor


class CombinedSensor(Sensor):
    """Combines different observations into one."""

    def __init__(self, entity: Entity, *sensors: Sensor):
        """Init the sensor."""
        assert [s.entity == entity for s in sensors]
        self.sensors = sensors

    def _reset(self, state: State) -> List[Observation]:
        """Reset all sensors."""
        return [s.reset(state) for s in self.sensors]

    def _step(self, state: State) -> List[Observation]:
        """Get observations from all sensors."""
        return [s.step(state) for s in self.sensors]


class EgoLocalizationSensor(Sensor):
    """Observation containing just the base entity information."""

    def _step(self, state: State) -> SingleEntityObservation:
        """Return the entity observation."""
        return SingleEntityObservation(
            self.entity, *state.get_entity_data(self.entity)
        )


@dataclass
class FutureCollisionObservation(SingleEntityObservation):
    """Observation with future collision information."""

    future_collision: bool


class FutureCollisionDetector(Sensor):
    """
    Detects any future collisions with the ego in the scenario.

    Entity trajectories are used to obtain their future position
    and compare t to the sensor's entity.
    """

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

    def _step(self, state: State) -> FutureCollisionObservation:
        """Produce an observation from the global state."""
        ents = {e: None for e in state.scenario.entities if e != self.entity}

        # check for collisions over the horizon
        future_collision = False
        for t in np.linspace(state.t, state.t + self.horizon, 10):
            ego_pose = self.entity.trajectory.position_at_t(t)
            for e in ents:
                ents[e] = e.trajectory.position_at_t(t)
            collisions = detect_collisions({self.entity: ego_pose}, ents)
            if len(collisions[self.entity]) > 0:
                future_collision = True
        return FutureCollisionObservation(
            self.entity,
            *state.get_entity_data(self.entity),
            future_collision,
        )


@dataclass
class CollisionObservation(SingleEntityObservation):
    """Observation with detected collisions."""

    collisions: Dict[Entity, List[Entity]]


class GlobalCollisionDetector(Sensor):
    """Returns collisions observed in the scene."""

    def _step(self, state: State) -> CollisionObservation:
        """Produce an observation from the global state."""
        return CollisionObservation(
            self.entity,
            *state.get_entity_data(self.entity),
            state.collisions(),
        )


@dataclass
class KeyboardObservation(SingleEntityObservation):
    """Observation with detected collisions."""

    last_keystroke: Dict[Entity, List[Entity]]


class KeyboardInputDetector(Sensor):
    """Detects keyboard input."""

    def _step(self, state: State) -> int:
        """Produce an observation from the global state."""
        return KeyboardObservation(
            self.entity,
            *state.get_entity_data(self.entity),
            state.last_keystroke,
        )
