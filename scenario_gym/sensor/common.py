"""Provides a selection of commonly used sensors."""
from copy import deepcopy
from typing import List

import numpy as np

from scenario_gym.entity import Entity
from scenario_gym.observation import Observation, SingleEntityPoseObservation
from scenario_gym.scenario.utils import detect_collisions
from scenario_gym.state import State

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
    """Returns position and orientation of the entity."""

    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        pass

    def _step(self, state: State) -> SingleEntityPoseObservation:
        """Produce an observation from the global state."""
        return SingleEntityPoseObservation(self.entity.pose)


class FutureCollisionDetector(Sensor):
    """
    Detects any future collisions in the scenario.

    Entity trajectories are used to obtain their future position
    and compare it to the sensor's entity.
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

    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        pass

    def _step(self, state: State) -> bool:
        """Produce an observation from the global state."""
        # make copeies of each entity to avoid modifying
        others = {
            a.entity: (deepcopy(a.entity), a)
            for a in state.scenario.agents.values()
        }

        # check for collisions over the horizon
        for t in np.linspace(state.t, state.t + self.horizon, 10):
            for e_ref, a in others.values():
                e_ref.pose = a.trajectory.position_at_t(t)
            other_ents, _ = map(list, zip(*others.values()))
            collisions = detect_collisions(other_ents)
            if len(collisions[others[self.entity][0]]) > 0:
                return True
        return False


class GlobalCollisionDetector(Sensor):
    """Returns collisions observed in the scene."""

    def _step(self, state: State):
        """Produce an observation from the global state."""
        return state.scenario.detect_collisions()


class KeyboardInputDetector(Sensor):
    """Detects keyboard input."""

    def _reset(self) -> None:
        """Reset the sensor at the start of the scenario."""
        pass

    def _step(self, state: State) -> int:
        """Produce an observation from the global state."""
        return state.last_keystroke
