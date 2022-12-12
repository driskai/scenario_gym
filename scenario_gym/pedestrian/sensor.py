from typing import List

from scenario_gym.entity import Entity, Pedestrian
from scenario_gym.pedestrian.observation import PedestrianObservation
from scenario_gym.sensor import Sensor
from scenario_gym.state import State


class PedestrianSensor(Sensor):
    """Returns observation (complete state) for pedestrian entities."""

    def __init__(
        self,
        entity: Entity,
        head_rot_angle: float = 0.0,
        distance_threshold: float = 1.0,
    ):
        """
        Init the pedestrian sensor.

        Parameters
        ----------
        entity : Entity
            The pedestrian entity.

        head_rot_angle : float
            Rotation angle of pedestrian head in rad wrt to heading
            (0 means looking forward).

        distance_threshold: float
            Only pedestrians within this distance of the entity will be considered.

        """
        super().__init__(entity)
        self.head_rot_angle = head_rot_angle
        self.distance_threshold = distance_threshold

    def _reset(self, state: State) -> PedestrianObservation:
        """Reset the sensor."""
        return self._step(state)

    def _step(self, state: State) -> PedestrianObservation:
        """Produce the pedestrian observation."""
        near_peds = self.get_nearby_pedestrians(state)
        return PedestrianObservation(
            self.entity,
            *state.get_entity_data(self.entity),
            self.head_rot_angle,
            near_peds,
            state.scenario.road_network.walkable_surface,
            state.scenario.road_network.impenetrable_surface,
        )

    def get_nearby_pedestrians(self, state: State) -> List[Entity]:
        """Get other pedestrians within a radius of the entity."""
        return [
            (e, state.poses[e], state.velocities[e])
            for e in state.get_entities_in_radius(
                *state.poses[self.entity][:2],
                self.distance_threshold,
            )
            if (isinstance(e, Pedestrian) or (e.type == "Pedestrian"))
            and (e != self.entity)
        ]
