from scenario_gym.entity import Entity
from scenario_gym.pedestrian.observation import PedestrianObservation
from scenario_gym.sensor import Sensor
from scenario_gym.state import State


class PedestrianSensor(Sensor):
    """Returns observation (complete state) for pedestrian entities."""

    def __init__(self, entity: Entity, head_rot_angle: float = 0.0):
        """
        Init the pedestrian sensor.

        Parameters
        ----------
        entity : Entity
            The pedestrian entity.

        head_rot_angle : float
            Rotation angle of pedestrian head in rad wrt to heading
            (0 means looking forward).

        """
        super().__init__(entity)
        self.head_rot_angle = head_rot_angle

    def _reset(self) -> None:
        """Reset the sensor."""
        pass

    def _step(self, state: State) -> PedestrianObservation:
        """Produce the pedestrian observation."""
        return PedestrianObservation(self.entity, self.head_rot_angle, state)
