from scenario_gym.entity import Entity
from scenario_gym.observation import Observation
from scenario_gym.state import State


class PedestrianObservation(Observation):
    """
    An observation for a pedestrian agent.

    Contains the objects and road elements detected within a given radius.

    Todo: Instead of returning complete state, return observation with only entities
    and pedestrian road info within a given radius by adapting RasterizedMapSensor
    or ObjectDetectionSensor.
    """

    def __init__(self, entity: Entity, head_rot_angle: float, state: State):
        """
        Init the pedestrian observation.

        Parameters
        ----------
        entity : Entity
            The pedestrian entity.

        head_rot_angle : float
            Rotation angle of pedestrian head in rad w.r.t. to heading
            (0 means looking forward).

        state : State
            Global state.

        """
        super().__init__()
        self.head_rot_angle = head_rot_angle
        self.state = state
