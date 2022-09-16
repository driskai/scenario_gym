from scenario_gym.action import Action


class PedestrianAction(Action):
    """A speed and heading angle update for pedestrian agents."""

    def __init__(self, speed: float, heading: float):
        """
        Init the action.

        Parameters
        ----------
        speed : float
            Velocity in m/s.

        heading : float
            Heading angle in rad.

        """
        self.speed = speed
        self.heading = heading
