from dataclasses import dataclass

from scenario_gym.action import Action


@dataclass
class PedestrianAction(Action):
    """A speed and heading angle update for pedestrian agents."""

    speed: float
    heading: float
