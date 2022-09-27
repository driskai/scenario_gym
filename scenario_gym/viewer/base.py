from abc import ABC, abstractmethod
from typing import Optional

from scenario_gym.state import State


class Viewer(ABC):
    """
    Base class for the scenario_gym visualization module.

    Subclasses will render a scenario rollout as a video. Each time
    a scenario is run a new instance of the viewer will be created
    which will render the rollout and then be destroyed.
    """

    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path

    @abstractmethod
    def render(self, state: State) -> Optional[int]:
        """
        Render the current state of the environment.

        Should optionally return the current key press.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the viewer."""
        raise NotImplementedError
