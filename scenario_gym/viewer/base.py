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

    def __init__(self):
        self.output_path: Optional[str] = None

    @abstractmethod
    def reset(self, output_path: str) -> None:
        """
        Reset the viewer at the start of a new scenario rollout.

        Parameters
        ----------
        output_path : str
            Path for the output video file.

        """
        self.output_path = output_path

    @abstractmethod
    def render(self, state: State) -> Optional[int]:
        """
        Render the current state of the environment.

        Should optionally return the current key press.

        Parameters
        ----------
        state : State
            Current global state of the gym.

        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the viewer."""
        raise NotImplementedError
