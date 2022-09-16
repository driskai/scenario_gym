import os
from typing import Optional

from scenario_gym.scenario import Scenario
from scenario_gym.xosc_interface import write_scenario


class ScenarioRecorder:
    """Record scenarios and write to OpenScenario."""

    def __init__(self, scenario: Optional[Scenario] = None):
        self.scenario = scenario

    @property
    def scenario(self) -> Optional[Scenario]:
        """Return the scenario being recorded."""
        return self._scenario

    @scenario.setter
    def scenario(self, s: Optional[Scenario]) -> None:
        """Set the scenario and turn on entity recording."""
        self._scenario = s
        if self._scenario is not None:
            for e in self._scenario.entities:
                e.record_trajectory = True

    def get_state(self) -> None:
        """Write the recorded scenario to an xosc file."""
        self.write_xml()

    def write_xml(self, out_path: Optional[str] = None, **kwargs) -> None:
        """
        Write the recorded scenario to an xosc file.

        Parameters
        ----------
        out_path : Optional[str]
            The filepath for the xosc file. By default will look for a
            recordings folder to use or will use the scenario filepath.

        kwargs:
            Keyword arguments for write_scenario.

        """
        if out_path is None:
            path = self.scenario.scenario_path
            s_dir = os.path.dirname(path)
            if os.path.exists(os.path.join(s_dir, "../Recordings")):
                out_path = os.path.join(
                    s_dir,
                    "../Recordings",
                    path.split("/")[-1],
                )
            else:
                out_path = self.scenario.scenario_file
        write_scenario(self.scenario, out_path, **kwargs)

    def close(self) -> None:
        """Close the recorder by turning off entity recording."""
        if self._scenario is not None:
            for e in self._scenario.entities:
                e.record_trajectory = False
