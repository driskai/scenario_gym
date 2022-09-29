import os
from typing import Any, Callable, Dict, List, Optional, Type, Union

from scenario_gym.agent import Agent, _create_agent
from scenario_gym.entity import Entity
from scenario_gym.metrics import Metric
from scenario_gym.recorder import ScenarioRecorder
from scenario_gym.scenario import Scenario
from scenario_gym.state import State
from scenario_gym.viewer import Viewer
from scenario_gym.viewer.opencv import OpenCVViewer
from scenario_gym.xosc_interface import import_scenario


class ScenarioGym:
    """The main class that loads and runs scenarios."""

    def __init__(
        self,
        timestep: float = 1.0 / 30.0,
        viewer_class: Type[Viewer] = OpenCVViewer,
        terminal_conditions: Optional[
            List[Union[str, Callable[[State], bool]]]
        ] = None,
        state_callbacks: Optional[List[Callable[[State], None]]] = None,
        metrics: Optional[List[Metric]] = None,
        **viewer_parameters,
    ):
        """
        Init the gym.

        All arguments for the constructor of the viewer class should be
        passed as keyword arguments which will be stored.

        Parameters
        ----------
        timestep: float
            Time between steps in the gym.

        viewer_class: Type[Viewer]
            Class type of the viewer that will be inisitalised.

        terminal_conditions : Optional[List[Union[str, Callable[[State], bool]]]]
            Conditions that if any are met will end the scenario.

        state_callbacks: Optional[List[Callable[[State], None]]]
            Additional methods to be called on the state after every step.

        metrics: List[Metric]
            List of metrics to measure.

        viewer_parameters:
            Keyword arguments for viewer_class.

        """
        self.timestep = timestep
        if viewer_class is OpenCVViewer and "fps" not in viewer_parameters:
            viewer_parameters["fps"] = int(1.0 / self.timestep)
        self.viewer_parameters = viewer_parameters.copy()

        if terminal_conditions is None:
            terminal_conditions = ["max_length"]
        self.terminal_conditions = terminal_conditions

        if state_callbacks is None:
            state_callbacks = []
        self.state_callbacks = state_callbacks

        self.viewer_class: Type[Viewer] = viewer_class
        self.viewer: Optional[Viewer] = None
        self.recorder: Optional[ScenarioRecorder] = None
        self.reset_gym()

        if metrics is not None:
            self.add_metrics(metrics)

    def reset_gym(self) -> None:
        """
        Reset the state of the gym.

        Closes the viewer, removes any metrics and unloads the scenario.
        """
        self.close_viewer()
        self.state = State(
            conditions=self.terminal_conditions,
            state_callbacks=self.state_callbacks,
        )
        self.recorder = None
        self.metrics = []

    def add_metrics(self, metrics: List[Metric]) -> None:
        """Add metrics to the gym."""
        self.metrics.extend(metrics)

    def load_scenario(
        self,
        scenario_path: str,
        create_agent: Callable[[Scenario, Entity], Optional[Agent]] = _create_agent,
        relabel: bool = False,
    ) -> None:
        """
        Load a scenario from a file.

        Parameters
        ----------
        scenario_path : str
            The OpenSCENARIO file to be loaded.

        create_agent : Callable[[str, Entity], Optional[Agent]]
            A function that returns an agent to control a given entity.

        relabel : bool
            If given, all entities will be relabeled to ego, vehicle_1,
            vehicle_2, ..., pedestrian_1, ..., entity_1, ...

        """
        scenario = import_scenario(
            scenario_path,
            relabel=relabel,
        )
        self._set_scenario(scenario, create_agent=create_agent)

    def _set_scenario(
        self,
        scenario: Scenario,
        create_agent: Callable[[Scenario, Entity], Optional[Agent]] = _create_agent,
    ) -> None:
        """
        Update the current scenario and create agents.

        Parameters
        ----------
        scenario : Scenario
            The scenario object.

        create_agent : Callable[[str, Entity], Optional[Agent]]
            A function that returns an agent to control a given entity.

        """
        self.state = State(
            conditions=self.terminal_conditions,
            state_callbacks=self.state_callbacks,
        )
        self.state.scenario = scenario
        if self.is_recording:
            self.recorder.scenario = scenario
        self.create_agents(create_agent=create_agent)
        self.reset_scenario()

    def create_agents(
        self,
        create_agent: Callable[[Scenario, Entity], Optional[Agent]] = _create_agent,
    ) -> None:
        """
        Create the agents for the scenario.

        Parameters
        ----------
        create_agent : Callable[[str, Entity], Optional[Agent]]
            A function that will return an agent for the given entity or
            return None if that entity does not need an agent. All entities
            without agents will replay their trajectories.

        """
        non_agents, non_agent_trajs = [], []
        for entity in self.state.scenario.entities:
            agent = create_agent(self.state.scenario, entity)
            if agent is not None:
                self.state.scenario.agents[entity.ref] = agent
            else:
                non_agents.append(entity)
                non_agent_trajs.append(entity.trajectory)
        self.state.scenario.non_agents.add_entities(non_agents, non_agent_trajs)

    def reset_scenario(self) -> None:
        """Reset the state to the beginning of the current scenario."""
        self.close_viewer()
        self.state.is_done = False
        self.state.t = Entity.INIT_PREV_T
        self.state.t = 0.0
        if self.state.scenario is not None:
            for agent in self.state.scenario.agents.values():
                agent.reset()
        self.state.scenario.non_agents.reset()

        for cb in self.state.state_callbacks:
            cb.reset(self.state)
        self.state.update_callbacks()
        for m in self.metrics:
            m.reset(self.state)

    def step(self) -> None:
        """Process a single step in the environment."""
        self.state.next_t = self.state.t + self.timestep

        # get the new poses
        new_poses = {}
        for agent in self.state.scenario.agents.values():
            new_poses[agent.entity] = agent.step(self.state)
        new_poses.update(self.state.scenario.non_agents.step(self.state))

        # update the poses and current time
        for e, p in new_poses.items():
            e.pose = p
        self.state.t = self.state.next_t
        self.state.update_callbacks()
        self.state.is_done = self.state.check_terminal()

        # metrics and rendering
        for m in self.metrics:
            m.step(self.state)
        if self.viewer is not None:
            self.state.last_keystroke = self.render()

    def rollout(self, render: bool = False, **kwargs) -> None:
        """Rollout the current scenario fully."""
        self.reset_scenario()
        if render:
            self.state.last_keystroke = self.render(**kwargs)
        while not self.state.is_done:
            self.step()
        for agent in self.state.scenario.agents.values():
            agent.finish(self.state)
        self.close()

    def render(self, video_path: Optional[str] = None) -> Optional[int]:
        """Render the state of the gym."""
        if self.viewer is None:
            self.setup_viewer()
        return self.viewer.render(self.state)

    def setup_viewer(self, video_path: Optional[str] = None) -> None:
        """Create the viewer when rendering is started."""
        if video_path is None:
            path = self.state.scenario.scenario_path
            video_dir = os.path.join(os.path.dirname(path), "../Recordings")
            if os.path.exists(video_dir):
                video_path = os.path.join(
                    video_dir,
                    os.path.splitext(
                        os.path.basename(path),
                    )[0]
                    + ".mp4",
                )
            else:
                video_path = (
                    os.path.splitext(self.state.scenario.scenario_path)[0] + ".mp4"
                )
        self.viewer = self.viewer_class(video_path, **self.viewer_parameters)

    def close_viewer(self) -> None:
        """Close the viewer if it exists."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def close(self) -> None:
        """Close the gym."""
        self.close_viewer()

    def record(self, close: bool = False) -> None:
        """
        Record the scenario and write to OpenScenario.

        Parameters
        ----------
        close : bool
            If given will close the recorder and remove trajectory tracking.

        """
        if not close:
            self.recorder = ScenarioRecorder(self.state.scenario)
        else:
            self.recorder.close()
            self.recorder = None
        if self.state.scenario is not None:
            self.reset_scenario()

    @property
    def is_recording(self) -> bool:
        """Return True if the gym is recording scenarios."""
        return self.recorder is not None

    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metric states."""
        values = {}
        for metric in self.metrics:
            value = metric.get_state()
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(k, str):
                        values[f"{metric.name}_{k}"] = v
            elif value is not None:
                values[metric.name] = value
        return values
