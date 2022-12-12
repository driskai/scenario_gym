import os
from typing import Any, Callable, Dict, List, Optional, Type, Union

from scenario_gym.agent import Agent, _create_agent
from scenario_gym.entity import Entity
from scenario_gym.metrics import Metric
from scenario_gym.scenario import Scenario
from scenario_gym.state import State
from scenario_gym.viewer import Viewer
from scenario_gym.xosc_interface import import_scenario


class ScenarioGym:
    """The main class that loads and runs scenarios."""

    INIT_PREV_T = -0.1

    @classmethod
    def run_scenarios(
        cls,
        paths: List[str],
        render: bool = False,
        **kwargs,
    ) -> None:
        """Rollout the scenarios in paths."""
        gym = cls(**kwargs)
        for path in paths:
            gym.load_scenario(path)
            gym.rollout(render=render)

    def __init__(
        self,
        timestep: float = 1.0 / 30.0,
        viewer_class: Optional[Type[Viewer]] = None,
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
        if viewer_class is None and "fps" not in viewer_parameters:
            viewer_parameters["fps"] = int(1.0 / self.timestep)
        self.viewer_parameters = viewer_parameters.copy()

        if terminal_conditions is None:
            terminal_conditions = ["max_length"]
        self.terminal_conditions = terminal_conditions

        if state_callbacks is None:
            state_callbacks = []
        self.state_callbacks = state_callbacks

        if viewer_class is None:
            self._get_viewer()
        else:
            self.viewer_class = viewer_class
            self._render_enabled = True
        self.state: Optional[State] = None
        self.viewer: Optional[Viewer] = None
        self.reset_gym()

        if metrics is not None:
            self.add_metrics(metrics)

    def _get_viewer(self) -> None:
        """Get the viewer if it is not provided."""
        try:
            from scenario_gym.viewer.opencv import OpenCVViewer

            self.viewer_class = OpenCVViewer
            self._render_enabled = True
        except ImportError:
            self._render_enabled = False
            self.viewer_class = None

    def reset_gym(self) -> None:
        """
        Reset the state of the gym.

        Closes the viewer, removes any metrics and unloads the scenario.
        """
        self.close()
        self.state = None
        self.metrics = []

    def add_metrics(self, metrics: List[Metric]) -> None:
        """Add metrics to the gym."""
        self.metrics.extend(metrics)

    def load_scenario(
        self,
        scenario_path: str,
        create_agent: Callable[[Scenario, Entity], Optional[Agent]] = _create_agent,
        relabel: bool = False,
        **kwargs,
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
            **kwargs,
        )
        self.set_scenario(scenario, create_agent=create_agent)

    def set_scenario(
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
            scenario=scenario,
        )
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
                self.state.agents[entity.ref] = agent
            else:
                non_agents.append(entity)
                non_agent_trajs.append(entity.trajectory)
        self.state.non_agents.add_entities(non_agents, non_agent_trajs)

    def reset_scenario(self) -> None:
        """Reset the state to the beginning of the current scenario."""
        self.close()
        self.state.reset(self.INIT_PREV_T, 0.0)
        for m in self.metrics:
            m.reset(self.state)

    def step(self) -> None:
        """Process a single step in the environment."""
        self.state.next_t = self.state.t + self.timestep

        # get the new poses
        new_poses = {}
        for agent in self.state.agents.values():
            new_poses[agent.entity] = agent.step(self.state)
        new_poses.update(self.state.non_agents.step(self.state))

        # update the poses and current time
        self.state.step(new_poses)

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
        for agent in self.state.agents.values():
            agent.finish(self.state)
        self.close()

    def render(self, video_path: Optional[str] = None) -> None:
        """Render the state of the gym."""
        if self.viewer is None:
            self.reset_viewer(video_path=video_path)
        return self.viewer.render(self.state)

    def reset_viewer(self, video_path: Optional[str] = None) -> None:
        """Reset the viewer at the start of a new rollout."""
        if self.viewer is None:
            if not self._render_enabled:
                raise ValueError(
                    "Rendering is disabled since no `viewer_class` was provided "
                    "and the default viewer could not be imported. Perhaps OpenCV "
                    "is not installed?"
                )
            self.viewer = self.viewer_class(**self.viewer_parameters)
        else:
            self.viewer.close()
        if video_path is None:
            path = self.state.scenario.path
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
                video_path = os.path.splitext(self.state.scenario.path)[0] + ".mp4"
        self.viewer.reset(video_path)

    def close(self) -> None:
        """Close the gym."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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
