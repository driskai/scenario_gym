from __future__ import annotations

from math import inf
from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from packaging import version

from scenario_gym.action import Action
from scenario_gym.agent import Agent
from scenario_gym.controller import VehicleController
from scenario_gym.entity import Entity
from scenario_gym.observation import Observation
from scenario_gym.scenario import Scenario
from scenario_gym.scenario_gym import ScenarioGym, _create_agent
from scenario_gym.sensor.map import RasterizedMapSensor
from scenario_gym.state import TERMINAL_CONDITIONS, State

try:
    import gym
    from gym import Env
    from gym.spaces import Box, Space
except ImportError as e:
    raise ImportError(
        "gym is required for this module. Install it with `pip install gym`."
    ) from e


class ScenarioGym(ScenarioGym, Env):
    """
    An OpenAI gym compatible version of the Scenario Gym.

    Provides an explicit interface to the observations of the
    ego agent. The agent must implement the reward method of the
    Agent class.
    """

    metadata = {"render_modes": []}
    _new_reset = version.parse(gym.__version__) >= version.parse("0.22.0")

    def __init__(
        self,
        action_space: Optional[Space] = None,
        observation_space: Optional[Space] = None,
        reward_range: Tuple[float, float] = (-inf, inf),
        terminal_conditions: Optional[
            List[Union[str, Callable[[State], bool]]]
        ] = None,
        timestep: float = 0.1,
        create_agent: Optional[
            Callable[[Scenario, Entity], Optional[Agent]]
        ] = None,
        select_scenario: Optional[
            Callable[[ScenarioGym], Union[Scenario, str]]
        ] = None,
        **kwargs,
    ):
        """
        Construct the ScenarioGym environment.

        Parameters
        ----------
        action_space : Optional[Space]
            The action space for the ego agent. If not given a Box space is
            assumed for acceleration and steering actions.

        observation_space : Optional[Space]
            The observation space for the ego agent. If not given a Box space is
            assumed for a rasterized map.

        reward_range : Tuple[float, float]
            Optional reward range parameter for gym.

        terminal_conditions : Optional[List[Union[str, Callable[[State], bool]]]]
            Terminal condtiions for the scenario gym. If not given then max_length,
            ego_collision and ego_off_road are used.

        timestep : float
            Timestep for the scenario_gym.

        create_agent : Optional[Callable[[Scenario, Entity], Optional[Agent]]]
            Create agent function for the gym. Should return an agent for the
            ego entity. If not given then `ScenarioGym.create_agent` will be used.

        select_scenario : Optional[Callable[[], Union[Scenario, str]]]
            Function that selects the scenario to be run each time `reset()` is
            called. Takes just self as argument and should return either the xosc
            filepath or the scenario object. If not given then
            `ScenarioGym.select_scenario` will be used.

        """
        if terminal_conditions is None:
            terminal_conditions = ["max_length", "ego_collision", "ego_off_road"]
        super().__init__(
            terminal_conditions=terminal_conditions,
            timestep=timestep,
            **kwargs,
        )
        if action_space is None:
            action_space = Box(
                low=np.array([-5.0, -0.9]).astype(np.float32),
                high=np.array([5.0, 0.9]).astype(np.float32),
                shape=(2,),
            )
        if observation_space is None:
            observation_space = Box(
                low=np.float32(0.0),
                high=np.float32(1.0),
                shape=(2, 128, 128),
            )
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = reward_range
        if create_agent is not None:
            self.create_agent = create_agent
        if select_scenario is not None:
            self.select_scenario = MethodType(select_scenario, self)

    def on_reset(self) -> None:
        """Run just before the reset is executed."""
        pass

    def after_reset(self) -> None:
        """Run just after the reset is executed."""
        pass

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[Dict] = None,
    ) -> Union[Observation, Tuple[Observation, Dict]]:
        """
        Reset the environment.

        Resets the state and computes the observation for the ego agent.

        Possible options:

        scenario : Union[Scenario, str]
            A chosen scenario object or filepath to the xosc to be used.

        """
        self.on_reset()

        if self._new_reset:
            super().reset(seed=seed)
        else:
            super().seed(seed)
        if (options is not None) and ("scenario" in options):
            s = options["scenario"]
        else:
            s = self.select_scenario()
        if s is not None:
            if isinstance(s, Scenario):
                self.set_scenario(s)
            else:
                self.load_scenario(s)
        elif self.state.scenario is None:
            raise ValueError("No scenario has been set.")
        else:
            self.reset_scenario()

        self.state.next_t = self.state.t + self.timestep
        ego_obs = self.ego_agent.sensor.step(self.state)

        self.after_reset()
        return (ego_obs, {}) if return_info else ego_obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        """
        Run one timestep of the environment.

        The action for the ego is processed and `step` is called on
        all other agents/entities. Then the state is processed and the
        reward is  for the

        Returns
        -------
        next_state : Observation
            Observation of the next state for the ego.

        reward : float
            The reward returned from the next state.

        done : bool
            Whether the next state is terminal.

        info : Dict
            Additional info.

        """
        if self.state.is_done:
            raise ValueError("Step called when state is terminal.")

        new_poses = {}
        for ref, agent in self.state.agents.items():
            if ref == "ego":
                agent.last_action = action
                new_poses[agent.entity] = agent.controller.step(self.state, action)
            else:
                new_poses[agent.entity] = agent.step(self.state)
        new_poses.update(self.state.non_agents.step(self.state))

        # update the poses and current time
        self.state.step(new_poses)

        # get reward of next state
        reward = self.ego_agent.reward(self.state)

        # rendering and metrics
        if self.viewer is not None:
            self.state.last_keystroke = self.render()
        for m in self.metrics:
            m.step(self.state)

        # process ego part of next state
        self.state.next_t = self.state.t + self.timestep
        ego_obs = self.ego_agent.sensor.step(self.state)

        if self.state.is_done:
            for agent in self.state.agents.values():
                agent.finish(self.state)

        return ego_obs, reward, self.state.is_done, {}

    def rollout(self, *args, **kwargs):
        """Raise an error if rollout is called with this env."""
        raise NotImplementedError("Rollout is not supported for this environment.")

    def render(
        self,
        mode: None = None,
        video_path: Optional[str] = None,
    ) -> Optional[int]:
        """Render the environment."""
        return super().render(video_path=video_path)

    def load_scenario(
        self, *args, create_agent: Optional[Callable] = None, **kwargs
    ) -> None:
        """
        Load a scenario from an OpenScenario file.

        Sets the default argument of `create_agent` to `self.create_agent`.
        """
        if create_agent is None:
            create_agent = self.create_agent
        super().load_scenario(*args, create_agent=create_agent, **kwargs)

    def set_scenario(
        self, *args, create_agent: Optional[Callable] = None, **kwargs
    ) -> None:
        """
        Set the scenario explicitly.

        Sets the default argument of `create_agent` to `self.create_agent`.
        """
        if create_agent is None:
            create_agent = self.create_agent
        super().set_scenario(*args, create_agent=create_agent, **kwargs)

    def select_scenario(self) -> Optional[Union[str, Scenario]]:
        """Update the scenario when reset is called."""
        return None

    def create_agents(
        self,
        create_agent: Callable[[Scenario, Entity], Optional[Agent]] = _create_agent,
    ) -> None:
        """Check there is an ego agent."""
        super().create_agents(create_agent=create_agent)
        try:
            self.ego_agent = self.state.agents["ego"]
        except KeyError as e:
            raise KeyError("No agent named ego.") from e

    @staticmethod
    def create_agent(scenario: Scenario, entity: Entity) -> Optional[Agent]:
        """Create the agents for the scenario."""
        if entity.ref == "ego":
            return RLAgent(
                entity,
                VehicleController(entity, max_steer=0.9, max_accel=5.0),
                MapOnlySensor(
                    entity, channels_first=True, height=30, width=30, n=128
                ),
            )


class MapOnlySensor(RasterizedMapSensor):
    """Sensor returning only the rasterized map."""

    def _step(self, state: State) -> np.ndarray:
        """Get the map from the base sensor's observation."""
        return super()._step(state).map


class RLAgent(Agent):
    """Example agent recieving negative rewards for collisions going off road."""

    def reward(self, state: State) -> Optional[float]:
        """Return the reward for the agent from the current state."""
        if state.is_done:
            if TERMINAL_CONDITIONS["ego_off_road"](state):
                return -1.0
            elif TERMINAL_CONDITIONS["ego_collision"](state):
                return -1.0
        return 0.01
