from abc import abstractmethod
from types import MethodType
from typing import Any, Callable, Optional, Tuple

from scenario_gym.agent import Agent
from scenario_gym.scenario_gym import ScenarioGym

try:
    from dm_env import Environment, TimeStep, restart, termination, transition
except ImportError:
    raise ImportError(
        "dm_env is required for this module. Install it with `pip install dm_env`."
    )


class ScenarioGym(ScenarioGym, Environment):
    """
    Scenario Gym subclass compatible with dm_env.

    This is still an abstract class which requires implementation of the method
    `observation_spec` and `action_spec`. This is because these are required as
    methods for `dm_env.Environment` but are not known by `ScenarioGym` until the
    ego agent is defined. Therefore subclasses should implement the specific specs
    required for the ego agent for the chosen experiment.
    """

    def __init__(
        self,
        *args,
        update_scenario: Optional[Callable[[ScenarioGym], None]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ego_agent: Optional[Agent] = None

        if update_scenario is not None:
            self.update_scenario = MethodType(update_scenario, self)

    def update_scenario(self) -> None:
        """Update the loaded scenario when reset is called."""
        pass

    def reset(self) -> TimeStep:
        """Reset the environment."""
        self.update_scenario()
        obs = self._reset()
        return restart(obs)

    def _reset(self) -> Any:
        """Reset the environment."""
        if self.state.scenario is None:
            raise ValueError("No scenario has been set.")
        self.reset_scenario()

        try:
            self.ego_agent = self.state.agents["ego"]
        except KeyError:
            raise KeyError("No agent named ego.")

        self.state.next_t = self.state.t + self.timestep
        return self.ego_agent.sensor.step(self.state)

    def step(self, action) -> TimeStep:
        """Process an action and get the next timetsep."""
        if (
            self.state.scenario is None
            or self.state.is_done
            or self.ego_agent is None
        ):
            return self.reset()

        obs, reward = self._step(action)

        if self.state.is_done:
            return termination(reward, obs)

        return transition(reward, obs)

    def _step(self, action) -> Tuple[Any, float]:
        """Process the given action."""
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
        for m in self.metrics:
            m.step(self.state)
        if self.viewer is not None:
            self.state.last_keystroke = self.render()

        # process ego part of next state
        self.state.next_t = self.state.t + self.timestep
        ego_obs = self.ego_agent.sensor.step(self.state)

        if self.state.is_done:
            for agent in self.state.scenario.agents.values():
                agent.finish(self.state)

        return ego_obs, reward

    @abstractmethod
    def observation_spec(self) -> Any:
        """Return the observation spec for the environment."""
        raise NotImplementedError()

    @abstractmethod
    def action_spec(self) -> Any:
        """Return the action spec for the environment."""
        raise NotImplementedError()

    def rollout(self, *args, **kwargs) -> None:
        """Raise an error if rollout is called with this env."""
        raise NotImplementedError("Rollout is not supported for this environment.")

    def reset_scenario(self) -> None:
        """Reset scenario and reference to old ego agent."""
        super().reset_scenario()
        self.ego_agent = None
