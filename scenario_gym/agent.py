from typing import Optional

from scenario_gym.action import Action, TeleportAction
from scenario_gym.controller import (
    Controller,
    PIDController,
    ReplayTrajectoryController,
)
from scenario_gym.entity import Entity
from scenario_gym.observation import Observation
from scenario_gym.scenario import Scenario
from scenario_gym.sensor import EgoLocalizationSensor, Sensor
from scenario_gym.state import State
from scenario_gym.trajectory import Trajectory
from scenario_gym.utils import ArrayLike


class Agent:
    """Base agent class. Processes observations to select an action."""

    def __init__(self, entity: Entity, controller: Controller, sensor: Sensor):
        """
        Construct an agent given an entity, controller and sensor.

        Parameters
        ----------
        entity : Entity
            The entity that the agent will control.

        controller : Controller
            The controller for the agent.

        sensor : Sensor:
            The sensor module for the agent.

        """
        self.entity = entity
        self.controller = controller
        self.sensor = sensor
        self._last_action: Optional[Action] = None
        self._last_reward: Optional[float] = None
        self._trajectory: Optional[Trajectory] = None

    def reset(self, state: State) -> None:
        """Reset the agent state at the start of the scenario."""
        self.last_action = None
        self.last_reward = None
        self.sensor.reset(state)
        self.controller.reset(state)
        self._reset()

    def step(self, state: State) -> ArrayLike:
        """Select an action from the current observation."""
        obs = self.sensor.step(state)
        action = self._step(obs)
        self.last_action = action
        return self.controller.step(state, action)

    def _reset(self) -> None:
        """Reset the agent state at the start of the scenario."""
        pass

    def _step(self, observation: Observation) -> Action:
        """Select an action from the current observation."""
        pass

    def finish(self, state: State) -> None:
        """Process the end of the scenario."""
        pass

    @property
    def trajectory(self) -> Trajectory:
        """
        Return the trajectory of the agent.

        By default this is the underlying entities trajectory but can be overridden.
        """
        return (
            self._trajectory
            if self._trajectory is not None
            else self.entity.trajectory
        )

    @trajectory.setter
    def trajectory(self, trajectory: Trajectory):
        self._trajectory = trajectory

    @property
    def last_action(self) -> Action:
        """Return the previous action selected by the agent."""
        return self._last_action

    @last_action.setter
    def last_action(self, action: Action) -> None:
        self._last_action = action

    def reward(self, state: State) -> Optional[float]:
        """Return and cache the reward for the agent from the current state."""
        r = self._reward(state)
        if r is not None:
            self.last_reward = r
        return r

    def _reward(self, state: State) -> Optional[float]:
        """Return the reward for the agent from the current state."""
        pass

    @property
    def last_reward(self) -> Optional[float]:
        """Get the last reward."""
        return self._last_reward

    @last_reward.setter
    def last_reward(self, reward: Optional[float]) -> None:
        self._last_reward = reward


class ReplayTrajectoryAgent(Agent):
    """Agent for entities that follow trajectories predefined in the scenario."""

    def _reset(self) -> None:
        """Reset the agent state at the start of the scenario."""
        pass

    def _step(self, observation: Observation) -> Action:
        """Return the agent's next pose."""
        new_pose = self.trajectory.position_at_t(observation.next_t)
        return TeleportAction(pose=new_pose)


class PIDAgent(Agent):
    """Agent following a specified trajectory with a PID controller."""

    def __init__(self, entity: Entity, **controller_kwargs):
        super().__init__(
            entity,
            PIDController(entity, **controller_kwargs),
            EgoLocalizationSensor(entity),
        )

    def _reset(self) -> None:
        """Reset the agent state at the start of the scenario."""
        pass

    def _step(self, observation: Observation) -> TeleportAction:
        """Get the next waypoint from the agent's trajectory."""
        pos = self.trajectory.position_at_t(observation.next_t)
        return TeleportAction(x=pos[0], y=pos[1], z=pos[2])


def _create_agent(scenario: Scenario, entity: Entity) -> Optional[Agent]:
    """
    Return a replay trajectory agent.

    This is the default create agent function used by the gym.

    Parameters
    ----------
    scenario : Scenario
        The scenario object.

    entity : Entity
        The specific entity within the scenario.

    """
    if entity.ref == "ego":
        controller = ReplayTrajectoryController(entity)
        sensor = EgoLocalizationSensor(entity)
        return ReplayTrajectoryAgent(entity, controller, sensor)
