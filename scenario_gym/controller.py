from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from scenario_gym.action import Action, TeleportAction, VehicleAction
from scenario_gym.entity import Entity
from scenario_gym.state import State
from scenario_gym.utils import ArrayLike


class Controller(ABC):
    """
    Base class for the controller. Takes the agent's action and returns the pose.

    When implementing a controller the _step method should return the
    new pose for the entity (as an np.ndarray). It may modify the state
    in other ways but the controller's step method will update the
    entities pose. This is to avoid errors with immutable arrays.
    """

    def __init__(self, entity: Entity):
        """Construct the controller from the entity."""
        self.entity = entity

    def reset(self, state: State) -> None:
        """Reset the controller parameters."""
        self._reset(state)

    def step(self, state: State, action: Action) -> ArrayLike:
        """Return the agent's next pose from the action."""
        return self._step(state, action)

    @abstractmethod
    def _reset(self, state: State) -> None:
        """Reset the controller parameters."""
        pass

    @abstractmethod
    def _step(self, state: State, action: Action) -> ArrayLike:
        """Return the agent's next pose from the action."""
        pass


class ReplayTrajectoryController(Controller):
    """A controller to replay preset trajectories."""

    def _reset(self, state: State) -> None:
        """Reset the controller parameters."""
        pass

    def _step(self, state: State, action: TeleportAction) -> ArrayLike:
        """Return the agent's next pose from the action."""
        return action.pose


class VehicleController(Controller):
    """
    A vehicle controller using a simple physical model.

    Allows acceleration and steering in a given range.
    """

    def __init__(
        self,
        entity: Entity,
        max_steer: float = 0.7,
        max_accel: float = 5.0,
        max_speed: Optional[float] = None,
        allow_reverse: bool = False,
    ):
        """
        Construct the controller from the entity.

        Parameters
        ----------
        entity : Entity
            The entity for the controller.

        max_steer : float
            The max allowed (absolute) steering angle.

        max_accel : float
            The max allowed (absolute) acceleration.

        max_speed : Optional[float]
            If given then the entity is limited to this max speed.

        allow_reverse : bool
            Allow the vehicle to move backwards. If False then the
            vehicle's speed is forced >= 0.

        """
        super().__init__(entity)
        self.max_steer = max_steer
        self.max_accel = max_accel
        self.allow_reverse = allow_reverse
        self.max_speed = max_speed

    def _reset(self, state: State) -> None:
        """Reset the controller parameters."""
        self.speed = np.linalg.norm(state.velocities[self.entity][:2])
        self.l = self.entity.catalog_entry.bounding_box.length

    def _step(
        self, state: State, action: Union[VehicleAction, np.ndarray]
    ) -> ArrayLike:
        """
        Return the agent's next pose from the action.

        Updates the heading based on the steering angle. Then calculates
        the new speed to return the new velocity.
        """
        if isinstance(action, VehicleAction):
            accel, steer = action.acceleration, action.steering
        else:
            accel, steer = action

        accel = np.clip(accel, -self.max_accel, self.max_accel)
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        pose = state.poses[self.entity].copy()
        dt = state.next_t - state.t
        h = pose[3]

        dx = self.speed * np.cos(h)
        dy = self.speed * np.sin(h)
        dh = self.speed * np.tan(steer) / self.l

        pose[[0, 1]] += np.array([dx, dy]) * dt
        pose[3] += dh * dt

        speed = self.speed + accel * dt
        if not self.allow_reverse:
            speed = np.maximum(0.0, speed)
        if self.max_speed is not None:
            speed = np.minimum(self.max_speed, speed)
        self.speed = speed

        return pose


class PIDController(VehicleController):
    """
    A PID controller for scenario gym agents.

    Selects acceleration and steering to get to a given waypoint. These
    are computed using a very simple PID controller for the steering
    and acceleration. The acceleration error and steering errors are based
    on the lateral and longitudinal error from the target in the vehicles
    local frame.
    """

    def __init__(
        self,
        entity: Entity,
        steer_Kp: float = 0.03054,
        steer_Kd: float = 1.5709,
        accel_Kp: float = 0.3753,
        accel_Kd: float = 1.8970,
        accel_Ki: float = 0.0204,
        **kwargs,
    ):
        """
        Construct the controller from the entity.

        entity : Entity
            The entity for the controller.

        steer_Kp : float
            The steering proportionality parameter.

        steer_Kd : float
            The steering derivative parameter.

        accel_Kp : float
            The acceleration proportionality parameter.

        accel_Kd : float
            The acceleration derivative parameter.

        accel_Ki : float
            The acceleration integral parameter.

        kwargs:
            Keyword arguments for the underlying vehicle model.
        """
        super(self.__class__, self).__init__(
            entity,
            **kwargs,
        )
        self.steer_Kp = steer_Kp
        self.steer_Kd = steer_Kd
        self.accel_Kp = accel_Kp
        self.accel_Ki = accel_Ki
        self.accel_Kd = accel_Kd

    def _reset(self, state: State) -> None:
        """Reset the controller parameters."""
        self.e_lon_prev = 0.0
        self.e_lon_int = 0.0
        self.e_lat_prev = 0.0
        super(self.__class__, self)._reset(state)

    def _step(self, state: State, action: TeleportAction) -> ArrayLike:
        """
        Return the agent's next pose from the action.

        Calculates the longitudinal and lateral error to use as error
        values for the acceleration and steering. Then the parameters
        are applied to produce vehicle action values for the vehicle
        controller.
        """
        # current and target positions
        target = action.pose[:2]
        pose = state.poses[self.entity].copy()
        cur, h = pose[:2], pose[3]
        speed = self.speed

        # error and derivatives
        e = target - cur  # (2,)
        R = np.array(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )
        e_lon, e_lat = R.dot(e)

        # steering
        if speed > 5.0 and speed <= 15:
            gain_adj = 1.0 - 0.9 * ((speed - 5.0)) / 10.0
        elif speed > 15:
            gain_adj = 0.1
        else:
            gain_adj = 1.0

        e_lat_D = (e_lat - self.e_lat_prev) / state.dt
        steer_Kp = self.steer_Kp * gain_adj
        steer_Kd = self.steer_Kd * gain_adj
        steer = steer_Kp * e_lat + steer_Kd * e_lat_D

        # acceleration
        e_lon_D = (e_lon - self.e_lon_prev) / state.dt
        e_lon_I = self.e_lon_int + e_lon * state.dt
        if abs(e_lon) > 0.1:
            accel = (
                self.accel_Kp * e_lon
                + self.accel_Kd * e_lon_D
                + self.accel_Ki * e_lon_I
            )
        else:
            accel = 0.0

        self.e_lat_prev = e_lat
        self.e_lon_prev = e_lon
        self.e_lon_int = e_lon_I
        return super(self.__class__, self)._step(state, VehicleAction(accel, steer))
