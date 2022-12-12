from typing import Optional

import numpy as np


class Action:
    """Base class for actions that agents commnicate to controllers."""

    pass


class TeleportAction(Action):
    """An action consiting of desired coordinates for the next pose."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        h: float = 0.0,
        r: float = 0.0,
        p: float = 0.0,
        pose: Optional[np.ndarray] = None,
    ):
        """
        Teleport action from coordinates or a pose.

        Parameters
        ----------
        x : float
            The x coordinate of the action.

        y : float
            The y coordinate of the action.

        z : float
            The z coordinate of the action.

        h : float
            The h coordinate of the action.

        r : float
            The r coordinate of the action.

        p : float
            The p coordinate of the action.

        pose : Optional[np.ndarray]
            The whole pose as a numpy array of shape (6,). Will
            overwrite any other coordinates passed.

        """
        self.x = pose[0] if pose is not None else x
        self.y = pose[1] if pose is not None else y
        self.z = pose[2] if pose is not None else z
        self.h = pose[3] if pose is not None else h
        self.r = pose[4] if pose is not None else r
        self.p = pose[5] if pose is not None else p

    @property
    def pose(self) -> np.ndarray:
        """Return a pose representation of the action as an array of shape (6,)."""
        return np.array([self.x, self.y, self.z, self.h, self.r, self.p])


class VehicleAction(Action):
    """An acceleration and a steering update."""

    def __init__(self, accel: float, steer: float):
        """
        Vehicle action from acceleration and steering.

        Parameters
        ----------
        accel : float
            The acceleration of the vehicle.

        steer : float
            The steering angle of angular velocity of the vehicle.

        """
        self.acceleration = accel
        self.steering = steer
