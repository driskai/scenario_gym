from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.linalg import norm


def inverse_direction(vector: Iterable, normalised: bool = True) -> List[float]:
    """
    Return the inverse of a 2D vector, Iterable -> Iterable.

    Uses clockwise-rotating sign convention: (x, y) --> (y, -x)

    Optional parameter normalised (default True):
    --  True: returns vector of unit length
    --  False: returns vector of same length as the input vector
    """
    assert len(vector) == 2, "Invalid vector dimension: {0}".format(len(vector))
    if normalised:
        n = norm([vector[1], vector[0]])
        return [vector[1] / n, -vector[0] / n]
    return [vector[1], -vector[0]]


def coord_change(
    vector: List[float],
    direction: List[float],
    centre: Tuple[float, float] = (0, 0),
) -> List[float]:
    """
    Change vector coordinates to new frame of reference.

    Uses Galilean transformation to change the components of <vector> to a new
    coordinate system defined by an origin at <centre> and with components parallel
    and perpendicular to <direction>

    Optional argument centre (default [0, 0])
    """
    assert len(vector) == 2, (
        "coord_change is implemented to work in 2D. Passed vector dimension: "
        + str(len(vector))
    )
    vector = np.array(vector)
    centre = np.array(centre)
    inv_dir = inverse_direction(direction)
    return [np.dot((vector - centre), inv_dir), np.dot(vector - centre, direction)]


def acceleration(
    entity_poses: List[List[float]],
    dt: float,
    parallel_velocity: bool = False,
    i: int = 0,
) -> List[float]:
    """
    Find entity acceleration from three consecutive poses.

    Optional argument parallel_velocity (default False):
    --  True: acceleration is resolved in coords para and perp to velocity
    --  False: acceleration remains in pose coords
    """
    try:
        entity_pose_2 = entity_poses[i + 2][1:3]
        entity_pose_1 = entity_poses[i + 1][1:3]
        entity_pose_0 = entity_poses[i][1:3]
    except IndexError:
        # If too few poses received, default with accel = [0, 0]
        return [0.0, 0.0]
    velocity_1 = (entity_pose_1 - entity_pose_2) / dt
    velocity_0 = (entity_pose_0 - entity_pose_1) / dt
    accel = np.array((velocity_0 - velocity_1) / dt)
    if not parallel_velocity:
        return accel
    # Resolve acceleration vector into [para to velocity, perp to velocity]
    return [
        np.dot(velocity_1, accel) / norm(velocity_1),
        np.dot([-velocity_1[1], velocity_1[0]] / norm(velocity_1), accel),
    ]


def ahead(ego: Dict, haz: Dict) -> bool:
    """
    Determine if ego is ahead of hazard.

    Takes two entities (ego and hazard) and returns True if the ego is ahead
    of the hazard when using ego's preferential frame of reference such that:
    --  ego is at position [0.0, 0.0]
    --  ego direction is [0, 1] ...
    --  ... requiring ego velocity to be parallel to y axis
    """
    ego_position = ego["position"][1]
    haz_position = haz["position"][1]
    return ego_position > haz_position


def direction(heading: float) -> list:
    """
    Turn heading into normalised direction vector.

    angle (rad) --> [x component, y component]
    Where 0 rad corresponds to positive x unit vector, and a positively
    increasing angle moves in counter-clockwise direction.
    """
    return [np.cos(heading), np.sin(heading)]
