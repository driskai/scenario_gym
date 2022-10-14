import numpy as np

from scenario_gym.entity import Entity


def is_stationary(e: Entity) -> bool:
    """
    Check if an entity is stationary for the entire scenario.

    Any nan values are replaced with 0s.
    """
    poses = e.recorded_poses
    return (
        len(
            np.unique(
                np.where(
                    np.isnan(poses[:, 1:]),
                    0.0,
                    poses[:, 1:],
                ),
                axis=0,
            )
        )
        <= 1
    )
