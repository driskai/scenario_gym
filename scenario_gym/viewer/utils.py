import math

import numpy as np


def rotate_coords(
    X: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Rotate xy coordinates in X by angle theta."""
    R = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ]
    )
    return X.dot(R.T)


def vec2pix(
    X: np.ndarray, mag: int = 20, h: float = 100.0, w: float = 100.0
) -> np.ndarray:
    """Convert xy coordinates to pixel values."""
    idxs = X + np.array([w / 2, h / 2])
    idxs[:, 0] = w - idxs[:, 0]
    return (mag * idxs).astype(np.int32)


def to_ego_frame(
    coords: np.ndarray,
    ego_pose: np.ndarray,
    vertical: bool = True,
) -> np.ndarray:
    """
    Convert coordinates in the global frame to the ego's frame.

    If vertical then the frame is rotated so the ego is aligned
    vertically (pi/2).
    """
    return rotate_coords(
        coords - ego_pose[None, :2],
        -ego_pose[3] - (math.pi / 2 if vertical else 0.0),
    )
