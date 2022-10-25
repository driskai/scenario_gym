from functools import cached_property
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import interp1d


class Trajectory:
    """
    A Scenario Gym representation of a trajectory.

    Note that trajectories consist of immutable arrays. To modify a trajectory
    one must copy the data and init a new one:
    ```
    new_data = trajectory.data.copy()
    # apply changes
    new_t = Trajectory(new_data)
    ```
    """

    _fields = ("t", "x", "y", "z", "h", "p", "r")
    t: Optional[NDArray] = None
    x: Optional[NDArray] = None
    y: Optional[NDArray] = None
    z: Optional[NDArray] = None
    h: Optional[NDArray] = None
    p: Optional[NDArray] = None
    r: Optional[NDArray] = None

    def __init__(self, data: NDArray, fields: Tuple[str] = _fields):
        """
        Trajectory constructor.

        Parameters
        ----------
        data : np.ndarray
            The trajectory data as a numpy array of (num_points, num_fields).
            By default the columns should be t, x, y, z, h, p, r otherwise the
            fields argument should be passed.

        fields : List[str]
            The field names for each column of data. Must contain t, x and y and
            must be a subset of _fields.

        """
        if not all(f in fields for f in ("t", "x", "y")):
            raise ValueError("Trajectory cannot be created with t, x and y values.")
        if data.ndim != 2 or data.shape[1] != len(fields):
            raise ValueError(
                f"Invalid shape: {data.shape}. Expected: (N, {len(fields)}). Either"
                " pass `fields` to specify the columns given or ensure that columns"
                f" for all of {self._fields} are provided."
            )

        _data: List[NDArray] = []
        for f in self._fields:
            d = data[:, fields.index(f)] if f in fields else np.zeros(data.shape[0])
            if f == "h":
                d = _resolve_heading(d)
            if f not in fields or (
                f in fields and np.isnan(data[:, fields.index(f)]).sum() != 0
            ):
                if f == "h" and data.shape[0] == 1:
                    d = np.zeros(1)
                elif f == "h" and data.shape[0] > 1:
                    t = _data[0]
                    fn = interp1d(
                        t,
                        np.array(_data[1:3]).T,
                        axis=0,
                        fill_value="extrapolate",
                    )
                    d = np.arctan2(*np.flip(fn(t + 1e-2) - fn(t - 1e-2), axis=1).T)
                elif f in ("z", "p", "r"):
                    d = np.zeros(data.shape[0])
                else:
                    raise ValueError(
                        f"Invalid values found for {f}. Values required for xyt."
                    )
            else:
                d = data[:, fields.index(f)]
            _data.append(d)
            setattr(self, f, d)

        # we will make the data readonly
        self.data = np.unique(np.array(_data).T, axis=0)
        self.data.flags.writeable = False

        self._interpolated: Optional[Callable[[ArrayLike], NDArray]] = None
        self._interpolated_s: Optional[Callable[[ArrayLike], NDArray]] = None

    def __len__(self) -> int:
        """Return the number of points in the trajectory."""
        return len(self.data)

    def __getitem__(self, idx: int) -> NDArray:
        """Get the idx'th point in the trajectory."""
        return self.data[idx]

    @cached_property
    def min_t(self) -> float:
        """Return the first timestamp of the trajectory."""
        return self.data[:, 0].min()

    @cached_property
    def max_t(self) -> float:
        """Return the final timestamp of the trajectory."""
        return self.data[:, 0].max()

    @cached_property
    def s(self) -> NDArray:
        """Return the distance travelled at each point."""
        ds = np.linalg.norm(np.diff(self.data[:, [1, 2]], axis=0), axis=1).cumsum()
        return np.hstack([[0.0], ds])

    @cached_property
    def arclength(self) -> float:
        """Total distance travelled."""
        return self.s[-1]

    def position_at_t(self, t: float) -> NDArray:
        """
        Compute the position of the entity at time t.

        Parameters
        ----------
        t : float
            The time at which the position is returned. Linearly interpolates
            the trajectory control points to find the position.

        Returns
        -------
        np.ndarray
            The position as a numpy array.

        """
        if self._interpolated is None:
            data = self.data
            if data.shape[0] == 1:
                data = np.repeat(data, 2, axis=0)
                data[-1, 0] += 1e-3
            self._interpolated = interp1d(
                data[:, 0],
                data[:, 1:],
                bounds_error=False,
                fill_value=(data[0, 1:], data[-1, 1:]),
                axis=0,
            )
        return self._interpolated(t)

    def position_at_s(self, s: float) -> NDArray:
        """
        Compute the position of the entity at distance travelled s.

        Parameters
        ----------
        s : float
            The arclength at which the position is returned. Linearly
            interpolates the trajectory control points to find the position.

        Returns
        -------
        np.ndarray
            The position as a numpy array.

        """
        if self._interpolated_s is None:
            data = self.data
            if data.shape[0] == 1:
                data = np.repeat(data, 2, axis=0)
                data[-1, 0] += 1e-3
            self._interpolated_s = interp1d(
                self.s,
                data[:, 1:],
                bounds_error=False,
                fill_value=(data[0, 1:], data[-1, 1:]),
                axis=0,
            )
        return self._interpolated_s(s)

    def translate(self, x: np.ndarray) -> "Trajectory":
        """
        Create a new trajectory by translating the current by x.

        Parameters
        ----------
        x : np.ndarray
            Translation quantity. Must broadcast to the data so must be
            a matrix, vector or scalar.

        Returns
        -------
        Trajectory
            The translated trajectory.

        """
        if x.ndim == 1:
            x = x[None, :]
        return self.__class__(self.data + x)

    def rotate(self, h: float) -> "Trajectory":
        """
        Create a new trajectory by rotating the current by h around O.

        Parameters
        ----------
        h : float
            The angle of rotation (about the origin).

        Returns
        -------
        Trajectory
            The rotated trajectory.

        """
        new_data = self.data.copy()
        xy = new_data[None, 0, [1, 2]]
        new_data[:, [1, 2]] = (new_data[:, [1, 2]] - xy).dot(
            np.array(
                [
                    [np.cos(h), np.sin(h)],
                    [-np.sin(h), np.cos(h)],
                ]
            )
        ) + xy
        new_data[:, 4] = (new_data[:, 4] + h) % (2.0 * np.pi)
        return self.__class__(new_data)


def _resolve_heading(h: NDArray) -> NDArray:
    """Update heading so that there are no large jumps."""
    deltas = np.diff(h) % (2 * np.pi)
    deltas = np.where(deltas > np.pi, deltas - 2 * np.pi, deltas)
    return np.hstack([h[0], deltas]).cumsum()
