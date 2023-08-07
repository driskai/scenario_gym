from __future__ import annotations

from copy import copy
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from scenario_gym.utils import ArrayLike, NDArray, cached_property


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
        perm = [fields.index(f) for f in self._fields if f in fields]
        data = data[:, perm]
        data = data[np.unique(data[:, 0], return_index=True)[1]]
        n = data.shape[0]

        _data: List[NDArray] = []
        for f in self._fields:
            d = data[:, perm.index(fields.index(f))] if f in fields else np.zeros(n)
            if f not in fields or (f in fields and np.isfinite(d).sum() != n):
                if f == "h" and n == 1:
                    d = np.zeros(1)
                elif f == "h" and n > 1:
                    t = _data[0]
                    fn = interp1d(
                        t,
                        np.array(_data[1:3]).T,
                        axis=0,
                        fill_value="extrapolate",
                    )
                    d = np.arctan2(*np.flip(fn(t + 1e-2) - fn(t - 1e-2), axis=1).T)
                    d = _resolve_heading(d)
                elif f in ("z", "p", "r"):
                    d = np.zeros(n)
                else:
                    raise ValueError(
                        f"Invalid values found for {f}. Values required for xyt."
                    )
            elif f == "h":
                d = _resolve_heading(d)
            _data.append(d)
            setattr(self, f, d)

        # we will make the data readonly
        self._data = np.array(_data).T.copy()
        self._data.flags.writeable = False

        self._interpolated: Optional[Callable[[ArrayLike], NDArray]] = None
        self._interpolated_s: Optional[Callable[[ArrayLike], NDArray]] = None
        self._grad_fn = None

    @property
    def data(self) -> NDArray:
        """
        Get the underlying trajectory data.

        Note this property has no setter. To modify the trajectory data one must
        copy the data and init a new trajectory:
        ```
        new_data = trajectory.data.copy()
        # apply changes
        new_t = Trajectory(new_data)
        ```
        """
        return self._data

    def __len__(self) -> int:
        """Return the number of points in the trajectory."""
        return len(self.data)

    def __getitem__(self, idx: int) -> NDArray:
        """Get the idx'th point in the trajectory."""
        return self.data[idx]

    @cached_property
    def min_t(self) -> float:
        """Return the first timestamp of the trajectory."""
        return self.t.min()

    @cached_property
    def max_t(self) -> float:
        """Return the final timestamp of the trajectory."""
        return self.t.max()

    @cached_property
    def s(self) -> NDArray:
        """Return the distance travelled at each point."""
        ds = np.linalg.norm(np.diff(self.data[:, [1, 2]], axis=0), axis=1).cumsum()
        return np.hstack([[0.0], ds])

    @cached_property
    def arclength(self) -> float:
        """Total distance travelled."""
        return self.s[-1]

    def position_at_t(
        self,
        t: Union[float, ArrayLike],
        extrapolate: Union[bool, Tuple[bool, bool]] = (False, False),
    ) -> Optional[NDArray]:
        """
        Compute the position of the entity at time t.

        Can vectorise over t.

        Parameters
        ----------
        t : float
            The time at which the position is returned. Linearly interpolates
            the trajectory control points to find the position.

        extrapolate : Union[bool, Tuple[bool, bool]]
            Whether to extrapolate the trajectory if the time given is outside
            of the range of the trajectory. If False then None will be returned
            for such times. If a Tuple is given then first and second elements
            correspond to whether to extrapolate before and after the trajectory
            respectively or to fix them.

        Returns
        -------
        Optional[np.ndarray]
            The position as a numpy array. If the time given is outside of the
            range of the trajectory and extrapolate is False then None is returned.

        """
        t = np.array(t)
        if self._interpolated is None:
            data = self.data
            if data.shape[0] == 1:
                data = np.repeat(data, 2, axis=0)
                data[-1, 0] += 1e-3
            self._interpolated = interp1d(
                data[:, 0],
                data[:, 1:],
                bounds_error=False,
                fill_value="extrapolate",
                axis=0,
            )
        if isinstance(extrapolate, tuple):
            ext_bck, ext_fwd = extrapolate
            extrapolate = True
        else:
            ext_bck = ext_fwd = extrapolate
        if t.ndim == 0:
            if not extrapolate and (t < self.min_t or t > self.max_t):
                return None
            elif t < self.min_t and not ext_bck:
                return self.data[0, 1:]
            elif t > self.max_t and not ext_fwd:
                return self.data[-1, 1:]
            return self._interpolated(t)
        poses = self._interpolated(t)
        if not ext_bck:
            poses = np.where(t[:, None] < self.min_t, self.data[0, None, 1:], poses)
        if not ext_fwd:
            poses = np.where(
                t[:, None] > self.max_t, self.data[-1, None, 1:], poses
            )
        return poses

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
            s_ = self.s
            s_, idx = np.unique(s_, return_index=True)
            data = data[idx]
            if data.shape[0] == 1:
                data = np.repeat(data, 2, axis=0)
                data[-1, 0] += 1e-3
                s_ = np.hstack([s_[0] - 1e-3, s[0]])
            self._interpolated_s = interp1d(
                s_,
                data,
                bounds_error=False,
                fill_value=(data[0, :], data[-1, :]),
                axis=0,
            )
        out = self._interpolated_s(s)
        out[..., 0] = np.where(s == 0, 0, out[..., 0])
        return out

    def velocity_at_t(
        self, t: Union[float, ArrayLike], eps: float = 1e-4
    ) -> NDArray:
        """
        Compute the velocity of the entity at time t.

        Parameters
        ----------
        t : float
            The time at which the velocity is returned.

        eps : float
            The epsilon used to compute the velocity.

        Returns
        -------
        np.ndarray
            The velocity as a numpy array.

        """
        t = np.array(t)
        inside = np.logical_and(self.min_t <= t, t <= self.max_t)
        v_in = (
            self.position_at_t(t + eps / 2, extrapolate=True)
            - self.position_at_t(t - eps / 2, extrapolate=True)
        ) / eps
        v_out = np.zeros(t.shape + (6,))

        if t.ndim >= 1:
            inside = inside.reshape(-1, 1)
        return np.where(inside, v_in, v_out)

    def is_stationary(self) -> bool:
        """Return True if the trajectory is stationary."""
        return is_stationary(self.data)

    def __copy__(self) -> Trajectory:
        """Create a copy of the trajectory."""
        return self.__class__(self.data.copy())

    def copy(self) -> Trajectory:
        """Create a copy of the trajectory."""
        return copy(self)

    def translate(self, x: np.ndarray) -> Trajectory:
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

    def rotate(self, h: float) -> Trajectory:
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

    def smooth_headings(self) -> Trajectory:
        """
        Create a new trajectory by smoothing the existing headings.

        Returns
        -------
        Trajectory
            The smoothed trajectory.

        """
        s = self.s

        d = np.arctan2(
            *np.flip(
                self.position_at_s(s + 1e-2)[:, 1:3]
                - self.position_at_s(s - 1e-2)[:, 1:3],
                axis=1,
            ).T
        )
        d = _resolve_heading(d)

        new_data = self.data.copy()
        new_data[:, 4] = d
        return self.__class__(new_data)

    def subsample(
        self,
        points_per_s: Optional[float] = None,
        points_per_t: Optional[float] = None,
        curvature: bool = False,
        **kwargs,
    ) -> Trajectory:
        """
        Create a new trajectory with a given frequency of control points.

        The control points can either be equally spaced across time or across arc by
        passing the keyword arguments `points_per_t` or `points_per_s` respectively.
        Exactly one of these keywords must be passed.

        Parameters
        ----------
        points_per_s : Optional[float]
            Number of control points per unit of arc.

        points_per_t : Optional[float]
            Number of control points per unit of time.

        curvature: bool
            If given will given curvature sampling to subsample the trajectory.

        """
        if (points_per_s is None) == (points_per_t is None):
            raise ValueError(
                "Exactly one of `points_per_s` or `points_per_t` must be supplied."
            )
        if curvature:
            return self.curvature_subsample(
                points_per_s=points_per_s,
                points_per_t=points_per_t,
                **kwargs,
            )
        if points_per_t:
            n = int(max(1, np.ceil((self.max_t - self.min_t) * points_per_t)))
            ts = np.linspace(self.min_t, self.max_t, n)
            data = self.position_at_t(ts)
            return self.__class__(np.concatenate([ts[:, None], data], axis=1))

        n = int(max(1, np.ceil(self.arclength * points_per_s)))
        ss = np.linspace(0, self.arclength, n)
        data = self.position_at_s(ss)
        return self.__class__(data)

    def curvature_subsample(
        self,
        points_per_s: Optional[float] = None,
        points_per_t: Optional[float] = None,
        eps: float = 1e-3,
        weight: float = 5.0,
    ) -> np.ndarray:
        """
        Subsample by sampling points arround high curvature areas.

        Parameters
        ----------
        points_per_s : Optional[float]
            Number of control points per unit of arc.

        points_per_t : Optional[float]
            Number of control points per unit of time.

        eps : float
            Epsilon parameter for computing gradients of the trajectory.

        weight : float
            Temperature for sampling distribution. Higher values give points more
            densley sampled around high curvature areas.

        """
        if points_per_s is not None:
            n = int(np.maximum(1, points_per_s * self.arclength))
        elif points_per_t is not None:
            n = int(np.maximum(1, points_per_t * self.max_t))
        else:
            raise ValueError(
                "Exactly one of `points_per_s` or `points_per_t` must be supplied."
            )
        s = self.s
        if self._grad_fn is None:
            fn = self.position_at_s
            grads = (fn(s + eps)[:, [1, 2]] - fn(s - eps)[:, [1, 2]]) / (2 * eps)
            self._grad_fn = interp1d(s, grads, axis=0, fill_value="extrapolate")
        grad_fn = self._grad_fn
        second_grad = (grad_fn(s[1:-1] + eps) - grad_fn(s[1:-1] - eps)) / (2 * eps)
        curv = np.linalg.norm(second_grad, axis=1)
        dist = np.exp(weight * curv) / np.exp(weight * curv).sum()
        num_points = int(np.clip(n - 2, 1, dist.shape[0]))
        idxs = np.random.choice(
            dist.shape[0],
            size=(num_points,),
            replace=False,
            p=dist,
        )
        s_vals = s[np.hstack([[0], 1 + np.sort(idxs), [s.shape[0] - 1]])]
        return self.__class__(fn(s_vals))

    def to_json(self) -> List[List[float]]:
        """Write the trajectory to a jsonable list."""
        return self.data.tolist()


def _resolve_heading(h: NDArray) -> NDArray:
    """Update heading so that there are no large jumps."""
    deltas = np.diff(h) % (2 * np.pi)
    deltas = np.where(deltas > np.pi, deltas - 2 * np.pi, deltas)
    return np.hstack([h[0], deltas]).cumsum()


def is_stationary(data: np.ndarray) -> bool:
    """
    Check if an entity is stationary for the entire scenario.

    Any nan values are replaced with 0s.
    """
    return (
        len(
            np.unique(
                np.where(
                    np.isnan(data[:, 1:]),
                    0.0,
                    data[:, 1:],
                ),
                axis=0,
            )
        )
        <= 1
    )
