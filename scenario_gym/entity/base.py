from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from shapely.geometry import Polygon

from scenario_gym.catalog_entry import BoundingBox, CatalogEntry
from scenario_gym.trajectory import Trajectory


class Entity(ABC):
    """
    An entity in the gym.

    An entity consists of a catalog entry and a pose. Note that poses
    are immutable arrays. Once a pose is set to the entity it cannot
    be changed, only overwritten. It could be modified e.g. by calling
    pose.copy() and then modifying the copied array and setting
    that back to the pose.
    """

    INIT_PREV_T = -0.1

    def __init__(
        self,
        catalog_entry: CatalogEntry,
        ref: Optional[str] = None,
    ):
        """
        Construct an entity.

        Parameters
        ----------
        catalog_entry: CatalogEntry
            The catalog entry used for the entity.

        ref : str
            The unique reference for the entity from the OpenScenario file.

        """
        self.ref = ref
        self.catalog_entry = catalog_entry
        self.record_trajectory = False
        self._trajectory: Optional[Trajectory] = None
        self._prev_pose: Optional[np.ndarray] = None
        self._pose: Optional[np.ndarray] = None
        self._t: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._recorded_poses: List[Tuple[float, np.ndarray]] = []
        self._box_points: Optional[np.ndarray] = None
        self._box_geom: Optional[np.ndarray] = None
        self._distance_travelled = 0.0

    def reset(self) -> None:
        """Reset the entity at the start of the scenario."""
        self._pose = None
        self._prev_pose = None
        self._recorded_poses.clear()
        self._distance_travelled = 0.0

    @property
    def pose(self) -> np.ndarray:
        """
        Return the entity's current pose.

        Current position of the entity given as an immutable array of shape
        (6,) with values: (x_coordinate, y_coordinate, z_coordinate, heading,
        roll, pitch).
        """
        return self._pose

    @pose.setter
    def pose(self, pose: ArrayLike) -> None:
        pose = np.array(pose)
        if self._pose is not None:
            self.prev_pose = self._pose
            self._distance_travelled += np.linalg.norm((pose - self.prev_pose)[:3])
        self._pose = pose
        self._pose.flags.writeable = False
        self._box_points = None
        self._box_geom = None

    @property
    def prev_pose(self) -> np.ndarray:
        """Return the entity's previous pose."""
        return self._prev_pose

    @prev_pose.setter
    def prev_pose(self, pose: ArrayLike) -> np.ndarray:
        self._prev_pose = np.array(pose)

    @property
    def t(self) -> float:
        """Return the current time in seconds (s)."""
        return self._t

    @t.setter
    def t(self, t: float) -> None:
        self.prev_t = self._t
        self._t = t
        if self.record_trajectory:
            self._recorded_poses.append((self._t, self.pose))
        return self._t

    @property
    def prev_t(self) -> float:
        """Return the previous time (s)."""
        return self._prev_t

    @prev_t.setter
    def prev_t(self, prev_t: float) -> None:
        self._prev_t = prev_t

    @property
    def dt(self):
        """Return the timestep between the current and previous time."""
        return self.t - self.prev_t

    @property
    def velocity(self) -> np.ndarray:
        """Return the entity's current velocity including angular components."""
        return (self.pose - self.prev_pose) / self.dt

    @property
    def distance_travelled(self) -> float:
        """Return the total distanced travelled since the initial time."""
        return self._distance_travelled

    def set_initial(
        self,
        t: float,
        pose: ArrayLike,
        prev_t: Optional[float] = None,
        prev_pose: Optional[ArrayLike] = None,
    ) -> None:
        """
        Set the initial position and velocity of the entity.

        Parameters
        ----------
        t : float
            The timestamp for the initial pose.

        pose : np.ndarray
            The initial pose.

        prev_t : float
            The timestamp for an initial previous pose. Used for
            setting the initial velocity.

        prev_pose : np.ndarray
            The initial previous pose. Used for setting the initial velocity.

        """
        self._t = t
        self._pose = pose
        if prev_t is not None and prev_pose is not None:
            self._prev_t = prev_t
            self._prev_pose = prev_pose
        if self.record_trajectory:
            if prev_t is not None and prev_pose is not None:
                self._recorded_poses.append((prev_t, prev_pose))
            self._recorded_poses.append((t, pose))

    @property
    def type(self) -> Optional[str]:
        """Get the catalog type of the entity. E.g. Vehicle, Pedestrian."""
        return self.catalog_entry.catalog_type.replace("Catalogs", "")

    def get_bounding_box_points(
        self, pose: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """
        Return the bounding box coordinates in the global frame for the given pose.

        Returns cached values if they are there.

        Parameters
        ----------
        pose : Optional[ArrayLike]
            An array of the entities pose. May broadcast (..., [x, y, (z), h, ...]).

        """
        if pose is not None:
            return self._get_bounding_box_points(pose=pose)

        if self._box_points is None:
            self._box_points = self._get_bounding_box_points()

        return self._box_points

    def _get_bounding_box_points(
        self, pose: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """
        Compute the bounding box coordinates in the global frame for the given pose.

        Returns in the order: RR, FR, FL, RL.

        Parameters
        ----------
        pose : Optional[ArrayLike]
            An array of the entities pose. May broadcast (..., [x, y, (z), h, ...]).

        """
        pose = pose if pose is not None else self.pose
        ref_xy, h = pose[..., :2], pose[..., 3 if pose.shape[-1] > 3 else 2]
        n = h.ndim
        R = np.array([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]]).transpose(
            *(tuple(i + 2 for i in range(n)) + (0, 1))
        )
        points = np.array(
            [
                [
                    self.bounding_box.center_x - 0.5 * self.bounding_box.length,
                    self.bounding_box.center_y + 0.5 * self.bounding_box.width,
                ],
                [
                    self.bounding_box.center_x + 0.5 * self.bounding_box.length,
                    self.bounding_box.center_y + 0.5 * self.bounding_box.width,
                ],
                [
                    self.bounding_box.center_x + 0.5 * self.bounding_box.length,
                    self.bounding_box.center_y - 0.5 * self.bounding_box.width,
                ],
                [
                    self.bounding_box.center_x - 0.5 * self.bounding_box.length,
                    self.bounding_box.center_y - 0.5 * self.bounding_box.width,
                ],
            ]
        )
        points = ref_xy[..., None, :] + np.einsum("ij,...jk->...ik", points, R)
        return points

    def get_bounding_box_geom(self, pose: Optional[ArrayLike] = None) -> Polygon:
        """
        Return a Polygon representing the bounding box global frame.

        Returns cached values if the pose is None and they are there.

        Parameters
        ----------
        pose : Optional[ArrayLike]
            An array of the entities pose. Only one may be given.

        """
        if pose is not None:
            return Polygon(self.get_bounding_box_points(pose=pose))
        if self._box_geom is None:
            self._box_geom = Polygon(self.get_bounding_box_points())
        return self._box_geom

    @property
    def recorded_poses(self) -> np.ndarray:
        """Return all previous recorded poses as a numpy array."""
        if not self._recorded_poses:
            return np.empty((0, 7))
        ts, poses = map(np.array, zip(*self._recorded_poses))
        return np.concatenate([ts[:, None], poses], axis=1)

    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box of the entity from its catalog entry."""
        return self.catalog_entry.bounding_box

    @property
    def trajectory(self) -> Trajectory:
        """Get the trajectory loaded from the OpenScenario file for the entity."""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory: Trajectory):
        self._trajectory = trajectory


class BatchReplayEntity:
    """
    A single object used to represent multiple entities.

    Will replay exact trajectories from OpenScenario files. Computation
    is vectorized for efficiency.
    """

    def __init__(self, timestep: Optional[float] = None):
        """Init the batch entity with no assigned entities."""
        self.entities: List[Entity] = []
        self.trajectories: List[Trajectory] = []
        self.timestep = timestep
        self.max_t = 0.0

    def reset(self) -> None:
        """Reset the entities at the start of the scenario."""
        if len(self.entities) > 0:
            for e, t in zip(self.entities, self.trajectories):
                e.reset()
                e.set_initial(
                    0.0,
                    e.trajectory.position_at_t(0.0),
                    Entity.INIT_PREV_T,
                    e.trajectory.position_at_t(Entity.INIT_PREV_T),
                )

    def step(self, state: "State") -> Dict[Entity, ArrayLike]:  # noqa: F821
        """
        Take a single step in the gym.

        Returns the pose of each entity at the next timestamp.
        """
        new_poses = {}
        if len(self.entities) > 0:
            pos = self.fn(state.next_t)  # (m, num_ents)
            for e, p in zip(self.entities, pos):
                new_poses[e] = p
        return new_poses

    def add_entities(
        self,
        entities: List[Entity],
        trajs: List[Trajectory],
    ) -> None:
        """
        Add entities that are to be batched together.

        This will reset the entities in the scenario so all entities
        must be passed at once.

        Parameters
        ----------
        entities : List[Entity]
            The entities to be used.

        trajs : List[Trajectory]
            The trajectory for each entity.

        """
        self.entities.clear()
        self.trajectories.clear()
        self.max_t = 0.0
        if len(entities) > 0:
            self.entities.extend(entities)
            self.trajectories.extend(trajs)

            num_ents = len(self.entities)
            datas = []
            for t in self.trajectories:
                d = np.nan_to_num(t.data)
                if d.shape[0] == 1:
                    d = np.repeat(d, 2, axis=0)
                    d[-1, 0] += 1e-1  # to prevent nan
                datas.append(d)

            m = datas[0].shape[1] - 1
            ts = np.array(
                sorted(list(set([t for d in datas for t in d[:, 0]])))
            )  # (N,)
            self.max_t = ts[-1]

            interpd = []
            for d in datas:
                x = interp1d(
                    d[:, 0],
                    d[:, 1:].T,
                    bounds_error=False,
                    fill_value=(d[0, 1:], d[-1, 1:]),
                )(
                    ts
                ).T  # (N, m)
                interpd.append(x)

            X = np.concatenate(interpd, axis=1)  # (N, num_ents * m)
            if self.timestep:
                all_ts = np.arange(0.0, self.max_t, self.timestep)
                all_Xs = interp1d(
                    ts,
                    X.T,
                    bounds_error=False,
                    fill_value=(X[0], X[-1]),
                )(all_ts).T
                self.fn = lambda t: all_Xs[np.abs(all_ts - t).argmin()].reshape(
                    num_ents, m
                )
            else:
                interp = interp1d(
                    ts,
                    X.T,
                    bounds_error=False,
                    fill_value=(X[0], X[-1]),
                )
                self.fn = lambda t: interp(t).reshape(num_ents, m)
