from __future__ import annotations

from copy import copy
from inspect import getfullargspec
from typing import Optional, Type

import numpy as np
from shapely.geometry import Polygon

from scenario_gym.catalog_entry import BoundingBox, CatalogEntry
from scenario_gym.trajectory import Trajectory
from scenario_gym.utils import ArrayLike, NDArray


class Entity:
    """
    An entity in the gym.

    An entity consists of a catalog entry and a pose. Note that poses
    are immutable arrays. Once a pose is set to the entity it cannot
    be changed, only overwritten. It could be modified e.g. by calling
    pose.copy() and then modifying the copied array and setting
    that back to the pose.
    """

    @classmethod
    def _catalog_entry_type(cls) -> Type[CatalogEntry]:
        """Get the type of catalog entry that is used for this entity."""
        args = getfullargspec(cls.__init__)
        ce = args.args[1]
        try:
            ce_type = args.annotations[ce]
        except KeyError:
            raise NotImplementedError(
                f"Subclass {cls.__name__} has no type annotation for catalog entry."
            )
        if not issubclass(ce_type, CatalogEntry):
            raise TypeError("Catalog entry type must be a catalog entry subclass.")
        return ce_type

    def __init__(
        self,
        catalog_entry: CatalogEntry,
        trajectory: Optional[Trajectory] = None,
        ref: Optional[str] = None,
    ):
        """
        Construct an entity.

        Parameters
        ----------
        catalog_entry: CatalogEntry
            The catalog entry used for the entity.

        trajectory : trajectory
            The trajectory for the entity.

        ref : Optional[str]
            The unique reference for the entity from the OpenScenario file.

        """
        self.ref = ref
        self.catalog_entry = catalog_entry
        self._trajectory = trajectory

    @property
    def trajectory(self) -> Trajectory:
        """Get the trajectory for the entity."""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, trajectory: Trajectory) -> None:
        """Set the trajectory for the entity."""
        self._trajectory = trajectory

    @property
    def bounding_box(self) -> BoundingBox:
        """Get the bounding box of the entity from its catalog entry."""
        return self.catalog_entry.bounding_box

    @property
    def type(self) -> Optional[str]:
        """Get the catalog type of the entity. E.g. Vehicle, Pedestrian."""
        return self.catalog_entry.catalog_type.replace("Catalogs", "")

    def __copy__(self) -> Entity:
        """Create a copy of an entity without copying the catalog_entry."""
        return self.__class__(
            self.catalog_entry,
            trajectory=self.trajectory.copy(),
            ref=self.ref,
        )

    def copy(self) -> Entity:
        """Create a copy of an entity without copying the catalog_entry."""
        return copy(self)

    def get_bounding_box_points(self, pose: ArrayLike) -> NDArray:
        """
        Compute the bounding box coordinates in the global frame for the given pose.

        Returns in the order: RR, FR, FL, RL.

        Parameters
        ----------
        pose : Optional[ArrayLike]
            An array of the entities pose. May broadcast (..., [x, y, (z), h, ...]).

        """
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

    def get_bounding_box_geom(self, pose: Optional[ArrayLike]) -> Polygon:
        """
        Return a Polygon representing the bounding box global frame.

        Returns cached values if the pose is None and they are there.

        Parameters
        ----------
        pose : Optional[ArrayLike]
            An array of the entities pose. Only one may be given.

        """
        return Polygon(self.get_bounding_box_points(pose))


class StaticEntity(Entity):
    """Used for entities with only one control point."""

    @Entity.trajectory.setter
    def trajectory(self, trajectory: Trajectory) -> None:
        """Check that the trajectory is static."""
        if trajectory.data.shape[0] != 1:
            raise ValueError(
                "Recieved multiple control points for static entity: {self.ref}"
            )
        self._trajectory = trajectory
