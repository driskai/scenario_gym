from dataclasses import dataclass
from typing import Any, Dict, Optional

from lxml.etree import Element

from scenario_gym.catalog_entry import (
    ArgsKwargs,
    BoundingBox,
    Catalog,
    CatalogEntry,
)
from scenario_gym.entity.base import Entity
from scenario_gym.trajectory import Trajectory


@dataclass
class PedestrianCatalogEntry(CatalogEntry):
    """Catalog entry for a pedestrian."""

    mass: Optional[float]

    xosc_names = ["Pedestrian"]

    @classmethod
    def load_data_from_xml(cls, catalog_name: str, element: Element) -> ArgsKwargs:
        """Load the vehicle from an xml element."""
        base_args, _ = super().load_data_from_xml(catalog_name, element)
        mass = element.attrib.get("mass")
        if mass is not None:
            mass = float(mass)
        return base_args + (mass,), {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the pedestrian from a dictionary."""
        return cls(
            Catalog(data["catalog"]["catalog_name"], data["catalog"]["rel_path"]),
            data["catalog_entry"],
            data["catalog_category"],
            data["catalog_type"],
            BoundingBox.from_dict(data["bounding_box"]),
            data.get("properties", {}),
            data.get("files", []),
            data.get("mass"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Write the pedestrian to a dictionary."""
        data = super().to_dict()
        data["mass"] = self.mass
        return data


class Pedestrian(Entity):
    """Entity class for pedestrians."""

    def __init__(
        self,
        catalog_entry: PedestrianCatalogEntry,
        trajectory: Optional[Trajectory] = None,
        ref: Optional[str] = None,
    ):
        super().__init__(catalog_entry, trajectory=trajectory, ref=ref)
