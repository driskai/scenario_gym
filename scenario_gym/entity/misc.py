from dataclasses import dataclass
from typing import Any, Dict, Optional

from lxml.etree import Element
from scenariogeneration import xosc

from scenario_gym.catalog_entry import (
    ArgsKwargs,
    BoundingBox,
    Catalog,
    CatalogEntry,
)
from scenario_gym.entity.base import Entity
from scenario_gym.trajectory import Trajectory


@dataclass
class MiscObjectCatalogEntry(CatalogEntry):
    """Catalog entry for a pedestrian."""

    mass: Optional[float]

    xosc_names = ["MiscObject"]

    @classmethod
    def load_data_from_xml(
        cls,
        element: Element,
        catalog: Optional[Catalog] = None,
    ) -> ArgsKwargs:
        """Load the vehicle from an xml element."""
        base_args, _ = super().load_data_from_xml(element, catalog=catalog)
        mass = element.attrib.get("mass")
        if mass is not None:
            mass = float(mass)
        return base_args + (mass,), {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the pedestrian from a dictionary."""
        catalog = (
            Catalog.from_dict(data["catalog"])
            if data.get("catalog") is not None
            else None
        )
        return cls(
            catalog,
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

    def to_xosc(self) -> xosc.MiscObject:
        """Write the pedestrian to xosc."""
        obj = xosc.MiscObject(
            self.catalog_entry,
            self.mass,
            getattr(
                xosc.MiscObjectCategory,
                self.catalog_category,
                xosc.MiscObjectCategory.none,
            ),
            self.bounding_box.to_xosc(),
        )
        for k, v in self.properties.items():
            obj.add_property(k, v)
        for f in self.files:
            obj.add_property_file(f)
        return obj


class MiscObject(Entity):
    """Entity class for pedestrians."""

    def __init__(
        self,
        catalog_entry: MiscObjectCatalogEntry,
        trajectory: Optional[Trajectory] = None,
        ref: Optional[str] = None,
    ):
        super().__init__(catalog_entry, trajectory=trajectory, ref=ref)
