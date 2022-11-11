from dataclasses import dataclass
from typing import Optional

from lxml.etree import Element

from scenario_gym.catalog_entry import ArgsKwargs, CatalogEntry
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


class Pedestrian(Entity):
    """Entity class for pedestrians."""

    def __init__(
        self,
        catalog_entry: PedestrianCatalogEntry,
        trajectory: Optional[Trajectory] = None,
        ref: Optional[str] = None,
    ):
        super().__init__(catalog_entry, trajectory=trajectory, ref=ref)
