from dataclasses import dataclass
from typing import Optional

from lxml.etree import Element

from scenario_gym.catalog_entry import ArgsKwargs, CatalogEntry, CatalogObject
from scenario_gym.entity.base import Entity


@dataclass
class Axle(CatalogObject):
    """A front or rear axle of a vehicle."""

    max_steering: float
    wheel_diameter: float
    track_width: float
    position_x: float
    position_z: float

    @classmethod
    def load_data_from_xml(cls, catalog_name: str, element: Element) -> ArgsKwargs:
        """Load the bounding box data form an xml element."""
        return (
            float(element.attrib["maxSteering"]),
            float(element.attrib["wheelDiameter"]),
            float(element.attrib["trackWidth"]),
            float(element.attrib["positionX"]),
            float(element.attrib["positionZ"]),
        ), {}


@dataclass
class VehicleCatalogEntry(CatalogEntry):
    """Catalog entry for a vehicle."""

    mass: float
    max_speed: float
    max_deceleration: float
    max_acceleration: float
    front_axle: Axle
    rear_axle: Axle

    xosc_names = ["Vehicle"]

    @classmethod
    def load_data_from_xml(cls, catalog_name: str, element: Element) -> ArgsKwargs:
        """Load the vehicle from an xml element."""
        base_args, _ = super().load_data_from_xml(catalog_name, element)
        performance = element.find("Performance")
        front_axle = element.find("Axles/FrontAxle")
        rear_axle = element.find("Axles/RearAxle")
        veh_args = (
            float(element.attrib["mass"]),
            float(performance.attrib["maxSpeed"]),
            float(performance.attrib["maxDeceleration"]),
            float(performance.attrib["maxAcceleration"]),
            Axle.from_xml(catalog_name, front_axle),
            Axle.from_xml(catalog_name, rear_axle),
        )
        return base_args + veh_args, {}


class Vehicle(Entity):
    """Class for vehicles."""

    def __init__(
        self,
        catalog_entry: VehicleCatalogEntry,
        ref: Optional[str] = None,
    ):
        super().__init__(catalog_entry, ref=ref)
