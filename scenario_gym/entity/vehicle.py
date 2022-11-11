from dataclasses import dataclass
from typing import Optional

from lxml.etree import Element

from scenario_gym.catalog_entry import ArgsKwargs, CatalogEntry, CatalogObject
from scenario_gym.entity.base import Entity
from scenario_gym.trajectory import Trajectory


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

    mass: Optional[float]
    max_speed: Optional[float]
    max_deceleration: Optional[float]
    max_acceleration: Optional[float]
    front_axle: Optional[Axle]
    rear_axle: Optional[Axle]

    xosc_names = ["Vehicle"]

    @classmethod
    def load_data_from_xml(cls, catalog_name: str, element: Element) -> ArgsKwargs:
        """Load the vehicle from an xml element."""
        base_args, _ = super().load_data_from_xml(catalog_name, element)
        performance = element.find("Performance")
        front_axle = element.find("Axles/FrontAxle")
        rear_axle = element.find("Axles/RearAxle")
        mass = float(element.attrib["mass"]) if "mass" in element.attrib else None
        if performance is not None:
            max_speed = float(performance.attrib["maxSpeed"])
            max_dec = float(performance.attrib["maxDeceleration"])
            max_acc = float(performance.attrib["maxAcceleration"])
        else:
            max_speed = max_dec = max_acc = None
        front_axle = (
            Axle.from_xml(catalog_name, front_axle)
            if front_axle is not None
            else None
        )
        rear_axle = (
            Axle.from_xml(catalog_name, rear_axle)
            if rear_axle is not None
            else None
        )
        veh_args = (
            mass,
            max_dec,
            max_acc,
            max_speed,
            front_axle,
            rear_axle,
        )
        return base_args + veh_args, {}


class Vehicle(Entity):
    """Class for vehicles."""

    def __init__(
        self,
        catalog_entry: VehicleCatalogEntry,
        trajectory: Optional[Trajectory] = None,
        ref: Optional[str] = None,
    ):
        super().__init__(catalog_entry, trajectory=trajectory, ref=ref)
