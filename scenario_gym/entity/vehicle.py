from dataclasses import dataclass
from typing import Any, Dict, Optional

from lxml.etree import Element

from scenario_gym.catalog_entry import (
    ArgsKwargs,
    BoundingBox,
    Catalog,
    CatalogEntry,
    CatalogObject,
)
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

    def to_dict(self) -> None:
        """Write the vehicle catalog entry to a dictionary."""
        return {
            "max_steering": self.max_steering,
            "wheel_diameter": self.wheel_diameter,
            "track_width": self.track_width,
            "position_x": self.position_x,
            "position_z": self.position_z,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the vehicle catalog entry from a dictionary."""
        return cls(
            data.get("max_steering"),
            data.get("wheel_diameter"),
            data.get("track_width"),
            data.get("position_x"),
            data.get("position_z"),
        )


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the vehicle from a dictionary."""
        return cls(
            Catalog(data["catalog"]["catalog_name"], data["catalog"]["rel_path"]),
            data["catalog_entry"],
            data["catalog_category"],
            data["catalog_type"],
            BoundingBox.from_dict(data["bounding_box"]),
            data.get("properties", {}),
            data.get("files", []),
            data.get("mass"),
            data.get("max_speed"),
            data.get("max_deceleration"),
            data.get("max_acceleration"),
            Axle.from_dict(data["front_axle"])
            if data.get("front_axle") is not None
            else None,
            Axle.from_dict(data["rear_axle"])
            if data.get("rear_axle") is not None
            else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Write the scenario to a dictionary."""
        data = super().to_dict()
        data.update(
            {
                "mass": self.mass,
                "max_speed": self.max_speed,
                "max_deceleration": self.max_deceleration,
                "max_acceleration": self.max_acceleration,
                "front_axle": self.front_axle.to_dict()
                if self.front_axle is not None
                else None,
                "rear_axle": self.rear_axle.to_dict()
                if self.rear_axle is not None
                else None,
            }
        )
        return data


class Vehicle(Entity):
    """Class for vehicles."""

    def __init__(
        self,
        catalog_entry: VehicleCatalogEntry,
        trajectory: Optional[Trajectory] = None,
        ref: Optional[str] = None,
    ):
        super().__init__(catalog_entry, trajectory=trajectory, ref=ref)
