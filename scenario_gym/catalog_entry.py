from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from lxml.etree import Element
from scenariogeneration import xosc

from scenario_gym.utils import ArgsKwargs, load_properties_from_xml


@dataclass(frozen=True)
class Catalog:
    """A catalog for catalog entries."""

    name: str
    group_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the catalog from a dictionary."""
        return cls(data["name"], data["group_name"])

    def to_dict(self) -> Dict[str, Any]:
        """Write the catalog to a dictionary."""
        return {"name": self.name, "group_name": self.group_name}


class CatalogObject(ABC):
    """
    Base class for objects loaded from catalogs.

    Subclasses should implement the `load_data_from_xml` class method wwhich takes
    the specific xml element that contains the data and returns the arguements
    and keyword arguments for the class constructor. It should not return the
    class itself since this way the methods can make use of loading methods from
    parent classes.

    The attribute xosc_namex can be set to the element names that the object
    represents. For example, if `xosc_names = ["Vehicle"]` then any elements with
    the tag `Vehicle` will be loaded by this entry. If not set then the class name
    will be used.
    """

    xosc_names: Optional[List[str]] = None

    @classmethod
    def from_xml(cls, element: Element, catalog: Optional[Catalog] = None):
        """Create the class from an xml element."""
        args, kwargs = cls.load_data_from_xml(element, catalog=catalog)
        return cls(*args, **kwargs)

    @abstractclassmethod
    def load_data_from_xml(
        cls,
        element: Element,
        catalog: Optional[Catalog] = None,
    ) -> ArgsKwargs:
        """Load the object from an xml element."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Create the object from a dictionary.

        Must be implemented to allow json serialization.
        """
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """
        Write the object to a dictionary.

        Must be implemented to allow json serialization.
        """
        raise NotImplementedError

    def to_xosc(self) -> xosc.VersionBase:
        """Write the object to an xosc object."""
        raise NotImplementedError


@dataclass
class BoundingBox(CatalogObject):
    """A bounding box defined by its length, width and center."""

    width: float
    length: float
    center_x: float
    center_y: float

    @classmethod
    def load_data_from_xml(
        cls,
        element: Element,
        catalog: Optional[Catalog] = None,
    ) -> ArgsKwargs:
        """Load the bounding box data form an xml element."""
        if element.tag != "BoundingBox":
            raise TypeError(f"Expected BoundingBox element not {element.tag}.")
        bb_center = element.find("Center")
        bb_dimensions = element.find("Dimensions")
        return (
            float(bb_dimensions.attrib["width"]),
            float(bb_dimensions.attrib["length"]),
            float(bb_center.attrib["x"]),
            float(bb_center.attrib["y"]),
        ), {}

    @classmethod
    def from_dict(cls, data: Dict[str, float]):
        """Load the bounding box from a dictionary."""
        return cls(
            data["width"],
            data["length"],
            data["center_x"],
            data["center_y"],
        )

    def to_dict(self) -> Dict[str, float]:
        """Write the bounding box to a jsonable dictionary."""
        return {
            "width": self.width,
            "length": self.length,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }

    def to_xosc(self) -> xosc.BoundingBox:
        """Write the bounding box to an xosc bounding box."""
        return xosc.BoundingBox(
            self.width,
            self.length,
            0.0,
            self.center_x,
            self.center_y,
            0.0,
        )


@dataclass
class CatalogEntry(CatalogObject):
    """
    A single catalog entry. Holds catalog information and a bounding box.

    Parameters
    ----------
    catalog : Optional[Catalog]
        The catalog from which the entry is loaded.

    catalog_entry : str
        The name of the specific entry.

    catalog_category : Optional[str]
        The category of the entry.

    catalog_type : str
        The catalog type e.g Vehicle or Pedestrian.

    bounding_box : BoundingBox
        The bounding box of the entry.

    properties : Dict[str, Union[float, str]]
        Any properties associated with the element.

    files: List[str]
        A list of filepaths for external files.

    """

    catalog: Optional[Catalog]
    catalog_entry: str
    catalog_category: Optional[str]
    catalog_type: str
    bounding_box: BoundingBox
    properties: Dict[str, Union[float, str]]
    files: List[str]

    @classmethod
    def load_data_from_xml(
        cls,
        element: Element,
        catalog: Optional[Catalog] = None,
    ) -> ArgsKwargs:
        """Load the catalog entry from an xml element."""
        entry_name = element.attrib["name"]
        cname = element.tag.lower() + "Category"
        category = element.attrib[cname] if cname in element.attrib else None
        bb = element.find("BoundingBox")
        bb = BoundingBox.from_xml(bb, catalog=catalog)
        properties, files = load_properties_from_xml(element)
        return (
            catalog,
            entry_name,
            category,
            element.tag,
            bb,
            properties,
            files,
        ), {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Load the catalog entry from a dictionary."""
        catalog = data.get("catalog", None)
        if catalog is not None:
            catalog = Catalog.from_dict(catalog)
        return cls(
            catalog,
            data["catalog_entry"],
            data["catalog_category"],
            data["catalog_type"],
            BoundingBox.from_dict(data["bounding_box"]),
            data.get("properties", {}),
            data.get("files", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Write the catalog entry to a dictionary."""
        return {
            "catalog": self.catalog.to_dict() if self.catalog else None,
            "catalog_entry": self.catalog_entry,
            "catalog_category": self.catalog_category,
            "catalog_type": self.catalog_type,
            "bounding_box": self.bounding_box.to_dict(),
            "properties": self.properties,
            "files": self.files,
        }

    def to_xosc(self) -> xosc.VersionBase:
        """Create an xosc entity object from the catalog entry."""
        obj = xosc.MiscObject(
            self.catalog_entry,
            1.0,
            getattr(
                xosc.MiscObjectCategory,
                self.catalog_category,
                xosc.MiscObjectCategory.none,
            ),
            self.catalog_category,
            self.bounding_box.to_xosc(),
        )
        for k, v in self.properties.items():
            obj.add_property(k, v)
        for f in self.files:
            obj.add_property_file(f)
        return obj
