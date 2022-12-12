from abc import ABC, abstractclassmethod
from contextlib import suppress
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from lxml.etree import Element

from scenario_gym.utils import ArgsKwargs


@dataclass(frozen=True)
class Catalog:
    """A catalog."""

    catalog_name: str
    rel_path: str


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
    def from_xml(cls, catalog: Catalog, element: Element):
        """Create the class from an xml element."""
        args, kwargs = cls.load_data_from_xml(catalog, element)
        return cls(*args, **kwargs)

    @abstractclassmethod
    def load_data_from_xml(cls, catalog: Catalog, element: Element) -> ArgsKwargs:
        """Load the object from an xml element."""
        raise NotImplementedError


@dataclass
class BoundingBox(CatalogObject):
    """A bounding box defined by its length, width and center."""

    width: float
    length: float
    center_x: float
    center_y: float

    @classmethod
    def load_data_from_xml(cls, catalog: Catalog, element: Element) -> ArgsKwargs:
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


@dataclass
class CatalogEntry(CatalogObject):
    """
    A single catalog entry. Holds catalog information and a bounding box.

    Parameters
    ----------
    catalog : Catalog
        The catalog from which the entry is loaded.

    catalog_entry : str
        The name of the specific entry.

    catalog_category : Optional[str]
        The category of the entry.

    catalog_type : str
        The catalog type e.g Vehicle or Pedestrian.

    properties : Dict[str, Union[float, str]]
        Any properties associated with the element.

    files: List[str]
        A list of filepaths for external files.

    """

    catalog: Catalog
    catalog_entry: str
    catalog_category: Optional[str]
    catalog_type: str
    bounding_box: BoundingBox
    properties: Dict[str, Union[float, str]]
    files: List[str]

    @classmethod
    def load_data_from_xml(cls, catalog: Catalog, element: Element) -> ArgsKwargs:
        """Load the catalog entry from an xml element."""
        entry_name = element.attrib["name"]
        cname = element.tag.lower() + "Category"
        category = element.attrib[cname] if cname in element.attrib else None
        bb = element.find("BoundingBox")
        bb = BoundingBox.from_xml(catalog, bb)
        properties, files = cls.load_properties_from_xml(element)
        return (
            catalog,
            entry_name,
            category,
            element.tag,
            bb,
            properties,
            files,
        ), {}

    @staticmethod
    def load_properties_from_xml(
        element: Element,
    ) -> Tuple[Dict[str, Union[str, float]], List[str]]:
        """
        Load properties from the xml element.

        These can be either `Property` or `File` elements. `Property` elements are
        given with `name` and `value` attributes (`name` must be unique for the
        entry) and are returned as a dict of values indexed by `name`. The value
        will be converted to a float if it can otherwise the string will be
        returned. `File` elements must have a `filepath` attribute which will
        be parsed. Multiple files can be stored with one entity.
        """
        files = []
        properties = {}
        prop = element.find("Properties")
        if prop is not None:
            for child in prop.findall("Property"):
                try:
                    v = child.attrib["value"]
                    with suppress(ValueError):
                        v = float(v)
                    properties[child.attrib["name"]] = v
                except KeyError:
                    raise RuntimeError(
                        "Property could not be loaded without `value` key."
                    )
            for file in prop.findall("File"):
                files.append(file.attrib["filepath"])
        return properties, files

    @property
    def catalog_name(self) -> str:
        """Get the name of the catalog file."""
        return self.catalog.catalog_name
