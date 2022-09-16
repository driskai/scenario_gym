from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    """A bounding box defined by its length, width and center."""

    width: float
    length: float
    center_x: float
    center_y: float


@dataclass
class CatalogEntry:
    """
    A single catalog entry. Holds catalog information and a bounding box.

    Parameters
    ----------
    catalog_name : str
        The name of the catalog file.

    catalog_entry : str
        The name of the specific entry.

    catalog_category : Optional[str]
        The category of the entry.

    catalog_type : str
        The catalog type e.g Vehicle or Pedestrian.

    """

    catalog_name: str
    catalog_entry: str
    catalog_category: Optional[str]
    catalog_type: str
    bounding_box: BoundingBox
