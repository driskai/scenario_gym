from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Type

from lxml import etree
from lxml.etree import Element

from scenario_gym.catalog_entry import Catalog, CatalogEntry
from scenario_gym.entity import Entity, Pedestrian, Vehicle

DEFAULT_ENTITY_TYPES = (Vehicle, Pedestrian)


def load_object(
    catalog: Catalog,
    entry: Element,
    entity_types: List[Type[Entity]],
    catalog_objects: List[Type[CatalogEntry]],
) -> Optional[Entity]:
    """Try to load a catalog entry with given catalog objects."""
    for Ent, Obj in zip(entity_types, catalog_objects):
        types = Obj.xosc_names if Obj.xosc_names is not None else [Obj.__name__]
        if entry.tag in types:
            obj = Obj.from_xml(catalog, entry)
            return Ent(obj)


@lru_cache(maxsize=None)
def read_catalog(
    catalog_file: str,
    relative_catalog_path: Optional[str] = None,
    entity_types: Optional[Tuple[Type[Entity]]] = None,
) -> Tuple[Catalog, Dict[str, Entity]]:
    """
    Read a catalog and return it's name and a dictionary of entities.

    Parameters
    ----------
    catalog_file : str
        Filepath of the catalog file.

    relative_catalog_path : Optional[str]
        Relative path used by OpenSCENARIO files when loading this catalog.

    entity_types : Optional[Tuple[Type[CatalogObject]]]
        Tuple of extra subclasses of CatalogObject that will be used when reading
        catalogs. Can be used for reading custom objects from catalog files. Must
        be immutable or lru_cache

    Returns
    -------
    catalog : Catalog
        The catalog object.

    entries : Dict[str, Entity]
        A dictionary mapping each entry name to an entity with that catalog entry.
        These can then be cloned to create entities using the chosen catalog entry.

    """
    if entity_types is None:
        entity_types = DEFAULT_ENTITY_TYPES
    else:
        entity_types = entity_types + DEFAULT_ENTITY_TYPES

    catalog_objects = [Ent._catalog_entry_type() for Ent in entity_types]

    et = etree.parse(catalog_file)
    osc_root = et.getroot()
    catalog_element = osc_root.find("Catalog")
    catalog_name = catalog_element.attrib["name"]
    catalog = Catalog(catalog_name, relative_catalog_path)
    entries = {}
    for element in catalog_element.getchildren():
        entry = load_object(catalog, element, entity_types, catalog_objects)
        if entry is None:
            entry = Entity(CatalogEntry.from_xml(catalog, element))
        entries[entry.catalog_entry.catalog_entry] = entry
    return catalog, entries
