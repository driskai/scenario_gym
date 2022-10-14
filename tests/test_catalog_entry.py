from dataclasses import dataclass
from typing import Optional

from lxml.etree import Element

from scenario_gym.catalog_entry import CatalogEntry
from scenario_gym.entity import Entity
from scenario_gym.xosc_interface import read_catalog


@dataclass
class CustomCatalogEntry(CatalogEntry):
    """Define a custom catalog entry which has an extra property."""

    mystery_property: float

    xosc_names = ["CustomObject"]

    @classmethod
    def load_data_from_xml(cls, catalog_name: str, element: Element):
        """Load from xml."""
        args, kwargs = super().load_data_from_xml(catalog_name, element)
        mystery_property = float(element.find("Mystery").attrib["value"])
        args = args + (mystery_property,)
        return args, kwargs


class CustomEntity(Entity):
    """Define a custom entity to use the custom catalog."""

    def __init__(
        self, catalog_entry: CustomCatalogEntry, ref: Optional[str] = None
    ):
        super().__init__(catalog_entry, ref=None)
        self.mystery = self.catalog_entry.mystery_property


def test_all_default_catalogs(all_catalogs):
    """Test the Scenario_Gym catalogs."""
    for catalog_file in all_catalogs.values():
        if "Scenario_Gym" in catalog_file:
            _ = read_catalog(catalog_file)


def test_custom_catalog(all_catalogs):
    """Test loading a custom catalog."""
    catalog_file = all_catalogs["Custom_Catalog/MiscCatalogs/CustomCatalog"]
    _, out = read_catalog(catalog_file, entity_types=(CustomEntity,))
    ent = out["misc_object"]
    assert ent.catalog_entry.catalog_entry == "misc_object"
    assert ent.catalog_entry.mystery_property == 100
    assert set(ent.catalog_entry.files) == set(["test.txt", "test2.txt"])
