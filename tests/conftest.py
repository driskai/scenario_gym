import importlib
import os
import warnings
from pathlib import Path
from typing import Callable, Dict

import pytest as pt


def pytest_addoption(parser):
    """Add an option to run speed tests."""
    parser.addoption(
        "--speed_tests",
        action="store_true",
        dest="speed_tests",
        help="Enable tests in test_speeds.py",
    )


@pt.fixture(scope="session")
def all_scenarios():
    """Get all scenarios in the module as a dictionary."""
    scenarios = [
        "1518e754-318f-4847-8a30-2dce552b4504",
        "3071b41f-903f-4465-a5bb-77262f2aa08a",
        "35915513-77b7-42bb-be37-68c28a5263f3",
        "379d4431-cadb-4401-8f74-2b474c67ccb2",
        "3e39a079-5653-440c-bcbe-24dc9f6bf0e6",
        "3fee6507-fd24-432f-b781-ca5676c834ef",
        "41dac6fa-6f83-461e-a145-08692da5f3c7",
        "511c3b32-1bd5-4c16-9649-4c92da443301",
        "682cd54d-b714-46ed-99d9-18937a181d88",
        "87219e81-085c-4183-b4de-deb3f746cbce",
        "9c324146-be03-4d4e-8112-eaf36af15c17",
        "a2281876-e0b4-4048-a08a-1ce69f94c085",
        "a5e43fe4-646a-49ba-82ce-5f0063776566",
        "a98d5c7d-76aa-49bf-b88c-97db5d5c7433",
        "d9726503-e04a-4e8b-b487-8805ef790c92",
        "e0b8abf3-7edb-436e-a5a4-63636f83c5ab",
        "e1bdb607-206b-4f40-9bc4-59ded182ecc8",
        "e56ae853-4266-4c30-865f-96737d87b601",
        "fbb6b5ca-3fcb-4a7b-9757-b8554a753e69",
        "no_references",
        "mixed_catalogs",
    ]
    base = os.path.join(os.path.dirname(__file__), "input_files", "Scenarios")
    scenarios = {f: os.path.join(base, f + ".xosc") for f in scenarios}
    return scenarios


@pt.fixture(scope="session")
def all_road_networks():
    """Get all road networks as a dictionary."""
    base = os.path.join(
        os.path.dirname(__file__),
        "input_files",
        "Road_Networks",
    )
    road_networks = [
        "Greenwich_Road_Network_002",
        "Greenwich_Road_Network_003",
        "Roundabout_Road_Network_001",
        "Rural_Road_Network",
        "Y_Intersection_Road_Network_001",
        "dRisk Unity 6-lane Intersection",
    ]
    return {r: os.path.join(base, r + ".json") for r in road_networks}


@pt.fixture(scope="session")
def all_xodr_networks():
    """Get all OpenDRIVE road networks as a dictionary."""
    base = os.path.join(
        os.path.dirname(__file__),
        "input_files",
        "Road_Networks",
    )
    road_networks = []
    return {r: os.path.join(base, r + ".xodr") for r in road_networks}


@pt.fixture(scope="session")
def all_catalogs():
    """Get all OpenDRIVE road networks as a dictionary."""
    base = os.path.join(
        os.path.dirname(__file__),
        "input_files",
        "Catalogs",
    )
    catalogs = [
        "Scenario_Gym/PedestrianCatalogs/ScenarioGymPedestrianCatalog",
        "Scenario_Gym/VehicleCatalogs/ScenarioGymVehicleCatalog",
        "Custom_Catalog/MiscCatalogs/CustomCatalog",
    ]
    return {r: os.path.join(base, r + ".xosc") for r in catalogs}


def import_all_fixtures() -> Dict[str, Callable]:
    """
    Import all fixtures from any submodule named fixtures.

    Returns a dict of functions which have the __pytest_wrapped__ attribute. These
    can be added to the global scope to allow importing from other modules.
    """
    modules = []
    base = Path(__file__).parent
    for path in base.rglob("*/fixtures.py"):
        if path.is_file():
            rel_path = path.relative_to(base)
            mod_path = str(Path(rel_path.parent, rel_path.stem)).replace("/", ".")
            modules.append(mod_path)

    fixtures = {
        k: v for k, v in globals().items() if hasattr(v, "__pytest_wrapped__")
    }
    for mod in modules:
        module = importlib.import_module(f"tests.{mod}")
        for name, obj in module.__dict__.items():
            if hasattr(obj, "__pytest_wrapped__"):
                if name in fixtures and fixtures[name].__module__ != obj.__module__:
                    warnings.warn(
                        f"Duplicate fixtures found: {name} in modules "
                        f"{obj.__module__} and {fixtures[name].__module__}."
                    )
                fixtures[name] = obj
    return fixtures


globals().update(import_all_fixtures())
