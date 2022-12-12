from copy import copy

import numpy as np
import pytest as pt

from scenario_gym import Trajectory
from scenario_gym.entity.base import Entity, StaticEntity
from scenario_gym.xosc_interface import import_scenario


@pt.fixture(scope="module")
def example_catalog_entry(all_scenarios):
    """Get an example catalog entry for use in entity tests."""
    scenario = import_scenario(
        all_scenarios["3fee6507-fd24-432f-b781-ca5676c834ef"]
    )
    return scenario.entities[0].catalog_entry


def test_copy_entity(example_catalog_entry):
    """Test deepcopying an entity."""
    e = Entity(
        example_catalog_entry,
        trajectory=Trajectory(
            np.array([[0.0, 1.0, 2.0]]),
            fields=["t", "x", "y"],
        ),
        ref="example",
    )
    e_new = copy(e)
    assert id(e_new.catalog_entry) == id(
        e.catalog_entry
    ), "Catalog entry should be the same."
    assert (id(e_new.trajectory) != id(e.trajectory)) and np.allclose(
        e_new.trajectory.data, e.trajectory.data
    ), "Trajectories should be the equal but different memory."


def test_static_entity(example_catalog_entry):
    """Test creating a static entity."""
    e_static = StaticEntity(example_catalog_entry, ref="static_ent")
    e_static.trajectory = Trajectory(
        np.array([[0.0, 1.0, 2.0]]),
        fields=["t", "x", "y"],
    )

    with pt.raises(ValueError):
        e_static.trajectory = Trajectory(
            np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
            fields=["t", "x", "y"],
        )
