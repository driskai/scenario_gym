import numpy as np
import pytest as pt

from scenario_gym import Trajectory
from scenario_gym.entity.base import StaticEntity
from scenario_gym.xosc_interface import import_scenario


def test_static_entity(all_scenarios):
    """Test creating a static entity."""
    scenario = import_scenario(
        all_scenarios["3fee6507-fd24-432f-b781-ca5676c834ef"]
    )
    e = scenario.entities[0]

    e_static = StaticEntity(e.catalog_entry, ref="static_ent")
    e_static.trajectory = Trajectory(
        np.array([[0.0, 1.0, 2.0]]),
        fields=["t", "x", "y"],
    )

    with pt.raises(ValueError):
        e_static.trajectory = Trajectory(
            np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
            fields=["t", "x", "y"],
        )
