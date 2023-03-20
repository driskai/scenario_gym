import numpy as np
import pytest as pt

from scenario_gym.catalog_entry import BoundingBox, CatalogEntry
from scenario_gym.entity import Entity
from scenario_gym.scenario import Scenario
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.state import detect_collisions
from scenario_gym.trajectory import Trajectory


@pt.fixture
def collision_scenario():
    """Create a scenario with two entities that collide."""
    box = BoundingBox(2.0, 5.0, 0.0, 0.0)
    ce = CatalogEntry("car", "car", "car", "car", box, {}, [])
    ego = Entity(ce, ref="ego")
    hazard = Entity(ce, ref="entity_1")

    ego.trajectory = Trajectory(
        np.array(
            [
                [0.0, 0, 0],
                [10, 20, 0],
            ]
        ),
        fields=["t", "x", "y"],
    )
    hazard.trajectory = Trajectory(
        np.array(
            [
                [0.0, 40, 0],
                [10, 20, 0],
            ]
        ),
        fields=["t", "x", "y"],
    )

    scenario = Scenario([ego, hazard])
    return scenario, ego, hazard


def test_detect_collisions(collision_scenario):
    """Test collisions are correctly detected."""
    s, ego, hazard = collision_scenario
    gym = ScenarioGym()
    gym.set_scenario(s)

    collisions = detect_collisions(
        {ego: gym.state.poses[ego]},
        others={hazard: gym.state.poses[hazard]},
    )
    assert not collisions[ego], "No collision at start of scenario"

    gym.rollout()

    collisions = detect_collisions(
        {ego: gym.state.poses[ego]},
        others={hazard: gym.state.poses[hazard]},
    )
    assert collisions[ego], "Collision at end of scenario not found."
