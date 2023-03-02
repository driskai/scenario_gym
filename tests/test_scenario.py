import os
import pickle
from copy import deepcopy
from tempfile import TemporaryDirectory, TemporaryFile

import numpy as np
import pytest as pt

from scenario_gym.road_network import RoadNetwork
from scenario_gym.scenario import Scenario
from scenario_gym.scenario.actions import UpdateStateVariableAction
from scenario_gym.xosc_interface import import_scenario


@pt.fixture
def example_scenario(all_scenarios):
    """Get a scenario to test."""
    s = import_scenario(all_scenarios["3e39a079-5653-440c-bcbe-24dc9f6bf0e6"])
    action = UpdateStateVariableAction(2.0, "TestAction", "ego", {"var": 1.0})
    s.add_action(action, inplace=True)
    return s


@pt.fixture
def example_entity(example_scenario):
    """Get an entity to use as an example."""
    return deepcopy(example_scenario.entities[0])


def test_ego(example_scenario):
    """Test getting the ego entity."""
    assert example_scenario.ego is not None, "Ego entity should not be None."
    ego = example_scenario.entity_by_name("ego")
    first = example_scenario.entities[0]
    assert (
        ego == first == example_scenario.ego
    ), "Here the ego entity should be first and named 'ego'."

    other = [e.copy() for e in example_scenario.entities]
    other[0].ref = "not_ego"
    s = Scenario(other)
    assert s.ego == s.entities[0], "Ego should still be the first entity."

    other[1].ref = "ego"
    s2 = Scenario(other)
    assert s2.ego == s.entities[1], "Ego should now be the second entity."


def test_length(example_scenario):
    """Test the length of the scenario is computed correctly."""
    l = max((e.trajectory.max_t for e in example_scenario.entities))
    assert example_scenario.length == l, "Incorrect length."


def test_deepcopy_scenario(example_scenario):
    """Test deepcopying a scenario."""
    s_new = deepcopy(example_scenario)
    assert len(s_new.entities) == len(example_scenario.entities)
    assert id(s_new.road_network) != id(
        example_scenario.road_network
    ), "Road networks should be different."


def test_copy_scenarios(example_entity):
    """Test copying scenario."""
    s = Scenario(
        [example_entity],
        road_network=RoadNetwork(),
        properties={"x": 1, "y": 2},
    )
    s_new = s.copy()
    assert id(s_new.road_network) == id(
        s.road_network
    ), "Road networks should be the same."
    assert s_new.properties == s.properties, "Properties should be the same."


def test_add_entity(example_scenario, example_entity):
    """Test adding an entity to a scenario."""
    s = example_scenario.copy()
    new_entity = example_entity.copy()
    new_entity.ref = "new_entity"
    s.add_entity(new_entity, inplace=True)
    assert len(s.entities) == len(example_scenario.entities) + 1
    assert new_entity in s.entities, "Entity should be in the scenario."
    assert s.entity_by_name("new_entity") == new_entity

    s2 = example_scenario.add_entity(new_entity, inplace=False)
    assert len(s2.entities) == len(example_scenario.entities) + 1
    assert new_entity in s2.entities, "Entity should be in the scenario."
    assert s2.entity_by_name("new_entity") == new_entity
    assert new_entity not in example_scenario.entities


def test_remove_entity(example_scenario):
    """Test removing an entity from a scenario."""
    s = example_scenario.copy()
    e = s.entities[-1]
    s.remove_entity(e, inplace=True)
    assert len(s.entities) == len(example_scenario.entities) - 1
    assert e not in s.entities, "Entity should not be in the scenario."

    s2 = example_scenario.copy()
    e = s2.entities[-1]
    s3 = s2.remove_entity(e, inplace=False)
    assert len(s3.entities) == len(s2.entities) - 1
    assert e in s2.entities, "Entity should be in the scenario."
    assert e not in s3.entities, "Entity should not be in the new scenario."


def test_describe(example_scenario):
    """Test the `describe` method."""
    example_scenario.describe()


def test_plot(example_scenario):
    """Test the `describe` method."""
    example_scenario.plot(show=False)


def test_translate(example_scenario):
    """Test the translating a scenario."""
    shift = np.arange(7) + 1
    new_scenario = example_scenario.translate(shift)
    ts = np.linspace(0.0, example_scenario.length, 10)
    for e, e_new in zip(example_scenario.entities, new_scenario.entities):
        ps = e.trajectory.position_at_t(ts)
        ps_new = e_new.trajectory.position_at_t(ts + shift[0])
        deltas = ps_new - ps
        assert np.allclose(deltas, shift[None, 1:])

    a_new = new_scenario.actions[0]
    a_old = example_scenario.actions[0]
    assert a_new.t == a_old.t + shift[0]


def test_to_dict(example_scenario, all_scenarios):
    """Test writing and reading the scenario from a dictionary."""
    base_dir = os.path.join(
        os.path.dirname(all_scenarios[example_scenario.name]),
        "../Road_Networks",
    )
    data = example_scenario.to_dict(road_network_path=base_dir)
    s2 = Scenario.from_dict(data)
    assert (
        example_scenario.road_network.name == s2.road_network.name
    ), "Road networks should have same name."
    assert all(
        e.ref == e2.ref for e, e2 in zip(example_scenario.entities, s2.entities)
    ), "Entities should be in the same order."
    ego, ego2 = example_scenario.entities[0], s2.entities[0]
    assert (
        ego.trajectory.data == ego2.trajectory.data
    ).all(), "Ego trajectories should be the same."
    assert ego.bounding_box == ego2.bounding_box


def test_jsonable(example_scenario):
    """Test reading and writing scenarios from json."""
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "example.json")
        example_scenario.to_json(path, road_network_path=None)
        s2 = Scenario.from_json(path)

    assert (
        example_scenario.road_network.name == s2.road_network.name
    ), "Road networks should have same name."
    assert all(
        e.ref == e2.ref for e, e2 in zip(example_scenario.entities, s2.entities)
    ), "Entities should be in the same order."
    ego, ego2 = example_scenario.entities[0], s2.entities[0]
    assert (
        ego.trajectory.data == ego2.trajectory.data
    ).all(), "Ego trajectories should be the same."


@pt.fixture()
def scenario_with_none_values(example_scenario):
    """Create a scenario with None values in one of the catalog entries."""
    s = deepcopy(example_scenario)
    s.entities[0].catalog_entry.front_axle.max_steering = None
    s.entities[0].catalog_entry.reat_axle = None
    return s


def test_jsonable_with_none(scenario_with_none_values):
    """Test reading and writing scenarios from json."""
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "example.json")
        scenario_with_none_values.to_json(path, road_network_path=None)
        s2 = Scenario.from_json(path)
    assert (
        s2.entities[0].catalog_entry.front_axle.max_steering is None
    ), "The max steering should be None."
    assert (
        scenario_with_none_values.road_network.name == s2.road_network.name
    ), "Road networks should have same name."
    assert all(
        e.ref == e2.ref
        for e, e2 in zip(
            scenario_with_none_values.entities,
            s2.entities,
        )
    ), "Entities should be in the same order."
    ego, ego2 = scenario_with_none_values.entities[0], s2.entities[0]
    assert (
        ego.trajectory.data == ego2.trajectory.data
    ).all(), "Ego trajectories should be the same."


def test_pickle_scenario(example_scenario):
    """Test pickling a scenario."""
    with TemporaryFile() as f:
        pickle.dump(example_scenario, f)
        f.seek(0)
        new_scenario = pickle.load(f)
    assert all(
        e_new.ref == e.ref
        for e_new, e in zip(new_scenario.entities, example_scenario.entities)
    ), "Different entities"
    assert new_scenario.length == example_scenario.length, "Different length."
    assert (
        new_scenario.road_network.name == example_scenario.road_network.name
    ), "Different road networks."
