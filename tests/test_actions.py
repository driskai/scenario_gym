import pytest as pt

from scenario_gym.scenario.actions import UserDefinedAction
from scenario_gym.xosc_interface import import_scenario


@pt.fixture(scope="module")
def scenario(all_scenarios):
    """Get a scenario with actions."""
    path = all_scenarios["1518e754-318f-4847-8a30-2dce552b4504"]
    return import_scenario(path)


def test_user_actions(scenario):
    """Test getting user actions."""
    assert scenario.actions, "There should be actions."

    act = scenario.actions[0]
    assert isinstance(act, UserDefinedAction), "Incorrect action type."
    assert act.action_class == "CustomCommandAction", "Incorrect action class."
    assert act.entity_ref == "ego", "Incorrect entity ref."
    assert act.action_variables == {"type": "turn_right"}, "Incorrect variables."
    assert act.t == 5.0, "Incorrect time."
