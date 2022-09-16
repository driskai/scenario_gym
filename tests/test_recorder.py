import numpy as np

from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.trajectory import Trajectory


def test_recorder() -> None:
    """
    Rollout a single scenario with recording enabled.

    Output the xosc then load it again and rollout the
    recorded version.

    """
    scenario_path = (
        "./tests/input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc"
    )
    out_path = scenario_path.replace("Scenarios", "Recordings").replace(
        ".xosc", "_test.xosc"
    )

    # rollout and record
    gym = ScenarioGym()
    gym.record()
    gym.load_scenario(scenario_path)
    gym.rollout()
    traj1 = gym.state.scenario.entities[0].trajectory

    gym.recorder.write_xml(out_path=out_path)

    # reload and test
    n_entities = len(gym.state.scenario.entities)
    n_stationary = sum(
        1 for t in gym.state.scenario.trajectories.values() if len(t) == 1
    )
    gym.load_scenario(out_path)
    traj2 = gym.state.scenario.entities[0].trajectory

    assert (
        len(gym.state.scenario.entities) == n_entities
    ), "New scenario has a different number of entities."
    assert n_stationary == sum(
        1 for t in gym.state.scenario.trajectories.values() if len(t) == 1
    ), "New scenario has a different number of stationary entities."
    assert all(
        [
            np.allclose(traj1.position_at_t(0.0), traj2.position_at_t(0.0)),
            np.allclose(traj1.position_at_t(5.0), traj2.position_at_t(5.0)),
            np.allclose(traj1.position_at_t(10.0), traj2.position_at_t(10.0)),
        ]
    ), "Recorded and true trajectories differ."

    ego = gym.state.scenario.entities[0]
    assert len(ego.recorded_poses) > 0, "Ego should have recorded poses."
    gym.reset_scenario()
    assert len(ego.recorded_poses) == 2, "Ego should have reset recorded poses."
    gym.rollout()
    assert len(ego.recorded_poses) > 0, "Ego should have recorded poses."
    gym.record(close=True)
    assert (len(ego.recorded_poses) == 0) and not (
        ego.record_trajectory
    ), "Ego should have no recorded poses."
    gym.rollout()
    assert (len(ego.recorded_poses) == 0) and not (
        ego.record_trajectory
    ), "Ego should have no recorded poses."


def test_recorder_with_resetting() -> None:
    """Test the recording behaviour when resetting, closing etc."""
    scenario_path = (
        "./tests/input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc"
    )
    gym = ScenarioGym()
    gym.record()

    # test always recording when resetting or reloading while in recording mode
    assert gym.is_recording, "Gym should be recording."
    gym.load_scenario(scenario_path)
    assert all(
        (e.record_trajectory for e in gym.state.scenario.entities)
    ), "Entities are not recording."
    gym.reset_scenario()
    assert all(
        (e.record_trajectory for e in gym.state.scenario.entities)
    ), "Entities are not recording."
    gym.load_scenario(scenario_path)
    assert all(
        (e.record_trajectory for e in gym.state.scenario.entities)
    ), "Entities are not recording."
    assert gym.is_recording, "Gym should be recording."

    # now close and test opposite behaviour
    gym.record(close=True)
    assert not any(
        (e.record_trajectory for e in gym.state.scenario.entities)
    ), "Entities are recording."
    gym.reset_scenario()
    assert not any(
        (e.record_trajectory for e in gym.state.scenario.entities)
    ), "Entities are recording."
    gym.load_scenario(scenario_path)
    assert not any(
        (e.record_trajectory for e in gym.state.scenario.entities)
    ), "Entities are recording."
    assert not gym.is_recording, "Gym should not be recording."


def test_all_stationary() -> None:
    """Enusre the recorder works when all entities are stationary."""
    scenario_path = (
        "./tests/input_files/Scenarios/a5e43fe4-646a-49ba-82ce-5f0063776566.xosc"
    )
    gym = ScenarioGym()
    gym.record()
    gym.load_scenario(scenario_path)

    for e in gym.state.scenario.entities:
        e.trajectory = Trajectory(np.zeros((1, 7)))
    gym.reset_scenario()
    gym.recorder.get_state()
