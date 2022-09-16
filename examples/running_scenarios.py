"""Running scenarios in the gym."""
import os

from scenario_gym.scenario_gym import ScenarioGym


def run_scenario():
    """Define the gym and run a scenario."""

    base = os.path.join(
        os.path.dirname(__file__),
        "../tests/input_files/Scenarios/",
    )
    s = "3fee6507-fd24-432f-b781-ca5676c834ef.xosc"
    filepath = os.path.join(base, s)

    gym = ScenarioGym(
        timestep=0.1,  # seconds per timestep
        terminal_conditions=["max_length"],  # end when the scenario ends
    )

    # load scenario from its xosc file
    gym.load_scenario(filepath)

    # run the scenario
    gym.rollout()

    # this is essentially the same as calling
    gym.reset_scenario()
    while not gym.state.is_done:
        gym.step()

    # but with rollout it is easier to render or record the output
    gym.rollout(render=True)
    print(f"Recording saved to {filepath.replace('Scenarios', 'Recordings')}.")


if __name__ == "__main__":
    run_scenario()
