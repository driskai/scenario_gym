import numpy as np

from scenario_gym import ScenarioGym
from scenario_gym.agent import PIDAgent


def test_pid_controller(all_scenarios):
    """Test the PID controller."""
    scenario_path = all_scenarios["a98d5c7d-76aa-49bf-b88c-97db5d5c7433"]
    gym = ScenarioGym(timestep=0.1)

    def create_agent(s, e):
        if e.ref == "ego":
            return PIDAgent(
                e,
                accel_Kp=2.0,
                max_accel=5.0,
                max_steer=np.pi / 90,
            )

    gym.load_scenario(
        scenario_path,
        create_agent=create_agent,
    )
    gym.rollout(render=False)
