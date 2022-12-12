"""Tune a PID controller with Bayesian Optimisation."""
import os
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from scenario_gym import Metric, ScenarioManager
from scenario_gym.agent import PIDAgent

try:
    import skopt
    from sklearn.model_selection import train_test_split
except ImportError:
    raise ImportError(
        """
`scikit-learn` and `scikit-optimize` are required for this example.
Install them with:
    ```pip install scikit-learn>=0.24.2 scikit-optimize>=0.9.0```
"""
    )


class Displacement(Metric):
    """Measure the average displacement over a scenario."""

    name = "displacement"

    def _reset(self, state):
        self.ego = state.scenario.entities[0]
        self.disp = 0.0
        self.n = 0

    def _step(self, state):
        self.disp += np.linalg.norm(
            state.poses[self.ego][:3]
            - self.ego.trajectory.position_at_t(state.t)[:3]
        )
        self.n += 1

    def get_state(self):
        return self.disp / self.n


class PIDManager(ScenarioManager):
    """Scenario manager for the experiment."""

    def __init__(
        self,
        controller_kwargs,
        **kwargs,
    ):
        super().__init__(**kwargs, metrics=[Displacement()])
        self.controller_kwargs = controller_kwargs

    def create_agent(self, s, entity):
        """
        Use a PID agent for the ego.

        See scenario_gym/agent.py for details.
        """
        if entity.ref == "ego":
            return PIDAgent(entity, **self.controller_kwargs)


def tune_parameters(
    paths: List[str],
    n_calls: int,
    x_max: float = 6.0,
    random_state: int = 0,
):
    """
    Tune the parameters of the PID controller

    Applies Bayesian optimisation to tune the parameters. The
    objective functino is the mean displacement between the
    target trajectory and the achieved trajectory over a batch
    of scenarios.

    Parameters
    ----------
    paths : List[str]
        Filepaths of the training scenarios as OpenScenario files.

    n_calls : int
        Number of times to run the optimisation loop.

    x_max : float
        Maximum value for each of the controller parameters.

    random_state : int
        Random state for the optimisation and initial params.

    """
    np.random.seed(random_state)

    # setup the manager with initial values
    manager = PIDManager(
        {
            "steer_Kp": x_max * np.random.rand(),
            "steer_Kd": x_max * np.random.rand(),
            "accel_Kp": x_max * np.random.rand(),
            "accel_Kd": x_max * np.random.rand(),
            "accel_Ki": x_max * np.random.rand(),
        }
    )

    # define the optimisastion space
    space = [
        skopt.space.Real(0.0, x_max, name="steer_Kp"),
        skopt.space.Real(0.0, x_max, name="steer_Kd"),
        skopt.space.Real(0.0, x_max, name="accel_Kp"),
        skopt.space.Real(0.0, x_max, name="accel_Kd"),
        skopt.space.Real(0.0, x_max, name="accel_Ki"),
    ]

    # define the objective function
    @skopt.utils.use_named_args(space)
    def evaluate_params(**params):
        manager.controller_kwargs.update(params)
        return np.mean(manager.run_scenarios(paths))

    # run Bayesian optimisation
    return skopt.gp_minimize(
        evaluate_params,
        space,
        n_calls=n_calls,
        verbose=True,
        random_state=random_state,
    )


def main(FLAGS):
    """Run the optimisation on the controller parameters."""

    if FLAGS.base_path is None:
        FLAGS.base_path = os.path.join(
            os.path.dirname(__file__), "..", "tests", "input_files", "Scenarios"
        )
    paths = [
        os.path.join(FLAGS.base_path, s)
        for s in os.listdir(FLAGS.base_path)
        if s.endswith("xosc")
    ]
    train, _ = train_test_split(paths, train_size=0.7, random_state=0)
    result = tune_parameters(
        train, FLAGS.n_calls, x_max=10.0, random_state=FLAGS.random_state
    )

    print(f"Optimal parameters: {result.x}")
    plt.figure(figsize=(20, 10))
    plt.scatter(
        list(range(len(result.func_vals))),
        result.func_vals,
        marker="x",
        s=20,
        label="Mean displacement",
    )
    plt.plot(np.minimum.accumulate(result.func_vals), label="Minimum so far")
    plt.legend()
    plt.title("Optimisation results for PID controller")
    plt.ylabel("Mean displacement")
    plt.xlabel("Step number")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--n_calls", type=int, default=150)
    parser.add_argument("--random_state", type=int, default=1)
    FLAGS = parser.parse_args()
    main(FLAGS)
