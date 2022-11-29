import inspect
import os
import warnings
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Type, Union

import yaml

from scenario_gym.agent import Agent, ReplayTrajectoryAgent
from scenario_gym.controller import ReplayTrajectoryController
from scenario_gym.entity import Entity
from scenario_gym.metrics import Metric
from scenario_gym.scenario import Scenario
from scenario_gym.scenario_gym import ScenarioGym
from scenario_gym.sensor import EgoLocalizationSensor


def load_keywords(obj: Type, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    """Find keyword arguments of the object."""
    sig = inspect.signature(obj.__init__)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default != inspect._empty and (exclude is None or k not in exclude)
    }


class ScenarioManager:
    """Provides functionality to manage running large numbers of scenarios."""

    GYM_PARAMETERS = load_keywords(ScenarioGym, exclude=["metrics"])
    PARAMETERS = {}

    @classmethod
    def generate_parser(cls) -> ArgumentParser:
        """Generate an argument parser for the manager."""
        parser = ArgumentParser(description=f"CLI for {cls.__name__}.")
        params = {
            **cls.GYM_PARAMETERS,
            **cls.VIEWER_PARAMETERS,
            **cls.PARAMETERS,
        }
        for k, v in params.items():
            if isinstance(v, bool):
                parser.add_argument(
                    f"--{k.replace(' ', '_')}",
                    action="store_false" if v else "store_true",
                )
            elif isinstance(v, (str, int, float)):
                parser.add_argument(
                    f"--{k.replace(' ', '_')}",
                    default=v,
                    type=type(v),
                )
            elif isinstance(v, (list, tuple)):
                parser.add_argument(
                    f"--{k.replace(' ', '_')}",
                    default=v,
                    nargs="+",
                )

            elif v is None:
                parser.add_argument(
                    f"--{k.replace(' ', '_')}",
                    default=v,
                    type=float,
                )
            else:
                warnings.warn(f"Type {type(v)} not supported.")
        return parser

    @classmethod
    def from_cli(cls, **kwargs):
        """Construct the manager from command line arguments."""
        parser = cls.generate_parser()
        args = parser.parse_args()
        return cls(
            **{
                k: v
                for k, v in args.__dict__.items()
                if k in cls.PARAMETERS and v is not None
            },
            **kwargs,
        )

    def __init__(
        self,
        config_path: Optional[str] = None,
        metrics: Optional[List[Metric]] = None,
        viewer_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Construct the manager from input parameters.

        Parameters
        ----------
        config_path : Optional[str]
            A path to a yaml file of the parameters to be used.

        metrics : Optional[List[Metric]]
            List of metrics to measure.

        viewer_params : Optional[Dict[str, Any]]
            Dictionary of parameters to use for the Viewer. Since the viewer can be
            a custom class its parameters should be passed as a dictionary so the
            manager will know which parameters should be passed for it.

        **kwargs:
            Parameters given as keywords. Will override any parameters
            from a config file.

        """
        self.load_params(path=config_path, **kwargs)
        self.metrics = metrics.copy() if metrics is not None else []
        self.viewer_params = viewer_params.copy() if viewer_params else {}

    def load_params(self, config_path: Optional[str] = None, **kwargs) -> None:
        """Load all parameters required and set them as attributes."""
        params = yaml.safe_load(open(config_path, "r")) if config_path else {}
        self.PARAMETERS = self.PARAMETERS.copy()
        self.combined_config = {
            **self.GYM_PARAMETERS,
            **self.PARAMETERS,
            **params,
            **kwargs,
        }
        for k, v in self.combined_config.items():
            if not hasattr(self, k):
                setattr(self, k, v.copy() if isinstance(v, (list, dict)) else v)

    @property
    def parameter_names(self) -> List[str]:
        """Return the names of all parameters."""
        return self.gym_parameters_names + self.viewer_parameters_names

    @property
    def parameters(self) -> List[str]:
        """Return all the parameters for the gym and viewer."""
        return {**self.gym_parameters, **self.viewer_parameters}

    @property
    def gym_parameter_names(self) -> List[str]:
        """Return the names of all gym parameters."""
        return list(self.GYM_PARAMETERS)

    @property
    def gym_parameters(self) -> Dict[str, Any]:
        """Return the parameters needed for the ScenarioGym constructor."""
        return {k: getattr(self, k) for k in self.GYM_PARAMETERS}

    @property
    def viewer_parameter_names(self) -> List[str]:
        """Return the names of all viewer parameters."""
        return list(self.viewer_params)

    @property
    def viewer_parameters(self) -> Dict[str, Any]:
        """Return the parameters needed for the rendering module."""
        return self.viewer_params

    def make_gym(self, **kwargs) -> ScenarioGym:
        """Create a gym instance with the given config."""
        return ScenarioGym(
            metrics=self.metrics,
            **self.gym_parameters,
            **self.viewer_parameters,
            **kwargs,
        )

    def create_agent(self, scenario: Scenario, entity: Entity) -> Agent:
        """
        Construct the agents when loading a scenario.

        Parameters
        ----------
        scenario : Scenario
            The scenario object.

        entity : Entity
            The specific entity within the scenario.

        """
        if entity.ref == "ego":
            controller = ReplayTrajectoryController(entity)
            sensor = EgoLocalizationSensor(entity)
            return ReplayTrajectoryAgent(entity, controller, sensor)

    def add_metric(self, m: Metric) -> None:
        """Add a metric to the manager."""
        self.metrics.append(m)

    def on_rollout_start(self, gym: ScenarioGym) -> None:
        """Run before the rollout when running scenarios."""
        pass

    def on_rollout_end(self, gym: ScenarioGym) -> None:
        """Run after the rollout when running scenarios."""
        pass

    def run_scenario(
        self,
        scenario: Union[str, Scenario],
        render: bool = False,
        record: bool = False,
        **kwargs,
    ) -> List[Any]:
        """
        Run a single scenario in the gym.

        Parameters
        ----------
        scenario : Union[str, Scenario],
            The filepath of the OpenScenario file for the scenario or
            the scenario object.

        render : bool
            Whether to render the scenario.

        record : bool
            Whether to record the scenario to OpenScenario.

        """
        gym = self.make_gym()
        if record:
            gym.record()
        if isinstance(scenario, str):
            gym.load_scenario(scenario, create_agent=self.create_agent)
        elif isinstance(scenario, Scenario):
            gym.set_scenario(scenario, create_agent=self.create_agent)
        else:
            raise ValueError(f"{scenario}: should be a scenario or a file.")
        self.on_rollout_start(gym)
        gym.rollout(render=render, **kwargs)
        self.on_rollout_end(gym)
        if record:
            gym.recorder.get_state()
        return [m.get_state() for m in self.metrics]

    def run_scenarios(
        self,
        scenarios: List[str],
        render: bool = False,
        record: bool = False,
        **kwargs,
    ) -> List[List[Any]]:
        """
        Run a batch of scenarios.

        Parameters
        ----------
        scenarios : List[Union[str, Scenario]]
            The filepaths of the OpenScenario files for the scenarios or
            the raw scenario objects.

        render : bool
            Whether to render each scenario.

        record : bool
            Whether to record each scenario to OpenScenario.

        Returns
        -------
        List[List[Any]]
            The values for each metric after each scenario.

        """
        results = []
        gym = self.make_gym()
        if record:
            gym.record()
        for scenario in scenarios:
            if isinstance(scenario, str):
                gym.load_scenario(scenario, create_agent=self.create_agent)
            elif isinstance(scenario, Scenario):
                gym._set_scenario(scenario, create_agent=self.create_agent)
            else:
                raise ValueError(f"{scenario}: should be a scenario or a file.")
            gym.rollout(render=render, **kwargs)
            if record:
                gym.recorder.get_state()
            results.append([m.get_state() for m in self.metrics])
        return results

    def save_config(self, path: str = "./params.yml") -> None:
        """
        Write the config parameters to a yaml file.

        Parameters
        ----------
        path : str
            The filepath for the output.

        """
        path = os.path.splitext(path)[0] + ".yml"
        with open(path, "w") as f:
            yaml.dump(self.combined_config, f)
