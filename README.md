# Scenario Gym - a lightweight framework for learning from scenario data

![Tests](https://github.com/driskai/scenario_gym/actions/workflows/test.yml/badge.svg)
![Python 3.7+](https://img.shields.io/badge/python-3.7+-brightgreen)

Scenario Gym is a universal autonomous driving simulation tool that allows fast execution of unconfined, complex scenarios containing a range of road users. It allows rich insight via customised metrics and includes a framework for designing intelligent agents for reactive simulation. It can be used for a variety of tasks relevant for AV development, such agent modelling, controller parameter tuning and deep reinforcement learning.

<p align="center">
<img src="https://raw.githubusercontent.com/driskai/scenario_gym/main/docs/source/_static/gym-ex1.gif" width="20%" />
&nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/driskai/scenario_gym/main/docs/source/_static/gym-ex2.gif" width="20%" />
&nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/driskai/scenario_gym/main/docs/source/_static/gym-ex3.gif" width="20%" />
</p>

## Overview

Scenario Gym defines a flexible scenario representation that is compatible with the OpenSCENARIO description language and OpenDRIVE road network representation. Entities can adopt predefined trajectories, or control themselves intelligently with a high-level goal (e.g. reach a target position) or via a complex trained policy. Scenarios are simulated synchronously in discrete time steps within which each agent selects an action and the pose of each entity is updated before moving to the next step.

Intelligent agents interact with the environment through a simple sensor-agent-controller architecture. This streamlines the agent design by splitting it into three components that emulate the design of autonomous agent systems. The sensor component produces a logical observation for the agent from the current global state of the environment. The agent then selects an action and passes it to the controller. The controller manages the physical model of the agent e.g. converting steering and acceleration commands into a new pose. This modular architecture provides reusability and quick iteration of agent designs, not only for vehicular agents but also pedestrians, bicycles and other entity types.

Custom metrics can be implemented to allow quick and specific yet comprehensive insights. Through the scenario representation these can be constructed to efficiently track statistics such as speeds and distances, to record events such as collisions and near misses or to capture more compound measures such as safe distances and risk measures.

<p align="center">
<img src="https://raw.githubusercontent.com/driskai/scenario_gym/main/docs/source/_static/system_overview.svg" width="80%">
</p>

## Installation
Install with `pip`:
```
pip install scenario_gym
```

To install extras for specific integrations or development requirements:
```
pip install scenario_gym[extra]
```

## Getting started
To run a scenario in OpenSCENARIO format:
```python
from scenario_gym import ScenarioGym

gym = ScenarioGym()
gym.load_scenario("path_to_xosc")
gym.rollout()
```
Several example scenarios are given in the `tests/input_files/Scenarios` directory.

## Intelligent Agents
Agents are defined by a subclass of `Agent` as well as a `Sensor` and a `Controller`. They use implement the `_step` method to produce actions from the observations which will be passed to the controller.
```python
from scenario_gym import Agent
from scenario_gym.sensor import RasterizedMapSensor
from scenario_gym.controller import VehicleController

class ExampleAgent(Agent):

    def __init__(self, entity):

        controller = VehicleController(entity)
        sensor = RasterizedMapSesnor(entity)
        super().__init__(
            entity,
            controller,
            sensor,
        )

    def _step(self, observation):
        action = ...
        return action
```

To run scenarios with intelligent agents we just define a `create_agent` method which will assign agents to each entity in the scenario. This is passed to the gym instance when loading a scenario. The function must take arguments `scenario` and `entity` and optionally return agents. If an agent is not returned for an entity then the entity will simply follow its predefined trajectory. For example, here we use the `ExampleAgent` implemented above for the ego only:
```python
def create_agent(scenario, entity):
    if entity.ref == "ego":
        return ExampleAgent(entity)

gym.load_scenario("path_to_xosc", create_agent=create_agent)
gym.rollout()
```

## Metrics
To track performance statistics or record events the `Metric` class can be used. These implement the `_reset` and `_step` method to maintin an internal state across the scenario and the `get_state` method to return their recorded data. A selection metrics are already implemented and can be run by passing them to the `ScenarioGym`:
```python
from scenario_gym.metrics import CollisionMetric, EgoAvgSpeed

gym = ScenarioGym(metrics=[CollisionMetric(), EgoAvgSpeed()])
gym.load_scenario("path_to_xosc")
gym.rollout()

gym.get_metrics()
```

## Deep reinforcement learning
For reinforcement learning applications Scenario Gym supports an OpenAI Gym compatible implementation. When creating the environment we need to specify the observation and action spaces used by the ego agent as well as our `create_agent` function. The observation from the ego agent's sensor will be returned by the environment and the action passed to `step` will be passed to the agent's controller.

```python
from scenario_gym.integrations.openaigym import ScenarioGym

env = ScenarioGym(
    observation_space=observation_space,
    action_space=action_space,
    create_agent=create_agent,
)
obs = env.reset()
action = model(obs)
obs, reward, done, info = env.step(action)
```

For more code examples please see the `examples` directory.

