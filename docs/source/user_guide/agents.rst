.. _agents-section:

Agents, Sensors and Controllers
===============================
Scenario Gym has a modular framework for implementing customised intelligent agents that splits an agent's decision process into three distinct stages. First a sensory module produces an observation from the global state. This observation will contain all of the information that the agent would have access to in a real life setting and can use to make decisions. The agent will then select an action from the observation. For example this could be acceleration and steering commands. Then a controller module will convert the action into an updated pose for the entity. This process is repeated for each agent at every timestep.

Agents
------
An :code:`Agent` is implented by providing a sensor, a controller and a method to select actions. The :code:`_reset` and :code:`_step` methods provide the ability to reset the agent's internal state at the start of the simulation and then to select an action at each timestep.

.. code-block::
    :caption: Skeleton code for implementing an agent.

    from scenario_gym import Agent

    class ExampleAgent(Agent):

        def __init__(self, entity):
            controller = ExampleController(entity)
            sensor = ExampleSensor(entity)
            super().__init__(entity, controller, sensor)
        
        def _reset(self):
            # reset internal state
            ...
        
        def _step(self, observation):
            # select the action
            action = ...
            return action

Sensors
-------
A :code:`Sensor` should produce an observation for the agent from the gym's :code:`State`. An observation should be lightweight dataclass that contain just the data an agent should be able to access at each timestep.

Observations
~~~~~~~~~~~~
Observations make use of the :code:`dataclasses` module in Python. With the dataclass API we can easily specify the information contained in each observation:

.. code-block::
    :caption: An observation containing the entity's pose and the current simulation time.

    from scenario_gym import Observation

    @dataclass
    class PoseAndTimeObservation(Observation):
        time: float
        pose: np.ndarray
    
Observations are also naturaly extensible. Here we combine a :code:`SingleEntityObservation` which contains all information about the agent's entity with a list of detected objects produced by an object detection module.

.. code-block::

    from scenario_gym.observation import SingleEntityObservation

    @dataclass
    class DectectedObjectsObservation(SingleEntityObservation):
        detected_objects: list

Implementing Sensors
~~~~~~~~~~~~~~~~~~~~
As with agents, sensors use the :code:`_reset` and :code:`_step` methods to reset their internal parameters and produce the observation at each timestep. The :code:`_reset` method should also return an initial observation for the sensor.

Combined Sensors
~~~~~~~~~~~~~~~~

Controllers
-----------
