Simulating Scenarios
====================
Scenario Gym simulates scenarios in discrete timesteps. In a single timestep the new pose of each entity is computed and updated synchronously. This is repeated until the simulation is complete. The typical workflow for running a simulating is to setup the environment (:code:`ScenarioGym`), load a scenario (e.g. via OpenSCENARIO with :code:`ScenarioGym.load_scenario`) and run the simulation with (:code:`ScenarioGym.rollout`):

.. code-block::
    :caption: Running a scenario.

    from scenario_gym import ScenarioGym
    gym = ScenarioGym()
    gym.load_scenario("path_to_xosc")
    gym.rollout()

The rollout method will run the simulation by calling :code:`gym.step` until a terminal state is reached:

.. code-block::
    :caption: Pseudocode of a rollout.

    def rollout(self):
        self.reset_scenario()
        while not self.state.is_done:
            self.step()

Simulation Setup
----------------
Before running a simulation we will want to configure the timestep and ending conditions of the simulation.

Timestep
~~~~~~~~
The timestep controls the amount of simulation time in each discrete step. A smaller value will result in a more accurate simulation but will need more steps to complete increasing the simulation time. When rendering the simulation the timestep is the inverse of the framerate.

.. code-block::
    :caption: Setting the timestep.

    gym = ScenarioGym(timestep=0.1)

Terminal Conditions
~~~~~~~~~~~~~~~~~~~
The simulation will run until any one of the terminal conditions is met. By default this will run for the length of the input scenario (determined by the trajectory of each entity) but can be customised to any condition provided as a function taking the current state and returning true if the simulation should end:

.. code-block::
    :caption: Implementing a custom terminal condition.

    def end_at_max_speed(state) -> bool:
        """End if the ego's speed is above 10m/s or time is above 1minute."""
        ego = state.scenario.entities[0]
        velocity = state.velocities[ego]
        speed = np.linalg.norm(velocity[:3])
        return speed > 10. or state.t > 60
    
    gym = ScenarioGym(terminal_conditions=[end_at_max_speed])

Some common conditions can be passed with a string identifier:

* :code:`max_length`: end when the time exceeds the length of the scenario,
* :code:`collision`: end if there is a collision between any entities,
* :code:`ego_collision`: end if there is a collision with the ego,
* :code:`ego_off_road`: end if the ego leaves the road surface.

Loading Scenarios
-----------------
Scenario Gym provides an interface to load scenarios from OpenSCENARIO files and to load road networks from OpenDRIVE and the custom JSON format via :code:`ScenarioGym.load_scenario`. Scenarios can be loaded from other sources by creating the :code:`Scenario` object and using :code:`ScenarioGym.set_scenario`:

.. code-block::
    :caption: Loading a scenario from a custom data source.

    scenario = load_custom_scenario(...)
    gym.set_scenario(scenario)
    gym.rollout()

When loading scenarios from OpenSCENARIO format the following directory structure will be assumed:

.. code-block::

    /path/to/data/
        Scenarios/
            example.xosc
        Catalogs/
            Scenario_Gym/
                VehicleCatalogs/
                    ScenarioGymVehicleCatalog.xosc
        Road_Networks/
            road_network1.json
            road_network2.xodr
        Recordings/

These directories will hold the catalog files, road networks and OpenSCENARIO files. The Recordings directory will hold any recorded simulations either as videos or other ouput formats such as OpenSCENARIO.

Assigning Agents
~~~~~~~~~~~~~~~~
A key feature of Scenario Gym is the ability to use intelligent agents to control entities in the scenario. To understand more about implementing agents see the section :ref:`agents-section`. When loading a scenario we will want to assign agents to relevant entities. This is done by defining a function that will return the agent that should control each entity. The function should take the scenario and the entity as inputs and return an agent if the entity should have one.

.. code-block::
    :caption: An example function to assign agents.

    def create_agent(scenario, entity):
        """Return agents for the ego and all pedestrians."""
        if entity.ref == "ego":
            return EgoAgent(entity)
        elif entity.catalog_entry.type == "Pedestrian":
            return PedestrianAgent(entity)
    
It is important to note that not every entity requires an agent. Entities without agents will follow their predefined trajectory from the scenario. Moreover the gym will batch the computation for these entities to improve speed. To not provide an agent for an entity the function should simply return :code:`None`. The function should be passed to :code:`load_scenario` or :code:`set_scenario` to create agents:

.. code-block::

    gym.load_scenario("path_to_xosc", create_agent=create_agent)
    gym.rollout()


Recording Simulations
---------------------
Simulations can be recorded in two main ways: rendering the simulation to a video or outputting the simulation as a new scenario.

Rendering
~~~~~~~~~
To produce a video of the simulation we just pass :code:`render=True` to :code:`rollout`. This will create a video and look to save it in a sensible location. If we are using the directory structure above then the :code:`Recordings` directory will be used. Otherwise the directory of the scenario (from `Scenario.scenario_path`) will be used. If neither of these exist or a different location is preffered the output path can be specified:

.. code-block::

    gym.rollout(render=True, vide_path="/path/to/result.mp4")

Output Scenarios
~~~~~~~~~~~~~~~~
When a simulation is complete an output scenario can be generated from the state via :code:`State.to_scenario` which can then be converted to the chosen output format.

.. code-block::
    :caption: Recording a simulation as an OpenSCENARIO file.

    from scenario_gym.xosc_interface import write_scenario

    gym.rollout()
    output_scenario = gym.state.to_scenario()
    
    # write to OpenSCENARIO
    write_scenario(output_scenario, "output_path.xosc")
