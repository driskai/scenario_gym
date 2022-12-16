Scenarios
=========

Scenario Gym represents scenarios as a collection of entities and a road network. Each entity has a catalog entry specifying the physical characteristics of the entity (e.g. the type such as car or pedestrian) and its bounding box. Entities also have trajectories which specify the position of the entity and its orientation at different points in time.

The :code:`Scenario` object holds the entities, road network and any other data together:
.. code-block::
    
    from scenario_gym import Scenario
    s = Scenario(entities, road_network=road_network, name="example_scenario")

From this object we can access entities and their properties:
.. code-block::
    
    ego = s.entities[0]
    trajectory = ego.trajectory
    bounding_box = ego.catalog_entry.bounding_box

Scenarios are considered static. They do not contain simulation specific information such as the current position or velocity of a particular entity.


Road Networks
-------------
Road networks are represented as a collection of :code:`RoadObjects`s. 

