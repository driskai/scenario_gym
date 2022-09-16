# Pedestrian Behaviour Modelling

Pedestrians are modelled as agents that can follow different behaviour models. The existing classes from the gym are adapted for pedestrians as well, and can be found in the [pedestrian folder](../pedestrian). 

A [PedestrianAgent](agent.py) executes each step a [PedestrianAction](action.py) that consists of a speed and heading angle. The speed and heading are obtained from a [PedestrianBehaviour](behaviour.py) class, which can be inherited to define new custom behaviour models. Currently, the implemented models are **random walk** and **social force**.

Pedestrian agents move along walkable surfaces, which are part of the [road network](../road_network.py). Walkable surfaces can be **Pavements** or **Crossings**, which are defined by shapely polygons generated from the scenario file. A route along a walkable surface can be obtained using the [RouteFinder](route.py) class, given a start and end point.

## Add pedestrian agents to scenario

To run an existing scenario and add additional pedestrian agents to it, follow the example file [pedestrian_example.py](../../examples/pedestrian_example.py).

First, the parameters for the pedestrian generation and behaviour models must be set.

```
general_params = {
                  "num_pedestrians": 20,
                  "speed": 5.0,   # base speed
                  "max_speed_factor": 1.3   # Allowable max speed compared to base speed
                  }
```

A ```create_agent``` function must be defined with the desired behaviour models that returns a PedestrianAgent object. 

```
def create_agent(self, sc: Scenario, entity: Entity) -> Agent:
    if entity.ref == "ego":
        sensor = EgoLocalizationSensor(entity)
        controller = ReplayTrajectoryController(entity)
        return ReplayTrajectoryAgent(entity, controller, sensor)
    elif entity.type == "Pedestrian":
        controller = PedestrianController(entity)
        sensor = PedestrianSensor(entity)
        base_speed = self.params_pedestrian["general"]["speed"]
        speed_desired = np.random.uniform(0.5 * base_speed, 1.5 * base_speed)   # random desired speed
        behaviour = SocialForce(self.params_pedestrian)
        # behaviour = RandomWalk(self.params_pedestrian)

        # Find route for pedestrian along walkable surface
        route_finder = RouteFinder(sc.road_network)
        start = entity.trajectory[0][[1, 2]]
        finish = entity.trajectory[-1][[1, 2]]
        route = route_finder.find_route(start, finish)
        return PedestrianAgent(entity, controller, sensor, route, speed_desired, behaviour)
```


## Implemented Models

### Social Force

<img src="../../docs/source/_static/social_force.png" width="100%">

The social force models estimates the direction and magnitude of pedestrians' movement as a sum of vectors or "forces":

* **Attraction to goal** (next edge on path).
* **Repulsion to other pedestrians**.
* **Attraction to other pedestrians** (group attraction).
* **Repulsion to boundaries** (closest point of closest boundary of walkable surface).

The original paper for the social force model can be found [here](https://arxiv.org/pdf/cond-mat/9805244.pdf).

Its parameters are:
```
"distance_threshold": 3,
"sight_weight": 0.5,
"sight_weight_use": True,
"sight_angle": 200,
"relaxation_time": 1.5,
"ped_repulse_V": 100.0,
"ped_repulse_sigma": 0.3,  
"ped_attract_C": 0.2,
"boundary_repulse_U": 10.0,  
"boundary_repulse_R": 0.2, 
```

### Random Walk Model

The trajectory of a pedestrian given by the scenario file is modified by adding a Normally distributed random noise. The parameters of this model are the means (biases) and variances for speed (longitudinal) and heading angle (lateral).

```
"bias_lon": 0.1, 
"bias_lat": 0.05, 
"std_lon": 0.02, 
"std_lat": 0.01
```
