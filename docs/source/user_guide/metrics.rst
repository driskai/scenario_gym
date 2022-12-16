Metrics
=======

Scenario Gym can provide detailed insight into scenario simulation via metrics. A metric can be used to track statistics within a simulation, detect events or measure agent performance. These work using a similar API to the :code:`Agent` object. To create a metric the user must implement the :code:`_reset`, :code:`_step` and :code:`get_state` methods. The :code:`_reset` method will reset the internal state of the metric, the :code:`_step` method will update the internal state at each timestep during the simulation and the :code:`get_state` method will return the current value of the metric.

For example, a metric to measure the ego's maximum speed would first reset the maximum observed speed to zero at the start of the simulation. Then at each timestep it would update the maximum by comparing it to the current speed of the ego. The :code:`get_state` method would return the current maximum speed achieved by the ego.

.. code-block::
    :caption: An example implemention of the maximum ego speed metric.

    from scenario_gym import Metric

    class EgoMaxSpeed(Metric):

        def _reset(self, state):
            self.max_speed = 0.
        
        def _step(self, state):
            ego = state.scenario.entities[0]
            velocity = state.velocities[ego]
            speed = np.linalg.norm(velocity[:3])
            self.max_speed = max((speed, self.max_speed))
        
        def get_state(self):
            return self.max_speed

A selection of metrics are already implemented in the :code:`scenario_gym.metrics` package.