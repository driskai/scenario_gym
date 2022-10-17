import numpy as np

from scenario_gym import ScenarioGym
from scenario_gym.viewer.opencv import OpenCVViewer


def test_viewer(all_scenarios):
    """Test that we can render with different options."""
    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym(
        timestep=0.1,
        viewer_class=OpenCVViewer,
        render_layers=list(OpenCVViewer._renderable_layers.keys()),
        **{
            f"{k}_color": tuple(map(int, np.random.randint(0, 255, 3)))
            for k in OpenCVViewer._renderable_layers
        },
        background=(0, 100, 0),
    )
    gym.load_scenario(scenario_path)
    for _ in range(1):
        gym.rollout(render=True)


def test_custom_viewer(all_scenarios):
    """Test rendering with a subclass of OpenCVViewer."""

    class CustomViewer(OpenCVViewer):
        """Viewer that changes colour for pedestrians and ego."""

        def get_entity_color(self, entity_idx: int, entity):
            if entity_idx == 0:
                return (0, 0, 240)  # red
            if entity.catalog_entry.catalog_category == "pedestrian":
                return (0, 240, 0)  # green
            return (200, 0, 0)  # blue

    scenario_path = all_scenarios["a5e43fe4-646a-49ba-82ce-5f0063776566"]

    gym = ScenarioGym(
        timestep=0.1,
        viewer_class=CustomViewer,
    )
    gym.load_scenario(scenario_path)
    gym.reset_scenario()
    gym.rollout(render=True)
