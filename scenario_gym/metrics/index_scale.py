from scenario_gym.state import State

from .base import Metric


# ---------- Index-scale Criticality Metrics ----------#
class SpaceOccupancyIndex(Metric):
    """
    Tracks SOI.

    The Space Occupancy Index (SOI) defines a personal space for the ego
    and counts violations by other participants.
    """

    name = "space_occupancy_index"

    def _reset(self, state: State) -> None:
        """Reset the space occupancy index."""
        self.ego = state.scenario.ego
        self.space_occupancy_index = {}
        self.predefined_radius = 15

    def _step(self, state: State) -> None:
        """
        SOI for every timestep.

        Capture the number of personal space incursions
        experienced by each actor.
        """
        # Personal space is defined by existing within a set radius.
        if (
            len(
                state.get_entities_in_radius(
                    *state.poses[self.ego][:2], self.predefined_radius
                )
            )
            > 1
        ):
            for entity in state.get_entities_in_radius(
                *state.poses[self.ego][:2], self.predefined_radius
            ):
                if entity == self.ego:
                    continue

                # state.t inclusion in the dictionary (line below). To
                # track/timestamp TTC in simulation.
                if state.t not in self.space_occupancy_index:
                    self.space_occupancy_index[state.t] = []
                self.space_occupancy_index[state.t].append(
                    {"space_occupancy_index": {f"{entity}": 1}}
                )

    def get_state(self) -> dict:
        """Return the space occupancy index ."""
        return self.space_occupancy_index
