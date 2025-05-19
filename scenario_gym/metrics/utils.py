from scenario_gym.state import State

from .base import Metric


# ---------- Collision Check ----------#
class CollisionTimestamp(Metric):
    """Checks if the ego vehicle collided with any object in the scenario."""

    name = "collision_timestamp"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.collision_check_and_timestamp = False
        self.t = 0.0

    def _step(self, state: State) -> None:
        if not self.collision_check_and_timestamp:
            if len(state.collisions()[state.scenario.entities[0]]) > 0:
                self.collision_check_and_timestamp = state.t

    def get_state(self) -> float | bool:
        """Return collision check and timestamp."""
        return self.collision_check_and_timestamp


# ---------- Process Lagging Metrics ----------#
class LaggingMetricsProcessor:
    """Class to calculate lagging metrics including TIT, TET, SOI."""

    def __init__(self, data):
        self.data = data
        self.entity_tet_duration = {}
        self.entity_tit_duration = {}
        self.space_occupancy_index_cumulative = {}

    def _get_sorted_timestamps(self):
        """Extract and sort the keys that are valid float timestamps."""
        return sorted(
            [float(ts) for ts in self.data if isinstance(ts, (int, float))]
        )

    def _calculate_tit_and_tet_duration(self, ttc_data, time_diff):
        """Calculate TIT and TET durations for each entity."""
        for entity, ttc in ttc_data.items():
            entity_type = self._get_entity_type(entity)
            threshold_ttc = 1 if entity_type == "pedestrian" else 0.5

            if 0 < ttc < threshold_ttc:
                # Calculate TET
                if entity not in self.entity_tet_duration:
                    self.entity_tet_duration[entity] = 0
                self.entity_tet_duration[entity] += time_diff

                # Calculate TIT
                if entity not in self.entity_tit_duration:
                    self.entity_tit_duration[entity] = 0
                self.entity_tit_duration[entity] += (
                    (1 / ttc) - (1 / threshold_ttc)
                ) * time_diff

    def _calculate_space_occupancy_index(self, soi_data, time_diff):
        """Calculate Space Occupancy Index for each entity."""
        for entity, _ in soi_data.items():
            if entity not in self.space_occupancy_index_cumulative:
                self.space_occupancy_index_cumulative[entity] = 0
            self.space_occupancy_index_cumulative[entity] += time_diff

    @staticmethod
    def _get_entity_type(entity):
        """Extract the entity type (pedestrian/vehicle) from the entity string."""
        parts = entity.split(".")
        if len(parts) > 1:
            return parts[-2].lower()
        return "unknown"

    def process_metrics(self):
        """Process the input data and compute lagging metrics."""
        timestamps = self._get_sorted_timestamps()

        for i in range(1, len(timestamps)):
            current_timestamp = timestamps[i]
            previous_timestamp = timestamps[i - 1]
            time_diff = current_timestamp - previous_timestamp

            current_simulation_data = self.data[current_timestamp]

            for metric_entry in current_simulation_data:
                if "time_to_collision" in metric_entry:
                    ttc_data = metric_entry["time_to_collision"]
                    self._calculate_tit_and_tet_duration(ttc_data, time_diff)

                if "space_occupancy_index" in metric_entry:
                    soi_data = metric_entry["space_occupancy_index"]
                    self._calculate_space_occupancy_index(soi_data, time_diff)

        # Ensure "Lagging Metrics Post-Runtime" exists
        if "Lagging Metrics Post-Runtime" not in self.data:
            self.data["Lagging Metrics Post-Runtime"] = {}

        # Add computed metrics to the "Lagging Metrics Post-Runtime"
        self.data["Lagging Metrics Post-Runtime"].update(
            {
                "time_exposed_time_to_collision": self.entity_tet_duration,
                "time_integrated_time_to_collision": self.entity_tit_duration,
                "cumulative_space_occupancy": self.space_occupancy_index_cumulative,
            }
        )

        return self.data

    def _lagging_metrics_term(self):
        """Normalised lagging metrics term."""
        # Initialize
        TET_term = 0
        TIT_term = 0
        CSO_term = 0
        avg_term = 0
        max_term = 0
        lagging_metrics_term = 0

        # Thresholds
        threshold_avg_speed = 7
        threshold_max_speed = 10
        threshold_TET = 0.4
        threshold_TIT = 1
        threshold_CSO = 5

        # Retrieve lagging metrics
        metrics = self.data
        lagging_metrics = metrics["Lagging Metrics Post-Runtime"]
        ego_avg_speed = lagging_metrics.get("ego_avg_speed", 0)
        ego_max_speed = lagging_metrics.get("ego_max_speed", 0)
        time_exposed_time_to_collision_dict = lagging_metrics.get(
            "time_exposed_time_to_collision", {}
        )
        time_integrated_time_to_collision_dict = lagging_metrics.get(
            "time_integrated_time_to_collision", {}
        )
        cumulative_space_occupancy_dict = lagging_metrics.get(
            "cumulative_space_occupancy", {}
        )

        # Handle avg and max speed
        avg_term = 1 if ego_avg_speed > threshold_avg_speed else 0
        max_term = 1 if ego_max_speed > threshold_max_speed else 0

        # Count values above TET and TIT threshold, and handle TET+TIT
        TET_count_above_threshold = sum(
            1
            for value in time_exposed_time_to_collision_dict.values()
            if value > threshold_TET
        )
        TIT_count_above_threshold = sum(
            1
            for value in time_integrated_time_to_collision_dict.values()
            if value > threshold_TIT
        )
        TET_term = 1 if TET_count_above_threshold > 1 else 0
        TIT_term = 1 if TIT_count_above_threshold > 1 else 0

        # Handle CSO
        CSO_count_above_threshold = sum(
            1
            for value in cumulative_space_occupancy_dict.values()
            if value > threshold_CSO
        )
        CSO_term = 1 if CSO_count_above_threshold > 1 else 0

        # Final lagging metrics term
        lagging_metrics_term = (
            TET_term + TIT_term + CSO_term + avg_term + max_term
        ) / 5

        return lagging_metrics_term
