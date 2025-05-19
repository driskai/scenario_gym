import numpy as np
import pandas as pd

from scenario_gym.metrics.normalisation import RiskMetricsNormalisation
from scenario_gym.metrics.utils import LaggingMetricsProcessor


class UnifiedRiskMetricCalculator:
    """Class for calculating unified and continuous risk metrics."""

    def __init__(self, metrics):
        """
        Initialize the class with metrics data.

        :param metrics: JSON-like dictionary containing the metrics data.
        """
        self.metrics = metrics
        self.normaliser = RiskMetricsNormalisation()

    def _get_metric_dataframe(self):
        """
        Convert metrics JSON data into separate Pandas DataFrames for each metric.

        :return: A dictionary of metric names and their corresponding DataFrames.
        """
        metric_dataframes = {}

        # Process each timestamp in the metrics data
        for timestamp, metrics_list in self.metrics.items():
            if timestamp == "Lagging Metrics Post-Runtime":
                continue  # Skip lagging metrics

            for metric in metrics_list:
                for metric_name, vehicle_data in metric.items():
                    if metric_name not in metric_dataframes:
                        metric_dataframes[metric_name] = []

                    # Create a row for this timestamp
                    row = {"timestamp": timestamp}

                    if isinstance(vehicle_data, dict):
                        for vehicle, value in vehicle_data.items():
                            row[f"{vehicle}"] = value

                    metric_dataframes[metric_name].append(row)

        # Convert lists of rows into DataFrames
        for metric_name, rows in metric_dataframes.items():
            df = pd.DataFrame(rows)
            df.set_index("timestamp", inplace=True)
            metric_dataframes[metric_name] = df

        return metric_dataframes

    def _calculate_unified_risk_metric(self, continuous_bool=False, timestep=None):
        """
        Calculate the unified risk metric based on the metrics provided.

        :param continuous_bool: Whether the calculation is for a continuous metric.
        :param timestep: The current timestep for continuous metrics.
        :return: Unified risk value.
        """
        dataframes_dict = self._get_metric_dataframe()
        metrics_list = []

        # Process each metric DataFrame
        for metric_name, df in dataframes_dict.items():
            if continuous_bool and timestep is not None:
                df = df[df.index <= timestep]

            # Map metric names to their corresponding normalisation methods
            metric_methods = {
                "delta_v": self.normaliser._delta_v_term,
                "space_occupancy_index": self.normaliser._space_occupancy_index_term,
                "time_to_collision": self.normaliser._time_to_collision_term,
                "distance_to_ego": self.normaliser._distance_to_ego_term,
                "latitudinal_risk": self.normaliser._latitudinal_or_longitudinal_risk_term,
                "longitudinal_risk": self.normaliser._latitudinal_or_longitudinal_risk_term,
            }

            if metric_name in metric_methods:
                term = metric_methods[metric_name](df)
                metrics_list.append(term)

        # Handle lagging metrics if not continuous
        if not continuous_bool:
            lagging_processor = LaggingMetricsProcessor(self.metrics)
            lagging_metrics_term = lagging_processor._lagging_metrics_term()
            metrics_list.append(lagging_metrics_term)

        # Calculate unified risk value
        weights = np.ones(len(metrics_list)) / len(metrics_list)
        unified_risk_value = np.dot(weights, np.array(metrics_list))
        unified_risk_value = (
            unified_risk_value if not np.isnan(unified_risk_value) else 0
        )

        return unified_risk_value

    def _calculate_continuous_unified_risk_metric(self):
        """
        Calculate the continuous unified risk metric for each timestep.

        Updates the metrics data with the continuous unified risk metric.
        """
        for timestep in self.metrics.keys():
            if isinstance(self.metrics[timestep], list):
                continuous_unified_risk_metric_value = (
                    self._calculate_unified_risk_metric(
                        continuous_bool=True, timestep=timestep
                    )
                )
                self.metrics[timestep].append(
                    {
                        "continuous_unified_risk_metric": continuous_unified_risk_metric_value
                    }
                )
