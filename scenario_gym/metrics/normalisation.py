import pandas as pd


class RiskMetricsNormalisation:
    """Class for calculating various risk metrics."""

    @staticmethod
    def _proportion_above_threshold(lst, threshold):
        """Calculate the proportion of values above a threshold."""
        if len(lst) == 0:
            return 0
        return sum(1 for x in lst if x > threshold) / len(lst)

    @staticmethod
    def _proportion_below_threshold(lst, threshold):
        """Calculate the proportion of values below a threshold."""
        if len(lst) == 0:
            return 0
        return sum(1 for x in lst if x < threshold) / len(lst)

    @staticmethod
    def _calculate_gradients(term_specific_dataframe):
        """Calculate gradients for each column in the DataFrame."""
        term_specific_dataframe = term_specific_dataframe.copy()
        timestamp = term_specific_dataframe.index
        for col in term_specific_dataframe.columns:
            col_diff = term_specific_dataframe[col].diff()
            timestamp_diff = timestamp.to_series().diff()
            gradient = col_diff / timestamp_diff
            term_specific_dataframe[f"{col}_gradient"] = gradient
        return term_specific_dataframe

    def _delta_v_term(self, term_specific_dataframe):
        """Calculate delta V term."""
        pedestrian_threshold = 5
        vehicle_threshold = 10

        normalized_values = {}
        for column in term_specific_dataframe.columns:
            if "pedestrian" in column:
                normalized_values[column] = self._proportion_above_threshold(
                    term_specific_dataframe[column], pedestrian_threshold
                )
            elif "vehicle" in column:
                normalized_values[column] = self._proportion_above_threshold(
                    term_specific_dataframe[column], vehicle_threshold
                )
            else:
                normalized_values[column] = 0

        normalized_df = pd.DataFrame(
            list(normalized_values.items()), columns=["Entity", "Normalized Value"]
        )
        return (
            normalized_df["Normalized Value"].mean()
            if len(normalized_df) > 1
            else normalized_df["Normalized Value"].iloc[0]
        )

    def _space_occupancy_index_term(self, term_specific_dataframe):
        """Calculate space occupancy index term."""
        has_vehicle = any(
            "vehicle" in item for item in term_specific_dataframe.columns
        )
        has_pedestrian = any(
            "pedestrian" in item for item in term_specific_dataframe.columns
        )

        if has_vehicle and has_pedestrian:
            return 1.0 if len(term_specific_dataframe.columns) > 5 else 0.5
        return 0.5 if len(term_specific_dataframe.columns) > 5 else 0.0

    def _time_to_collision_term(self, term_specific_dataframe):
        """Calculate time to collision term."""
        pedestrian_threshold = 1.5
        vehicle_threshold = 1

        normalized_values = {}
        for column in term_specific_dataframe.columns:
            if "pedestrian" in column:
                normalized_values[column] = self._proportion_below_threshold(
                    term_specific_dataframe[column], pedestrian_threshold
                )
            elif "vehicle" in column:
                normalized_values[column] = self._proportion_below_threshold(
                    term_specific_dataframe[column], vehicle_threshold
                )
            else:
                normalized_values[column] = 0

        normalized_df = pd.DataFrame(
            list(normalized_values.items()), columns=["Entity", "Normalized Value"]
        )
        return (
            normalized_df["Normalized Value"].mean()
            if len(normalized_df) > 1
            else normalized_df["Normalized Value"].iloc[0]
        )

    def _distance_to_ego_term(self, term_specific_dataframe):
        """Calculate distance to ego term."""
        gradient_df = self._calculate_gradients(term_specific_dataframe)
        vehicle_thresholds = (4, 2)
        pedestrian_thresholds = (10, 1)

        results = {}
        for col in gradient_df.columns:
            if not col.endswith("_gradient"):
                if "vehicle" in col.lower():
                    value_threshold, gradient_threshold = vehicle_thresholds
                elif "pedestrian" in col.lower():
                    value_threshold, gradient_threshold = pedestrian_thresholds
                else:
                    continue

                gradient_col = f"{col}_gradient"
                if gradient_col in gradient_df.columns:
                    condition = (gradient_df[col] < value_threshold) & (
                        gradient_df[gradient_col].abs() > gradient_threshold
                    )
                    matching_timestamps = gradient_df.index[condition].tolist()
                    if matching_timestamps:
                        results[col] = matching_timestamps

        has_vehicle = any("vehicle" in key.lower() for key in results)
        has_pedestrian = any("pedestrian" in key.lower() for key in results)

        if has_vehicle and has_pedestrian:
            return 1
        return 0.5 if has_vehicle else 1 if has_pedestrian else 0

    def _latitudinal_or_longitudinal_risk_term(self, term_specific_dataframe):
        """Calculate latitudinal or longitudinal risk term."""
        average_non_zero_list = []
        for column in term_specific_dataframe.columns:
            non_zero_values = term_specific_dataframe[column][
                term_specific_dataframe[column] != 0
            ]
            if len(non_zero_values) > 0:
                average_non_zero_list.append(non_zero_values.mean())

        return (
            sum(average_non_zero_list) / len(average_non_zero_list)
            if len(average_non_zero_list) > 0
            else 0
        )
