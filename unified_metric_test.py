import json
import os

from scenario_gym import ScenarioGym
from scenario_gym.metrics import (
    CollisionTimestamp,
    DeltaV,
    DistanceToEgo,
    EgoAvgSpeed,
    EgoMaxSpeed,
    LaggingMetricsProcessor,
    SafeLatDistance,
    SafeLongDistance,
    SpaceOccupancyIndex,
    TimeToCollision,
    UnifiedRiskMetricCalculator,
)
from scenario_gym.xosc_interface import import_scenario

# ---------- Instantiating variables ----------#
# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Create directories for metric evaluations and recordings
directories = ["saved_recordings/", "saved_metric_evaluations/"]

for directory in directories:
    dir_path = os.path.join(current_dir, directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Paths for input/outputs
default_save_recordings_path = os.path.join(current_dir, directories[0])
default_save_metrics_path = os.path.join(current_dir, directories[1])
default_input_files_path = os.path.join(current_dir, "tests/input_files/Scenarios/")

# User/Developer Controlled Variables --> Would recommend to NOT change these.
plot_bool = False
render_bool = True
print_metrics_bool = True
unify_metrics_bool = True

# If False, then only evaluates metrics for one OpenScenario file.
# If True, evaluates metrics for all files in tests/input_files/Scenarios
loop_bool = True
openscenario_file = default_input_files_path + "5-vehicles-congested-collision.xosc"


def load_and_rollout_scenario(
    openscenario_file,
    default_save_recordings_path,
    default_save_metrics_path,
    render_bool,
    print_metrics_bool,
    unify_metrics_bool,
    plot_bool,
) -> None:
    """Load input OpenScenario file, and saves recording and metric evaluation."""
    # String handling for saved .mp4
    file_name, file_extension = os.path.splitext(openscenario_file)

    # Simulating scenario
    gym = ScenarioGym(
        metrics=[
            TimeToCollision(),
            DistanceToEgo(),
            SafeLongDistance(),
            SafeLatDistance(),
            SpaceOccupancyIndex(),
            DeltaV(),
            EgoAvgSpeed(),
            EgoMaxSpeed(),
            CollisionTimestamp(),
        ]
    )
    # gym = ScenarioGym()
    gym.load_scenario(openscenario_file)
    gym.rollout(
        render=render_bool,
        video_path=default_save_recordings_path
        + os.path.basename(file_name)
        + ".mp4",
    )

    # Print risk metric values
    if print_metrics_bool:
        metrics = gym.get_metrics()

        processor = LaggingMetricsProcessor(metrics)
        metrics = processor.process_metrics()

        calculator = UnifiedRiskMetricCalculator(metrics)

        if unify_metrics_bool:
            if (
                metrics["Lagging Metrics Post-Runtime"]["collision_timestamp"]
                != False
            ):
                unified_risk_value = 1
                calculator._calculate_continuous_unified_risk_metric()
            else:
                unified_risk_value = calculator._calculate_unified_risk_metric()
                calculator._calculate_continuous_unified_risk_metric()

            metrics["Lagging Metrics Post-Runtime"][
                "unified_risk_value"
            ] = unified_risk_value

        # Saving metrics json file and storing in saved_metric_evaluations
        save_metrics_path = (
            default_save_metrics_path + os.path.basename(file_name) + ".json"
        )
        with open(save_metrics_path, "w") as json_file:
            json.dump(metrics, json_file, indent=4)

    # Show static trajectory plot
    if plot_bool:
        scenario = import_scenario(openscenario_file)
        scenario.plot()


if loop_bool:
    # Loop through all files in the directory
    for filename in os.listdir(default_input_files_path):
        file_path = os.path.join(
            default_input_files_path,
            filename,
        )
        if os.path.isfile(file_path):
            load_and_rollout_scenario(
                file_path,
                default_save_recordings_path,
                default_save_metrics_path,
                render_bool,
                print_metrics_bool,
                unify_metrics_bool,
                plot_bool,
            )
            print("Rolled out:", filename)
else:
    load_and_rollout_scenario(
        openscenario_file,
        default_save_recordings_path,
        default_save_metrics_path,
        render_bool,
        print_metrics_bool,
        unify_metrics_bool,
        plot_bool,
    )
