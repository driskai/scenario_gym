"""Parameters for pedestrian behavior models."""

# General parameters for all models
general_params = {"speed": 0.4, "max_speed_factor": 1.3}

# Random walk parameters
random_walk = {
    "bias_lon": 0.1,
    "bias_lat": 0.05,
    "std_lon": 0.2,
    "std_lat": 0.1,
}

social_force = {
    "distance_threshold": 3,
    "sight_weight": 0.5,
    "sight_angle": 200,
    "relaxation_time": 1.5,
    "ped_repulse_V": 2.1,
    "ped_repulse_sigma": 0.3,
    "ped_attract_C": 0.2,
    "boundary_repulse_U": 10,
    "boundary_repulse_R": 0.2,
}

# Add models to be used here
models_params = {"random_walk": random_walk, "social_force": social_force}

params = {"general": general_params, "models": models_params}
