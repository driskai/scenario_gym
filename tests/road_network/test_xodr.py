from scenario_gym.road_network import RoadNetwork


def test_xodr_networks(all_xodr_networks):
    """Test that we can create RoadNetwork from OpenDRIVE."""
    for _, path in all_xodr_networks.items():
        try:
            _ = RoadNetwork.create_from_xodr(path)
        except Exception as e:
            raise Exception(f"Error while reading {path}: ") from e
