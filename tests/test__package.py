import os

from packaging import version


def test_scenario_gym():
    """Test importing the package."""
    import scenario_gym  # noqa F401


def test_version():
    """Make sure the version mathces."""
    import scenario_gym

    v_module = version.parse(scenario_gym.__version__)

    setup_pth = os.path.join(
        os.path.dirname(__file__),  # scenario_gym/tests
        "../setup.py",
    )
    with open(setup_pth, "r") as f:
        setup = f.read()
    v_setup = version.parse(setup.split("version=")[-1].split(",")[0].strip("'\""))
    assert v_module == v_setup, "Setup version does not equal module version."
