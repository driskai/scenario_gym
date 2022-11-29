from setuptools import find_packages, setup

requires = [
    "lxml>=4.7.1",
    "matplotlib>=3.4.2",
    "numpy>=1.21.0",
    "opencv-python>=4.5.3.56",
    "pyxodr @ git+https://github.com/driskai/pyxodr@develop",
    "PyYAML>=6.0",
    "setuptools>=58.0.4",
    "scenariogeneration>=0.7.4",
    "scipy>=1.6.3",
    "Shapely>=1.8.0",
]
extras = {
    "gym": ["gym>=0.21.0"],
    "hooks": [
        "black~=22.3.0",
        "flake8~=3.9.2",
        "isort~=5.10.1",
        "pre-commit~=2.16.0",
        "pydocstyle~=6.1.1",
        "importlib-metadata<5.0",
    ],
    "docs": ["Sphinx~=4.4.0"],
    "integrations": ["pandas>=1.1.5"],
    "examples": ["torch~=1.11.0"],
    "testing": ["pytest~=6.2.4"],
}
extras["dev"] = list(
    set().union(
        extras["gym"],
        extras["hooks"],
        extras["docs"],
        extras["testing"],
        extras["integrations"],
    )
)
extras["all"] = list(set().union(*extras.values()))

with open("README.md", "r") as f:
    long_description = "".join(f.readlines()[:3])

setup(
    author="dRISK AI",
    author_email="hamish@drisk.ai",
    description=(
        "scenario_gym - a lightweight framework for learning from scenario data."
    ),
    extras_require=extras,
    install_requires=requires,
    # license=...,
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="scenario_gym",
    packages=find_packages(
        where=".",
        include=["scenario_gym", "scenario_gym.*"],
    ),
    version="0.2.0",
)
