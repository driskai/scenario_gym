# Guidelines for contributing

Thank you for your interest in contributing to `scenario_gym`! We are always eager for new features, examples and integrations.

## Types of changes
Here you will find a guide for how to propose different types of changes to `scenario_gym`.

### Bug fixes and small tweaks
Create a pull request with your changes. Include a description of the need for the change and what you have done in the PR. If the bug is non-trivial then consider adding a test for it.

### Larger problems or changes
If you want to propose a more significant change then create an issue with the label `enhancement` or `bug` as well as your pull request. Larger changes should always include tests.

### New features
Create an issue with the label `enhancement` and include a description of the feature.

### New examples
Create an issue with the label `example` and include a description of the new example. These should ideally be limited to one file (that is not too long) and scripts are preferred over notebooks. If your example doesn't fit into the repository but you'd like to share it then host it on your own profile and we can add a link to it in the `README`. If your example has additional requirements to the base `scenario_gym` package then consider wrapping these imports in a `try` block and raise a helpful error message to the user.

### Other changes
For any other changes, questions or problems create an issue using the label you feel most appropriate. We will be happy to discuss it there!

## Testing
All tests should use `pytest`. In `tests/conftest.py` you will find some useful fixtures to make loading available scenarios and road networks easier. Make sure your test requires different input files before adding new ones.

## Style guidelines
The code style is `black`. The pre-commit hooks will ensure that all style requirements are met. These should be setup by installing the package with the `dev` extra and installing the hooks:
```
pip install "scenario_gym[dev] @ git+https://github.com/driskai/scenario_gym"
cd scenario_gym
pre-commit install
```