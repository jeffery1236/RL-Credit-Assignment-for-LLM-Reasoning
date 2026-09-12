"""Smoke tests: verify the project skeleton imports and runs before any domain
logic exists. These should stay green throughout Phase 1 as modules are added.
"""

import importlib
import subprocess
import sys

import pytest

PACKAGES = ["env", "oracle", "policies", "analysis", "experiments", "inference"]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package):
    importlib.import_module(package)


EXPERIMENT_MODULES = [
    "experiments.render_dag",
    "experiments.profile_environment",
    "experiments.freeze_dataset",
    "experiments.evaluate_allowed_action_llm_policy",
    "experiments.compare_menu_ordering_sensitivity",
    "experiments.evaluate_countdown_capability",
    "experiments.evaluate_policy_bridge",
]


@pytest.mark.parametrize("module", EXPERIMENT_MODULES)
def test_experiment_module_runs_with_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
