import shutil

import pytest
import rstcheck_core.checker  # type: ignore
from ert.plugins.plugin_manager import ErtPluginManager

from ccs_scripts.hook_implementations import forward_model_steps


@pytest.fixture
def expected_jobs() -> list[str]:
    return [
        "CO2_CONTAINMENT",
        "CO2_PLUME_AREA",
        "CO2_PLUME_EXTENT",
        "CO2_CSV_ARROW_CONVERTER",
        "GRID3D_CO2_MASS_MAP",
        "GRID3D_AGGREGATE_MAP",
        "GRID3D_MIGRATION_TIME",
    ]


# Avoid category inflation. Add to this list when it makes sense:
ACCEPTED_JOB_CATEGORIES = ["modelling", "utility"]


def test_hook_implementations(expected_jobs):
    """Test that we have the correct set of jobs installed,
    nothing more, nothing less"""
    plugin_m = ErtPluginManager(plugins=[forward_model_steps])

    installable_jobs = plugin_m.get_installable_jobs()
    assert set(installable_jobs.keys()) == set(expected_jobs)

    installable_workflow_jobs = plugin_m.get_installable_workflow_jobs()
    assert set(installable_workflow_jobs.keys()) == set(expected_jobs)


@pytest.mark.integration
def test_executables(expected_jobs):
    """Test executables listed in job configurations exist in $PATH"""
    plugin_m = ErtPluginManager(plugins=[forward_model_steps])
    for job in expected_jobs:
        assert shutil.which(plugin_m.get_installable_jobs()[job])


def test_hook_implementations_job_docs():
    """For each installed job, we require the associated
    description string to be nonempty, and valid RST markup"""

    plugin_m = ErtPluginManager(plugins=[forward_model_steps])

    installable_jobs = plugin_m.get_installable_jobs()

    docs = plugin_m.get_documentation_for_jobs()
    assert set(docs.keys()) == set(installable_jobs.keys())

    for job_name in installable_jobs.keys():
        desc = docs[job_name]["description"]
        assert desc != ""
        assert not list(rstcheck_core.checker.check_source(desc))
        category = docs[job_name]["category"]
        assert category != "other"
        assert category.split(".")[0] in ACCEPTED_JOB_CATEGORIES
