################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Regression guard for the tensile_generator tuning templates.

Background (rocm-libraries#8789): the tuning template shipped under
``Tensile/Utilities/tensile_generator/tuning_template.yaml`` is fed -- via
``tensile_config_generator.py`` -- into Tensile, which validates every
solution parameter with ``checkParametersAreValid``. When the template drifts
ahead of (or behind) ``validParameters`` it references an obsolete parameter
name and Tensile aborts with::

    Invalid parameter name: GlobalSplitUCoalesced

These tests pin the template to the in-tree ``validParameters`` /
``globalParameters`` registries so the template can never silently drift
again. If a parameter is renamed/removed in Common, this test fails until the
template is updated to match -- catching the bug in CI instead of in a user's
tuning run.
"""

from pathlib import Path

import pytest
import yaml

from Tensile.Common.ValidParameters import (
    checkParametersAreValid,
    validParameters,
)
from Tensile.Common.GlobalParameters import globalParameters

# Top-level "Architecture"-style global keys that are consumed directly by the
# argument reader / debug-config maker rather than living in the
# ``globalParameters`` dict. Keep in sync with the allowlist in
# Tensile/Common/GlobalParameters.py.
EXTRA_GLOBAL_KEYS = {
    "PrintSolutionRejectionReason",
}

GENERATOR_DIR = (
    Path(__file__).resolve().parents[2]
    / "Utilities"
    / "tensile_generator"
)
TUNING_TEMPLATE = GENERATOR_DIR / "tuning_template.yaml"


def _load_template(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _solution_params(cfg):
    """Yield (name, values) for every solution parameter used in a template."""
    benchmarkProblems = cfg.get("BenchmarkProblems") or []
    for group in benchmarkProblems:
        # group is [ProblemType, BenchmarkConfig]
        if not isinstance(group, list) or len(group) < 2:
            continue
        benchmarkConfig = group[1]
        if not isinstance(benchmarkConfig, dict):
            continue
        for section in ("BenchmarkCommonParameters", "ForkParameters"):
            for entry in (benchmarkConfig.get(section) or []):
                if not isinstance(entry, dict):
                    continue
                for name, values in entry.items():
                    yield name, values


def test_tuning_template_exists():
    assert TUNING_TEMPLATE.is_file(), f"missing template: {TUNING_TEMPLATE}"


def test_tuning_template_solution_params_are_valid():
    """Every solution parameter in tuning_template.yaml must pass the same
    ``checkParametersAreValid`` gate Tensile applies to a generated config.

    This directly guards rocm-libraries#8789: a stale param name such as
    ``GlobalSplitUCoalesced`` would raise ``Invalid parameter name`` here.
    """
    cfg = _load_template(TUNING_TEMPLATE)
    params = list(_solution_params(cfg))
    assert params, "no solution parameters found in tuning_template.yaml"

    invalid = []
    for name, values in params:
        try:
            checkParametersAreValid((name, values), validParameters)
        except Exception as exc:  # noqa: BLE001 - surface name + message
            invalid.append(f"{name}: {exc}")

    assert not invalid, (
        "tuning_template.yaml references solution parameters that are not "
        "valid against Tensile/Common/ValidParameters.py. Update the template "
        "(or validParameters) so they stay in sync:\n  - "
        + "\n  - ".join(invalid)
    )


def test_tuning_template_global_params_are_known():
    """Top-level GlobalParameters in the template must be recognized keys."""
    cfg = _load_template(TUNING_TEMPLATE)
    used = set((cfg.get("GlobalParameters") or {}).keys())
    known = set(globalParameters.keys()) | EXTRA_GLOBAL_KEYS
    unknown = sorted(used - known)
    assert not unknown, (
        "tuning_template.yaml uses unrecognized GlobalParameters keys "
        f"(not in globalParameters): {unknown}"
    )
