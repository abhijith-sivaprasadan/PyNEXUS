"""Numerical isolation checks for independent-task distribution."""

import copy
from pathlib import Path

import pytest
import yaml

from optimization.scenario_sweep import compare_results, scenarios, solve_scenario


def test_scenario_order_does_not_leak_model_state():
    cfg = yaml.safe_load(Path("configs/tiny_test.yaml").read_text())
    original = copy.deepcopy(cfg)
    cases = [scenarios()[0], scenarios()[-1]]
    forward = [solve_scenario(spec, cfg, 4) for spec in cases]
    reverse = [solve_scenario(spec, cfg, 4) for spec in reversed(cases)]
    compare_results(forward, reverse)
    assert cfg == original
    assert all(r["verification"]["passed"] for r in forward)


def test_reference_check_rejects_missing_scenario():
    cfg = yaml.safe_load(Path("configs/tiny_test.yaml").read_text())
    result = solve_scenario(scenarios()[0], cfg, 4)
    with pytest.raises(AssertionError, match="scenarios"):
        compare_results([result], [])
