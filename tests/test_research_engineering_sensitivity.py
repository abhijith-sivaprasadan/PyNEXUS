"""Hermetic tests for registered, controlled WP4 engineering experiments."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from research.engineering_sensitivity import PROTOCOL, perturb_config, run_case
from research.run_reference import CONFIG


def fixture():
    cfg = yaml.safe_load(CONFIG.read_text())
    data = pd.DataFrame(
        {
            "wind_available_mw": [0.0, 145.0, 30.0],
            "electricity_price": [10.0, -30.0, 20.0],
        }
    )
    return cfg, data


def test_all_registered_controls_are_numerical_and_isolated():
    original, data = fixture()
    cases = yaml.safe_load(PROTOCOL.read_text())["registered_cases"]
    assert len(cases) == 17 and "baseline" in cases
    for name, entry in cases.items():
        changed, altered = perturb_config(original, entry, data)
        assert original == yaml.safe_load(CONFIG.read_text())
        assert changed != original or not altered.equals(data) or name == "baseline"
        assert all(np.isfinite(altered["wind_available_mw"]))
        assert (altered["wind_available_mw"] <= 150 + 1e-6).all()


def test_reject_unknown_or_nonnumeric_config_drift():
    cfg, frame = fixture()
    with pytest.raises(ValueError, match="Unregistered"):
        perturb_config(cfg, {"path": "nonsense.field", "value": 1.0}, frame)
    with pytest.raises(ValueError, match="finite"):
        perturb_config(cfg, {"path": "hydrogen_demand.daily_average_kg", "value": np.nan}, frame)
    with pytest.raises(ValueError, match="no perturbation"):
        perturb_config(cfg, {"path": None, "value": 5}, frame)


def test_baseline_and_one_control_run_with_independent_solver_verification(tmp_path):
    b = run_case("baseline", tmp_path / "reference", hours=24, time_limit=180)
    x = run_case("slow_ramp", tmp_path / "ramp", hours=24, time_limit=180)
    for path, result in ((tmp_path / "reference", b), (tmp_path / "ramp", x)):
        assert result["status"] == "VERIFIED_PARTIAL_24H_SMOKE_NOT_ANNUAL"
        assert result["stage_a_verified"] and result["stage_b_verified"]
        assert (path / "stage_a.csv").exists()
        assert (path / "stage_b.csv").exists()
        assert json.loads((path / "manifest.json").read_text())["status"] == result["status"]
    assert b["case_config_sha256"] != x["case_config_sha256"]
    with pytest.raises(FileExistsError):
        run_case("baseline", tmp_path / "reference")


def test_never_silently_replace_invalid_or_unregistered_inputs(tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        run_case("unregistered", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
