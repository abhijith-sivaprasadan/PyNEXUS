"""WP2 registered matrix: scenario isolation, algebra, and short-horizon smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from research.interventions import (
    PROTOCOL,
    analyse_contrasts,
    config_for_case,
    evaluate_case,
    registered_cases,
    run_matrix,
)


@pytest.fixture
def registration():
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


@pytest.fixture
def original():
    return yaml.safe_load(Path("configs/annual.yaml").read_text(encoding="utf-8"))


def test_matrix_is_exactly_registered_17_unique_cases(registration):
    cases = registered_cases(registration["order"])
    assert len(cases) == 17
    assert len({name for name, _ in cases}) == 17
    assert cases[0] == ("baseline", ())
    assert cases[-1] == ("all", ("W", "E", "G", "S", "eta"))
    assert sum(len(items) == 2 for _, items in cases) == 10


def test_every_single_factor_changes_only_one_registered_config_field(registration, original):
    for name in registration["order"]:
        changed, scale = config_for_case(original, registration, [name])
        active = registration["interventions"][name]
        section, field = active["field"].split(".", 1)
        assert changed[section][field] == pytest.approx(active["target"])
        assert changed["hydrogen_demand"] == original["hydrogen_demand"]
        assert changed["hydrogen_storage"]["initial_level_kg"] == 10000
        assert changed["hydrogen_storage"]["final_level_min_kg"] == 10000
        if name == "W":
            assert scale == pytest.approx(1.5)
        else:
            assert scale == pytest.approx(1.0)
        for section_name, settings in original.items():
            if not isinstance(settings, dict):
                continue
            for key, value in settings.items():
                if (section_name, key) != (section, field):
                    assert changed[section_name][key] == value


def test_all_interventions_apply_without_mutating_source(registration, original):
    snapshot = json.dumps(original, sort_keys=True)
    changed, multiplier = config_for_case(original, registration, registration["order"])
    assert multiplier == pytest.approx(1.5)
    assert changed["electrolyzer"]["rated_power_mw"] == 125
    assert changed["electrolyzer"]["nominal_efficiency"] == pytest.approx(0.75)
    assert changed["grid"]["connection_capacity_mw"] == 100
    assert changed["hydrogen_storage"]["capacity_kg"] == 40000
    assert json.dumps(original, sort_keys=True) == snapshot


def test_efficiency_update_is_consumed_by_dispatch_conversion(registration, original, tmp_path):
    old, _ = config_for_case(original, registration, ())
    new, _ = config_for_case(original, registration, ("eta",))
    old_path, new_path = tmp_path / "old.yaml", tmp_path / "new.yaml"
    old_path.write_text(yaml.safe_dump(old), encoding="utf-8")
    new_path.write_text(yaml.safe_dump(new), encoding="utf-8")
    assert ElectrolyzerDispatchOptimizer(str(new_path)).h2_coeff > (
        ElectrolyzerDispatchOptimizer(str(old_path)).h2_coeff
    )


def test_pairwise_formula_calculates_real_fractional_interaction():
    cases = []
    base = 0.30
    effects = {"W": 0.05, "E": 0.02, "G": 0.08, "S": 0.0, "eta": 0.01}
    order = ["W", "E", "G", "S", "eta"]
    for name, active in registered_cases(order):
        shortfall = base - sum(effects.get(a, 0) for a in active)
        if name == "W+E":
            shortfall -= 0.03
        cases.append(
            {
                "scenario": name,
                "status": "VERIFIED_TWO_STAGE_SOLVE",
                "metrics": {"stage_a_shortfall_fraction": shortfall},
            }
        )
    report = analyse_contrasts(cases, order)
    we = next(x for x in report["pairwise_interactions"] if x["pair"] == "W+E")
    assert we["interaction_fraction"] == pytest.approx(0.03)
    assert report["single_benefits_fraction"]["S"] == pytest.approx(0)


def test_bad_reference_value_fails_closed(registration, original):
    wrong = original.copy()
    wrong["grid"] = dict(original["grid"], connection_capacity_mw=75)
    with pytest.raises(ValueError, match="reference mismatch"):
        config_for_case(wrong, registration, ("G",))


def test_reference_and_wind_intervention_short_solve(tmp_path, registration, original):
    source = pd.DataFrame(
        {
            "wind_available_mw": np.array([10.0, 100.0, 20.0, 100.0]),
            "electricity_price": np.array([12.0, 18.0, 21.0, 6.0]),
            "carbon_intensity": np.array([200.0] * 4),
        }
    )
    baseline = evaluate_case(
        ("baseline", ()), original, source, registration, tmp_path / "baseline", time_limit=60
    )
    wind = evaluate_case(
        ("W", ("W",)), original, source, registration, tmp_path / "W", time_limit=60
    )
    assert baseline["status"] == "VERIFIED_TWO_STAGE_SOLVE"
    assert wind["metrics"]["wind_available_mwh"] == pytest.approx(
        1.5 * baseline["metrics"]["wind_available_mwh"]
    )
    assert wind["metrics"]["stage_a_shortfall_kg"] <= (
        baseline["metrics"]["stage_a_shortfall_kg"] + 1e-3
    )
    for case in (baseline, wind):
        assert all(case["verification_a"].values())
        assert all(case["verification_b"].values())
        assert (tmp_path / case["scenario"] / "stage_a.csv").exists()
        assert (tmp_path / case["scenario"] / "stage_b.csv").exists()
        dispatch = pd.read_csv(tmp_path / case["scenario"] / "stage_b.csv")
        online = dispatch["online_status"].to_numpy(dtype=int)
        power = dispatch["power_optimized_mw"].to_numpy(dtype=float)
        assert set(online).issubset({0, 1})
        assert np.all(online[power > 1e-3] == 1)


def test_duplicate_output_path_never_overwrites(tmp_path):
    path = tmp_path / "prior"
    path.mkdir()
    with pytest.raises(FileExistsError):
        run_matrix(path, hours=4, scenario_filter="baseline")
