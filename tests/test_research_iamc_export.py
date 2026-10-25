"""WP6 registry, physical accounting, provenance and real pyam round-trip tests."""

import copy
import json

import pytest

from research.iamc_export import (
    SCENARIOS,
    WP4_ANALYSIS_SHA256,
    YEARS,
    build_iamc_rows,
    load_registered_wp4,
    validate_synthesis,
    variable_dictionary,
    verify_pyam_roundtrip,
)


def representative_synthesis():
    results, metrics = [], []
    fractions = {
        "baseline": 0.20,
        "W": 0.16,
        "E": 0.12,
        "G": 0.00,
        "S": 0.20,
        "eta": 0.15,
        "E+S": 0.10,
    }
    for year in YEARS:
        hours = 8784 if year == 2024 else 8760
        demand = 50_000 * hours / 24
        for scenario in SCENARIOS:
            fraction = fractions[scenario]
            results.append(
                {
                    "year": year,
                    "scenario": scenario,
                    "unserved_h2_kg": demand * fraction,
                    "unserved_fraction": fraction,
                    "benefit_percentage_points": 100 * (fractions["baseline"] - fraction),
                    "wind_resource_mwh": 456789.0,
                    "wind_capacity_factor": 0.35,
                    "hours": hours,
                }
            )
        metrics.append(
            {
                "year": year,
                "provider": "Open-Meteo explicit ERA5 (NOT original CDS-2023 input)",
                "source_api_sha256": "a" * 64,
                "baseline_unserved_fraction": fractions["baseline"],
                "wind_capacity_factor": 0.35,
                "storage_conditional_benefit_after_E_pp": 2.0,
                "E_plus_S_interaction_pp": 2.0,
            }
        )
    return {
        "years": list(YEARS),
        "scenario_count": 35,
        "weather_year_count": 5,
        "annual_results": results,
        "annual_metrics": metrics,
    }


def test_complete_year_synthesis_and_local_variable_dictionary():
    definitions = variable_dictionary()
    assert definitions["registry_status"] == "LOCAL_PYNEXUS_DICTIONARY_NOT_OFFICIAL_IAMC"
    assert len(definitions["variables"]) == 8
    data = representative_synthesis()
    assert validate_synthesis(data, 50_000)
    exported = build_iamc_rows(data, 50_000, definitions)
    assert len(exported) == 250
    assert exported["year"].nunique() == 5
    assert exported["scenario"].nunique() == 7
    assert exported["variable"].nunique() == 8
    assert (exported["unit"] == "percentage point").sum() == 40
    assert (exported["year"] == 2024).sum() == 50
    unique = ["model", "scenario", "region", "variable", "unit", "year"]
    assert not exported.duplicated(unique).any()
    store = exported[
        (exported["year"] == 2024)
        & (exported["scenario"] == "E+S")
        & (exported["variable"].str.endswith("|Electrolyser Storage Interaction"))
    ]
    assert store.iloc[0]["value"] == pytest.approx(2.0)
    demand = exported[
        (exported["year"] == 2024)
        & (exported["scenario"] == "baseline")
        & (exported["variable"].str.endswith("|Hydrogen|Demand"))
    ]
    assert demand.iloc[0]["value"] == 18_300_000


def test_pyam_real_wide_csv_roundtrip(tmp_path):
    pytest.importorskip("pyam")
    frame = build_iamc_rows(representative_synthesis(), 50_000, variable_dictionary())
    written = tmp_path / "out.csv"
    count, version = verify_pyam_roundtrip(frame, written)
    assert written.is_file()
    assert count == len(frame) == 250
    assert version


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        (lambda x: x["annual_results"].pop(), "Missing annual"),
        (lambda x: x["annual_results"][0].update({"year": 2020}), "Scenario-year"),
        (lambda x: x["annual_results"][0].update({"hours": 8784}), "Leap-year"),
        (lambda x: x["annual_results"][0].update({"unserved_h2_kg": 1}), "identity"),
        (lambda x: x["annual_results"][1].update({"benefit_percentage_points": 4.5}), "benefit"),
        (lambda x: x["annual_results"][7].update({"wind_resource_mwh": 0}), "resource"),
        (lambda x: x["annual_metrics"][0].update({"E_plus_S_interaction_pp": 0}), "interaction"),
        (lambda x: x["annual_metrics"][0].update({"provider": "CDS"}), "source"),
    ],
)
def test_reject_corrupt_annual_records(mutator, pattern):
    result = copy.deepcopy(representative_synthesis())
    mutator(result)
    with pytest.raises(ValueError, match=pattern):
        validate_synthesis(result, 50_000)


def test_negative_or_nonfinite_metrics_fail_closed():
    result = representative_synthesis()
    result["annual_results"][0]["unserved_fraction"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_synthesis(result, 50_000)
    result = representative_synthesis()
    result["annual_results"][0]["unserved_fraction"] = 1.2
    with pytest.raises(ValueError, match="maximum"):
        validate_synthesis(result, 50_000)


def test_reject_arbitrary_relabelled_synthesis_even_if_status_claims_verified(tmp_path):
    source = tmp_path / "analysis.json"
    source.write_text(
        json.dumps(
            {
                "status": "COMPLETE_5_YEAR_35_SCENARIO_COMPARISON",
                "schema_version": "pynexus.wp4.interannual.v1",
            }
        ),
        encoding="utf-8",
    )
    assert len(WP4_ANALYSIS_SHA256) == 64
    with pytest.raises(ValueError, match="bytes differ"):
        load_registered_wp4(source)
