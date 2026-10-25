"""WP3 diagnostics: numerical logic, boundary discipline, and fail-closed guards."""

import json

import pytest

from research.attribution import _assert_close, _markdown, analyse_archive


def test_independent_comparison_never_accepts_bad_values():
    _assert_close(1.00000001, 1.0)
    with pytest.raises(ValueError, match="observed"):
        _assert_close(2, 1, label="unit-bound")
    with pytest.raises(ValueError, match="nonfinite"):
        _assert_close(float("nan"), 1.0)


def test_missing_archive_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyse_archive(tmp_path / "missing")


def test_partial_wp2_not_reportable(tmp_path):
    folder = tmp_path / "partial"
    folder.mkdir()
    (folder / "summary.json").write_text(
        json.dumps({
            "schema_version": "pynexus.wp2.result.v1",
            "status": "VERIFIED_PARTIAL_OR_SMOKE",
            "hours": 8760,
            "failure_count": 0,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete verified"):
        analyse_archive(folder)


def test_causal_language_stays_qualified():
    result = {
        "wp2_source_run_id": "control-run",
        "wp2_source_artifact": "artifact",
        "wp2_summary_sha256": "abcd",
        "registered_protocol_sha256": "efgh",
        "wp2_source_commit": "revision",
        "case_diagnostics": [{
            "scenario": "baseline",
            "annual_unserved_fraction_stage_a": 0.15875,
            "optimistic_unavoidable_shortfall_kg": 2_895_630,
            "unserved_above_optimistic_bound_kg": 1_653,
            "shortfall_hours_stage_a": 5000,
            "shortfall_coincident_grid_capacity_hours": 4000,
            "storage_max_level_kg_stage_a": 16000,
            "simultaneous_import_export_hours_stage_a": 0,
            "simultaneous_import_export_hours_stage_b": 0,
            "simultaneous_storage_charge_discharge_hours_stage_a": 0,
            "simultaneous_storage_charge_discharge_hours_stage_b": 0,
        }],
        "research_focus": {
            "baseline_unavoidable_fraction_of_observed_shortfall": 0.9994,
            "storage_only_shortfall_change_kg": 0.0,
            "storage_conditional_benefit_after_electrolyser_expansion_kg": 343792.77,
            "electrolyser_storage_interaction_percentage_points": 1.884,
        },
    }
    text = _markdown(result)
    assert "not causal attribution" in text.lower() or "not independent causes" in text
    assert "one ERA5-derived weather year" in text
    assert "certification" in text
    assert "343792.770" in text
