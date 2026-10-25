"""Matrix archiving and reproducible synthesis fail closed on missing evidence."""

import json

import pytest

from research import batch_evidence


def test_expected_wp4_and_wp5_matrices_are_each_exactly_seventeen_cases():
    for study in ("wp4", "wp5"):
        names = batch_evidence.expected_names(study)
        assert len(names) == 17
        assert len(set(names)) == 17
    assert "E+S" in batch_evidence.expected_names("wp5")


@pytest.mark.parametrize("kind", ["wp4", "wp5"])
def test_missing_annual_case_is_never_counted_as_completed(tmp_path, kind):
    root = tmp_path / "evidence"
    root.mkdir()
    with pytest.raises(ValueError, match="Incomplete/extraneous"):
        batch_evidence.generate(kind, root, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_verified_wp4_complete_synthesis_computes_adequacy_with_common_source(
    tmp_path, monkeypatch
):
    root = tmp_path / "evidence"
    root.mkdir()
    cases = batch_evidence.expected_names("wp4")
    for name in cases:
        (root / f"wp4-engineering-full-year-{name}").mkdir()

    def validated(_folder, name):
        return {
            "scenario": name,
            "source_sha256": "unchanged_wind",
            "protocol_sha256": "registered_protocol",
            "adequacy_percent": 85 if name == "baseline" else 90,
        }

    monkeypatch.setattr(batch_evidence, "verify_wp4", validated)
    archive = batch_evidence.generate("wp4", root, tmp_path / "summary")
    assert archive["scenario_count"] == 17
    assert archive["fully_costed_adequacy_frontier_established"] is False
    assert archive["records"][0]["adequacy_improvement_pp_vs_baseline"] == 0
    assert archive["records"][1]["adequacy_improvement_pp_vs_baseline"] == 5
    saved = json.loads((tmp_path / "summary" / "audit.json").read_text())
    assert saved["source_sha256"]["source_sha256"] == "unchanged_wind"


def test_cross_case_price_hash_mismatch_rejects_all_wp5_results(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    names = batch_evidence.expected_names("wp5")
    for name in names:
        (root / f"wp5-observed-full-year-{name}").mkdir()

    def validated(_folder, name):
        return {
            "scenario": name,
            "price_csv_sha256": "bad" if name == "all" else "good",
            "source_wind_sha256": "identical",
            "base_config_sha256": "identical",
            "adequacy_percent": 90,
        }

    monkeypatch.setattr(batch_evidence, "verify_wp5", validated)
    with pytest.raises(ValueError, match="different base sources"):
        batch_evidence.generate("wp5", root, tmp_path / "summary")
