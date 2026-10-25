"""WP7 evidence and numerical-claim guards; archived end-to-end CI is separate."""

import json

import pytest

from research.paper_wp7 import EXPECTED, SHA, close, pinned_json, render_markdown


def test_matrix_registration_and_hashes():
    assert len(EXPECTED) == len(set(EXPECTED)) == 17
    assert len(SHA) == 4
    assert all(len(v) == 64 for v in SHA.values())


def test_refuse_untrusted_synthesis(tmp_path):
    doc = tmp_path / "summary.json"
    doc.write_text(json.dumps({"status": "COMPLETE_REGISTERED_MATRIX"}))
    with pytest.raises(ValueError, match="SHA-256"):
        pinned_json(doc, SHA["wp2_summary"], "WP2")
    with pytest.raises(ValueError, match="SHA-256"):
        pinned_json(tmp_path / "missing.json", SHA["wp4_analysis"], "WP4")


def test_numerical_claims_fail_closed():
    close(5.0, 5.0, "matching")
    with pytest.raises(ValueError, match="independent"):
        close(5.1, 5.0, "altered")
    with pytest.raises(ValueError, match="independent"):
        close(float("nan"), 5.0, "nonfinite")


def test_unresolved_result_placeholders_fail(tmp_path):
    page = tmp_path / "paper.md"
    page.write_text("{baseline_shortfall_pct} {es_extra_delivery_t}")
    with pytest.raises(ValueError, match="Missing registered"):
        render_markdown(page, {"baseline_shortfall_pct": "15.9"})
    actual = render_markdown(
        page, {"baseline_shortfall_pct": "15.9", "es_extra_delivery_t": "343.79"}
    )
    assert actual == "15.9 343.79"


def test_declared_placeholder_must_not_survive_rendering(tmp_path):
    page = tmp_path / "paper.md"
    page.write_text("{baseline_shortfall_pct}")
    with pytest.raises(ValueError, match="Unresolved"):
        render_markdown(page, {"baseline_shortfall_pct": "{baseline_fake}"})
