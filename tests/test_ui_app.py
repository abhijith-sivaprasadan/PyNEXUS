"""Smoke tests for ui/app.py using Streamlit's official AppTest framework.

These are not a substitute for actually running the app in a browser (done
once manually to produce the README screenshots) — they catch import/syntax
errors and exceptions during a normal render, which is what a CI environment
without a browser can actually verify.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest.from_file resolves a relative path against the file that calls
# it (i.e. this test file's directory), not the working directory. A
# relative "ui/app.py" was tried first and passed locally, then failed in
# CI with FileNotFoundError looking for tests/ui/app.py — an absolute path
# sidesteps the ambiguity entirely rather than depending on resolution
# behaviour that isn't consistent across environments.
APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def test_app_loads_without_exceptions() -> None:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_app_has_four_tabs() -> None:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert len(at.tabs) == 4


def test_reconciliation_tab_computes_with_default_values() -> None:
    """The reconciliation tab needs no file upload or solve — it should show
    a real, non-error result with the pre-filled default sensor values."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception
    # Default values (production=1000, to_store=200, from_store=50,
    # to_pipeline=850) are the exactly-balanced fixture from
    # tests/test_reconciliation.py — must report no gross error.
    metric_labels = [m.label for m in at.tabs[3].metric]
    assert any("Chi-square" in label for label in metric_labels)
