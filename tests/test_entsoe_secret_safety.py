"""Security tests: ENTSO-E transport errors must never echo API credentials."""

import pytest

from data.entsoe import fetch_entsoe_prices


def test_query_error_does_not_print_entsoe_token(monkeypatch):
    entsoe = pytest.importorskip("entsoe")
    secret = "test_entsoe_secret_do_not_expose_123456"
    monkeypatch.setenv("ENTSOE_API_KEY", secret)

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == secret

        def query_day_ahead_prices(self, country_code, start, end):
            raise RuntimeError(f"upstream URL securityToken={secret}&documentType=A44")

    monkeypatch.setattr(entsoe, "EntsoePandasClient", FakeClient)
    with pytest.raises(RuntimeError, match="ENTSO-E query failed") as caught:
        fetch_entsoe_prices("NL", "2023-01-01", "2023-01-02", "Europe/Amsterdam")
    assert secret not in str(caught.value)
    assert "securityToken" not in str(caught.value)


def test_client_construction_error_does_not_print_entsoe_token(monkeypatch):
    entsoe = pytest.importorskip("entsoe")
    secret = "test_entsoe_secret_do_not_expose_654321"
    monkeypatch.setenv("ENTSOE_API_KEY", secret)

    def fail_client(*args, **kwargs):
        raise ValueError(f"Cannot connect with securityToken={secret}")

    monkeypatch.setattr(entsoe, "EntsoePandasClient", fail_client)
    with pytest.raises(RuntimeError, match="ENTSO-E client could not be created") as caught:
        fetch_entsoe_prices("NL", "2023-01-01", "2023-01-02", "Europe/Amsterdam")
    assert secret not in str(caught.value)
    assert "securityToken" not in str(caught.value)
