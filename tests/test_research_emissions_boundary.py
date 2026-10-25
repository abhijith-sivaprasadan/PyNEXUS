"""Unit tests for strictly scoped operational carbon arithmetic; no real CO2 data."""

import numpy as np
import pandas as pd
import pytest

from research.emissions_accounting import grid_import_emissions


def dispatch(imports=(10.0, 0.0), exports=(0.0, 5.0)):
    return pd.DataFrame(
        {
            "grid_import_mw": imports,
            "grid_export_mw": exports,
            "power_optimized_mw": [15.0, 20.0],
        }
    )


def test_two_hour_grid_only_no_export_credit():
    result = grid_import_emissions(dispatch(), [100.0, 500.0], timestep_hours=1)
    assert result["grid_import_emissions_kg_co2"] == pytest.approx(1000.0)
    assert result["grid_export_mwh_not_credited"] == pytest.approx(5.0)
    assert result["scope"] == "OPERATIONAL_GRID_IMPORT_ONLY_NO_EXPORT_CREDIT"


@pytest.mark.parametrize("intensities", [[20.0], [20.0, -1.0], [np.nan, 3.0]])
def test_reject_noncomparable_emissions_factors(intensities):
    with pytest.raises(ValueError):
        grid_import_emissions(dispatch(), intensities)


def test_reject_simultaneous_grid_exchange():
    with pytest.raises(ValueError, match="Simultaneous"):
        grid_import_emissions(dispatch(imports=(1.0, 2.0)), [100.0, 100.0])


def test_no_electricity_use_double_counted_as_imports():
    r = grid_import_emissions(
        dispatch(imports=(0.0, 0.0), exports=(0.0, 0.0)),
        [500.0, 500.0],
    )
    assert r["grid_import_emissions_kg_co2"] == 0.0
