"""Explicit, source-limited electricity-grid emissions accounting.

This is a post-dispatch reporting boundary, NOT the legacy minimize_emissions
objective (which mixes kg CO2 and shortfall penalty units). It is also NOT
life-cycle carbon accounting or RFNBO certification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def grid_import_emissions(dispatch, grid_kg_co2_per_mwh, *, timestep_hours=1.0):
    """Compute only physically purchased grid-electricity emissions in kg CO2.

    An externally documented time-aligned grid-intensity vector is required.
    Wind is assigned no emissions in this operational direct-electricity scope;
    grid exports receive no avoided-emissions credit. No missing values,
    broadcasting or alignment guesses are permitted.
    """
    if not isinstance(dispatch, pd.DataFrame) or not {
        "grid_import_mw",
        "grid_export_mw",
        "power_optimized_mw",
    }.issubset(dispatch.columns):
        raise ValueError("Dispatch must contain explicit grid power flow columns")
    g = np.asarray(dispatch["grid_import_mw"], dtype=float)
    exp = np.asarray(dispatch["grid_export_mw"], dtype=float)
    electrolyser = np.asarray(dispatch["power_optimized_mw"], dtype=float)
    intensity = np.asarray(grid_kg_co2_per_mwh, dtype=float)
    if (
        intensity.ndim != 1
        or intensity.shape != g.shape
        or not np.isfinite(intensity).all()
        or (intensity < 0).any()
        or not np.isfinite(g).all()
        or not np.isfinite(exp).all()
        or not np.isfinite(electrolyser).all()
        or (g < -1e-7).any()
        or (exp < -1e-7).any()
        or (electrolyser < -1e-7).any()
    ):
        raise ValueError(
            "Grid-intensity and dispatch inputs must be aligned, finite and nonnegative"
        )
    if not np.isfinite(timestep_hours) or timestep_hours <= 0:
        raise ValueError("Invalid timestep duration")
    if np.any(np.minimum(g, exp) > 1e-6):
        raise ValueError("Simultaneous imports and exports: grid emissions boundary ambiguous")
    hourly = g * intensity * timestep_hours
    return {
        "scope": "OPERATIONAL_GRID_IMPORT_ONLY_NO_EXPORT_CREDIT",
        "grid_import_emissions_kg_co2": float(hourly.sum()),
        "grid_import_mwh": float(g.sum() * timestep_hours),
        "grid_export_mwh_not_credited": float(exp.sum() * timestep_hours),
        "emission_intensity_input_required": "EXTERNAL_TIME_ALIGNED_GRID_KG_CO2_PER_MWH",
        "not_claimed": [
            "electricity life-cycle CO2e",
            "offshore-wind embodied emissions",
            "hydrogen certification or RFNBO",
            "additionality or marginal displacement",
            "legacy minimize_emissions objective validity",
        ],
    }
