"""Optimistic, independent energy-capacity bounds for adequacy experiments."""
from __future__ import annotations

import numpy as np


def optimistic_shortfall_lower_bound_kg(
    optimiser, wind_power_mw, *, grid_enabled, storage_enabled
):
    """Lower bound on unavoidable H2 shortfall, without solving a MILP.

    The bound ignores ramping and electrolyser minimum-load rules, making
    generation MORE flexible than the real model. Therefore an optimum may
    have more unmet hydrogen, never less. It is not a feasibility certificate.
    """
    wind = np.asarray(wind_power_mw, dtype=float)
    if wind.ndim != 1 or not len(wind) or not np.isfinite(wind).all() or (wind < 0).any():
        raise ValueError("Expected finite, nonnegative one-dimensional wind data")
    grid = optimiser.grid_connection_mw if grid_enabled else 0.0
    power_cap = min(
        optimiser.p_max,
        optimiser.max_h2_kg_s * 3600.0 / optimiser.h2_coeff,
    )
    optimistic_power = np.minimum(np.minimum(wind + grid, power_cap), optimiser.p_max)
    optimistic_production = float(optimistic_power.sum() * optimiser.h2_coeff * optimiser.dt)
    initial_inventory_credit = (
        max(0.0, optimiser.storage_initial_kg - optimiser.storage_final_min_kg)
        if storage_enabled else 0.0
    )
    optimistic_delivery = optimistic_production + initial_inventory_credit
    required = optimiser.hourly_demand_kg * len(wind) * optimiser.dt
    return max(0.0, float(required - optimistic_delivery))
