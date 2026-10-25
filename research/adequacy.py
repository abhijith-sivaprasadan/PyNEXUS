"""Two-stage physical-adequacy-first solver for WP1.

Stage A minimizes total unmet hydrogen (kg), independent of prices.
Stage B minimizes net grid settlement (EUR), subject to Stage A's optimal
shortfall plus a declared numerical tolerance.

The economics exclude onsite wind CAPEX/OPEX and opportunity costs.
This module does not claim a project-wide levelised hydrogen cost.
"""

from __future__ import annotations

import numpy as np

from optimization.verification import verify_dispatch
from research.physical_bounds import optimistic_shortfall_lower_bound_kg


def solve_adequacy_first(
    optimiser,
    wind_power_mw,
    electricity_prices,
    *,
    enable_storage=True,
    numerical_slack_kg=None,
    grid_exchange_mode="legacy",
):
    """Return independently verified Stage A and Stage B dispatch results.

    The same optimiser instance, physical configuration and arrays are used
    for both stages. This study currently requires a grid-connected hourly
    model with heat coupling disabled. An annual run is not claimed until an
    actual solved/verified manifest has been archived.
    """
    wind = np.asarray(wind_power_mw, dtype=float)
    prices = np.asarray(electricity_prices, dtype=float)
    if wind.ndim != 1 or prices.shape != wind.shape or not np.isfinite(wind).all():
        raise ValueError("wind and price inputs must be finite matching 1D arrays")
    if not np.isfinite(prices).all() or len(wind) == 0:
        raise ValueError("prices must be finite and the horizon nonempty")
    if optimiser.mip_gap > 1e-4:
        raise ValueError("Research adequacy solves require mip_rel_gap <= 1e-4")

    common = dict(
        demand_mode="hourly",
        enable_grid=True,
        enable_storage=enable_storage,
        grid_exchange_mode=grid_exchange_mode,
    )
    stage_a = optimiser.optimize(wind, prices, objective="minimize_unserved", **common)
    if stage_a["status"] != "optimal":
        raise RuntimeError(f"Stage A did not solve to reported optimality: {stage_a['status']}")
    verified_a = verify_dispatch(
        optimiser,
        stage_a["results_df"],
        stage_a["objective_value"],
        "minimize_unserved",
        "hourly",
        enable_grid=True,
        enable_storage=enable_storage,
    )
    if not verified_a["passed"]:
        raise RuntimeError(f"Independent Stage A verification failed: {verified_a['checks']}")

    total_demand = optimiser.hourly_demand_kg * len(wind) * optimiser.dt
    default_tol = max(1e-4, 1e-8 * total_demand)
    tol = default_tol if numerical_slack_kg is None else float(numerical_slack_kg)
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("numerical_slack_kg must be finite and nonnegative")
    lower_bound = optimistic_shortfall_lower_bound_kg(
        optimiser,
        wind,
        grid_enabled=True,
        storage_enabled=enable_storage,
    )
    if float(stage_a["objective_value"]) + 1e-4 < lower_bound:
        raise RuntimeError("Stage A violates independent physical shortfall bound")
    cap = float(stage_a["objective_value"]) + tol

    stage_b = optimiser.optimize(
        wind,
        prices,
        objective="minimize_cash_cost",
        shortfall_cap_kg=cap,
        **common,
    )
    if stage_b["status"] != "optimal":
        raise RuntimeError(f"Stage B did not solve to reported optimality: {stage_b['status']}")
    verified_b = verify_dispatch(
        optimiser,
        stage_b["results_df"],
        stage_b["objective_value"],
        "minimize_cash_cost",
        "hourly",
        enable_grid=True,
        enable_storage=enable_storage,
        shortfall_cap_kg=cap,
    )
    if not verified_b["passed"]:
        frame = stage_b["results_df"]
        p = frame["power_optimized_mw"].to_numpy(dtype=float)
        u = frame["online_status"].to_numpy(dtype=float)
        lower = float(np.min(p - optimiser.p_min * u))
        upper = float(np.max(p - optimiser.p_max * u))
        raise RuntimeError(
            "Independent Stage B verification failed: "
            f"{verified_b['checks']}; min(p-p_min*u)={lower:.12g} MW; "
            f"max(p-p_max*u)={upper:.12g} MW; "
            f"min(p)={np.min(p):.12g}; max(p)={np.max(p):.12g}"
        )

    final_unserved = float(stage_b["results_df"]["demand_slack_kg_h"].sum() * optimiser.dt)
    if final_unserved > cap + 1e-5:
        raise RuntimeError("Stage B exceeded the declared adequacy allowance")
    if grid_exchange_mode == "exclusive":
        for label, stage in (("A", stage_a), ("B", stage_b)):
            df = stage["results_df"]
            overlap = np.minimum(
                df["grid_import_mw"].to_numpy(dtype=float),
                df["grid_export_mw"].to_numpy(dtype=float),
            )
            if np.any(overlap > 1e-6):
                raise RuntimeError(f"Exclusive grid mode violated by Stage {label}")
    return {
        "total_demand_kg": total_demand,
        "optimistic_shortfall_lower_bound_kg": lower_bound,
        "stage_a_unserved_kg": float(stage_a["objective_value"]),
        "stage_b_unserved_kg": final_unserved,
        "numerical_allowance_kg": tol,
        "shortfall_cap_kg": cap,
        "stage_b_grid_settlement_eur": float(stage_b["objective_value"]),
        "stage_a": stage_a,
        "stage_b": stage_b,
        "verification_a": verified_a,
        "verification_b": verified_b,
        "grid_exchange_mode": grid_exchange_mode,
        "economics_scope": "net grid imports less exports only; excludes wind CAPEX/OPEX",
    }
