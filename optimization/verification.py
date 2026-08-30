"""Independent numerical checks on exported dispatch, not solver expressions."""

import numpy as np


def verify_dispatch(opt, frame, objective_value, objective, demand_mode, carbon=None):
    tolerance = 1e-5
    p = frame["power_optimized_mw"].to_numpy(dtype=float)
    u = frame["online_status"].to_numpy(dtype=float)
    wind = frame["wind_available_mw"].to_numpy(dtype=float)
    slack = frame["demand_slack_kg_h"].to_numpy(dtype=float)
    h2 = p * opt.h2_coeff
    cost = p * frame["electricity_price"].to_numpy(dtype=float) * opt.dt
    cost_objective = cost.sum() + opt.DEMAND_PENALTY * slack.sum() * opt.dt
    emissions_objective = None
    if carbon is not None:
        emissions_objective = float(
            np.dot(p, carbon) * opt.dt + opt.DEMAND_PENALTY * slack.sum() * opt.dt
        )
    recomputed = cost_objective if objective == "minimize_cost" else emissions_objective
    checks = {
        "finite_results": bool(np.isfinite(frame.select_dtypes(include="number")).all().all()),
        "nonnegativity": bool(np.all(p >= -tolerance) and np.all(slack >= -tolerance)),
        "binary_linkage": bool(np.all(np.minimum(abs(u), abs(u - 1)) <= tolerance)),
        "wind_limits": bool(np.all(p <= wind + tolerance)),
        "electrolyser_bounds": bool(
            np.all(p >= opt.p_min * u - tolerance) and np.all(p <= opt.p_max * u + tolerance)
        ),
        "ramps": bool(np.all(np.abs(np.diff(p)) <= opt.max_ramp_mw + tolerance)),
        "pipeline_capacity": bool(np.all(h2 <= opt.max_h2_kg_s * 3600 + tolerance)),
        "demand_balance": bool(
            np.all(h2 + slack >= opt.hourly_demand_kg - tolerance)
            if demand_mode == "hourly"
            else h2.sum() * opt.dt >= opt.hourly_demand_kg * len(p) * opt.dt - tolerance
        ),
        "hydrogen_identity": bool(
            np.allclose(frame["h2_produced_kg_h"], h2, atol=tolerance, rtol=1e-8)
        ),
        "curtailment_identity": bool(
            np.allclose(frame["curtailment_mw"], wind - p, atol=tolerance, rtol=1e-8)
        ),
        "cost_identity": bool(np.allclose(frame["cost_eur"], cost, atol=tolerance, rtol=1e-8)),
        "objective_recomputed": bool(
            recomputed is not None
            and np.isclose(objective_value, recomputed, atol=tolerance, rtol=1e-8)
        ),
    }
    return {
        "passed": all(checks.values()),
        "absolute_tolerance": tolerance,
        "relative_identity_tolerance": 1e-8,
        "checks": checks,
        "cost_objective": float(cost_objective),
        "emissions_objective": emissions_objective,
    }
