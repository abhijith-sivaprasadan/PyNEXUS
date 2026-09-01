"""Independent numerical checks on exported dispatch, not solver expressions."""

import numpy as np


def verify_dispatch(
    opt,
    frame,
    objective_value,
    objective,
    demand_mode,
    carbon=None,
    enable_storage=False,
    enable_grid=False,
    enable_heat=False,
):
    tolerance = 1e-5
    p = frame["power_optimized_mw"].to_numpy(dtype=float)
    u = frame["online_status"].to_numpy(dtype=float)
    wind = frame["wind_available_mw"].to_numpy(dtype=float)
    slack = frame["demand_slack_kg_h"].to_numpy(dtype=float)
    h2 = p * opt.h2_coeff
    dt = opt.dt

    checks = {
        "finite_results": bool(np.isfinite(frame.select_dtypes(include="number")).all().all()),
        "nonnegativity": bool(np.all(p >= -tolerance) and np.all(slack >= -tolerance)),
        "binary_linkage": bool(np.all(np.minimum(abs(u), abs(u - 1)) <= tolerance)),
        "electrolyser_bounds": bool(
            np.all(p >= opt.p_min * u - tolerance) and np.all(p <= opt.p_max * u + tolerance)
        ),
        "ramps": bool(np.all(np.abs(np.diff(p)) <= opt.max_ramp_mw + tolerance)),
        "pipeline_capacity": bool(np.all(h2 <= opt.max_h2_kg_s * 3600 + tolerance)),
        "hydrogen_identity": bool(
            np.allclose(frame["h2_produced_kg_h"], h2, atol=tolerance, rtol=1e-8)
        ),
    }

    # --- Wind/grid balance and curtailment: the constraint dispatch.py builds
    # depends on enable_grid, so the independent check must branch the same way. ---
    if enable_grid:
        g_imp = frame["grid_import_mw"].to_numpy(dtype=float)
        g_exp = frame["grid_export_mw"].to_numpy(dtype=float)
        curtail = frame["curtailment_mw"].to_numpy(dtype=float)
        checks["grid_balance"] = bool(
            np.allclose(wind + g_imp, p + g_exp + curtail, atol=tolerance, rtol=1e-8)
        )
        checks["grid_connection_limit"] = bool(
            np.all(g_imp <= opt.grid_connection_mw + tolerance)
            and np.all(g_exp <= opt.grid_connection_mw + tolerance)
            and np.all(g_imp >= -tolerance)
            and np.all(g_exp >= -tolerance)
        )
        cost = p * frame["electricity_price"].to_numpy(dtype=float) * dt
        cost = cost + g_imp * frame["electricity_price"].to_numpy(dtype=float) * dt
        cost = cost - g_exp * frame["electricity_price"].to_numpy(dtype=float) * dt
    else:
        checks["wind_limits"] = bool(np.all(p <= wind + tolerance))
        checks["curtailment_identity"] = bool(
            np.allclose(frame["curtailment_mw"], wind - p, atol=tolerance, rtol=1e-8)
        )
        cost = p * frame["electricity_price"].to_numpy(dtype=float) * dt

    checks["cost_identity"] = bool(np.allclose(frame["cost_eur"], cost, atol=tolerance, rtol=1e-8))

    # --- Hydrogen demand balance and storage: same branching as dispatch.py's
    # h2_demand_hourly rule. ---
    h2_net = h2.copy()
    if enable_storage:
        h_in = frame["storage_charge_kg_h"].to_numpy(dtype=float)
        h_out = frame["storage_discharge_kg_h"].to_numpy(dtype=float)
        s = frame["storage_level_kg"].to_numpy(dtype=float)
        s_prev = np.concatenate(([opt.storage_initial_kg], s[:-1]))
        loss = opt.storage_loss_fraction_per_hour * s_prev * dt
        checks["storage_balance"] = bool(
            np.allclose(s, s_prev + (h_in - h_out) * dt - loss, atol=tolerance, rtol=1e-8)
        )
        checks["storage_terminal"] = bool(s[-1] >= opt.storage_final_min_kg - tolerance)
        checks["storage_charge_limit"] = bool(np.all(h_in <= h2 + tolerance))
        h2_net = h2 - h_in + h_out

    checks["demand_balance"] = bool(
        np.all(h2_net + slack >= opt.hourly_demand_kg - tolerance)
        if demand_mode == "hourly"
        else h2.sum() * dt >= opt.hourly_demand_kg * len(p) * dt - tolerance
    )

    cost_objective = cost.sum() + opt.DEMAND_PENALTY * slack.sum() * dt
    emissions_objective = None
    if carbon is not None:
        emissions = np.dot(p, carbon) * dt + opt.DEMAND_PENALTY * slack.sum() * dt
        if enable_grid:
            emissions += np.dot(frame["grid_import_mw"].to_numpy(dtype=float), carbon) * dt
        emissions_objective = float(emissions)

    # --- Heat coupling: additional balance/recursion checks, and the coupled
    # objective's boiler cost / hydrogen+heat value terms. ---
    if enable_heat:
        q_wh = frame["waste_heat_recovered_mw"].to_numpy(dtype=float)
        q_boiler = frame["boiler_output_mw"].to_numpy(dtype=float)
        q_dump = frame["heat_dumped_mw"].to_numpy(dtype=float)
        heat_slack = frame["heat_slack_mw"].to_numpy(dtype=float)
        heat_demand = frame["heat_demand_mw"].to_numpy(dtype=float)

        checks["waste_heat_identity"] = bool(
            np.allclose(
                q_wh, p * (1 - opt.eta_linearized) * opt.recoverable_heat_fraction, atol=tolerance
            )
        )

        if "heat_storage_level_mwh" in frame:
            q_hs_in = frame["heat_storage_charge_mw"].to_numpy(dtype=float)
            q_hs_out = frame["heat_storage_discharge_mw"].to_numpy(dtype=float)
            e_hs = frame["heat_storage_level_mwh"].to_numpy(dtype=float)
            e_prev = np.concatenate(([opt.heat_storage_initial_mwh], e_hs[:-1]))
            hloss = opt.heat_storage_loss_fraction_per_hour * e_prev * dt
            checks["heat_storage_balance"] = bool(
                np.allclose(e_hs, e_prev + (q_hs_in - q_hs_out) * dt - hloss, atol=tolerance)
            )
            checks["heat_storage_terminal"] = bool(
                e_hs[-1] >= opt.heat_storage_final_min_mwh - tolerance
            )
        else:
            q_hs_in = np.zeros_like(q_wh)
            q_hs_out = np.zeros_like(q_wh)

        checks["heat_charge_limit"] = bool(np.all(q_hs_in + q_dump <= q_wh + tolerance))
        delivered = q_wh - q_hs_in - q_dump + q_hs_out + q_boiler + heat_slack
        checks["heat_balance"] = bool(np.all(delivered >= heat_demand - tolerance))
        checks["boiler_limit"] = bool(np.all(q_boiler <= opt.boiler_max_output_mw + tolerance))

        heat_delivered = q_wh - q_hs_in - q_dump + q_hs_out
        h2_delivered = h2_net
        boiler_cost = (opt.boiler_fuel_cost_eur_per_mwh * q_boiler * dt).sum()
        heat_penalty = opt.HEAT_DEMAND_PENALTY * heat_slack.sum() * dt
        h2_value = opt.hydrogen_value_per_kg * (h2_delivered * dt).sum()
        heat_value = opt.heat_value_per_mwh * (heat_delivered * dt).sum()

        if objective == "minimize_cost":
            cost_objective = cost_objective + boiler_cost + heat_penalty - h2_value - heat_value
        elif emissions_objective is not None:
            boiler_emissions = (
                opt.boiler_emission_factor_kg_co2_per_mwh * q_boiler * dt
            ).sum()
            emissions_objective = emissions_objective + boiler_emissions + heat_penalty

    recomputed = cost_objective if objective == "minimize_cost" else emissions_objective
    checks["objective_recomputed"] = bool(
        recomputed is not None
        and np.isclose(objective_value, recomputed, atol=tolerance, rtol=1e-8)
    )

    return {
        "passed": all(checks.values()),
        "absolute_tolerance": tolerance,
        "relative_identity_tolerance": 1e-8,
        "checks": checks,
        "cost_objective": float(cost_objective),
        "emissions_objective": emissions_objective,
    }
