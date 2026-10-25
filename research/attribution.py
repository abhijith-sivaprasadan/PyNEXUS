"""WP3: independent constraint attribution from the frozen, verified WP2 archive.

This is a post-solve analysis; no new optimizer objective or scenarios.
It checks artifact hashes, reconstructs physical and economic identities,
then computes structural production bounds and co-occurring constraints.
Coincidence of an active constraint and shortage is not causal attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from optimization.dispatch import ElectrolyzerDispatchOptimizer
from optimization.verification import verify_dispatch
from research.interventions import (
    PROTOCOL,
    analyse_contrasts,
    config_for_case,
    registered_cases,
    source_inputs,
)
from research.physical_bounds import optimistic_shortfall_lower_bound_kg
from research.run_reference import INPUTS, sha256

TOL_MW = 1e-3
TOL_H2_KG = 1e-3
DATA_SOURCE_RUN = "40148299968"
ARTIFACT_NAME = "wp2-2023-17-case-study"


def _assert_close(actual, expected, *, atol=1e-5, label="value"):
    if not np.isfinite(actual) or not np.isfinite(expected):
        raise ValueError(f"{label} is nonfinite")
    if not np.isclose(actual, expected, rtol=1e-8, atol=atol):
        raise ValueError(f"{label}: observed {actual} != expected {expected}")


def _load_case(source_dir, case, summary_record, source, base_cfg, protocol):
    label, active = case
    folder = source_dir / label
    if (
        summary_record["scenario"] != label
        or summary_record["status"] != "VERIFIED_TWO_STAGE_SOLVE"
    ):
        raise ValueError(f"Missing independently verified WP2 case {label}")
    if tuple(summary_record["active_interventions"]) != active:
        raise ValueError(f"Wrong WP2 intervention label: {label}")
    record_path = folder / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    # The in-summary record must be identical to the independently archived record.
    if record != summary_record:
        raise ValueError(f"Mismatch between WP2 summary and case record: {label}")
    cfgpath = folder / "config.yaml"
    if sha256(cfgpath) != record["config_sha256"]:
        raise ValueError(f"Scenario config hash mismatch: {label}")
    cfg = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))
    expected_cfg, wind_scale = config_for_case(base_cfg, protocol, active)
    expected_cfg["simulation"]["time_horizon_hours"] = len(source)
    expected_cfg["optimization"]["time_limit_seconds"] = 600
    expected_cfg["optimization"]["mip_gap"] = protocol["mip_rel_gap"]
    if cfg != expected_cfg:
        raise ValueError(f"Unregistered configuration modification: {label}")
    opt = ElectrolyzerDispatchOptimizer(str(cfgpath))

    frames = {}
    for stage, objective, objective_value, cap in (
        ("a", "minimize_unserved", record["metrics"]["stage_a_shortfall_kg"], None),
        (
            "b",
            "minimize_cash_cost",
            record["metrics"]["stage_b_grid_settlement_eur_synthetic_prices"],
            record["metrics"]["stage_a_shortfall_kg"]
            + max(1e-4, 1e-8 * record["metrics"]["demand_h2_kg"]),
        ),
    ):
        name = f"stage_{stage}.csv"
        path = folder / name
        if sha256(path) != record["dispatch_hashes"][name]:
            raise ValueError(f"Dispatch hash mismatch: {label}/{name}")
        frame = pd.read_csv(path)
        if len(frame) != len(source) or not np.array_equal(
            frame["timestep"].to_numpy(), np.arange(len(source))
        ):
            raise ValueError(f"Wrong or nonchronological time rows: {label}/{name}")
        expected_wind = source["wind_available_mw"].to_numpy(dtype=float) * wind_scale
        if not np.allclose(frame["wind_available_mw"], expected_wind, atol=1e-7, rtol=1e-9):
            raise ValueError(f"Weather-input mismatch: {label}/{name}")
        if not np.allclose(
            frame["electricity_price"], source["electricity_price"], atol=1e-8, rtol=1e-9
        ):
            raise ValueError(f"Electricity-price input mismatch: {label}/{name}")
        evidence = verify_dispatch(
            opt,
            frame,
            float(objective_value),
            objective,
            "hourly",
            enable_grid=True,
            enable_storage=True,
            shortfall_cap_kg=cap,
        )
        if not evidence["passed"]:
            raise ValueError(
                f"Independent WP3 re-verification failed: {label}/{name}: {evidence['checks']}"
            )
        if record[f"verification_{stage}"] != evidence["checks"]:
            raise ValueError(f"Verification ledger changed: {label}/{name}")
        frames[stage] = frame

    a, b = frames["a"], frames["b"]
    hours = len(a)
    dt = opt.dt
    demand = float(a["h2_demand_kg_h"].sum() * dt)
    shortfall = float(a["demand_slack_kg_h"].sum() * dt)
    _assert_close(shortfall, record["metrics"]["stage_a_shortfall_kg"], label=label + " shortfall")
    _assert_close(demand, record["metrics"]["demand_h2_kg"], label=label + " demand")
    _assert_close(
        float(b["demand_slack_kg_h"].sum() * dt),
        record["metrics"]["stage_b_shortfall_kg"],
        label=label + " stage B shortfall",
    )
    physical_bound = optimistic_shortfall_lower_bound_kg(
        opt,
        a["wind_available_mw"].to_numpy(dtype=float),
        grid_enabled=True,
        storage_enabled=True,
    )
    _assert_close(
        physical_bound,
        record["metrics"]["independent_lower_bound_kg"],
        label=label + " optimistic physical bound",
    )
    if shortfall < physical_bound - 1e-3:
        raise ValueError(f"{label}: computed dispatch beats mathematical shortfall lower bound")

    grid_power = opt.grid_connection_mw
    source_power = a["wind_available_mw"].to_numpy(dtype=float)
    power = a["power_optimized_mw"].to_numpy(dtype=float)
    grid_import = a["grid_import_mw"].to_numpy(dtype=float)
    curtail = a["curtailment_mw"].to_numpy(dtype=float)
    stock = a["storage_level_kg"].to_numpy(dtype=float)
    charge = a["storage_charge_kg_h"].to_numpy(dtype=float)
    discharge = a["storage_discharge_kg_h"].to_numpy(dtype=float)
    slip = a["demand_slack_kg_h"].to_numpy(dtype=float)
    short = slip > TOL_H2_KG
    cap_power = min(opt.p_max, opt.max_h2_kg_s * 3600.0 / opt.h2_coeff)
    independent_instant_power = np.minimum(source_power + grid_power, cap_power)
    max_instant_h2 = independent_instant_power * opt.h2_coeff
    demand_per_hour = a["h2_demand_kg_h"].to_numpy(dtype=float)

    # These are overlapping diagnostic flags, not an additive decomposition
    # or binding dual values. They identify co-occurrence in Stage A dispatch.
    grid_at_cap = grid_import >= grid_power - TOL_MW
    electrolyser_at_cap = power >= opt.p_max - TOL_MW
    pipeline_at_cap = power * opt.h2_coeff >= opt.max_h2_kg_s * 3600 - TOL_H2_KG
    store_empty = stock <= TOL_H2_KG
    store_full = stock >= opt.storage_capacity_kg - TOL_H2_KG
    curtailing = curtail > TOL_MW
    storing = charge > TOL_H2_KG
    withdrawing = discharge > TOL_H2_KG

    # Transparent accounting: annual optimistic production, independent of
    # binary/ramp/storage charging. Uses the existing WP1 bound definition.
    optimistic_production_kg = float(max_instant_h2.sum() * dt)
    # Production-limited energy deficit after eligible inventory credit.
    inventory_credit_kg = max(0.0, opt.storage_initial_kg - opt.storage_final_min_kg)
    energy_deficit_lb_kg = max(0.0, demand - optimistic_production_kg - inventory_credit_kg)
    _assert_close(energy_deficit_lb_kg, physical_bound, label=label + " energy bound")
    residual_over_bound = shortfall - physical_bound

    def hours_mask(mask):
        return int(np.count_nonzero(mask))

    # Consecutive-hour duration, not chronological solver dual analysis.
    def longest_run(mask):
        v = np.asarray(mask, dtype=bool)
        if not v.any():
            return 0
        boundaries = np.flatnonzero(np.diff(np.r_[False, v, False].astype(int)))
        return int((boundaries[1::2] - boundaries[::2]).max())

    stage_b_imp = b["grid_import_mw"].to_numpy(dtype=float)
    stage_b_exp = b["grid_export_mw"].to_numpy(dtype=float)
    stage_b_chg = b["storage_charge_kg_h"].to_numpy(dtype=float)
    stage_b_dis = b["storage_discharge_kg_h"].to_numpy(dtype=float)
    monthly = []
    months = pd.date_range("2023-01-01", periods=hours, freq="h").month.to_numpy()
    for month in range(1, 13):
        sel = months == month
        monthly.append(
            {
                "scenario": label,
                "month": month,
                "hours": hours_mask(sel),
                "unserved_h2_kg_stage_a": float(slip[sel].sum() * dt),
                "wind_available_mwh": float(source_power[sel].sum() * dt),
                "grid_import_mwh_stage_a": float(grid_import[sel].sum() * dt),
                "grid_import_mwh_stage_b": float(stage_b_imp[sel].sum() * dt),
                "hours_shortfall_stage_a": hours_mask(short & sel),
            }
        )

    stats = {
        "scenario": label,
        "interventions": list(active),
        "annual_h2_demand_kg": demand,
        "annual_unserved_kg_stage_a": shortfall,
        "annual_unserved_fraction_stage_a": shortfall / demand,
        "optimistic_unavoidable_shortfall_kg": physical_bound,
        "unserved_above_optimistic_bound_kg": max(0.0, residual_over_bound),
        "bound_fraction_of_observed_shortfall": (
            physical_bound / shortfall if shortfall > 1e-6 else None
        ),
        "optimistic_production_kg": optimistic_production_kg,
        "annual_wind_availability_mwh": float(source_power.sum() * dt),
        "annual_actual_h2_production_kg_stage_a": float(a["h2_produced_kg_h"].sum() * dt),
        "shortfall_hours_stage_a": hours_mask(short),
        "longest_consecutive_shortfall_hours_stage_a": longest_run(short),
        "shortfall_coincident_grid_capacity_hours": hours_mask(short & grid_at_cap),
        "shortfall_coincident_electrolyser_capacity_hours": hours_mask(short & electrolyser_at_cap),
        "shortfall_coincident_pipeline_capacity_hours": hours_mask(short & pipeline_at_cap),
        "shortfall_coincident_storage_empty_hours": hours_mask(short & store_empty),
        "shortfall_coincident_wind_curtailment_hours": hours_mask(short & curtailing),
        "grid_import_at_cap_hours_stage_a": hours_mask(grid_at_cap),
        "electrolyser_at_cap_hours_stage_a": hours_mask(electrolyser_at_cap),
        "pipeline_at_cap_hours_stage_a": hours_mask(pipeline_at_cap),
        "storage_at_capacity_hours_stage_a": hours_mask(store_full),
        "storage_at_zero_hours_stage_a": hours_mask(store_empty),
        "storage_charge_hours_stage_a": hours_mask(storing),
        "storage_discharge_hours_stage_a": hours_mask(withdrawing),
        "storage_max_level_kg_stage_a": float(stock.max()),
        "storage_min_level_kg_stage_a": float(stock.min()),
        "storage_charge_total_kg_stage_a": float(charge.sum() * dt),
        "storage_discharge_total_kg_stage_a": float(discharge.sum() * dt),
        "simultaneous_import_export_hours_stage_a": hours_mask(
            (grid_import > TOL_MW) & (a["grid_export_mw"].to_numpy() > TOL_MW)
        ),
        "simultaneous_import_export_hours_stage_b": hours_mask(
            (stage_b_imp > TOL_MW) & (stage_b_exp > TOL_MW)
        ),
        "simultaneous_storage_charge_discharge_hours_stage_a": hours_mask(storing & withdrawing),
        "simultaneous_storage_charge_discharge_hours_stage_b": hours_mask(
            (stage_b_chg > TOL_H2_KG) & (stage_b_dis > TOL_H2_KG)
        ),
        "stage_b_synthetic_grid_settlement_eur": float(
            (b["electricity_price"] * (b["grid_import_mw"] - b["grid_export_mw"])).sum() * dt
        ),
        "instantaneous_max_production_less_than_demand_hours": hours_mask(
            max_instant_h2 + TOL_H2_KG < demand_per_hour
        ),
        "stage_a_independent_verification": True,
        "stage_b_independent_verification": True,
    }
    _assert_close(
        stats["stage_b_synthetic_grid_settlement_eur"],
        record["metrics"]["stage_b_grid_settlement_eur_synthetic_prices"],
        label=label + " grid settlement",
    )
    # Stage A physical adequacy, Stage B net market settlement only.
    # Explicitly avoid interpreting coincident flags as an additive cause split.
    return stats, monthly


def analyse_archive(source_dir):
    """Verify all 17 frozen cases and return qualified WP3 diagnostics."""
    source_dir = Path(source_dir)
    summary_path = source_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary["schema_version"] != "pynexus.wp2.result.v1"
        or summary["status"] != "COMPLETE_REGISTERED_MATRIX"
        or summary["hours"] != 8760
        or summary["failure_count"] != 0
    ):
        raise ValueError("WP3 requires complete verified 2023 WP2 archive")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if sha256(PROTOCOL) != summary["protocol_sha256"]:
        raise ValueError("WP2 experimental registration was modified")
    if sha256(INPUTS) != summary["input_sha256"]:
        raise ValueError("Original 2023 inputs no longer match WP2 source")
    source, cfg, mode = source_inputs(protocol, hours=8760)
    if mode != summary["historical_input_provenance_mode"]:
        raise ValueError("WP2 input provenance mode differs from current source")
    cases = registered_cases(protocol["order"])
    expected_names = [name for name, _ in cases]
    actual_names = [r["scenario"] for r in summary["cases"]]
    if actual_names != expected_names:
        raise ValueError("The exact pre-registered 17 scenario order is required")
    stats, monthly = [], []
    for case, record in zip(cases, summary["cases"], strict=True):
        s, m = _load_case(source_dir, case, record, source, cfg, protocol)
        stats.append(s)
        monthly.extend(m)

    # Reconstruct registered interaction contrasts instead of trusting report.
    contrasts = analyse_contrasts(summary["cases"], protocol["order"])
    original = summary["contrasts"]
    _assert_close(
        contrasts["baseline_shortfall_fraction"],
        original["baseline_shortfall_fraction"],
        label="baseline contrast",
    )
    for name, observed in contrasts["single_benefits_fraction"].items():
        _assert_close(observed, original["single_benefits_fraction"][name], label=name + " benefit")
    for p, q in zip(
        contrasts["pairwise_interactions"], original["pairwise_interactions"], strict=True
    ):
        if p["pair"] != q["pair"]:
            raise ValueError("Pair label changed")
        _assert_close(p["interaction_fraction"], q["interaction_fraction"], label=p["pair"])

    lookup = {v["scenario"]: v for v in stats}
    b, e, s, es = (lookup[k] for k in ("baseline", "E", "S", "E+S"))
    interaction = (
        b["annual_unserved_kg_stage_a"]
        - es["annual_unserved_kg_stage_a"]
        - (b["annual_unserved_kg_stage_a"] - e["annual_unserved_kg_stage_a"])
        - (b["annual_unserved_kg_stage_a"] - s["annual_unserved_kg_stage_a"])
    )
    interaction_pp = interaction / b["annual_h2_demand_kg"] * 100
    original_es = next(x for x in contrasts["pairwise_interactions"] if x["pair"] == "E+S")
    _assert_close(
        interaction_pp, original_es["interaction_fraction"] * 100, label="E+S interaction"
    )

    focus = {
        "baseline_unavoidable_fraction_of_observed_shortfall": b[
            "bound_fraction_of_observed_shortfall"
        ],
        "baseline_shortfall_above_independent_energy_bound_kg": b[
            "unserved_above_optimistic_bound_kg"
        ],
        "storage_only_shortfall_change_kg": (
            b["annual_unserved_kg_stage_a"] - s["annual_unserved_kg_stage_a"]
        ),
        "storage_conditional_benefit_after_electrolyser_expansion_kg": (
            e["annual_unserved_kg_stage_a"] - es["annual_unserved_kg_stage_a"]
        ),
        "electrolyser_storage_interaction_percentage_points": interaction_pp,
        "grid_upgrade_unserved_h2_kg": lookup["G"]["annual_unserved_kg_stage_a"],
        "grid_upgrade_is_grid_power_dependence_not_renewable_certification": True,
    }
    return {
        "schema_version": "pynexus.wp3.attribution.v1",
        "status": "VERIFIED_POST_SOLVE_DIAGNOSTICS",
        "wp2_source_run_id": DATA_SOURCE_RUN,
        "wp2_source_artifact": ARTIFACT_NAME,
        "wp2_summary_sha256": sha256(summary_path),
        "wp2_source_commit": summary.get("research_version"),
        "registered_protocol_sha256": sha256(PROTOCOL),
        "source_input_sha256": sha256(INPUTS),
        "scenario_count": len(stats),
        "case_diagnostics": stats,
        "monthly_diagnostics": monthly,
        "interaction_contrasts": contrasts,
        "research_focus": focus,
        "limitations": [
            "Capacity ceilings are optimistic mathematical bounds, not observed causal attribution.",
            "Active-hour counts overlap; no additive allocation of shortage to constraints is valid.",
            "Stage A minimizes unmet hydrogen; Stage B dispatch responds to synthetic prices.",
            "No proof that grid imports create renewable or low-carbon hydrogen.",
            "No independently observed hydrogen facility validation or interannual robustness.",
            "Simultaneous import/export or storage flows indicate model degeneracy, not profitability.",
        ],
    }


def _markdown(result):
    lines = [
        "# WP3: 2023 constraint attribution — verified archived results",
        "",
        "**Evidence class:** author-executed mathematical diagnostics on verified WP2 dispatch.",
        "No industrial validation, causal dual inference, market-price evidence or certification claims.",
        "",
        f"WP2 source: GitHub Actions run {result['wp2_source_run_id']}, "
        f"artifact \`{result['wp2_source_artifact']}\`.",
        "",
        "## Main case comparisons",
        "",
        "| Scenario | Shortfall (%) | Bound (t H2) | Observed minus bound (t H2) "
        "| Shortfall hours | Grid-cap coincidence hours | Storage peak (t) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["case_diagnostics"]:
        lines.append(
            f"| {row['scenario']} | {row['annual_unserved_fraction_stage_a'] * 100:.3f} "
            f"| {row['optimistic_unavoidable_shortfall_kg'] / 1000:.2f} "
            f"| {row['unserved_above_optimistic_bound_kg'] / 1000:.2f} "
            f"| {row['shortfall_hours_stage_a']} "
            f"| {row['shortfall_coincident_grid_capacity_hours']} "
            f"| {row['storage_max_level_kg_stage_a'] / 1000:.2f} |"
        )
    focus = result["research_focus"]
    lines.extend(
        [
            "",
            "## Structural deductions and limitations",
            "",
            f"- Baseline: {focus['baseline_unavoidable_fraction_of_observed_shortfall'] * 100:.3f}% "
            "of its reported unmet hydrogen is at least the independent optimistic "
            "annual energy-capacity lower bound. This explains **why storage volume "
            "alone cannot resolve the dominant shortage** in this particular system; "
            "it does not establish the relative marginal values of technologies.",
            f"- Storage-only improvement: {focus['storage_only_shortfall_change_kg']:.3f} kg.",
            f"- Adding storage after electrolyser expansion: "
            f"{focus['storage_conditional_benefit_after_electrolyser_expansion_kg']:.3f} kg "
            "additional delivery under the registered assumptions.",
            f"- E+S finite interaction: "
            f"{focus['electrolyser_storage_interaction_percentage_points']:+.3f} percentage points.",
            "- Zero unserved hydrogen in the grid expansion case means the *model* "
            "meets demand with a larger grid allowance, not that delivered hydrogen "
            "meets renewable-energy certification or has low life-cycle emissions.",
            "- Grid/plant/storage cap counts are overlapping **co-occurrences** "
            "with shortfall, not independent causes or statistically identified effects.",
            "- The dataset has one ERA5-derived weather year and synthetic prices. "
            "Negative net settlement is not project profitability.",
            "",
            "## Import/export and storage circulation diagnostics",
            "",
            "| Scenario | Simultaneous grid exchange Stage A / B hours "
            "| Simultaneous charge/discharge Stage A / B hours |",
            "|---|---:|---:|",
        ]
    )
    for row in result["case_diagnostics"]:
        lines.append(
            f"| {row['scenario']} | "
            f"{row['simultaneous_import_export_hours_stage_a']} / "
            f"{row['simultaneous_import_export_hours_stage_b']} | "
            f"{row['simultaneous_storage_charge_discharge_hours_stage_a']} / "
            f"{row['simultaneous_storage_charge_discharge_hours_stage_b']} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Source archive summary SHA-256: \`{result['wp2_summary_sha256']}\`",
            f"- Registered WP2 protocol SHA-256: \`{result['registered_protocol_sha256']}\`",
            f"- Parent source revision reported by WP2: \`{result['wp2_source_commit']}\`",
            "- Each archived stage CSV, YAML and verification ledger was checked against "
            "the archived record and the unchanged registered source inputs.",
            "- Monthly scenario data and all statistics are machine-readable in "
            "\`monthly.csv\`, \`diagnostics.csv\` and \`analysis.json\`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp2-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Research archive output already exists")
    analysis = analyse_archive(args.wp2_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(
        json.dumps(analysis, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",

    )
    pd.DataFrame(analysis["case_diagnostics"]).to_csv(args.output / "diagnostics.csv", index=False)
    pd.DataFrame(analysis["monthly_diagnostics"]).to_csv(args.output / "monthly.csv", index=False)
    (args.output / "RESEARCH_FINDINGS.md").write_text(_markdown(analysis), encoding="utf-8")
    f = analysis["research_focus"]
    print(
        "WP3 VERIFIED: "
        f"baseline_bound_fraction={f['baseline_unavoidable_fraction_of_observed_shortfall']:.6f}; "
        f"E+S_storage_conditional_gain_kg="
        f"{f['storage_conditional_benefit_after_electrolyser_expansion_kg']:.3f}; "
        f"E+S_interaction_pp={f['electrolyser_storage_interaction_percentage_points']:+.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
