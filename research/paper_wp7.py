"""WP7: generate an auditable manuscript from fixed WP2–WP4 and WP6 artifacts.

No results are invented or sourced from prose. Earlier hourly optimizations are
not rerun: original archives are hash pinned, then hourly Stage-A/Stage-B KPIs
are recalculated for all 17 frozen WP2 cases. WP4 carries earlier independent
35-case hourly validation; its exact synthesis bytes are pinned here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHA = {
    "wp2_summary": "c181159e139001b78e5d4da879c2b227bf05ab2c2c889603d899e2c527f705d0",
    "wp3_analysis": "9965af08e2a258100c816ef0d037d37bcdc4c83aa2e7a33f594546d253cfc204",
    "wp4_analysis": "ee1b475d1034a151d1e2e91b85ec3047aed6f0b756d042159d2305cfa466e907",
    "wp6_manifest": "e5a6ddad0a229aa5ff4abd9ebef78a8e7db578b7d291451aab0abc292683dade",
}
WP2_RUN = 40148299968
WP3_RUN = 40150436630
WP4_RUN = 40157038537
WP6_RUN = 40190690775
EXPECTED = (
    "baseline",
    "W",
    "E",
    "G",
    "S",
    "eta",
    "W+E",
    "W+G",
    "W+S",
    "W+eta",
    "E+G",
    "E+S",
    "E+eta",
    "G+S",
    "G+eta",
    "S+eta",
    "all",
)
WEATHER_SCENARIOS = ("baseline", "W", "E", "G", "S", "eta", "E+S")
YEARS = (2021, 2022, 2023, 2024, 2025)
MODEL = "PyNEXUS|OpenMeteo-ERA5|Fixed-System-Adequacy"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_json(path, hash_value, label):
    if not Path(path).is_file() or digest(path) != hash_value:
        raise ValueError(f"{label}: archived bytes missing or SHA-256 mismatch")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(value, expected, label, *, atol=1e-3):
    if not np.isfinite(float(value)) or not np.isclose(
        float(value), expected, rtol=1e-9, atol=atol
    ):
        raise ValueError(f"{label}: independent numerical check failed")


def audit_wp2(root):
    """Recompute saved Stage A/Stage B annual outcomes from archived hourly CSVs."""
    root = Path(root)
    summary_path = root / "summary.json"
    summary = pinned_json(summary_path, SHA["wp2_summary"], "WP2 summary")
    if (
        summary.get("schema_version") != "pynexus.wp2.result.v1"
        or summary.get("status") != "COMPLETE_REGISTERED_MATRIX"
        or summary.get("hours") != 8760
        or summary.get("failure_count") != 0
        or summary.get("is_full_year") is not True
        or summary.get("price_data_kind") != "synthetic_daily_sinusoid_NOT_historical_market_data"
    ):
        raise ValueError("Incomplete/misidentified WP2 experiment")
    if tuple(r["scenario"] for r in summary["cases"]) != EXPECTED:
        raise ValueError("WP2 17-case registration mismatch")
    rows = []
    for r in summary["cases"]:
        name = r["scenario"]
        if r.get("status") != "VERIFIED_TWO_STAGE_SOLVE":
            raise ValueError(f"WP2 {name}: unverified scenario")
        folder = root / name
        if digest(folder / "config.yaml") != r["config_sha256"]:
            raise ValueError(f"WP2 {name}: config hash mismatch")
        for stage in ("a", "b"):
            csv = folder / f"stage_{stage}.csv"
            if digest(csv) != r["dispatch_hashes"][csv.name]:
                raise ValueError(f"WP2 {name} stage {stage}: hourly dispatch tampered")
        # Recompute directly from saved hourly results instead of copying KPIs.
        a = pd.read_csv(
            folder / "stage_a.csv",
            usecols=["timestep", "demand_slack_kg_h", "h2_demand_kg_h"],
        )
        b = pd.read_csv(
            folder / "stage_b.csv",
            usecols=[
                "timestep",
                "demand_slack_kg_h",
                "electricity_price",
                "grid_import_mw",
                "grid_export_mw",
            ],
        )
        if len(a) != 8760 or len(b) != 8760:
            raise ValueError(f"WP2 {name}: incomplete hourly timeline")
        if not np.array_equal(a["timestep"], np.arange(8760)) or not np.array_equal(
            b["timestep"], np.arange(8760)
        ):
            raise ValueError(f"WP2 {name}: invalid hourly index")
        if not np.isfinite(a.select_dtypes(include="number").to_numpy()).all():
            raise ValueError(f"WP2 {name}: nonfinite Stage A values")
        if not np.isfinite(b.select_dtypes(include="number").to_numpy()).all():
            raise ValueError(f"WP2 {name}: nonfinite Stage B values")
        met = r["metrics"]
        a_short = float(a["demand_slack_kg_h"].sum())
        b_short = float(b["demand_slack_kg_h"].sum())
        demand = float(a["h2_demand_kg_h"].sum())
        settlement = float(
            (b["electricity_price"] * (b["grid_import_mw"] - b["grid_export_mw"])).sum()
        )
        close(demand, met["demand_h2_kg"], name + " demand")
        close(a_short, met["stage_a_shortfall_kg"], name + " Stage A")
        close(b_short, met["stage_b_shortfall_kg"], name + " Stage B")
        close(
            settlement,
            met["stage_b_grid_settlement_eur_synthetic_prices"],
            name + " Stage B settlement",
        )
        close(a_short / demand, met["stage_a_shortfall_fraction"], name + " fraction", atol=1e-10)
        allowance = max(1e-4, 1e-8 * demand)
        if b_short + 1e-4 < a_short or b_short > a_short + allowance + 1e-4:
            raise ValueError(f"WP2 {name}: Stage B violates Stage A shortfall envelope")
        rows.append(
            {
                "scenario": name,
                "unserved_h2_t": a_short / 1000,
                "unserved_percent": 100 * a_short / demand,
                "delivered_h2_t": (demand - a_short) / 1000,
                "electrolyser_mw": met["electrolyser_capacity_mw"],
                "grid_mw": met["grid_capacity_mw"],
                "wind_mw": met["wind_nameplate_mw"],
                "storage_t": met["storage_capacity_kg"] / 1000,
            }
        )
    return summary, pd.DataFrame(rows)


def audit_wp3(root, wp2_summary, wp2_df):
    result = pinned_json(Path(root) / "analysis.json", SHA["wp3_analysis"], "WP3 analysis")
    if (
        result.get("schema_version") != "pynexus.wp3.attribution.v1"
        or result.get("status") != "VERIFIED_POST_SOLVE_DIAGNOSTICS"
        or result.get("scenario_count") != 17
        or result.get("wp2_summary_sha256") != SHA["wp2_summary"]
    ):
        raise ValueError("WP3 failed verified WP2 linkage")
    diagnostics = pd.DataFrame(result["case_diagnostics"])
    if tuple(diagnostics["scenario"]) != EXPECTED:
        raise ValueError("WP3 scenario registration mismatch")
    for row in wp2_df.to_dict("records"):
        d = diagnostics.loc[diagnostics["scenario"] == row["scenario"]].iloc[0]
        close(d["annual_unserved_kg_stage_a"], row["unserved_h2_t"] * 1000, "WP3 cross-check")
        close(
            d["optimistic_unavoidable_shortfall_kg"] + d["unserved_above_optimistic_bound_kg"],
            d["annual_unserved_kg_stage_a"],
            "optimistic bound decomposition",
        )
        if not d["stage_a_independent_verification"] or not d["stage_b_independent_verification"]:
            raise ValueError("WP3 missing numerical verification")
    base = diagnostics.set_index("scenario")
    effect = float(
        base.loc["E", "annual_unserved_kg_stage_a"] - base.loc["E+S", "annual_unserved_kg_stage_a"]
    )
    close(
        effect,
        result["research_focus"]["storage_conditional_benefit_after_electrolyser_expansion_kg"],
        "WP3 E+S",
    )
    return result, diagnostics


def audit_wp4(root):
    result = pinned_json(Path(root) / "analysis.json", SHA["wp4_analysis"], "WP4 synthesis")
    if (
        result.get("schema_version") != "pynexus.wp4.interannual.v1"
        or result.get("status") != "COMPLETE_5_YEAR_35_SCENARIO_COMPARISON"
        or result.get("scenario_count") != 35
        or result.get("weather_year_count") != 5
        or tuple(result.get("years", [])) != YEARS
    ):
        raise ValueError("Incomplete WP4 five-year synthesis")
    frame = pd.DataFrame(result["annual_results"])
    expected = [(y, name) for y in YEARS for name in WEATHER_SCENARIOS]
    if list(zip(frame["year"], frame["scenario"], strict=True)) != expected:
        raise ValueError("WP4 registered scenario/year order mismatch")
    for year in YEARS:
        group = frame.loc[frame["year"] == year].set_index("scenario")
        h = 8784 if year == 2024 else 8760
        if (group["hours"] != h).any():
            raise ValueError("WP4 hourly calendar error")
        if not np.isfinite(group[["unserved_fraction", "unserved_h2_kg"]].to_numpy()).all():
            raise ValueError("WP4 nonfinite data")
        for row in group.to_dict("records"):
            close(
                row["unserved_fraction"] * (50_000 * h / 24), row["unserved_h2_kg"], "WP4 H2 mass"
            )
        m = next(z for z in result["annual_metrics"] if z["year"] == year)
        close(
            100
            * (
                group.loc["E", "unserved_fraction"]
                + group.loc["S", "unserved_fraction"]
                - group.loc["baseline", "unserved_fraction"]
                - group.loc["E+S", "unserved_fraction"]
            ),
            m["E_plus_S_interaction_pp"],
            "WP4 E+S interaction",
            atol=1e-7,
        )
    return result, frame


def audit_wp6(root, wp4):
    root = Path(root)
    manifest = pinned_json(root / "manifest.json", SHA["wp6_manifest"], "WP6 manifest")
    if (
        manifest.get("status") != "HASH_PINNED_WP4_SYNTHESIS_PYAM_ROUNDTRIP_PASSED"
        or manifest.get("source_analysis_sha256") != SHA["wp4_analysis"]
        or manifest.get("row_count") != 250
        or manifest.get("source_run_id") != WP4_RUN
    ):
        raise ValueError("WP6 round-trip provenance does not match WP4")
    for name, expected in manifest["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("WP6 exported file hash mismatch")
    data = pd.read_csv(root / "iamc_long.csv")
    if len(data) != 250 or data["model"].nunique() != 1 or data["model"].iloc[0] != MODEL:
        raise ValueError("WP6 malformed IAMC export")
    if data.duplicated(["model", "scenario", "region", "variable", "unit", "year"]).any():
        raise ValueError("WP6 duplicate IAMC coordinate")
    if not np.isfinite(data["value"].to_numpy()).all():
        raise ValueError("WP6 nonfinite exported data")
    demand = data.loc[data["variable"] == "PyNEXUS|Hydrogen|Unserved Demand|Stage A"]
    for row in wp4["annual_results"]:
        one = demand.loc[(demand["year"] == row["year"]) & (demand["scenario"] == row["scenario"])]
        if len(one) != 1:
            raise ValueError("WP6 missing unserved hydrogen IAMC value")
        close(one.iloc[0]["value"], row["unserved_h2_kg"], "WP6 to WP4 exchange")
    return manifest


def do_figures(output, wp2, wp3, wp4):
    figdir = output / "figures"
    figdir.mkdir()
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170}
    )
    files = []

    def save(name):
        plt.tight_layout()
        for ext in ("png", "svg"):
            f = figdir / f"{name}.{ext}"
            plt.savefig(f, bbox_inches="tight")
            files.append(f)
        plt.close()

    # Figure 1: all 17 original CDS-2023 scenarios, ordered by preregistration.
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = [
        "#374151"
        if s == "baseline"
        else ("#2563eb" if s == "G" else ("#15803d" if s == "E+S" else "#94a3b8"))
        for s in wp2["scenario"]
    ]
    ax.barh(wp2["scenario"], wp2["unserved_percent"], color=palette)
    ax.invert_yaxis()
    ax.set_xlabel("Stage A unmet configured H₂ demand (%)")
    ax.set_ylabel("Registered 2023 intervention")
    ax.set_title("Fig. 1 | Physical adequacy, original CDS-derived 2023 wind")
    ax.set_xlim(0, max(wp2["unserved_percent"]) * 1.13)
    for y, x in enumerate(wp2["unserved_percent"]):
        ax.text(x + 0.16, y, f"{x:.2f}", va="center", fontsize=8)
    save("fig1_wp2_2023_interventions")

    # Figure 2: identify 2021–2025 weather effects, do not pool original 2023 CDS.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for scenario, color, marker in (
        ("baseline", "#334155", "o"),
        ("W", "#ea580c", "s"),
        ("E", "#2563eb", "^"),
        ("E+S", "#15803d", "D"),
        ("eta", "#7c3aed", "v"),
        ("G", "#64748b", "x"),
    ):
        rows = wp4[wp4["scenario"] == scenario].sort_values("year")
        ax.plot(
            rows["year"],
            rows["unserved_fraction"] * 100,
            label=scenario,
            marker=marker,
            lw=1.9,
            color=color,
        )
    ax.set_xticks(YEARS)
    ax.set_xlabel("ERA5 weather year (fixed technology vintage)")
    ax.set_ylabel("Stage A unmet configured H₂ demand (%)")
    ax.set_title("Fig. 2 | Controlled weather-year adequacy screening")
    ax.legend(ncol=3, frameon=False, fontsize=9)
    save("fig2_wp4_weather_adequacy")

    # Figure 3: E+S interaction within each provider family; 2023 CDS separate.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    interactions = []
    for year in YEARS:
        rows = wp4.loc[wp4["year"] == year].set_index("scenario")
        gain = 100 * (rows.loc["E", "unserved_fraction"] - rows.loc["E+S", "unserved_fraction"])
        interactions.append(gain)
    ax.plot(YEARS, interactions, "o-", color="#15803d", lw=2, label="Open-Meteo ERA5 (WP4)")
    original = wp3["research_focus"]["electrolyser_storage_interaction_percentage_points"]
    ax.scatter(
        [2023], [original], marker="X", s=105, color="#be123c", label="Original CDS 2023 (WP3)"
    )
    ax.set_xticks(YEARS)
    ax.set_xlabel("Weather year")
    ax.set_ylabel("E+S finite interaction (percentage points)")
    ax.set_title("Fig. 3 | Conditional storage value after electrolyser expansion")
    ax.legend(frameon=False, fontsize=9)
    save("fig3_storage_interaction_provenance")

    # Figure 4: bound is an optimistic mathematical lower bound, NOT causal allocation.
    diag = pd.DataFrame(wp3["case_diagnostics"]).set_index("scenario").loc[["baseline", "E", "E+S"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = ["baseline", "E", "E+S"]
    lower = diag["optimistic_unavoidable_shortfall_kg"].to_numpy() / 1000
    residual = diag["unserved_above_optimistic_bound_kg"].to_numpy() / 1000
    ax.bar(labels, lower, color="#1d4ed8", label="Optimistic unavoidable bound")
    ax.bar(labels, residual, bottom=lower, color="#f97316", label="Gap above optimistic bound")
    ax.set_ylabel("Stage A annual unserved H₂ (tonnes)")
    ax.set_title("Fig. 4 | Structural production bound vs actual dispatch shortfall")
    ax.legend(frameon=False, fontsize=9)
    save("fig4_optimistic_bound_diagnostics")
    return files


def write_tables(out, wp2, wp3, wp4):
    table_dir = out / "tables"
    table_dir.mkdir()
    files = []
    a = table_dir / "table1_2023_17_scenarios.csv"
    wp2.to_csv(a, index=False, float_format="%.12g")
    files.append(a)
    w = wp4.pivot(index="year", columns="scenario", values="unserved_fraction")
    interactions = []
    for year in YEARS:
        row = w.loc[year]
        interactions.append(100 * (row["E"] + row["S"] - row["baseline"] - row["E+S"]))
    baseline = wp4.loc[wp4["scenario"] == "baseline"].set_index("year")
    result = pd.DataFrame(
        {
            "year": YEARS,
            "reference_wind_cf_percent": baseline.loc[
                list(YEARS), "wind_capacity_factor"
            ].to_numpy()
            * 100,
            "baseline_shortfall_percent": w.loc[list(YEARS), "baseline"].to_numpy() * 100,
            "E_shortfall_percent": w.loc[list(YEARS), "E"].to_numpy() * 100,
            "E_plus_S_shortfall_percent": w.loc[list(YEARS), "E+S"].to_numpy() * 100,
            "G_shortfall_percent": w.loc[list(YEARS), "G"].to_numpy() * 100,
            "ES_interaction_percentage_points": interactions,
        }
    )
    b = table_dir / "table2_weather_robustness.csv"
    result.to_csv(b, index=False, float_format="%.12g")
    files.append(b)
    selected = (
        pd.DataFrame(wp3["case_diagnostics"])
        .set_index("scenario")
        .loc[["baseline", "E", "E+S", "G"]]
    )
    cols = [
        "annual_unserved_kg_stage_a",
        "optimistic_unavoidable_shortfall_kg",
        "unserved_above_optimistic_bound_kg",
        "shortfall_hours_stage_a",
        "storage_max_level_kg_stage_a",
        "storage_at_capacity_hours_stage_a",
    ]
    c = table_dir / "table3_constraint_bounds.csv"
    selected[cols].to_csv(c, float_format="%.12g")
    files.append(c)
    return files


def make_values(wp2, wp3, wp4):
    row = wp2.set_index("scenario")
    year_table = wp4.pivot(index="year", columns="scenario", values="unserved_fraction")
    e = wp3["research_focus"]
    interactions = [
        100
        * (
            year_table.loc[y, "E"]
            + year_table.loc[y, "S"]
            - year_table.loc[y, "baseline"]
            - year_table.loc[y, "E+S"]
        )
        for y in YEARS
    ]
    extra = [
        50_000
        * (8784 if y == 2024 else 8760)
        / 24
        * (year_table.loc[y, "E"] - year_table.loc[y, "E+S"])
        / 1000
        for y in YEARS
    ]
    vals = {
        "baseline_shortfall_t": f"{row.loc['baseline', 'unserved_h2_t']:,.2f}",
        "baseline_shortfall_pct": f"{row.loc['baseline', 'unserved_percent']:.3f}",
        "baseline_h2_demand_t": f"{row.loc['baseline', 'unserved_h2_t'] / (row.loc['baseline', 'unserved_percent'] / 100):,.0f}",
        "wind_single_pct": f"{row.loc['W', 'unserved_percent']:.3f}",
        "electrolyser_single_pct": f"{row.loc['E', 'unserved_percent']:.3f}",
        "grid_single_pct": f"{row.loc['G', 'unserved_percent']:.3f}",
        "storage_single_pct": f"{row.loc['S', 'unserved_percent']:.3f}",
        "efficiency_single_pct": f"{row.loc['eta', 'unserved_percent']:.3f}",
        "es_shortfall_t": f"{row.loc['E+S', 'unserved_h2_t']:,.2f}",
        "es_shortfall_pct": f"{row.loc['E+S', 'unserved_percent']:.3f}",
        "es_extra_delivery_t": f"{e['storage_conditional_benefit_after_electrolyser_expansion_kg'] / 1000:.2f}",
        "es_interaction_pp": f"{e['electrolyser_storage_interaction_percentage_points']:.3f}",
        "optimistic_bound_t": f"{(row.loc['baseline', 'unserved_h2_t'] * 1000 - e['baseline_shortfall_above_independent_energy_bound_kg']) / 1000:.2f}",
        "baseline_bound_share_pct": f"{e['baseline_unavoidable_fraction_of_observed_shortfall'] * 100:.4f}",
        "baseline_gap_above_bound_t": f"{e['baseline_shortfall_above_independent_energy_bound_kg'] / 1000:.2f}",
        "baseline_max_storage_t": f"{float(next(z for z in wp3['case_diagnostics'] if z['scenario'] == 'baseline')['storage_max_level_kg_stage_a']) / 1000:.2f}",
        "wp4_min_shortfall_pct": f"{year_table['baseline'].min() * 100:.3f}",
        "wp4_max_shortfall_pct": f"{year_table['baseline'].max() * 100:.3f}",
        "wp4_min_interaction_pp": f"{min(interactions):.3f}",
        "wp4_max_interaction_pp": f"{max(interactions):.3f}",
        "wp4_min_extra_t": f"{min(extra):.2f}",
        "wp4_max_extra_t": f"{max(extra):.2f}",
    }
    singles = row.loc[list(EXPECTED)]
    vals["table_2023_markdown"] = (
        "**Table 1.** Original CDS-2023 registered interventions (Stage A).\n\n"
        "| Scenario | Unserved H₂ (t/y) | Unserved demand (%) |\n"
        "|:--|--:|--:|\n"
        + "".join(
            f"| {name} | {record['unserved_h2_t']:,.2f} | {record['unserved_percent']:.3f} |\n"
            for name, record in singles.iterrows()
        )
    )
    vals["table_weather_markdown"] = (
        "**Table 2.** Open-Meteo ERA5 weather-year outcomes (not the original CDS-2023 source).\n\n"
        "| Year | Reference wind CF (%) | Baseline shortfall (%) | E+S shortfall (%) | E+S interaction (pp) |\n"
        "|--:|--:|--:|--:|--:|\n"
        + "".join(
            f"| {year} | {wp4[(wp4['year'] == year) & (wp4['scenario'] == 'baseline')]['wind_capacity_factor'].iloc[0] * 100:.2f} | "
            f"{year_table.loc[year, 'baseline'] * 100:.3f} | "
            f"{year_table.loc[year, 'E+S'] * 100:.3f} | "
            f"{interaction:.3f} |\n"
            for year, interaction in zip(YEARS, interactions, strict=True)
        )
    )
    d = pd.DataFrame(wp3["case_diagnostics"]).set_index("scenario")
    vals["table_bounds_markdown"] = (
        "**Table 3.** Optimistic mathematical shortfall bound, original CDS-2023.\n\n"
        "| Scenario | Optimistic unavoidable (t/y) | Above bound (t/y) | Storage maximum (t) |\n"
        "|:--|--:|--:|--:|\n"
        + "".join(
            f"| {name} | {d.loc[name, 'optimistic_unavoidable_shortfall_kg'] / 1000:,.2f} | "
            f"{d.loc[name, 'unserved_above_optimistic_bound_kg'] / 1000:,.2f} | "
            f"{d.loc[name, 'storage_max_level_kg_stage_a'] / 1000:.2f} |\n"
            for name in ("baseline", "E", "E+S", "G")
        )
    )
    return vals


def render_markdown(template_path, values):
    text = Path(template_path).read_text(encoding="utf-8")
    # Explicit small dictionary; reject unfilled format keys rather than silent text.
    try:
        result = text.format_map(values)
    except KeyError as exc:
        raise ValueError(f"Missing registered manuscript result placeholder: {exc}") from exc
    if "{baseline_" in result or "{wp4_" in result or "{es_" in result:
        raise ValueError("Unresolved manuscript results placeholder")
    return result


def build(wp2, wp3, wp4, wp6, template, output):
    output = Path(output)
    if output.exists() or output.with_name(output.name + ".incomplete").exists():
        raise FileExistsError("WP7 refuses to overwrite existing evidence")
    summary, t2 = audit_wp2(wp2)
    attribution, t3 = audit_wp3(wp3, summary, t2)
    synthesis, t4 = audit_wp4(wp4)
    metadata = audit_wp6(wp6, synthesis)
    values = make_values(t2, attribution, t4)
    partial = output.with_name(output.name + ".incomplete")
    partial.mkdir(parents=True, exist_ok=False)
    figures = do_figures(partial, t2, attribution, t4)
    tables = write_tables(partial, t2, attribution, t4)
    paper = partial / "MANUSCRIPT.md"
    paper.write_text(render_markdown(template, values), encoding="utf-8")
    audit = partial / "claim_audit.json"
    audit.write_text(
        json.dumps(
            {
                "rendered_values": values,
                "source_sha256": SHA,
                "synthesis_source_revision": synthesis["source_revision"],
                "wp2_source_revision": summary["research_version"],
                "previous_wp6_pyam_version": metadata["pyam_version"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "pynexus.wp7.manuscript.v1",
        "status": "GENERATED_FROM_BYTE_PINNED_VERIFIED_ARCHIVES",
        "source_workflows": {"wp2": WP2_RUN, "wp3": WP3_RUN, "wp4": WP4_RUN, "wp6": WP6_RUN},
        "original_hourly_wp2_stage_a_b_recalculated": 17,
        "wp4_weather_scenarios_from_previously_hourly_verified_synthesis": 35,
        "wp6_pyam_exchange_source_cells_crosschecked": 250,
        "wp7_reoptimised_hourly_scenarios": 0,
        "not_a_new_external_physical_or_market_validation": True,
        "synthetic_electricity_prices_not_market_evidence": True,
        "source_sha256": SHA,
        "file_sha256": {
            str(p.relative_to(partial)): digest(p) for p in [paper, audit, *figures, *tables]
        },
        "references_reviewed_separately_not_cited_as_evidence_of_model_outputs": True,
    }
    manifest = partial / "manifest.json"
    manifest.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.rename(output)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("wp2", "wp3", "wp4", "wp6", "template", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    a = parser.parse_args(argv)
    record = build(a.wp2, a.wp3, a.wp4, a.wp6, a.template, a.output)
    print(
        "WP7 PASS: 17 hourly source pairs, 35 prior WP4 results, 250 IAMC cells; manuscript and 4 figures generated"
    )
    print(json.dumps(record["source_sha256"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())