"""PyNEXUS — Streamlit GUI.

A thin visual layer over the exact same code path `pynexus solve`/`verify`
use (`ElectrolyzerDispatchOptimizer.optimize`, `verify_dispatch`,
`OffshoreWindFarm`, `optimization.reconciliation`). No separate logic and
no capability this doesn't already have via the CLI — this only makes the
existing model runnable and inspectable without writing Python.

Run with: streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from components.wind_turbine import OffshoreWindFarm  # noqa: E402
from optimization.dispatch import ElectrolyzerDispatchOptimizer  # noqa: E402
from optimization.reconciliation import (  # noqa: E402
    CHI_SQUARE_1_CRIT_999,
    reconcile_hydrogen_balance,
)
from optimization.verification import verify_dispatch  # noqa: E402

st.set_page_config(page_title="PyNEXUS", page_icon="\U0001f4a8", layout="wide")

CONFIGS = {
    "config.yaml (168h reference)": "config.yaml",
    "configs/tiny_test.yaml (4h)": "configs/tiny_test.yaml",
    "configs/annual.yaml (8,760h)": "configs/annual.yaml",
}


def synthetic_inputs(count: int, dt: float) -> pd.DataFrame:
    """Identical formula to optimization/cli.py's --synthetic generator."""
    t = np.arange(count) * dt
    return pd.DataFrame(
        {
            "wind_available_mw": 140.0 + 10.0 * np.sin(2 * np.pi * t / 24),
            "electricity_price": 50.0 + 20.0 * np.cos(2 * np.pi * t / 24),
            "carbon_intensity": 200.0 + 100.0 * np.sin(2 * np.pi * t / 24),
        }
    )


def synthetic_heat_demand(count: int, dt: float) -> np.ndarray:
    """Identical formula to optimization/cli.py's --enable-heat auto-generation."""
    t = np.arange(count) * dt
    return 12.0 + 6.0 * np.cos(2 * np.pi * (t % 24 - 7) / 24)


def load_optimizer(config_choice: str) -> tuple[ElectrolyzerDispatchOptimizer, dict, int]:
    config_path = ROOT / CONFIGS[config_choice]
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    opt = ElectrolyzerDispatchOptimizer(str(config_path))
    count = int(cfg["simulation"]["time_horizon_hours"] / opt.dt)
    return opt, cfg, count


def kpi_row(result: dict, opt: ElectrolyzerDispatchOptimizer) -> None:
    row1 = st.columns(3)
    row1[0].metric("Status", result["status"])
    row1[1].metric("Solve time", f"{result['solve_wall_time_s']:.2f} s")
    if result["objective_value"] is not None:
        row1[2].metric("Objective (EUR or kgCO2)", f"{result['objective_value']:,.0f}")

    row2 = st.columns(3)
    row2[0].metric("Variables", f"{result['variables']:,}")
    row2[1].metric("Binaries", f"{result['binary_variables']:,}")
    df = result["results_df"]
    if df is not None:
        demand_met_frac = 1 - df["demand_slack_kg_h"].sum() / (opt.hourly_demand_kg * len(df))
        row2[2].metric("Demand met", f"{demand_met_frac:.1%}")


def dispatch_charts(df: pd.DataFrame) -> None:
    st.subheader("Wind vs. dispatch")
    st.line_chart(df.set_index("timestep")[["wind_available_mw", "power_optimized_mw"]])

    st.subheader("Hydrogen production vs. demand")
    st.line_chart(df.set_index("timestep")[["h2_produced_kg_h", "h2_demand_kg_h"]])

    if "grid_import_mw" in df.columns and (
        df["grid_import_mw"].abs().sum() > 0 or df["grid_export_mw"].abs().sum() > 0
    ):
        st.subheader("Grid exchange")
        st.line_chart(df.set_index("timestep")[["grid_import_mw", "grid_export_mw"]])

    if "storage_level_kg" in df.columns:
        st.subheader("Hydrogen storage level")
        st.line_chart(df.set_index("timestep")[["storage_level_kg"]])

    if "waste_heat_recovered_mw" in df.columns:
        st.subheader("Heat balance")
        heat_cols = ["waste_heat_recovered_mw", "boiler_output_mw", "heat_demand_mw"]
        st.line_chart(df.set_index("timestep")[heat_cols])
        if "heat_storage_level_mwh" in df.columns:
            st.line_chart(df.set_index("timestep")[["heat_storage_level_mwh"]])


st.title("PyNEXUS")
st.caption(
    "Electricity + hydrogen + heat dispatch model (Pyomo/HiGHS). This GUI calls the exact "
    "same `ElectrolyzerDispatchOptimizer.optimize` / `verify_dispatch` code path as "
    "`pynexus solve` / `verify` — nothing new underneath. Source: "
    "[github.com/abhijith-sivaprasadan/PyNEXUS](https://github.com/abhijith-sivaprasadan/PyNEXUS)."
)

tab_run, tab_browse, tab_cf, tab_recon = st.tabs(
    ["Run dispatch", "Browse past runs", "Capacity factor", "Data reconciliation"]
)

# --- Tab 1: Run dispatch -----------------------------------------------------
with tab_run:
    left, right = st.columns([1, 2])

    with left:
        config_choice = st.selectbox("Config", list(CONFIGS.keys()))
        opt, cfg, count = load_optimizer(config_choice)
        st.caption(f"{count:,} timesteps at {opt.dt}h resolution")

        source = st.radio(
            "Input source",
            ["Synthetic (deterministic)", "Upload wind CSV", "Upload full inputs CSV"],
        )

        wind_csv = None
        inputs_csv = None
        if source == "Upload wind CSV":
            wind_csv = st.file_uploader(
                "CSV with a wind_speed_hub_ms column (e.g. from `fetch-era5-year --output-csv`)",
                type="csv",
            )
        elif source == "Upload full inputs CSV":
            inputs_csv = st.file_uploader(
                "CSV with wind_available_mw, electricity_price, [carbon_intensity], "
                "[heat_demand_mw] columns",
                type="csv",
            )

        objective = st.selectbox("Objective", ["minimize_cost", "minimize_emissions"])
        demand_mode = st.selectbox("Demand mode", ["cumulative", "hourly"])

        st.markdown("**Phase A/B features** (opt-in, off by default — see `docs/formulation.md`)")
        enable_storage = st.checkbox("Hydrogen storage")
        enable_grid = st.checkbox("Grid import/export")
        enable_heat = st.checkbox("Heat coupling (waste heat + storage + boiler)")
        if enable_heat:
            st.caption(
                "Heat demand auto-generated with the same synthetic profile as `pynexus solve`."
            )
        if (enable_storage or enable_heat) and demand_mode != "hourly":
            st.warning(
                "Storage/heat only affect dispatch in hourly demand mode — see docs/formulation.md."
            )

        run_clicked = st.button("Run dispatch", type="primary")

    with right:
        if run_clicked:
            try:
                if source == "Synthetic (deterministic)":
                    inputs = synthetic_inputs(count, opt.dt)
                elif source == "Upload wind CSV":
                    if wind_csv is None:
                        st.error("Upload a wind CSV first.")
                        st.stop()
                    wind_df = pd.read_csv(wind_csv)
                    farm = OffshoreWindFarm(str(ROOT / CONFIGS[config_choice]))
                    wind_mw = farm.power_output_mw_from_hub_height(
                        wind_df["wind_speed_hub_ms"].to_numpy()
                    )
                    t = np.arange(count) * opt.dt
                    inputs = pd.DataFrame(
                        {
                            "wind_available_mw": wind_mw,
                            "electricity_price": 50.0 + 20.0 * np.cos(2 * np.pi * t / 24),
                            "carbon_intensity": 200.0 + 100.0 * np.sin(2 * np.pi * t / 24),
                        }
                    )
                else:
                    if inputs_csv is None:
                        st.error("Upload an inputs CSV first.")
                        st.stop()
                    inputs = pd.read_csv(inputs_csv)

                if len(inputs) != count:
                    st.error(
                        f"Input has {len(inputs)} rows; the selected config needs {count} "
                        f"({cfg['simulation']['time_horizon_hours']}h / {opt.dt}h steps)."
                    )
                    st.stop()

                if enable_heat and "heat_demand_mw" not in inputs.columns:
                    inputs["heat_demand_mw"] = synthetic_heat_demand(count, opt.dt)

                carbon = (
                    inputs["carbon_intensity"].to_numpy() if "carbon_intensity" in inputs else None
                )
                heat_demand_mw = inputs["heat_demand_mw"].to_numpy() if enable_heat else None

                with st.spinner(f"Solving {count:,}-timestep MILP..."):
                    result = opt.optimize(
                        inputs["wind_available_mw"],
                        inputs["electricity_price"],
                        objective=objective,
                        demand_mode=demand_mode,
                        carbon_intensity=carbon,
                        enable_storage=enable_storage,
                        enable_grid=enable_grid,
                        enable_heat=enable_heat,
                        heat_demand_mw=heat_demand_mw,
                    )

                st.session_state["last_result"] = result
                st.session_state["last_opt"] = opt
                st.session_state["last_objective"] = objective
                st.session_state["last_demand_mode"] = demand_mode
                st.session_state["last_flags"] = (enable_storage, enable_grid, enable_heat)
                st.session_state["last_carbon"] = carbon

            except (ValueError, KeyError) as exc:
                st.error(f"Rejected before solving: {exc}")
                st.stop()

        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            opt_r = st.session_state["last_opt"]

            if result["status"] != "optimal":
                st.error(f"Solve did not reach an optimum: **{result['status']}**")
                st.caption(
                    "An infeasible or time-limited solve is a real result to report, not to "
                    "hide — see docs/reproducibility.md's 2026-09-01 section for a worked "
                    "example (real annual demand exceeding real wind supply)."
                )
                kpi_row(result, opt_r)
            else:
                kpi_row(result, opt_r)
                df = result["results_df"]

                if st.checkbox("Independently verify this result", value=True):
                    enable_storage_f, enable_grid_f, enable_heat_f = st.session_state["last_flags"]
                    evidence = verify_dispatch(
                        opt_r,
                        df,
                        result["objective_value"],
                        st.session_state["last_objective"],
                        st.session_state["last_demand_mode"],
                        st.session_state["last_carbon"],
                        enable_storage=enable_storage_f,
                        enable_grid=enable_grid_f,
                        enable_heat=enable_heat_f,
                    )
                    if evidence["passed"]:
                        st.success("All independent checks passed (optimization/verification.py)")
                    else:
                        st.error("Verification FAILED — see checks below")
                    with st.expander("Verification checks"):
                        st.json(evidence["checks"])

                dispatch_charts(df)

                st.download_button(
                    "Download dispatch.csv",
                    df.to_csv(index=False),
                    file_name="dispatch.csv",
                    mime="text/csv",
                )
        elif not run_clicked:
            st.info("Configure a run on the left and click **Run dispatch**.")

# --- Tab 2: Browse past runs --------------------------------------------------
with tab_browse:
    st.caption(
        "Loads a committed `run_manifest.json` / `dispatch.csv` pair from `outputs/` — the "
        "same artefacts `pynexus solve` writes. Includes the real annual runs from "
        "docs/reproducibility.md's 2026-09-01 section."
    )
    run_dirs = sorted(p.parent for p in (ROOT / "outputs").rglob("run_manifest.json"))
    if not run_dirs:
        st.info("No committed runs found under outputs/.")
    else:
        labels = {str(p.relative_to(ROOT)).replace("\\", "/"): p for p in run_dirs}
        choice = st.selectbox("Run", list(labels.keys()))
        run_dir = labels[choice]
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

        cols = st.columns(5)
        cols[0].metric("Status", manifest["result"]["termination_condition"])
        cols[1].metric("Snapshots", f"{manifest['snapshots']:,}")
        cols[2].metric("Wall time", f"{manifest['result']['solve_wall_time_s']:.2f} s")
        if manifest["result"]["objective_value"] is not None:
            cols[3].metric("Objective", f"{manifest['result']['objective_value']:,.0f}")
        cols[4].metric("Verified", str(manifest.get("verified", "n/a")))

        with st.expander("Full manifest"):
            st.json(manifest)

        dispatch_path = run_dir / "dispatch.csv"
        if dispatch_path.exists():
            df = pd.read_csv(dispatch_path)
            dispatch_charts(df)
        else:
            st.warning(
                f"No dispatch.csv in this run (termination: "
                f"{manifest['result']['termination_condition']}) — the solve did not reach "
                "an optimum, so no dispatch table was produced. That is the recorded result."
            )

# --- Tab 3: Capacity factor ---------------------------------------------------
with tab_cf:
    st.caption(
        "Real ERA5 wind (Phase C1/C5) run through the same OffshoreWindFarm model "
        "pynexus solve --wind-csv uses. See docs/validation.md for the full write-up."
    )
    wind_csv_path = ROOT / "data" / "raw" / "era5_2023_wind.csv"
    if not wind_csv_path.exists():
        st.info(
            "No fetched ERA5 wind found at data/raw/era5_2023_wind.csv. Run "
            "`pynexus fetch-era5-year --config config.yaml --year 2023` first."
        )
    else:
        wind_df = pd.read_csv(wind_csv_path, index_col=0, parse_dates=True)
        farm = OffshoreWindFarm(str(ROOT / "config.yaml"))
        power_mw = farm.power_output_mw_from_hub_height(wind_df["wind_speed_hub_ms"].to_numpy())
        cf = power_mw.mean() / farm.farm_rated_mw

        cols = st.columns(3)
        cols[0].metric("Annual capacity factor", f"{cf:.1%}")
        cols[1].metric(
            "Annual mean power", f"{power_mw.mean():.1f} MW / {farm.farm_rated_mw:.0f} MW rated"
        )
        cols[2].metric("Total annual energy", f"{power_mw.sum():,.0f} MWh")

        st.subheader("Hub-height wind speed distribution")
        st.bar_chart(np.histogram(wind_df["wind_speed_hub_ms"], bins=30)[0])

        st.subheader("Comparison against published North Sea figures")
        st.caption(
            'Elizalde, A., Akhtar, N., Geyer, B., and Schrum, C.: "Uncertainty in North Sea '
            'offshore wind power...", *Wind Energy Science*, 11, 1077–1095, 2026. '
            "https://doi.org/10.5194/wes-11-1077-2026"
        )
        st.table(
            pd.DataFrame(
                {
                    "Scenario": [
                        "No wake / no losses",
                        "With wake / with losses",
                        "Broader literature range",
                    ],
                    "Published (15 MW)": ["0.61", "0.49", "0.23–0.52 (mean 0.35)"],
                    "This model": [
                        f"{cf / (farm.wake_loss_factor * farm.electrical_loss_factor * farm.availability):.3f}",
                        f"{cf:.3f}",
                        f"{cf:.3f}",
                    ],
                }
            )
        )

# --- Tab 4: Data reconciliation ------------------------------------------------
with tab_recon:
    st.caption(
        "Weighted least squares + chi-square(1) gross-error test on the hydrogen mass "
        "balance `production = to_store - from_store + to_pipeline` — same method as "
        "ThermoTwin's thermal reconciliation. See docs/reconciliation.md."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        production = st.number_input("Production (kg/h)", value=1000.0)
        var_production = st.number_input("Variance (production)", value=4.0, min_value=0.01)
    with c2:
        to_store = st.number_input("To store (kg/h)", value=200.0)
        var_to_store = st.number_input("Variance (to_store)", value=1.0, min_value=0.01)
    with c3:
        from_store = st.number_input("From store (kg/h)", value=50.0)
        var_from_store = st.number_input("Variance (from_store)", value=1.0, min_value=0.01)
    with c4:
        to_pipeline = st.number_input("To pipeline (kg/h)", value=850.0)
        var_to_pipeline = st.number_input("Variance (to_pipeline)", value=4.0, min_value=0.01)

    result = reconcile_hydrogen_balance(
        production,
        to_store,
        from_store,
        to_pipeline,
        var_production,
        var_to_store,
        var_from_store,
        var_to_pipeline,
    )

    st.subheader("Reconciliation result")
    st.table(
        pd.DataFrame(
            {
                "Measured": result.measured,
                "Adjusted": result.adjusted,
                "Adjustment": result.adjustment,
            }
        ).T
    )

    cols = st.columns(3)
    cols[0].metric("Raw imbalance (kg/h)", f"{result.raw_imbalance:.2f}")
    cols[1].metric("Chi-square(1) statistic", f"{result.test_statistic:.2f}")
    cols[2].metric(
        f"Gross error (>{CHI_SQUARE_1_CRIT_999})?",
        "YES" if result.gross_error else "no",
        delta=None,
    )
    if result.gross_error:
        st.error(
            "Flagged: these four readings are mutually inconsistent beyond plausible "
            "measurement noise."
        )
    else:
        st.success("Consistent with the stated measurement uncertainties.")

    st.info(
        "This deliberately does not name a 'likely culprit' sensor. With one balance "
        "equation across four sensors, that ranking is mathematically determined by the "
        "configured variances alone, independent of which sensor is actually biased — "
        "see docs/reconciliation.md for the proof and the numerical check that caught it."
    )
