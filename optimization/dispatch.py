# optimization/dispatch.py
# ============================================================
# MILP Dispatch Optimizer — Layer 3
# ============================================================
# Optimises electrolyzer dispatch over a time horizon given:
#   - Wind power availability (from wind turbine model)
#   - Electricity prices (from ENTSO-E or synthetic)
#   - Hydrogen demand profile
#   - Physical constraints from Layers 1 & 2
#
# Decision variable:
#   p[t] = electrical power to electrolyzer at hour t (MW)
#
# Objective (switchable):
#   COST:      minimise sum( price[t] * p[t] * dt )
#   EMISSIONS: minimise sum( carbon_intensity[t] * p[t] * dt )
#
# Key constraints:
#   1. Electrolyzer operating bounds (min/max load)
#   2. Ramp rate limit (inter-timestep coupling)
#   3. Hydrogen demand satisfaction (soft constraint with slack)
#   4. Pipeline pressure feasibility
#   5. Wind availability (can't use more than wind produces)
#
# Why MILP and not just LP?
#   The electrolyzer has a BINARY on/off state. It must either
#   be off (0 MW) or on (>=10 MW min load). This integer
#   variable u[t] in {0,1} makes it a Mixed Integer Linear Program.
#
# Why soft demand constraint?
#   Hard hourly constraints go infeasible when wind drops below
#   the power needed to meet demand. Soft constraints use a
#   demand_slack variable with a heavy penalty instead.
#
# NOTE: variable named 'demand_slack' not 'slack' — Pyomo 6.10.0
#   appsi_highs has an internal conflict with any Var named 'slack'.
#
# Solver: HiGHS via Pyomo appsi interface (free, no license)
# ============================================================

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("WARNING: Pyomo not installed. Run: pip install pyomo highspy")

from components.electrolyzer import (PEMElectrolyzer,
                                      efficiency_at_load,
                                      LHV_HYDROGEN_KWH_PER_KG)
from components.wind_turbine  import OffshoreWindFarm
from components.pipeline      import HydrogenPipeline


# --- Load config --------------------------------------------
def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parent.parent
    full_path = root / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


# --- H2 output linearisation --------------------------------
def linear_h2_coefficient(nominal_load_fraction: float = 0.80,
                           nominal_efficiency: float = 0.70) -> float:
    """
    kg H2 per MWh of electricity at nominal load.
    Linearises the nonlinear h2(p) = p * eta(p) / LHV function
    by fixing eta at the nominal 80% operating point.
    """
    eta = efficiency_at_load(nominal_load_fraction, nominal_efficiency)
    return (eta * 1000.0) / LHV_HYDROGEN_KWH_PER_KG   # kg/MWh


# --- Core optimizer class -----------------------------------

class ElectrolyzerDispatchOptimizer:
    """
    MILP optimizer for electrolyzer dispatch.

    Usage
    -----
    >>> opt = ElectrolyzerDispatchOptimizer()
    >>> result = opt.optimize(
    ...     wind_power_mw=wind_array,
    ...     electricity_prices=price_array,
    ...     objective="minimize_cost"
    ... )
    >>> opt.print_solution_summary(result)
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)

        e = cfg["electrolyzer"]
        self.p_rated       = e["rated_power_mw"]
        self.p_min_frac    = e["min_load_fraction"]
        self.p_min         = self.p_rated * self.p_min_frac
        self.p_max         = self.p_rated * e["max_load_fraction"]
        self.ramp_per_hour = e.get("ramp_rate_per_hour", 0.40)
        self.max_ramp_mw   = self.ramp_per_hour * self.p_rated
        self.nominal_eta   = e["nominal_efficiency"]

        self.pipeline    = HydrogenPipeline(config_path)
        self.max_h2_kg_s = self.pipeline.max_feasible_flow_kg_s

        d = cfg["hydrogen_demand"]
        self.hourly_demand_kg = d["daily_average_kg"] / 24.0

        self.h2_coeff = linear_h2_coefficient(0.80, self.nominal_eta)

        s = cfg["simulation"]
        self.dt = s["time_step_hours"]

        o = cfg["optimization"]
        self.solver_name = o.get("solver", "highs")
        self.time_limit  = o.get("time_limit_seconds", 300)
        self.mip_gap     = o.get("mip_gap", 0.01)

        self.DEMAND_PENALTY = 1000.0   # EUR per kg unmet H2

    def optimize(self,
                 wind_power_mw: np.ndarray,
                 electricity_prices: np.ndarray,
                 objective: str = "minimize_cost",
                 demand_mode: str = "cumulative",
                 carbon_intensity: np.ndarray = None) -> dict:
        """
        Run MILP optimization.

        Parameters
        ----------
        wind_power_mw : np.ndarray
            Available wind power at each timestep (MW).
        electricity_prices : np.ndarray
            Day-ahead electricity price (EUR/MWh).
        objective : str
            "minimize_cost" or "minimize_emissions"
        demand_mode : str
            "hourly" (soft per-hour) or "cumulative" (hard total)
        carbon_intensity : np.ndarray, optional
            Required for emissions objective (kg CO2/MWh).

        Returns
        -------
        dict with solution data and results DataFrame
        """
        if not PYOMO_AVAILABLE:
            raise RuntimeError("Pyomo not installed. pip install pyomo highspy")

        wind_power_mw      = np.asarray(wind_power_mw, dtype=float)
        electricity_prices = np.asarray(electricity_prices, dtype=float)
        T = len(wind_power_mw)
        assert len(electricity_prices) == T, "Arrays must match length"

        model   = pyo.ConcreteModel(name="ElectrolyzerDispatch")
        model.T = pyo.RangeSet(0, T - 1)

        # Parameters
        model.wind      = pyo.Param(model.T,
                           initialize={t: float(wind_power_mw[t]) for t in range(T)})
        model.price     = pyo.Param(model.T,
                           initialize={t: float(electricity_prices[t]) for t in range(T)})
        model.demand_kg = pyo.Param(initialize=self.hourly_demand_kg)

        # Decision variables
        model.p = pyo.Var(model.T, domain=pyo.NonNegativeReals,
                           bounds=(0, self.p_max))
        model.u = pyo.Var(model.T, domain=pyo.Binary)

        # demand_slack: shortfall variable
        # Named 'demand_slack' NOT 'slack' — avoid Pyomo 6.10.0 appsi bug
        model.demand_slack = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        # Constraints
        def wind_limit(model, t):
            return model.p[t] <= model.wind[t]
        model.c_wind = pyo.Constraint(model.T, rule=wind_limit)

        def min_load(model, t):
            return model.p[t] >= self.p_min * model.u[t]
        model.c_min_load = pyo.Constraint(model.T, rule=min_load)

        def max_load(model, t):
            return model.p[t] <= self.p_max * model.u[t]
        model.c_max_load = pyo.Constraint(model.T, rule=max_load)

        def ramp_up(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.p[t] - model.p[t-1] <= self.max_ramp_mw
        model.c_ramp_up = pyo.Constraint(model.T, rule=ramp_up)

        def ramp_down(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.p[t-1] - model.p[t] <= self.max_ramp_mw
        model.c_ramp_down = pyo.Constraint(model.T, rule=ramp_down)

        max_h2_per_hour = self.max_h2_kg_s * 3600.0
        def pipeline_cap(model, t):
            return model.p[t] * self.h2_coeff <= max_h2_per_hour
        model.c_pipeline = pyo.Constraint(model.T, rule=pipeline_cap)

        if demand_mode == "hourly":
            def h2_demand_hourly(model, t):
                return (model.p[t] * self.h2_coeff
                        + model.demand_slack[t] >= model.demand_kg)
            model.c_demand = pyo.Constraint(model.T, rule=h2_demand_hourly)
        elif demand_mode == "cumulative":
            total_demand = self.hourly_demand_kg * T
            def h2_demand_cumul(model):
                return (sum(model.p[t] * self.h2_coeff * self.dt
                            for t in model.T) >= total_demand)
            model.c_demand = pyo.Constraint(rule=h2_demand_cumul)

        # Objective
        if objective == "minimize_cost":
            def cost_obj(model):
                return (sum(model.price[t] * model.p[t] * self.dt
                            for t in model.T)
                        + sum(self.DEMAND_PENALTY * model.demand_slack[t]
                              for t in model.T))
            model.objective = pyo.Objective(rule=cost_obj, sense=pyo.minimize)

        elif objective == "minimize_emissions":
            if carbon_intensity is None:
                raise ValueError("carbon_intensity required")
            model.carbon = pyo.Param(model.T,
                            initialize={t: float(carbon_intensity[t])
                                        for t in range(T)})
            def emissions_obj(model):
                return (sum(model.carbon[t] * model.p[t] * self.dt
                            for t in model.T)
                        + sum(self.DEMAND_PENALTY * model.demand_slack[t]
                              for t in model.T))
            model.objective = pyo.Objective(rule=emissions_obj, sense=pyo.minimize)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Solve
        solver = SolverFactory("appsi_highs")
        solver.options["time_limit"]  = self.time_limit
        solver.options["mip_rel_gap"] = self.mip_gap

        sol    = solver.solve(model, tee=False)
        status = str(sol.solver.termination_condition)

        if "optimal" not in status.lower() and "feasible" not in status.lower():
            print(f"WARNING: Solver status = {status}")
            return {"status": status, "objective_value": None,
                    "power_schedule": None, "results_df": None}

        # Extract solution
        power_schedule   = np.array([pyo.value(model.p[t]) for t in range(T)])
        online_status    = np.array([pyo.value(model.u[t]) for t in range(T)])
        slack_values     = np.array([pyo.value(model.demand_slack[t])
                                      for t in range(T)])

        h2_produced_kg_h = power_schedule * self.h2_coeff
        cost_profile     = electricity_prices * power_schedule * self.dt
        demand_met_bool  = (h2_produced_kg_h + slack_values
                            >= self.hourly_demand_kg - 1e-3)

        results_df = pd.DataFrame({
            "timestep"           : np.arange(T),
            "wind_available_mw"  : wind_power_mw,
            "power_optimized_mw" : power_schedule,
            "online_status"      : online_status.astype(int),
            "electricity_price"  : electricity_prices,
            "h2_produced_kg_h"   : h2_produced_kg_h,
            "h2_demand_kg_h"     : self.hourly_demand_kg,
            "demand_slack_kg_h"  : slack_values,
            "demand_met"         : demand_met_bool,
            "cost_eur"           : cost_profile,
        })

        return {
            "status"           : status,
            "objective_value"  : pyo.value(model.objective),
            "power_schedule"   : power_schedule,
            "online_status"    : online_status,
            "h2_produced_kg_h" : h2_produced_kg_h,
            "cost_profile"     : cost_profile,
            "slack_values"     : slack_values,
            "results_df"       : results_df,
        }

    def print_solution_summary(self, result: dict):
        """Print formatted optimization result summary."""
        if result["results_df"] is None:
            print(f"Optimization failed: {result['status']}")
            return

        df          = result["results_df"]
        total_slack = df["demand_slack_kg_h"].sum()

        print("\n" + "=" * 55)
        print("OPTIMIZATION RESULT SUMMARY")
        print("=" * 55)
        print(f"  Solver status:           {result['status']}")
        print(f"  Objective value:         {result['objective_value']:.2f}")
        print(f"  Hours optimized:         {len(df)}")
        print(f"  Avg power dispatch:      {df['power_optimized_mw'].mean():.1f} MW")
        print(f"  Electrolyzer online:     {df['online_status'].sum()}/{len(df)} hours")
        print(f"  Total H2 produced:       {df['h2_produced_kg_h'].sum()/1000:.2f} tonnes")
        print(f"  Total H2 demand:         {df['h2_demand_kg_h'].sum()/1000:.2f} tonnes")
        print(f"  Unmet demand (slack):    {total_slack:.0f} kg total")
        print(f"  Hours demand fully met:  {df['demand_met'].sum()}/{len(df)}")
        print(f"  Total electricity cost:  EUR {df['cost_eur'].sum():.0f}")
        print(f"  Avg electricity price:   EUR {df['electricity_price'].mean():.1f}/MWh")
        print("=" * 55)

    def compare_objectives(self,
                            wind_power_mw: np.ndarray,
                            electricity_prices: np.ndarray,
                            carbon_intensity: np.ndarray) -> tuple:
        """
        Run cost and emissions objectives and compare side by side.
        Demonstrates that cheapest != lowest-carbon dispatch.
        """
        print("  Running cost minimization...")
        r_cost = self.optimize(wind_power_mw, electricity_prices,
                               objective="minimize_cost",
                               demand_mode="cumulative")

        print("  Running emissions minimization...")
        r_emis = self.optimize(wind_power_mw, electricity_prices,
                               objective="minimize_emissions",
                               demand_mode="cumulative",
                               carbon_intensity=carbon_intensity)

        if r_cost["results_df"] is None or r_emis["results_df"] is None:
            print("One or both optimizations failed.")
            return None, r_cost, r_emis

        df_c = r_cost["results_df"]
        df_e = r_emis["results_df"]

        emis_of_cost_opt = (df_c["power_optimized_mw"]
                             * carbon_intensity * self.dt).sum()
        emis_of_emis_opt = (df_e["power_optimized_mw"]
                             * carbon_intensity * self.dt).sum()
        cost_of_emis_opt = (df_e["power_optimized_mw"]
                             * electricity_prices * self.dt).sum()
        cost_of_cost_opt = df_c["cost_eur"].sum()

        comparison = pd.DataFrame({
            "Metric": [
                "Total cost (EUR)",
                "Total emissions (kg CO2)",
                "Total H2 produced (kg)",
                "Avg power dispatch (MW)",
                "Hours at full load",
                "Hours offline",
            ],
            "Cost-optimal": [
                f"{cost_of_cost_opt:.0f}",
                f"{emis_of_cost_opt:.0f}",
                f"{df_c['h2_produced_kg_h'].sum():.0f}",
                f"{df_c['power_optimized_mw'].mean():.1f}",
                f"{(df_c['power_optimized_mw'] >= self.p_max * 0.99).sum()}",
                f"{(df_c['online_status'] == 0).sum()}",
            ],
            "Emissions-optimal": [
                f"{cost_of_emis_opt:.0f}",
                f"{emis_of_emis_opt:.0f}",
                f"{df_e['h2_produced_kg_h'].sum():.0f}",
                f"{df_e['power_optimized_mw'].mean():.1f}",
                f"{(df_e['power_optimized_mw'] >= self.p_max * 0.99).sum()}",
                f"{(df_e['online_status'] == 0).sum()}",
            ]
        })
        return comparison, r_cost, r_emis


# --- Visualisation ------------------------------------------

def plot_optimization_result(result: dict,
                              title: str = "Optimized Dispatch",
                              save_path: str = None):
    """Three-panel: dispatch, H2 vs demand, price vs dispatch."""
    if result["results_df"] is None:
        print("No results to plot.")
        return

    df    = result["results_df"]
    hours = df["timestep"].values
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.fill_between(hours, df["wind_available_mw"],
                    alpha=0.2, color="#2196F3", label="Wind available")
    ax.step(hours, df["power_optimized_mw"],
            color="#2196F3", linewidth=2.0, where="post",
            label="Optimized dispatch")
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2        = axes[1]
    demand_val = df["h2_demand_kg_h"].iloc[0]
    ax2.step(hours, df["h2_produced_kg_h"],
             color="#4CAF50", linewidth=2.0, where="post",
             label="H2 produced (kg/h)")
    ax2.axhline(demand_val, color="orange", linestyle="--",
                linewidth=1.5, label="Hourly demand")
    ax2.fill_between(hours, df["h2_produced_kg_h"], demand_val,
                     where=df["h2_produced_kg_h"] >= demand_val,
                     alpha=0.2, color="green", label="Surplus")
    ax2.fill_between(hours, df["h2_produced_kg_h"], demand_val,
                     where=df["h2_produced_kg_h"] < demand_val,
                     alpha=0.2, color="red", label="Deficit")
    ax2.set_ylabel("H2 (kg/hour)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3  = axes[2]
    ax3c = ax3.twinx()
    ax3.bar(hours, df["electricity_price"], alpha=0.4,
            color="#FF9800", label="Electricity price")
    ax3c.step(hours, df["power_optimized_mw"],
              color="#2196F3", linewidth=1.5, where="post", alpha=0.8)
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Price (EUR/MWh)", color="#FF9800")
    ax3c.set_ylabel("Power dispatch (MW)", color="#2196F3")
    ax3.set_title("Price Signal vs Dispatch")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


def plot_objective_comparison(r_cost: dict, r_emis: dict,
                               carbon_intensity: np.ndarray,
                               save_path: str = None):
    """Side-by-side cost-optimal vs emissions-optimal dispatch."""
    if r_cost["results_df"] is None or r_emis["results_df"] is None:
        return

    df_c  = r_cost["results_df"]
    df_e  = r_emis["results_df"]
    hours = df_c["timestep"].values
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col, (df, label, color) in enumerate([
        (df_c, "Cost-optimal",      "#2196F3"),
        (df_e, "Emissions-optimal", "#4CAF50")
    ]):
        ax = axes[0, col]
        ax.fill_between(hours, df["wind_available_mw"],
                        alpha=0.15, color=color)
        ax.step(hours, df["power_optimized_mw"],
                color=color, linewidth=2.0, where="post")
        ax.set_title(f"{label} — Power Dispatch")
        ax.set_ylabel("Power (MW)")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1, col]
        hourly_emissions = df["power_optimized_mw"] * carbon_intensity
        ax2.bar(hours, hourly_emissions, color=color, alpha=0.7)
        ax2.set_title(f"{label} — Hourly Emissions (kg CO2)")
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("kg CO2/hour")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("PyNEXUS — Cost vs Emissions Objective Comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


# --- Sanity check -------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("MILP Dispatch Optimizer — Sanity Check")
    print("=" * 55)

    opt = ElectrolyzerDispatchOptimizer()

    print(f"\nOptimizer parameters:")
    print(f"  Electrolyzer rated:    {opt.p_rated} MW")
    print(f"  Min load:              {opt.p_min} MW")
    print(f"  Max ramp/hour:         {opt.max_ramp_mw} MW/h")
    print(f"  H2 coefficient:        {opt.h2_coeff:.2f} kg/MWh")
    print(f"  Hourly H2 demand:      {opt.hourly_demand_kg:.0f} kg/h")
    print(f"  Demand penalty:        EUR {opt.DEMAND_PENALTY}/kg unmet")

    np.random.seed(42)
    T = 48

    wind_raw = np.clip(
        10 + 5 * np.sin(np.linspace(0, 4*np.pi, T))
        + np.random.normal(0, 2, T), 0, 30
    )
    wind_48h = OffshoreWindFarm().power_output_mw(wind_raw)

    price_48h = np.clip(
        60 + 30 * np.sin(np.linspace(-np.pi/2, 4*np.pi - np.pi/2, T))
        + np.random.normal(0, 8, T), 5, 200
    )

    ci_48h = np.clip(
        200 + 100 * np.sin(np.linspace(0, 4*np.pi, T))
        + np.random.normal(0, 20, T), 50, 400
    )

    print("\n[Test 1] Cost minimization — hourly demand (soft constraint):")
    r1 = opt.optimize(wind_48h, price_48h,
                      objective="minimize_cost",
                      demand_mode="hourly")
    opt.print_solution_summary(r1)

    print("\n[Test 2] Cost minimization — cumulative demand:")
    r2 = opt.optimize(wind_48h, price_48h,
                      objective="minimize_cost",
                      demand_mode="cumulative")
    opt.print_solution_summary(r2)

    if r1["objective_value"] and r2["objective_value"]:
        saving = (r1["results_df"]["cost_eur"].sum()
                - r2["results_df"]["cost_eur"].sum())
        print(f"\n  Cost saving (cumulative vs hourly): EUR {saving:.0f}")
        print(f"  (Temporal flexibility has monetary value)")

    print("\n[Test 3] Cost vs emissions objective comparison:")
    comparison, r_cost, r_emis = opt.compare_objectives(
        wind_48h, price_48h, ci_48h
    )
    if comparison is not None:
        print(comparison.to_string(index=False))

    print("\n[Test 4] Plotting cost-optimal dispatch (48h)...")
    plot_optimization_result(
        r2,
        title="PyNEXUS — MILP Cost-Optimal Dispatch (48h, cumulative demand)"
    )

    if r_cost["results_df"] is not None:
        print("[Test 5] Plotting cost vs emissions comparison...")
        plot_objective_comparison(r_cost, r_emis, ci_48h)

    print("\nDone. Place this file in optimization/dispatch.py")
