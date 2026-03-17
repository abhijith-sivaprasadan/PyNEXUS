# assets/green_hydrogen_asset.py
# ============================================================
# Green Hydrogen Asset — Multi-Commodity Coupling Layer
# ============================================================
# This is Layer 2: it connects the three Layer 1 components
# into a single coupled energy asset.
#
# Physical chain at each timestep:
#
#   ERA5 wind speed (m/s at 10m)
#       ↓  wind_turbine.power_output_mw()
#   Wind power available (MW)
#       ↓  [curtailment logic if power > electrolyzer rated]
#   Power to electrolyzer (MW)
#       ↓  electrolyzer.simulate_timeseries()
#   H2 mass flow requested (kg/s)
#       ↓  pipeline.constrained_flow()
#   H2 mass flow delivered (kg/s)
#       ↓  compare to hydrogen_demand
#   Demand met / unmet (kg/h)
#
# Key coupling decisions (be ready to defend these):
#
#   1. Wind power directly drives the electrolyzer — no grid
#      intermediary. This is the "islanded" green hydrogen
#      model: all wind electricity goes to electrolysis.
#      A more complex model (Layer 3) will add grid interaction.
#
#   2. Electrolyzer ramp constraint creates INTER-TIMESTEP
#      coupling. You cannot optimise each hour independently —
#      the previous hour's operating point constrains the next.
#      This is what makes it a DYNAMIC model, not static.
#
#   3. Pipeline pressure constraint creates CROSS-COMMODITY
#      coupling. Too much wind → too much H2 → pipeline
#      pressure drops → curtailment needed. The electricity
#      dispatch must respect the hydrogen network physics.
#      This is the multi-commodity nexus.
#
#   4. System KPIs track where energy is lost:
#      - Wind curtailment (wind > electrolyzer capacity)
#      - Electrolyzer ramp curtailment (ramp rate limit)
#      - Pipeline curtailment (pressure constraint)
#      - Demand unmet (H2 delivery < demand)
#      Together these reveal the system bottleneck.
# ============================================================

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Import Layer 1 components
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components.electrolyzer import PEMElectrolyzer
from components.wind_turbine  import OffshoreWindFarm
from components.pipeline      import HydrogenPipeline


# --- Load config --------------------------------------------
def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parent.parent
    full_path = root / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


# --- Core class ---------------------------------------------

class GreenHydrogenAsset:
    """
    Coupled wind-electrolyzer-pipeline asset model.

    Simulates a green hydrogen production system where an
    offshore wind farm powers a PEM electrolyzer, which feeds
    hydrogen into a transmission pipeline to a demand node.

    This is the direct Python analogue of the PyDOLPHYN
    asset-level simulation — physics-based, dynamic,
    multi-commodity.

    Usage
    -----
    >>> asset = GreenHydrogenAsset()
    >>> wind_speeds = np.array([8.0, 10.0, 12.0, ...])  # ERA5 data
    >>> results = asset.simulate(wind_speeds)
    >>> asset.print_kpis(results)
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)

        # Instantiate Layer 1 components
        self.wind_farm    = OffshoreWindFarm(config_path)
        self.electrolyzer = PEMElectrolyzer(config_path)
        self.pipeline     = HydrogenPipeline(config_path)

        # Demand parameters
        d = cfg["hydrogen_demand"]
        self.daily_demand_kg        = d["daily_average_kg"]
        self.hourly_demand_kg       = self.daily_demand_kg / 24.0

        # Simulation parameters
        s = cfg["simulation"]
        self.timestep_hours         = s["time_step_hours"]

    # --- Core simulation ------------------------------------

    def simulate(self, wind_speed_10m: np.ndarray) -> pd.DataFrame:
        """
        Run a full coupled simulation over a wind speed time series.

        At each timestep:
          1. Compute wind farm power output
          2. Apply electrolyzer input limits (curtail excess wind)
          3. Run electrolyzer with ramp constraints
          4. Apply pipeline pressure constraint
          5. Compare delivery to demand

        Parameters
        ----------
        wind_speed_10m : np.ndarray
            Wind speed at 10m height (m/s) — ERA5 input

        Returns
        -------
        pd.DataFrame
            Full time-series results with all system variables
            and curtailment/loss breakdown
        """
        wind_speed_10m = np.asarray(wind_speed_10m, dtype=float)
        n              = len(wind_speed_10m)
        dt             = self.timestep_hours

        # --- Step 1: Wind farm power output -----------------
        wind_power_mw = self.wind_farm.power_output_mw(wind_speed_10m)

        # --- Step 2: Electrolyzer input (curtail excess) ----
        # Wind power may exceed electrolyzer rated capacity.
        # Excess is curtailed (spilled) — in a real system it
        # would go to the grid, but in this islanded model it
        # is lost. Layer 3 will add grid export.
        elec_rated     = self.electrolyzer.rated_power_mw
        power_to_elec  = np.minimum(wind_power_mw, elec_rated)
        wind_curtailed = wind_power_mw - power_to_elec   # MW curtailed

        # --- Step 3: Electrolyzer simulation ----------------
        # This applies ramp rate constraints internally.
        # Returns actual power consumed + H2 output per timestep.
        elec_result = self.electrolyzer.simulate_timeseries(
            power_to_elec, timestep_hours=dt
        )

        actual_power_mw = elec_result["power_input_mw"].values
        h2_requested    = elec_result["h2_output_kg_s"].values   # kg/s
        efficiency      = elec_result["efficiency"].values
        load_fraction   = elec_result["load_fraction"].values

        # Ramp curtailment: difference between what wind offered
        # and what the electrolyzer could actually consume
        ramp_curtailed_mw = power_to_elec - actual_power_mw

        # --- Step 4: Pipeline constraint --------------------
        # Pipeline may not be able to accept all H2 produced.
        # constrained_flow() returns max feasible flow.
        h2_delivered = np.array([
            self.pipeline.constrained_flow(q) for q in h2_requested
        ])
        pipeline_curtailed = h2_requested - h2_delivered   # kg/s

        outlet_pressure = np.array([
            self.pipeline.outlet_pressure(q) for q in h2_delivered
        ])

        # --- Step 5: Demand satisfaction --------------------
        hourly_demand_kg_s = self.hourly_demand_kg / 3600.0   # kg/s

        demand_met    = np.minimum(h2_delivered, hourly_demand_kg_s)
        demand_unmet  = np.maximum(hourly_demand_kg_s - h2_delivered, 0)
        demand_excess = np.maximum(h2_delivered - hourly_demand_kg_s, 0)

        # --- Energy accounting (MWh per timestep) -----------
        wind_energy_mwh      = wind_power_mw    * dt
        elec_energy_mwh      = actual_power_mw  * dt
        curtailed_energy_mwh = wind_curtailed   * dt + ramp_curtailed_mw * dt

        # H2 in kg per timestep
        h2_produced_kg  = h2_requested  * 3600.0 * dt
        h2_delivered_kg = h2_delivered  * 3600.0 * dt
        h2_demand_kg    = np.full(n, self.hourly_demand_kg * dt)

        return pd.DataFrame({
            # Inputs
            "wind_speed_10m_ms"      : wind_speed_10m,
            # Wind farm
            "wind_power_mw"          : wind_power_mw,
            "wind_curtailed_mw"      : wind_curtailed,
            "power_to_electrolyzer_mw": actual_power_mw,
            # Electrolyzer
            "load_fraction"          : load_fraction,
            "efficiency"             : efficiency,
            "h2_produced_kg_s"       : h2_requested,
            "h2_produced_kg"         : h2_produced_kg,
            # Pipeline
            "h2_delivered_kg_s"      : h2_delivered,
            "h2_delivered_kg"        : h2_delivered_kg,
            "pipeline_curtailed_kg_s": pipeline_curtailed,
            "outlet_pressure_bar"    : outlet_pressure,
            # Demand
            "h2_demand_kg"           : h2_demand_kg,
            "demand_met_kg_s"        : demand_met,
            "demand_unmet_kg_s"      : demand_unmet,
            "demand_excess_kg_s"     : demand_excess,
            # Energy accounting
            "wind_energy_mwh"        : wind_energy_mwh,
            "electrolyzer_energy_mwh": elec_energy_mwh,
            "curtailed_energy_mwh"   : curtailed_energy_mwh,
            # Ramp curtailment
            "ramp_curtailed_mw"      : ramp_curtailed_mw,
        })

    # --- KPI calculation ------------------------------------

    def compute_kpis(self, results: pd.DataFrame) -> dict:
        """
        Compute system-level KPIs from simulation results.

        These are the metrics you'd report to a client or use
        to compare design alternatives — exactly what TNO does.

        Returns
        -------
        dict of KPI name → value
        """
        total_wind_energy    = results["wind_energy_mwh"].sum()
        total_elec_energy    = results["electrolyzer_energy_mwh"].sum()
        total_h2_produced    = results["h2_produced_kg"].sum()
        total_h2_delivered   = results["h2_delivered_kg"].sum()
        total_h2_demand      = results["h2_demand_kg"].sum()
        total_curtailed      = results["curtailed_energy_mwh"].sum()

        # Capacity factor
        farm_rated = self.wind_farm.farm_rated_mw
        hours      = len(results) * self.timestep_hours
        cf         = total_wind_energy / (farm_rated * hours)

        # System efficiency: H2 energy out / wind energy in
        lhv_kwh_per_kg   = 33.33
        h2_energy_mwh    = total_h2_delivered * lhv_kwh_per_kg / 1000.0
        system_efficiency = h2_energy_mwh / total_wind_energy if total_wind_energy > 0 else 0

        # Curtailment breakdown
        wind_curtail_mwh  = results["wind_curtailed_mw"].sum() * self.timestep_hours
        ramp_curtail_mwh  = results["ramp_curtailed_mw"].sum() * self.timestep_hours
        pipe_curtail_kg   = results["pipeline_curtailed_kg_s"].sum() * 3600

        # Demand satisfaction
        demand_coverage = (total_h2_delivered / total_h2_demand * 100
                           if total_h2_demand > 0 else 0)

        # Electrolyzer utilisation
        online_hours    = (results["load_fraction"] > 0).sum() * self.timestep_hours
        utilisation     = online_hours / hours * 100

        return {
            "simulation_hours"          : hours,
            "wind_energy_mwh"           : round(total_wind_energy, 1),
            "electrolyzer_energy_mwh"   : round(total_elec_energy, 1),
            "wind_capacity_factor_pct"  : round(cf * 100, 1),
            "h2_produced_tonnes"        : round(total_h2_produced / 1000, 2),
            "h2_delivered_tonnes"       : round(total_h2_delivered / 1000, 2),
            "h2_demand_tonnes"          : round(total_h2_demand / 1000, 2),
            "demand_coverage_pct"       : round(demand_coverage, 1),
            "system_efficiency_pct"     : round(system_efficiency * 100, 1),
            "wind_curtailment_mwh"      : round(wind_curtail_mwh, 1),
            "ramp_curtailment_mwh"      : round(ramp_curtail_mwh, 1),
            "pipeline_curtailment_kg"   : round(pipe_curtail_kg, 1),
            "electrolyzer_utilisation_pct": round(utilisation, 1),
        }

    def print_kpis(self, results: pd.DataFrame):
        """Print formatted KPI summary — useful for quick checks."""
        kpis = self.compute_kpis(results)
        print("\n" + "=" * 50)
        print("SYSTEM KPI SUMMARY")
        print("=" * 50)
        print(f"  Simulation hours:          {kpis['simulation_hours']:.0f} h")
        print(f"  Wind energy in:            {kpis['wind_energy_mwh']:.1f} MWh")
        print(f"  Wind capacity factor:      {kpis['wind_capacity_factor_pct']:.1f}%")
        print(f"  Electrolyzer utilisation:  {kpis['electrolyzer_utilisation_pct']:.1f}%")
        print(f"  H2 produced:               {kpis['h2_produced_tonnes']:.2f} tonnes")
        print(f"  H2 delivered:              {kpis['h2_delivered_tonnes']:.2f} tonnes")
        print(f"  H2 demand:                 {kpis['h2_demand_tonnes']:.2f} tonnes")
        print(f"  Demand coverage:           {kpis['demand_coverage_pct']:.1f}%")
        print(f"  System efficiency (W2H):   {kpis['system_efficiency_pct']:.1f}%")
        print("-" * 50)
        print(f"  Wind curtailment:          {kpis['wind_curtailment_mwh']:.1f} MWh")
        print(f"  Ramp curtailment:          {kpis['ramp_curtailment_mwh']:.1f} MWh")
        print(f"  Pipeline curtailment:      {kpis['pipeline_curtailment_kg']:.1f} kg")
        print("=" * 50)

    # --- Visualisation --------------------------------------

    def plot_results(self, results: pd.DataFrame,
                     title: str = "Green Hydrogen Asset — Simulation Results",
                     save_path: str = None):
        """
        Four-panel system overview plot.

        Panel 1: Wind power and electrolyzer input
        Panel 2: H2 production and delivery vs demand
        Panel 3: Pipeline outlet pressure
        Panel 4: Curtailment breakdown
        """
        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)
        hours = np.arange(len(results))

        # Panel 1: Wind and electrolyzer power
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.fill_between(hours, results["wind_power_mw"],
                         alpha=0.3, color="#2196F3", label="Wind available")
        ax1.plot(hours, results["power_to_electrolyzer_mw"],
                 color="#2196F3", linewidth=1.5, label="To electrolyzer")
        ax1.axhline(self.electrolyzer.rated_power_mw, color="red",
                    linestyle="--", alpha=0.5, linewidth=1,
                    label=f"Elec. rated ({self.electrolyzer.rated_power_mw} MW)")
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Power (MW)")
        ax1.set_title("Wind Farm → Electrolyzer")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Panel 2: H2 production vs demand
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(hours, results["h2_produced_kg"] / 1000,
                 color="#4CAF50", linewidth=1.5, label="H2 produced (t)")
        ax2.plot(hours, results["h2_delivered_kg"] / 1000,
                 color="#8BC34A", linewidth=1.5,
                 linestyle="--", label="H2 delivered (t)")
        ax2.axhline(self.hourly_demand_kg / 1000, color="orange",
                    linestyle=":", linewidth=1.5,
                    label=f"Demand ({self.hourly_demand_kg/1000:.1f} t/h)")
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("H₂ (tonnes/hour)")
        ax2.set_title("H2 Production & Delivery vs Demand")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Pipeline outlet pressure
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(hours, results["outlet_pressure_bar"],
                 color="#9C27B0", linewidth=1.5)
        ax3.axhline(self.pipeline.min_outlet_pressure, color="red",
                    linestyle="--", alpha=0.7,
                    label=f"Min pressure ({self.pipeline.min_outlet_pressure} bar)")
        ax3.axhline(self.pipeline.inlet_pressure_bar, color="gray",
                    linestyle=":", alpha=0.5,
                    label=f"Inlet ({self.pipeline.inlet_pressure_bar} bar)")
        ax3.set_xlabel("Hour")
        ax3.set_ylabel("Pressure (bar)")
        ax3.set_title("Pipeline Outlet Pressure")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Curtailment breakdown
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.stackplot(hours,
                      results["wind_curtailed_mw"],
                      results["ramp_curtailed_mw"],
                      labels=["Wind curtailment (MW)",
                               "Ramp curtailment (MW)"],
                      colors=["#FF9800", "#F44336"],
                      alpha=0.7)
        ax4.set_xlabel("Hour")
        ax4.set_ylabel("Curtailed power (MW)")
        ax4.set_title("Curtailment Breakdown")
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        return fig


# --- Sanity check -------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("Green Hydrogen Asset — Coupled Simulation Check")
    print("=" * 55)

    asset = GreenHydrogenAsset()

    print(f"\nAsset configuration:")
    print(f"  Wind farm rated:       {asset.wind_farm.farm_rated_mw:.0f} MW")
    print(f"  Electrolyzer rated:    {asset.electrolyzer.rated_power_mw:.0f} MW")
    print(f"  Pipeline max flow:     {asset.pipeline.max_feasible_flow_kg_s:.2f} kg/s")
    print(f"  H2 demand:             {asset.hourly_demand_kg:.0f} kg/h")

    # --- Test 1: Calm conditions (wind below rated) ---------
    print("\n[Test 1] 24h — moderate wind (avg 8 m/s):")
    np.random.seed(10)
    wind_calm = np.clip(8 + np.random.normal(0, 1.5, 24), 3, 20)
    results_calm = asset.simulate(wind_calm)
    asset.print_kpis(results_calm)

    # --- Test 2: Strong wind (frequent full output) ---------
    print("\n[Test 2] 24h — strong wind (avg 14 m/s):")
    wind_strong = np.clip(14 + np.random.normal(0, 2, 24), 3, 28)
    results_strong = asset.simulate(wind_strong)
    asset.print_kpis(results_strong)

    # --- Test 3: Highly variable wind (ramp constraints) ----
    print("\n[Test 3] 24h — highly variable wind (ramp stress test):")
    # Alternate between low and high wind — maximises ramp events
    wind_variable = np.tile([5.0, 15.0], 12)
    results_var = asset.simulate(wind_variable)
    asset.print_kpis(results_var)

    # --- Test 4: Full plot of variable wind case ------------
    print("\n[Test 4] Plotting variable wind results...")
    asset.plot_results(results_var,
                       title="PyNEXUS — Green H2 Asset: Variable Wind Stress Test")

    print("\nDone. Place this file in assets/green_hydrogen_asset.py")
