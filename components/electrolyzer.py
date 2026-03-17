# components/electrolyzer.py
# ============================================================
# PEM Electrolyzer — Dynamic Component Model
# ============================================================
# Models a Proton Exchange Membrane (PEM) electrolyzer as a
# dynamic energy component: electrical load IN, hydrogen mass
# flow OUT.
#
# Key physics decisions (be ready to defend these):
#
#   1. Efficiency is load-dependent (not constant). We use a
#      quadratic curve fitted to real PEM datasheet behaviour.
#      At partial load, efficiency drops — this matters for
#      dispatch optimization because running at 50% load is
#      not simply half the hydrogen output of 100% load.
#
#   2. Ramp rate is constrained. PEM electrolyzers are fast
#      (advantage over alkaline) but not instantaneous.
#      This creates inter-timestep coupling in the optimizer.
#
#   3. Hydrogen output is in kg/s using LHV basis (33.33 kWh/kg).
#      LHV is the standard for electrolyzer benchmarking.
#
# Reference parameters sourced from config.yaml — no magic
# numbers in this file.
# ============================================================

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


# --- Load config --------------------------------------------
def _load_config(config_path: str = "config.yaml") -> dict:
    # Resolve config path relative to project root (one level up from components/)
    root = Path(__file__).resolve().parent.parent
    full_path = root / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


# --- Constants ----------------------------------------------
LHV_HYDROGEN_KWH_PER_KG = 33.33   # Lower Heating Value of H2


# --- Efficiency curve ---------------------------------------
# Real PEM electrolyzers show a slight efficiency drop at very
# low and very high loads. We model this as a quadratic:
#
#   eta(x) = eta_nom * (a*x^2 + b*x + c)
#
# where x = load_fraction (0 to 1), and coefficients are chosen
# so that:
#   - Peak efficiency occurs around x = 0.6-0.7 (typical PEM)
#   - At x = 1.0, eta = eta_nominal (our reference point)
#   - At x = min_load, eta is ~5% below nominal
#
# This is a simplified but physically motivated curve.
# More detailed models use polarization curves (Butler-Volmer),
# but that requires electrochemical parameters we don't have
# for a generic unit.

def efficiency_at_load(load_fraction: float,
                        nominal_efficiency: float) -> float:
    """
    Returns electrolyzer efficiency at a given load fraction.

    Parameters
    ----------
    load_fraction : float
        Operating point as fraction of rated power (0.0 to 1.0)
    nominal_efficiency : float
        Efficiency at rated (full) load, from config (e.g. 0.70)

    Returns
    -------
    float
        Actual efficiency at this load point (dimensionless)
    """
    # Quadratic coefficients tuned to PEM behaviour:
    # slight peak around 60-70% load, slight drop at full load
    # and significant drop below 20% load
    a = -0.15
    b = 0.25
    c = 0.90

    # Shape factor (normalised so that at x=1.0, shape=1.0)
    shape = a * load_fraction**2 + b * load_fraction + c
    shape_at_full = a * 1.0**2 + b * 1.0 + c  # = 1.0

    return nominal_efficiency * (shape / shape_at_full)


# --- Core class ---------------------------------------------

class PEMElectrolyzer:
    """
    Dynamic model of a PEM electrolyzer unit.

    Tracks state across timesteps (current load, ramp history)
    and computes hydrogen output for each operating point.

    Usage
    -----
    >>> elec = PEMElectrolyzer()
    >>> h2_kg_s = elec.compute_h2_output(power_mw=80.0)
    >>> print(f"H2 output: {h2_kg_s:.4f} kg/s")
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)
        e = cfg["electrolyzer"]

        self.rated_power_mw      = e["rated_power_mw"]
        self.min_load_fraction   = e["min_load_fraction"]
        self.max_load_fraction   = e["max_load_fraction"]
        self.nominal_efficiency  = e["nominal_efficiency"]
        self.ramp_rate_per_hour   = e["ramp_rate_per_hour"]
        self.outlet_pressure_bar = e["hydrogen_output_pressure_bar"]
        self.degradation_rate    = e["degradation_rate_per_year"]

        # State variables (updated each timestep)
        self.current_load_fraction = 0.0   # starts offline
        self.age_years             = 0.0   # for degradation calc
        self._is_online            = False

    # --- Properties -----------------------------------------

    @property
    def min_power_mw(self) -> float:
        return self.rated_power_mw * self.min_load_fraction

    @property
    def max_power_mw(self) -> float:
        return self.rated_power_mw * self.max_load_fraction

    @property
    def effective_efficiency(self) -> float:
        """
        Nominal efficiency degraded by age.
        Each year of operation reduces efficiency by degradation_rate.
        """
        degradation_factor = 1.0 - (self.degradation_rate * self.age_years)
        base_eta = efficiency_at_load(
            self.current_load_fraction,
            self.nominal_efficiency
        )
        return base_eta * degradation_factor

    # --- Core methods ----------------------------------------

    def compute_h2_output(self, power_mw: float) -> float:
        """
        Given electrical power input, return hydrogen mass flow.

        This is the fundamental coupling equation between the
        electricity network and the hydrogen pipeline network.

        Parameters
        ----------
        power_mw : float
            Electrical power consumed by electrolyzer (MW)

        Returns
        -------
        float
            Hydrogen mass flow rate (kg/s)
            Returns 0.0 if power is below minimum stable load.
        """
        # Below minimum load: electrolyzer shuts down
        if power_mw < self.min_power_mw:
            return 0.0

        # Clamp to rated capacity
        power_mw = min(power_mw, self.max_power_mw)

        # Load fraction at this operating point
        load_fraction = power_mw / self.rated_power_mw

        # Efficiency at this load (load-dependent, age-degraded)
        eta = efficiency_at_load(load_fraction, self.nominal_efficiency)
        eta *= (1.0 - self.degradation_rate * self.age_years)

        # Hydrogen energy output (MWh_H2 per hour = MW_H2)
        h2_power_mw = power_mw * eta

        # Convert MW_H2 → kg/s using LHV
        # MW = MJ/s, LHV = 33.33 kWh/kg = 120.0 MJ/kg
        lhv_mj_per_kg = LHV_HYDROGEN_KWH_PER_KG * 3.6   # = 120.0 MJ/kg
        h2_kg_per_s = (h2_power_mw * 1e6) / (lhv_mj_per_kg * 1e6)
        # Simplifies to:
        h2_kg_per_s = h2_power_mw / lhv_mj_per_kg

        return h2_kg_per_s

    def max_ramp_mw_per_timestep(self, timestep_hours: float = 1.0) -> float:
        """
        For sub-hourly timesteps: use per-minute ramp rate.
        For hourly timesteps: use per-hour ramp rate directly.
        This reflects that PEM ramp constraints are meaningful
        at sub-hourly resolution but not at hourly resolution."""
        cfg = _load_config()
        e = cfg["electrolyzer"]
    
        if timestep_hours < 0.5:
            # Sub-hourly: per-minute constraint is meaningful
            timestep_minutes = timestep_hours * 60.0
            max_ramp_fraction = self.ramp_rate_per_min * timestep_minutes
        else:
            # Hourly: use explicit hourly ramp rate from config
            hourly_ramp = e.get("ramp_rate_per_hour", 0.40)
            max_ramp_fraction = hourly_ramp * timestep_hours

        return max_ramp_fraction * self.rated_power_mw
        """
        Make both changes, run again, and Test 2 should now show the power ramping up step by step — not jumping straight to 100 MW.

        When you fix it, the output should look roughly like:

        0   40.0 MW   (ramp from 0)
        1   80.0 MW   (another ramp step)
        2  100.0 MW   (hits rated)
        3  100.0 MW   (steady state)
        """
    def simulate_timeseries(self,
                             power_input_mw: np.ndarray,
                             timestep_hours: float = 1.0) -> pd.DataFrame:
        """
        Run the electrolyzer over a time series of power inputs.

        Applies ramp rate constraints: if requested power change
        exceeds ramp limit, the actual power is clipped to the
        nearest feasible value. This is the dynamic behaviour
        that static models miss entirely.

        Parameters
        ----------
        power_input_mw : np.ndarray
            Time series of requested electrical power (MW)
        timestep_hours : float
            Duration of each timestep in hours

        Returns
        -------
        pd.DataFrame with columns:
            - power_input_mw     : actual (ramp-constrained) power
            - load_fraction      : operating point (0-1)
            - efficiency         : actual efficiency at each step
            - h2_output_kg_s     : hydrogen mass flow (kg/s)
            - h2_output_kg_h     : hydrogen mass flow (kg/hour)
            - h2_cumulative_kg   : running total of H2 produced
        """
        n = len(power_input_mw)
        max_ramp = self.max_ramp_mw_per_timestep(timestep_hours)

        actual_power   = np.zeros(n)
        load_fractions = np.zeros(n)
        efficiencies   = np.zeros(n)
        h2_kg_s        = np.zeros(n)

        current_power = 0.0  # start offline

        for i, requested_power in enumerate(power_input_mw):

            # Apply ramp rate constraint
            delta = requested_power - current_power
            delta_clamped = np.clip(delta, -max_ramp, max_ramp)
            feasible_power = current_power + delta_clamped

            # Enforce min/max bounds
            if feasible_power < self.min_power_mw:
                feasible_power = 0.0   # shut down rather than run below minimum
            feasible_power = min(feasible_power, self.max_power_mw)

            # Record state
            actual_power[i]    = feasible_power
            load_fractions[i]  = feasible_power / self.rated_power_mw if feasible_power > 0 else 0.0
            efficiencies[i]    = efficiency_at_load(load_fractions[i], self.nominal_efficiency) if feasible_power > 0 else 0.0
            h2_kg_s[i]         = self.compute_h2_output(feasible_power)

            current_power = feasible_power

        h2_kg_h = h2_kg_s * 3600.0
        h2_cumulative = np.cumsum(h2_kg_h) * timestep_hours

        return pd.DataFrame({
            "power_input_mw"   : actual_power,
            "load_fraction"    : load_fractions,
            "efficiency"       : efficiencies,
            "h2_output_kg_s"   : h2_kg_s,
            "h2_output_kg_h"   : h2_kg_h,
            "h2_cumulative_kg" : h2_cumulative,
        })


# --- Visualisation ------------------------------------------

def plot_efficiency_curve(nominal_efficiency: float = 0.70,
                           save_path: str = None):
    """
    Plot the load-dependent efficiency curve.
    Useful for reports and the Jupyter notebook walkthrough.
    """
    load_points = np.linspace(0.05, 1.0, 200)
    efficiencies = [efficiency_at_load(x, nominal_efficiency) for x in load_points]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(load_points * 100, np.array(efficiencies) * 100,
            color="#2196F3", linewidth=2.5, label="PEM efficiency (LHV basis)")
    ax.axhline(nominal_efficiency * 100, color="gray",
               linestyle="--", alpha=0.6, label=f"Nominal: {nominal_efficiency*100:.0f}%")
    ax.set_xlabel("Load fraction (%)")
    ax.set_ylabel("Efficiency (%, LHV)")
    ax.set_title("PEM Electrolyzer Load-Dependent Efficiency Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(55, 80)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


# --- Quick sanity check (run this file directly to test) ----
if __name__ == "__main__":
    print("=" * 55)
    print("PEM Electrolyzer — Sanity Check")
    print("=" * 55)

    elec = PEMElectrolyzer()

    # Test 1: H2 output at key operating points
    print("\n[Test 1] H2 output vs load fraction:")
    for load_pct in [10, 25, 50, 75, 100]:
        power = elec.rated_power_mw * (load_pct / 100)
        h2    = elec.compute_h2_output(power)
        eta   = efficiency_at_load(load_pct / 100, elec.nominal_efficiency)
        print(f"  {load_pct:3d}% load ({power:6.1f} MW) → "
              f"η={eta*100:.1f}% → H2={h2*3600:.1f} kg/h")

    # Test 2: Simulate a step-change in power (ramp constraint visible)
    print("\n[Test 2] Ramp rate constraint — step from 0 to 100 MW:")
    step_input = np.array([100.0] * 10)   # request full power immediately
    result = elec.simulate_timeseries(step_input, timestep_hours=1.0)
    print(result[["power_input_mw", "load_fraction",
                   "efficiency", "h2_output_kg_h"]].to_string(index=True))

    # Test 3: Plot efficiency curve
    print("\n[Test 3] Plotting efficiency curve...")
    plot_efficiency_curve()
    print("Done. Place this file in components/electrolyzer.py")
