# components/wind_turbine.py
# ============================================================
# Offshore Wind Turbine — Power Curve Model
# ============================================================
# Models a wind turbine farm as a power source whose output
# depends on wind speed at hub height.
#
# Key physics decisions (be ready to defend these):
#
#   1. Power curve uses cubic relationship between cut-in and
#      rated wind speed (P ~ v³ follows from kinetic energy
#      in wind: KE = ½mv² and mass flow ~ v, so P ~ v³).
#      This is the fundamental wind energy equation.
#
#   2. Wind speed at hub height is extrapolated from ERA5
#      reference height (10m) using the wind shear power law.
#      Offshore shear exponent α = 0.11 (smoother than onshore).
#
#   3. Farm-level output applies a wake loss factor — turbines
#      downstream receive less wind. Typically 10-15% for
#      offshore arrays. This is a real engineering correction
#      that naive models omit.
#
#   4. Availability factor accounts for planned/unplanned
#      maintenance downtime (~95% for modern offshore turbines).
#
# The output of this model feeds DIRECTLY into the electrolyzer:
#   wind_power_mw → electrolyzer.compute_h2_output()
# That chain is the multi-commodity coupling centrepiece.
# ============================================================

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path


# --- Load config --------------------------------------------
def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parent.parent
    full_path = root / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


# --- Wind shear correction ----------------------------------
# ERA5 data is at 10m reference height.
# Turbine hub is at 120m.
# We correct using the power law: v(h) = v_ref * (h/h_ref)^alpha
# Offshore alpha = 0.11 (IEC standard for open sea conditions)

OFFSHORE_SHEAR_EXPONENT = 0.11
ERA5_REFERENCE_HEIGHT_M = 10.0


def wind_speed_at_hub_height(v_10m: float,
                              hub_height_m: float) -> float:
    """
    Extrapolate wind speed from ERA5 10m reference to hub height
    using the wind shear power law.

    Parameters
    ----------
    v_10m : float or np.ndarray
        Wind speed at 10m height (m/s) from ERA5
    hub_height_m : float
        Turbine hub height (m)

    Returns
    -------
    float or np.ndarray
        Wind speed at hub height (m/s)
    """
    return v_10m * (hub_height_m / ERA5_REFERENCE_HEIGHT_M) ** OFFSHORE_SHEAR_EXPONENT


# --- Single turbine power curve -----------------------------

def single_turbine_power_mw(wind_speed_ms: float,
                             cut_in: float,
                             rated_speed: float,
                             cut_out: float,
                             rated_power_mw: float) -> float:
    """
    Power output of a single turbine at given wind speed.

    Regions:
      - Below cut-in:         0 MW (too little wind)
      - Cut-in to rated:      P ~ v³ (cubic, kinetic energy)
      - Rated to cut-out:     rated_power_mw (pitch control caps output)
      - Above cut-out:        0 MW (safety shutdown)

    Parameters
    ----------
    wind_speed_ms : float
        Wind speed at hub height (m/s)
    cut_in : float
        Cut-in wind speed (m/s)
    rated_speed : float
        Rated wind speed — full power above this (m/s)
    cut_out : float
        Cut-out wind speed — shutdown above this (m/s)
    rated_power_mw : float
        Nameplate capacity of single turbine (MW)

    Returns
    -------
    float
        Power output (MW)
    """
    v = wind_speed_ms

    if v < cut_in or v >= cut_out:
        return 0.0
    elif v >= rated_speed:
        return rated_power_mw
    else:
        # Cubic scaling between cut-in and rated
        # Normalised so that at v=rated_speed, output = rated_power_mw
        cubic_fraction = (v**3 - cut_in**3) / (rated_speed**3 - cut_in**3)
        return rated_power_mw * cubic_fraction


# --- Vectorised version for time-series ---------------------
# Using np.vectorize for clean application over arrays

_turbine_power_vec = np.vectorize(single_turbine_power_mw,
                                   excluded=["cut_in", "rated_speed",
                                             "cut_out", "rated_power_mw"])


# --- Core class ---------------------------------------------

class OffshoreWindFarm:
    """
    Offshore wind farm model.

    Takes wind speed time series (from ERA5) and returns farm-level
    power output after applying hub height correction, wake losses,
    and availability factor.

    Usage
    -----
    >>> farm = OffshoreWindFarm()
    >>> wind_speeds_ms = np.array([8.0, 10.0, 15.0, 5.0])
    >>> power_mw = farm.power_output_mw(wind_speeds_ms)
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)
        w = cfg["wind_turbine"]

        self.rated_power_mw     = w["rated_power_mw"]
        self.n_turbines         = w["n_turbines"]
        self.hub_height_m       = w["hub_height_m"]
        self.cut_in_ms          = w["cut_in_wind_speed_ms"]
        self.rated_ms           = w["rated_wind_speed_ms"]
        self.cut_out_ms         = w["cut_out_wind_speed_ms"]

        # Farm-level correction factors
        # Wake loss: downstream turbines receive less wind
        # Typical offshore value: 10-12% (DNV GL / Orsted benchmarks)
        self.wake_loss_factor   = 0.90    # 10% wake loss

        # Availability: accounts for maintenance downtime
        # Modern offshore: ~95% (O&M included)
        self.availability       = 0.95

        # Total rated capacity
        self.farm_rated_mw = self.rated_power_mw * self.n_turbines

    # --- Core methods ----------------------------------------

    def power_output_mw(self, wind_speed_10m: np.ndarray) -> np.ndarray:
        """
        Farm power output from ERA5 10m wind speed time series.

        Pipeline:
          ERA5 wind speed (10m)
            → hub height correction (power law)
            → single turbine power curve (cubic / capped)
            → multiply by n_turbines
            → apply wake loss factor
            → apply availability factor
            = farm output (MW)

        Parameters
        ----------
        wind_speed_10m : np.ndarray
            Wind speed at 10m height (m/s) — ERA5 native output

        Returns
        -------
        np.ndarray
            Farm-level power output (MW) at each timestep
        """
        wind_speed_10m = np.asarray(wind_speed_10m, dtype=float)

        # Step 1: correct to hub height
        v_hub = wind_speed_at_hub_height(wind_speed_10m, self.hub_height_m)

        # Step 2: single turbine power curve
        p_single = _turbine_power_vec(
            wind_speed_ms=v_hub,
            cut_in=self.cut_in_ms,
            rated_speed=self.rated_ms,
            cut_out=self.cut_out_ms,
            rated_power_mw=self.rated_power_mw
        )

        # Step 3: scale to farm, apply wake and availability
        p_farm = p_single * self.n_turbines * self.wake_loss_factor * self.availability

        return p_farm

    def capacity_factor(self, wind_speed_10m: np.ndarray) -> float:
        """
        Annual capacity factor for given wind speed time series.
        CF = actual_energy / (rated_power × hours)
        Typical offshore North Sea: 45-55%
        """
        p_farm = self.power_output_mw(wind_speed_10m)
        cf = p_farm.mean() / self.farm_rated_mw
        return cf

    def simulate_timeseries(self,
                             wind_speed_10m: np.ndarray,
                             timestep_hours: float = 1.0) -> pd.DataFrame:
        """
        Full time-series simulation with all intermediate values.

        Useful for debugging, validation, and the Jupyter walkthrough.

        Parameters
        ----------
        wind_speed_10m : np.ndarray
            ERA5 wind speed at 10m (m/s)
        timestep_hours : float
            Duration of each timestep

        Returns
        -------
        pd.DataFrame with columns:
            - wind_speed_10m     : ERA5 input
            - wind_speed_hub     : after hub height correction
            - power_single_mw    : single turbine output
            - power_farm_mw      : full farm output (wake + availability)
            - capacity_factor    : instantaneous CF
            - energy_mwh         : energy in this timestep
        """
        wind_speed_10m = np.asarray(wind_speed_10m, dtype=float)
        v_hub = wind_speed_at_hub_height(wind_speed_10m, self.hub_height_m)

        p_single = _turbine_power_vec(
            wind_speed_ms=v_hub,
            cut_in=self.cut_in_ms,
            rated_speed=self.rated_ms,
            cut_out=self.cut_out_ms,
            rated_power_mw=self.rated_power_mw
        )

        p_farm = p_single * self.n_turbines * self.wake_loss_factor * self.availability

        return pd.DataFrame({
            "wind_speed_10m"  : wind_speed_10m,
            "wind_speed_hub"  : v_hub,
            "power_single_mw" : p_single,
            "power_farm_mw"   : p_farm,
            "capacity_factor" : p_farm / self.farm_rated_mw,
            "energy_mwh"      : p_farm * timestep_hours,
        })


# --- Visualisation ------------------------------------------

def plot_power_curve(config_path: str = "config.yaml",
                     save_path: str = None):
    """
    Plot the single turbine power curve and hub height correction.
    Two panels: (1) power curve, (2) wind shear profile.
    """
    cfg = _load_config(config_path)
    w = cfg["wind_turbine"]
    farm = OffshoreWindFarm(config_path)

    wind_speeds = np.linspace(0, 30, 300)

    # Single turbine power at hub height
    p_single = _turbine_power_vec(
        wind_speed_ms=wind_speeds,
        cut_in=w["cut_in_wind_speed_ms"],
        rated_speed=w["rated_wind_speed_ms"],
        cut_out=w["cut_out_wind_speed_ms"],
        rated_power_mw=w["rated_power_mw"]
    )

    # Farm output from 10m ERA5 wind speed
    p_farm = farm.power_output_mw(wind_speeds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 1: Power curve
    ax = axes[0]
    ax.plot(wind_speeds, p_single, color="#2196F3",
            linewidth=2.5, label="Single turbine (hub height)")
    ax.plot(wind_speeds, p_farm, color="#4CAF50",
            linewidth=2, linestyle="--",
            label=f"Farm ({w['n_turbines']} turbines, wake+avail)")
    ax.axvline(w["cut_in_wind_speed_ms"], color="gray",
               linestyle=":", alpha=0.7, label="Cut-in")
    ax.axvline(w["rated_wind_speed_ms"], color="orange",
               linestyle=":", alpha=0.7, label="Rated")
    ax.axvline(w["cut_out_wind_speed_ms"], color="red",
               linestyle=":", alpha=0.7, label="Cut-out")
    ax.set_xlabel("Wind speed at hub height (m/s)")
    ax.set_ylabel("Power output (MW)")
    ax.set_title("Offshore Wind Farm Power Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Wind shear — 10m ERA5 vs hub height
    v_10m = np.linspace(0, 25, 100)
    v_hub = wind_speed_at_hub_height(v_10m, w["hub_height_m"])
    ax2 = axes[1]
    ax2.plot(v_10m, v_hub, color="#9C27B0", linewidth=2.5)
    ax2.plot([0, 25], [0, 25], color="gray", linestyle="--",
             alpha=0.5, label="1:1 reference")
    ax2.set_xlabel("Wind speed at 10m ERA5 (m/s)")
    ax2.set_ylabel(f"Wind speed at {w['hub_height_m']}m hub (m/s)")
    ax2.set_title(f"Wind Shear Correction (α={OFFSHORE_SHEAR_EXPONENT})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig


# --- Quick sanity check -------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("Offshore Wind Farm — Sanity Check")
    print("=" * 55)

    farm = OffshoreWindFarm()

    # Test 1: Power at key wind speeds
    print("\n[Test 1] Farm power at key wind speeds:")
    test_speeds_10m = [2, 5, 8, 10, 13, 15, 20, 26]
    for v in test_speeds_10m:
        v_hub = wind_speed_at_hub_height(v, farm.hub_height_m)
        p = farm.power_output_mw(np.array([float(v)]))[0]
        cf = p / farm.farm_rated_mw
        print(f"  v_10m={v:3d} m/s → v_hub={v_hub:5.1f} m/s "
              f"→ P={p:6.1f} MW  CF={cf*100:.0f}%")

    # Test 2: Simulate a 24-hour synthetic wind profile
    print("\n[Test 2] 24-hour synthetic wind profile:")
    # Realistic North Sea diurnal pattern — speeds vary 6-16 m/s
    np.random.seed(42)
    wind_24h = 10 + 4 * np.sin(np.linspace(0, 2*np.pi, 24)) \
               + np.random.normal(0, 1.0, 24)
    wind_24h = np.clip(wind_24h, 0, 30)

    result = farm.simulate_timeseries(wind_24h)
    print(result[["wind_speed_10m", "wind_speed_hub",
                   "power_farm_mw", "capacity_factor"]].round(2).to_string())

    cf_avg = farm.capacity_factor(wind_24h)
    print(f"\n  Average capacity factor: {cf_avg*100:.1f}%")
    print(f"  Total energy (24h): {result['energy_mwh'].sum():.0f} MWh")

    # Test 3: Plot power curve
    print("\n[Test 3] Plotting power curve...")
    plot_power_curve()
    print("Done. Place this file in components/wind_turbine.py")
