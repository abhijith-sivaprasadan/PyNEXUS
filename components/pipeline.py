# components/pipeline.py
# ============================================================
# Hydrogen Pipeline — Steady-State Flow Model
# ============================================================
# Models a single hydrogen pipeline segment connecting the
# electrolyzer outlet to the demand node.
#
# Key physics decisions (be ready to defend these):
#
#   1. We use the Weymouth equation for compressible gas flow.
#      This is the industry standard for gas transmission
#      pipelines (used by ENTSOG, Gasunie, and in pandapipes).
#      It assumes isothermal flow — valid for buried pipelines
#      where soil temperature stabilises the gas.
#
#   2. Hydrogen is NOT an ideal gas at pipeline pressures.
#      We apply the van der Waals compressibility correction (Z).
#      For H2 at 30 bar and 15°C, Z ≈ 1.006 — small but
#      physically correct. At higher pressures (>100 bar) this
#      matters much more.
#
#   3. Friction factor uses the Chen approximation to the
#      Colebrook-White equation (explicit, avoids iteration).
#      This is what pandapipes uses internally.
#
#   4. We track both mass flow (kg/s) AND pressure drop.
#      Pressure drop is the key constraint: if outlet pressure
#      falls below minimum delivery pressure, the pipeline is
#      capacity-constrained regardless of electrolyzer output.
#      THIS is the multi-commodity coupling constraint that
#      links the hydrogen network to the electricity dispatch.
#
# Physical chain:
#   electrolyzer H2 output (kg/s)
#     → pipeline inlet (pressure, flow)
#     → Weymouth equation (pressure drop)
#     → outlet pressure check (feasibility)
#     → demand node delivery (kg/s, bar)
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


# --- Physical constants -------------------------------------
R_UNIVERSAL       = 8314.0    # J / (kmol·K)
M_HYDROGEN        = 2.016     # kg/kmol
R_HYDROGEN        = R_UNIVERSAL / M_HYDROGEN   # J/(kg·K) = 4124.0
T_STANDARD_K      = 288.15    # 15°C in Kelvin (pipeline standard)
P_STANDARD_PA     = 101325.0  # 1 atm in Pa

# Van der Waals constants for hydrogen
VDW_A_H2          = 0.02476   # Pa·m⁶/mol²
VDW_B_H2          = 2.661e-5  # m³/mol


# --- Hydrogen gas properties --------------------------------

def compressibility_factor_h2(pressure_bar: float,
                               temperature_k: float = T_STANDARD_K) -> float:
    """
    Van der Waals compressibility factor Z for hydrogen.
    Uses first-order virial expansion: Z = 1 + (b - a/RT) * P/RT
    At 30 bar, 15C: Z ~ 1.02
    """
    R_MOL = 8.314          # J/(mol·K) — molar gas constant
    P_pa  = pressure_bar * 1e5

    a = VDW_A_H2           # Pa·m6/mol2
    b = VDW_B_H2           # m3/mol

    RT = R_MOL * temperature_k                 # J/mol
    Z  = 1.0 + (b - a / RT) * (P_pa / RT)

    return max(float(Z), 1.0)


def hydrogen_density_kg_m3(pressure_bar: float,
                            temperature_k: float = T_STANDARD_K) -> float:
    """
    Hydrogen gas density at given pressure and temperature.
    Uses real gas equation: ρ = P / (Z * R_H2 * T)
    """
    P_pa = pressure_bar * 1e5
    Z    = compressibility_factor_h2(pressure_bar, temperature_k)
    return P_pa / (Z * R_HYDROGEN * temperature_k)


# --- Friction factor -----------------------------------------

def friction_factor_chen(diameter_m: float,
                          roughness_m: float,
                          reynolds: float) -> float:
    """
    Chen (1979) explicit approximation to Colebrook-White equation.
    Avoids iterative solution, accuracy within 0.1% for Re > 3000.
    Same approach used in pandapipes gas flow solver.
    """
    if reynolds < 2300:
        return 64.0 / reynolds   # Laminar: Hagen-Poiseuille

    eps_D = roughness_m / diameter_m
    A = (eps_D / 3.7065) - (5.0452 / reynolds) * np.log10(
        (eps_D**1.1098 / 2.8257) + (5.8506 / reynolds**0.8981)
    )
    return (-2.0 * np.log10(A)) ** (-2)


def reynolds_number_h2(mass_flow_kg_s: float,
                        diameter_m: float,
                        pressure_bar: float,
                        temperature_k: float = T_STANDARD_K) -> float:
    """
    Reynolds number for hydrogen flow in a pipe.
    Re = 4 * ṁ / (π * D * μ)
    Dynamic viscosity of H2 at ~15°C ≈ 8.9e-6 Pa·s
    """
    mu_h2 = 8.9e-6
    return 4.0 * mass_flow_kg_s / (np.pi * diameter_m * mu_h2)


# --- Weymouth equation --------------------------------------

def weymouth_outlet_pressure(mass_flow_kg_s: float,
                              inlet_pressure_bar: float,
                              length_km: float,
                              diameter_m: float,
                              roughness_mm: float = 0.046,
                              temperature_k: float = T_STANDARD_K) -> float:
    """
    Weymouth equation for compressible gas pipeline flow.

    Given mass flow and inlet pressure, computes outlet pressure.
    Industry standard for gas transmission pipelines.

    Weymouth (isothermal, compressible):
      P_out² = P_in² - (8 * f * L * Z * R * T * ṁ²) / (π² * D⁵)

    Returns outlet pressure in bar.
    Returns 0.0 if flow physically exceeds pipeline capacity.
    """
    if mass_flow_kg_s <= 0:
        return inlet_pressure_bar

    L_m        = length_km * 1000.0
    roughness_m = roughness_mm / 1000.0
    P_in_pa    = inlet_pressure_bar * 1e5

    Z  = compressibility_factor_h2(inlet_pressure_bar, temperature_k)
    Re = reynolds_number_h2(mass_flow_kg_s, diameter_m, inlet_pressure_bar)
    f  = friction_factor_chen(diameter_m, roughness_m, Re)

    # Weymouth pressure drop (Pa²)
    delta_P2 = (8.0 * f * L_m * Z * R_HYDROGEN * temperature_k
                * mass_flow_kg_s**2) / (np.pi**2 * diameter_m**5)

    P_out_sq = P_in_pa**2 - delta_P2

    if P_out_sq <= 0:
        return 0.0

    return np.sqrt(P_out_sq) / 1e5   # convert Pa → bar


def max_feasible_flow(inlet_pressure_bar: float,
                      min_outlet_pressure_bar: float,
                      length_km: float,
                      diameter_m: float,
                      roughness_mm: float = 0.046,
                      temperature_k: float = T_STANDARD_K) -> float:
    """
    Binary search for maximum flow that keeps outlet above minimum pressure.
    This is the pipeline capacity constraint used in the optimizer.
    """
    low, high = 0.0, 20.0

    for _ in range(50):
        mid   = (low + high) / 2.0
        P_out = weymouth_outlet_pressure(
            mid, inlet_pressure_bar, length_km,
            diameter_m, roughness_mm, temperature_k
        )
        if P_out >= min_outlet_pressure_bar:
            low = mid
        else:
            high = mid

    return low


# --- Core class ---------------------------------------------

class HydrogenPipeline:
    """
    Steady-state hydrogen pipeline model.

    Given H2 mass flow from the electrolyzer at each timestep,
    computes outlet pressure and checks feasibility against
    minimum delivery pressure.

    This is the hydrogen network side of the multi-commodity
    coupling. The feasibility check links back to electricity
    dispatch: if the electrolyzer produces too much H2,
    the pipeline becomes the bottleneck constraint.

    Usage
    -----
    >>> pipe = HydrogenPipeline()
    >>> P_out = pipe.outlet_pressure(mass_flow_kg_s=1.5)
    >>> feasible = pipe.is_feasible(mass_flow_kg_s=1.5)
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = _load_config(config_path)
        p   = cfg["pipeline"]

        self.length_km           = p["length_km"]
        self.diameter_m          = p["diameter_m"]
        self.roughness_mm        = p["roughness_mm"]
        self.inlet_pressure_bar  = p["inlet_pressure_bar"]
        self.min_outlet_pressure = p["min_outlet_pressure_bar"]
        self.max_flow_kg_s       = p["max_flow_kg_per_s"]
        self.temperature_k       = T_STANDARD_K

    @property
    def max_feasible_flow_kg_s(self) -> float:
        """Physics-limited maximum flow (may differ from config max)."""
        return max_feasible_flow(
            self.inlet_pressure_bar,
            self.min_outlet_pressure,
            self.length_km,
            self.diameter_m,
            self.roughness_mm,
            self.temperature_k
        )

    def outlet_pressure(self, mass_flow_kg_s: float) -> float:
        """Outlet pressure for given mass flow (bar)."""
        return weymouth_outlet_pressure(
            mass_flow_kg_s,
            self.inlet_pressure_bar,
            self.length_km,
            self.diameter_m,
            self.roughness_mm,
            self.temperature_k
        )

    def pressure_drop(self, mass_flow_kg_s: float) -> float:
        """Pressure drop across pipeline (bar)."""
        return self.inlet_pressure_bar - self.outlet_pressure(mass_flow_kg_s)

    def is_feasible(self, mass_flow_kg_s: float) -> bool:
        """
        Check if flow satisfies minimum outlet pressure.
        This is the pipeline capacity constraint for the optimizer.
        """
        return self.outlet_pressure(mass_flow_kg_s) >= self.min_outlet_pressure

    def constrained_flow(self, requested_flow_kg_s: float) -> float:
        """
        Return feasible flow given pipeline constraints.
        If requested exceeds capacity, return max feasible.
        Called by the asset layer during coupled simulation.
        """
        if self.is_feasible(requested_flow_kg_s):
            return requested_flow_kg_s
        return self.max_feasible_flow_kg_s

    def simulate_timeseries(self,
                             h2_flow_kg_s: np.ndarray) -> pd.DataFrame:
        """
        Simulate pipeline over a time series of H2 flows
        from the electrolyzer.

        Returns
        -------
        pd.DataFrame with columns:
            - requested_flow_kg_s   : electrolyzer output
            - feasible_flow_kg_s    : after pipeline constraint
            - curtailed_kg_s        : flow that couldn't be delivered
            - outlet_pressure_bar   : pipeline outlet pressure
            - pressure_drop_bar     : pressure loss
            - feasible              : bool above min pressure?
        """
        h2_flow_kg_s = np.asarray(h2_flow_kg_s, dtype=float)
        n = len(h2_flow_kg_s)

        feasible_flow    = np.zeros(n)
        curtailed        = np.zeros(n)
        outlet_pressures = np.zeros(n)
        pressure_drops   = np.zeros(n)
        feasibility      = np.zeros(n, dtype=bool)

        for i, flow in enumerate(h2_flow_kg_s):
            ff               = self.constrained_flow(flow)
            P_out            = self.outlet_pressure(ff)
            feasible_flow[i] = ff
            curtailed[i]     = flow - ff
            outlet_pressures[i] = P_out
            pressure_drops[i]   = self.inlet_pressure_bar - P_out
            feasibility[i]      = self.is_feasible(flow)

        return pd.DataFrame({
            "requested_flow_kg_s" : h2_flow_kg_s,
            "feasible_flow_kg_s"  : feasible_flow,
            "curtailed_kg_s"      : curtailed,
            "outlet_pressure_bar" : outlet_pressures,
            "pressure_drop_bar"   : pressure_drops,
            "feasible"            : feasibility,
        })


# --- Visualisation ------------------------------------------

def plot_pipeline_characteristics(config_path: str = "config.yaml",
                                   save_path: str = None):
    """
    Two-panel: outlet pressure and pressure drop vs mass flow.
    Shows the pipeline capacity constraint visually.
    """
    pipe  = HydrogenPipeline(config_path)
    flows = np.linspace(0, pipe.max_flow_kg_s * 1.5, 200)

    outlets = [pipe.outlet_pressure(q) for q in flows]
    drops   = [pipe.pressure_drop(q)   for q in flows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(flows, outlets, color="#2196F3", linewidth=2.5)
    ax.axhline(pipe.min_outlet_pressure, color="red", linestyle="--",
               label=f"Min delivery: {pipe.min_outlet_pressure} bar")
    ax.axvline(pipe.max_feasible_flow_kg_s, color="orange", linestyle=":",
               label=f"Max feasible: {pipe.max_feasible_flow_kg_s:.2f} kg/s")
    ax.set_xlabel("H₂ mass flow (kg/s)")
    ax.set_ylabel("Outlet pressure (bar)")
    ax.set_title(f"Pipeline Outlet Pressure\n"
                 f"(L={pipe.length_km} km, D={pipe.diameter_m} m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, pipe.inlet_pressure_bar * 1.1)

    ax2 = axes[1]
    ax2.plot(flows, drops, color="#E91E63", linewidth=2.5)
    ax2.set_xlabel("H₂ mass flow (kg/s)")
    ax2.set_ylabel("Pressure drop (bar)")
    ax2.set_title("Pipeline Pressure Drop vs Flow\n(Weymouth equation)")
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
    print("Hydrogen Pipeline — Sanity Check")
    print("=" * 55)

    pipe = HydrogenPipeline()

    print(f"\nPipeline config:")
    print(f"  Length:          {pipe.length_km} km")
    print(f"  Diameter:        {pipe.diameter_m} m")
    print(f"  Inlet pressure:  {pipe.inlet_pressure_bar} bar")
    print(f"  Min outlet:      {pipe.min_outlet_pressure} bar")

    # Test 1: pressure drop at key flow rates
    print("\n[Test 1] Outlet pressure vs mass flow:")
    for q in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        P_out = pipe.outlet_pressure(q)
        dP    = pipe.pressure_drop(q)
        feas  = "OK" if pipe.is_feasible(q) else "INFEASIBLE"
        print(f"  m={q:.1f} kg/s → P_out={P_out:.2f} bar  "
              f"dP={dP:.2f} bar  [{feas}]")

    # Test 2: max feasible flow
    q_max   = pipe.max_feasible_flow_kg_s
    binding = "physics" if q_max < pipe.max_flow_kg_s else "config"
    print(f"\n[Test 2] Max feasible flow: {q_max:.3f} kg/s")
    print(f"  Config limit:       {pipe.max_flow_kg_s:.1f} kg/s")
    print(f"  Binding constraint: {binding}")

    # Test 3: pipeline response to variable electrolyzer output
    print("\n[Test 3] Pipeline response to variable H2 flow:")
    sim_flows = np.array([0.0, 0.5, 1.0, q_max * 0.9,
                           q_max, q_max * 1.1, q_max * 1.5])
    result = pipe.simulate_timeseries(sim_flows)
    print(result.round(3).to_string())

    # Test 4: H2 density and compressibility
    print(f"\n[Test 4] H2 real gas properties:")
    for p_bar in [1, 10, 20, 30]:
        rho = hydrogen_density_kg_m3(p_bar)
        Z   = compressibility_factor_h2(p_bar)
        print(f"  P={p_bar:3d} bar → rho={rho:.3f} kg/m3  Z={Z:.4f}")

    # Test 5: plot
    print("\n[Test 5] Plotting pipeline characteristics...")
    plot_pipeline_characteristics()
    print("Done. Place this file in components/pipeline.py")
