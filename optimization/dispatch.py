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

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory

    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("WARNING: Pyomo not installed. Run: pip install pyomo highspy")

from components.electrolyzer import LHV_HYDROGEN_KWH_PER_KG, efficiency_at_load
from components.pipeline import HydrogenPipeline
from components.wind_turbine import OffshoreWindFarm


# --- Load config --------------------------------------------
def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parent.parent
    full_path = root / config_path
    with open(full_path, "r") as f:
        return yaml.safe_load(f)


# --- H2 output linearisation --------------------------------
def linear_h2_coefficient(
    nominal_load_fraction: float = 0.80, nominal_efficiency: float = 0.70
) -> float:
    """
    kg H2 per MWh of electricity at nominal load.
    Linearises the nonlinear h2(p) = p * eta(p) / LHV function
    by fixing eta at the nominal 80% operating point.
    """
    eta = efficiency_at_load(nominal_load_fraction, nominal_efficiency)
    return (eta * 1000.0) / LHV_HYDROGEN_KWH_PER_KG  # kg/MWh


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
        self.p_rated = e["rated_power_mw"]
        self.p_min_frac = e["min_load_fraction"]
        self.p_min = self.p_rated * self.p_min_frac
        self.p_max = self.p_rated * e["max_load_fraction"]
        self.ramp_per_hour = e.get("ramp_rate_per_hour", 0.40)
        self.max_ramp_mw = self.ramp_per_hour * self.p_rated
        self.nominal_eta = e["nominal_efficiency"]

        self.pipeline = HydrogenPipeline(config_path)
        self.max_h2_kg_s = self.pipeline.max_feasible_flow_kg_s

        d = cfg["hydrogen_demand"]
        self.hourly_demand_kg = d["daily_average_kg"] / 24.0

        self.h2_coeff = linear_h2_coefficient(0.80, self.nominal_eta)

        s = cfg["simulation"]
        self.dt = s["time_step_hours"]
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("time_step_hours must be finite and positive")
        self.max_ramp_mw *= self.dt

        o = cfg["optimization"]
        self.solver_name = o.get("solver", "highs")
        self.time_limit = o.get("time_limit_seconds", 300)
        self.mip_gap = o.get("mip_gap", 0.01)
        self.threads = o.get("threads", 1)
        self.random_seed = o.get("random_seed", 0)
        if self.solver_name not in {"highs", "appsi_highs"}:
            raise ValueError("Only the HiGHS solver is supported")

        self.DEMAND_PENALTY = 1000.0  # EUR per kg unmet H2

        # --- Hydrogen storage (Phase A1, opt-in via optimize(enable_storage=True)) ---
        hs = cfg.get("hydrogen_storage", {})
        self.storage_capacity_kg = hs.get("capacity_kg", 0.0)
        self.storage_max_charge_kg_h = hs.get("max_charge_rate_kg_h", 0.0)
        self.storage_max_discharge_kg_h = hs.get("max_discharge_rate_kg_h", 0.0)
        self.storage_initial_kg = hs.get("initial_level_kg", 0.0)
        self.storage_final_min_kg = hs.get("final_level_min_kg", self.storage_initial_kg)
        # Fractional loss per hour of stored inventory (boil-off/leakage). Compressed
        # gaseous H2 storage: near-zero: liquid H2 cryogenic boil-off: ~0.1-1%/day
        # (~0.004-0.04%/h). Default assumes compressed gas (this repo's pipeline
        # outlet pressure is 30 bar, gaseous, not cryogenic) — see docs/assumptions.md.
        self.storage_loss_fraction_per_hour = hs.get("loss_fraction_per_hour", 0.0)

        # --- Grid import/export (Phase A2, opt-in via optimize(enable_grid=True)) ---
        g = cfg.get("grid", {})
        self.grid_connection_mw = g.get("connection_capacity_mw", 0.0)
        # Exported power displaces grid generation, which could be credited as
        # avoided emissions in the emissions objective. This model does NOT
        # credit it (see docs/formulation.md): grid displacement factors are
        # controversial/context-dependent (marginal vs. average generation mix)
        # and crediting them without a sourced marginal-emissions factor would
        # be exactly the kind of unsourced number this repo's honesty rules
        # forbid. Only import emissions are counted.
        self.credit_export_emissions = False

        # --- Heat coupling (Phase B1-B3, opt-in via optimize(enable_heat=True)) ---
        # Fraction of the electrolyser's rejected energy (p*(1-eta)) that is
        # actually recoverable at useful temperature. PEM stack coolant is
        # typically 50-80 C, which suits district heating or low-temperature
        # industrial demand, NOT high-grade process heat — this is a stated
        # assumption/range, not a sourced figure for a specific stack design.
        self.recoverable_heat_fraction = e.get("recoverable_heat_fraction", 0.5)
        # The linearised electrolyser efficiency implied by h2_coeff, i.e. the
        # SAME efficiency the MILP's hydrogen output already assumes (nominal
        # 80% load point) — waste heat must use this, not the full nonlinear
        # efficiency_at_load() curve from components/electrolyzer.py, or the
        # energy balance p = h2_power + waste_heat would not close within the
        # optimizer's own (already-linearised) hydrogen accounting.
        self.eta_linearized = self.h2_coeff * LHV_HYDROGEN_KWH_PER_KG / 1000.0

        hst = cfg.get("heat_storage", {})
        self.heat_storage_capacity_mwh = hst.get("capacity_mwh_th", 0.0)
        self.heat_storage_max_charge_mw = hst.get("max_charge_rate_mw_th", 0.0)
        self.heat_storage_max_discharge_mw = hst.get("max_discharge_rate_mw_th", 0.0)
        self.heat_storage_initial_mwh = hst.get("initial_level_mwh_th", 0.0)
        self.heat_storage_final_min_mwh = hst.get("final_level_min_mwh_th", self.heat_storage_initial_mwh)
        # Standing thermal losses (insulated hot-water/thermal store): a
        # stated illustrative range, not a vessel-specific measurement.
        self.heat_storage_loss_fraction_per_hour = hst.get("loss_fraction_per_hour", 0.0)

        b = cfg.get("boiler", {})
        self.boiler_max_output_mw = b.get("max_output_mw_th", 0.0)
        self.boiler_fuel_cost_eur_per_mwh = b.get("fuel_cost_eur_per_mwh_th", 0.0)
        self.boiler_emission_factor_kg_co2_per_mwh = b.get("emission_factor_kg_co2_per_mwh_th", 0.0)

        econ = cfg.get("economics", {})
        # Both illustrative EUR values, not sourced offtake contracts — see
        # docs/assumptions.md. Only used when enable_heat=True, which turns
        # the cost objective from "minimise cost of meeting fixed demand"
        # into "minimise net cost after valuing both delivered outputs" —
        # documented explicitly in docs/formulation.md since this is a
        # deliberate, consequential modelling choice, not a detail.
        self.hydrogen_value_per_kg = econ.get("hydrogen_value_per_kg", 0.0)
        self.heat_value_per_mwh = econ.get("heat_value_per_mwh", 0.0)
        self.HEAT_DEMAND_PENALTY = 1000.0  # EUR per MWh-th unmet, mirrors DEMAND_PENALTY

    def optimize(
        self,
        wind_power_mw: np.ndarray,
        electricity_prices: np.ndarray,
        objective: str = "minimize_cost",
        demand_mode: str = "cumulative",
        carbon_intensity: np.ndarray = None,
        enable_storage: bool = False,
        enable_grid: bool = False,
        enable_heat: bool = False,
        heat_demand_mw: np.ndarray = None,
    ) -> dict:
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
        enable_storage : bool
            Add hydrogen storage (s/h_in/h_out, terminal condition). Default
            False preserves the exact pre-Phase-A model and its locked
            reference objective values (docs/reproducibility.md). Only
            meaningful in combination with demand_mode="hourly" — see
            docs/formulation.md.
        enable_grid : bool
            Add grid import/export against `grid.connection_capacity_mw`.
            Default False preserves the exact pre-Phase-A model.
        enable_heat : bool
            Add electrolyser waste-heat recovery, heat storage, backup
            boiler, and heat demand, and switch the cost objective to value
            hydrogen and heat delivered rather than only penalise shortfall
            — see docs/formulation.md's Phase B section; this is a real
            change in what the objective represents, not a minor addition.
            Requires `heat_demand_mw`. Only affects `objective="minimize_cost"`
            (the emissions objective is unchanged when heat is enabled,
            other than counting boiler emissions).
        heat_demand_mw : np.ndarray, optional
            Required when enable_heat=True: thermal demand (MW-th) at each
            timestep, same length as wind_power_mw.

        Returns
        -------
        dict with solution data and results DataFrame
        """
        if not PYOMO_AVAILABLE:
            raise RuntimeError("Pyomo not installed. pip install pyomo highspy")

        def vector(values, name):
            array = np.asarray(values, dtype=float)
            if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a non-empty finite 1-D array")
            return array

        wind_power_mw = vector(wind_power_mw, "wind_power_mw")
        electricity_prices = vector(electricity_prices, "electricity_prices")
        T = len(wind_power_mw)
        if len(electricity_prices) != T:
            raise ValueError("Arrays must match length")
        if (wind_power_mw < 0).any():
            raise ValueError("wind_power_mw must be non-negative")
        if demand_mode not in {"hourly", "cumulative"}:
            raise ValueError(f"Unknown demand_mode: {demand_mode}")
        if objective == "minimize_emissions" and carbon_intensity is None:
            raise ValueError("carbon_intensity required")
        if carbon_intensity is not None:
            carbon_intensity = vector(carbon_intensity, "carbon_intensity")
            if len(carbon_intensity) != T or (carbon_intensity < 0).any():
                raise ValueError("carbon_intensity must match horizon and be non-negative")
        if enable_heat:
            if heat_demand_mw is None:
                raise ValueError("enable_heat requires heat_demand_mw")
            heat_demand_mw = vector(heat_demand_mw, "heat_demand_mw")
            if len(heat_demand_mw) != T or (heat_demand_mw < 0).any():
                raise ValueError("heat_demand_mw must match horizon and be non-negative")

        model = pyo.ConcreteModel(name="ElectrolyzerDispatch")
        model.T = pyo.RangeSet(0, T - 1)

        # Parameters
        model.wind = pyo.Param(model.T, initialize={t: float(wind_power_mw[t]) for t in range(T)})
        model.price = pyo.Param(
            model.T, initialize={t: float(electricity_prices[t]) for t in range(T)}
        )
        model.demand_kg = pyo.Param(initialize=self.hourly_demand_kg)

        # Decision variables
        model.p = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.p_max))
        model.u = pyo.Var(model.T, domain=pyo.Binary)

        # demand_slack: shortfall variable
        # Named 'demand_slack' NOT 'slack' — avoid Pyomo 6.10.0 appsi bug
        model.demand_slack = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        if enable_grid:
            if self.grid_connection_mw <= 0:
                raise ValueError("enable_grid requires grid.connection_capacity_mw > 0 in config")
            model.g_imp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.grid_connection_mw))
            model.g_exp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.grid_connection_mw))
            model.curtail = pyo.Var(model.T, domain=pyo.NonNegativeReals)

            def grid_balance(model, t):
                # sources = sinks: wind + import = electrolyser load + export + curtailment.
                # Replaces the plain wind_limit constraint used when grid is disabled.
                return model.wind[t] + model.g_imp[t] == model.p[t] + model.g_exp[t] + model.curtail[t]

            model.c_grid_balance = pyo.Constraint(model.T, rule=grid_balance)
        else:

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
            return model.p[t] - model.p[t - 1] <= self.max_ramp_mw

        model.c_ramp_up = pyo.Constraint(model.T, rule=ramp_up)

        def ramp_down(model, t):
            if t == 0:
                return pyo.Constraint.Skip
            return model.p[t - 1] - model.p[t] <= self.max_ramp_mw

        model.c_ramp_down = pyo.Constraint(model.T, rule=ramp_down)

        max_h2_per_hour = self.max_h2_kg_s * 3600.0

        def pipeline_cap(model, t):
            return model.p[t] * self.h2_coeff <= max_h2_per_hour

        model.c_pipeline = pyo.Constraint(model.T, rule=pipeline_cap)

        if enable_storage:
            if self.storage_capacity_kg <= 0:
                raise ValueError("enable_storage requires hydrogen_storage.capacity_kg > 0 in config")
            model.s = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, self.storage_capacity_kg))
            model.h_in = pyo.Var(
                model.T, domain=pyo.NonNegativeReals, bounds=(0, self.storage_max_charge_kg_h)
            )
            model.h_out = pyo.Var(
                model.T, domain=pyo.NonNegativeReals, bounds=(0, self.storage_max_discharge_kg_h)
            )

            def storage_charge_limit(model, t):
                # Can't divert more hydrogen to storage than was actually produced.
                return model.h_in[t] <= model.p[t] * self.h2_coeff

            model.c_storage_charge_limit = pyo.Constraint(model.T, rule=storage_charge_limit)

            def storage_balance(model, t):
                previous = self.storage_initial_kg if t == 0 else model.s[t - 1]
                loss = self.storage_loss_fraction_per_hour * previous * self.dt
                return model.s[t] == previous + (model.h_in[t] - model.h_out[t]) * self.dt - loss

            model.c_storage_balance = pyo.Constraint(model.T, rule=storage_balance)

            # Terminal condition: without this the optimiser empties the store on
            # the last timestep (free energy with no penalty for leaving it
            # depleted), which is a classic storage-model artefact.
            def storage_terminal(model):
                return model.s[T - 1] >= self.storage_final_min_kg

            model.c_storage_terminal = pyo.Constraint(rule=storage_terminal)

        if enable_heat:
            if self.boiler_max_output_mw <= 0:
                raise ValueError("enable_heat requires boiler.max_output_mw_th > 0 in config")
            model.heat_demand = pyo.Param(
                model.T, initialize={t: float(heat_demand_mw[t]) for t in range(T)}
            )
            # Waste heat is a deterministic function of p[t], not a free variable:
            # everything the electrolyser doesn't convert to hydrogen (at the
            # SAME linearised efficiency the MILP's hydrogen output already
            # assumes) is rejected as heat, of which only a stated fraction is
            # recoverable at useful temperature.
            model.q_wh = pyo.Expression(
                model.T,
                rule=lambda model, t: model.p[t]
                * (1 - self.eta_linearized)
                * self.recoverable_heat_fraction,
            )
            model.q_boiler = pyo.Var(
                model.T, domain=pyo.NonNegativeReals, bounds=(0, self.boiler_max_output_mw)
            )
            model.q_dump = pyo.Var(model.T, domain=pyo.NonNegativeReals)
            model.heat_slack = pyo.Var(model.T, domain=pyo.NonNegativeReals)

            if self.heat_storage_capacity_mwh > 0:
                model.e_hs = pyo.Var(
                    model.T, domain=pyo.NonNegativeReals, bounds=(0, self.heat_storage_capacity_mwh)
                )
                model.q_hs_in = pyo.Var(
                    model.T, domain=pyo.NonNegativeReals, bounds=(0, self.heat_storage_max_charge_mw)
                )
                model.q_hs_out = pyo.Var(
                    model.T,
                    domain=pyo.NonNegativeReals,
                    bounds=(0, self.heat_storage_max_discharge_mw),
                )

                def heat_storage_balance(model, t):
                    previous = self.heat_storage_initial_mwh if t == 0 else model.e_hs[t - 1]
                    loss = self.heat_storage_loss_fraction_per_hour * previous * self.dt
                    return (
                        model.e_hs[t]
                        == previous + (model.q_hs_in[t] - model.q_hs_out[t]) * self.dt - loss
                    )

                model.c_heat_storage_balance = pyo.Constraint(model.T, rule=heat_storage_balance)

                def heat_storage_terminal(model):
                    return model.e_hs[T - 1] >= self.heat_storage_final_min_mwh

                model.c_heat_storage_terminal = pyo.Constraint(rule=heat_storage_terminal)

            else:
                # No heat storage configured: charge/discharge are fixed at zero
                # rather than omitted, so the balance below can reference them
                # unconditionally regardless of whether storage is sized.
                model.q_hs_in = pyo.Param(model.T, initialize=0.0)
                model.q_hs_out = pyo.Param(model.T, initialize=0.0)

            def heat_charge_limit(model, t):
                # Can't divert more heat to storage+dump than was actually
                # recovered — mirrors c_storage_charge_limit for hydrogen.
                return model.q_hs_in[t] + model.q_dump[t] <= model.q_wh[t]

            model.c_heat_charge_limit = pyo.Constraint(model.T, rule=heat_charge_limit)

            def heat_balance(model, t):
                # Recovered heat, net of what's diverted to storage or dumped,
                # plus storage withdrawal and boiler backup, must cover demand
                # (soft: heat_slack absorbs any shortfall) — same pattern as
                # the hydrogen demand balance with storage enabled.
                delivered = model.q_wh[t] - model.q_hs_in[t] - model.q_dump[t]
                return (
                    delivered + model.q_hs_out[t] + model.q_boiler[t] + model.heat_slack[t]
                    >= model.heat_demand[t]
                )

            model.c_heat_balance = pyo.Constraint(model.T, rule=heat_balance)

        if demand_mode == "hourly":

            def h2_demand_hourly(model, t):
                produced = model.p[t] * self.h2_coeff
                if enable_storage:
                    produced = produced - model.h_in[t] + model.h_out[t]
                return produced + model.demand_slack[t] >= model.demand_kg

            model.c_demand = pyo.Constraint(model.T, rule=h2_demand_hourly)
        elif demand_mode == "cumulative":
            total_demand = self.hourly_demand_kg * T * self.dt

            def h2_demand_cumul(model):
                return sum(model.p[t] * self.h2_coeff * self.dt for t in model.T) >= total_demand

            model.c_demand = pyo.Constraint(rule=h2_demand_cumul)

        # Objective
        if objective == "minimize_cost":

            def cost_obj(model):
                terms = sum(model.price[t] * model.p[t] * self.dt for t in model.T) + sum(
                    self.DEMAND_PENALTY * model.demand_slack[t] * self.dt for t in model.T
                )
                if enable_grid:
                    terms += sum(model.price[t] * model.g_imp[t] * self.dt for t in model.T)
                    terms -= sum(model.price[t] * model.g_exp[t] * self.dt for t in model.T)
                if enable_heat:
                    # Coupled objective (Phase B3): boiler fuel cost and
                    # heat-shortfall penalty are added; hydrogen and heat
                    # VALUE are subtracted, turning this from "minimise cost
                    # of meeting a fixed demand" into "minimise net cost
                    # after valuing both delivered outputs". See
                    # docs/formulation.md for why this weighting choice is
                    # made explicit rather than left implicit.
                    terms += sum(
                        self.boiler_fuel_cost_eur_per_mwh * model.q_boiler[t] * self.dt
                        for t in model.T
                    )
                    terms += sum(
                        self.HEAT_DEMAND_PENALTY * model.heat_slack[t] * self.dt for t in model.T
                    )
                    h2_delivered = sum(model.p[t] * self.h2_coeff * self.dt for t in model.T)
                    if enable_storage:
                        h2_delivered = sum(
                            (model.p[t] * self.h2_coeff - model.h_in[t] + model.h_out[t]) * self.dt
                            for t in model.T
                        )
                    terms -= self.hydrogen_value_per_kg * h2_delivered
                    heat_delivered = sum(
                        (model.q_wh[t] - model.q_hs_in[t] - model.q_dump[t] + model.q_hs_out[t])
                        * self.dt
                        for t in model.T
                    )
                    terms -= self.heat_value_per_mwh * heat_delivered
                return terms

            model.objective = pyo.Objective(rule=cost_obj, sense=pyo.minimize)

        elif objective == "minimize_emissions":
            if carbon_intensity is None:
                raise ValueError("carbon_intensity required")
            model.carbon = pyo.Param(
                model.T, initialize={t: float(carbon_intensity[t]) for t in range(T)}
            )

            def emissions_obj(model):
                terms = sum(model.carbon[t] * model.p[t] * self.dt for t in model.T) + sum(
                    self.DEMAND_PENALTY * model.demand_slack[t] * self.dt for t in model.T
                )
                if enable_grid:
                    # Import carries grid carbon intensity. Export is NOT credited
                    # as avoided emissions — see self.credit_export_emissions and
                    # docs/formulation.md for why.
                    terms += sum(model.carbon[t] * model.g_imp[t] * self.dt for t in model.T)
                if enable_heat:
                    terms += sum(
                        self.boiler_emission_factor_kg_co2_per_mwh * model.q_boiler[t] * self.dt
                        for t in model.T
                    )
                    terms += sum(
                        self.HEAT_DEMAND_PENALTY * model.heat_slack[t] * self.dt for t in model.T
                    )
                return terms

            model.objective = pyo.Objective(rule=emissions_obj, sense=pyo.minimize)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Solve
        solver = SolverFactory("appsi_highs")
        solver.options["time_limit"] = self.time_limit
        solver.options["mip_rel_gap"] = self.mip_gap
        solver.options["threads"] = self.threads
        solver.options["random_seed"] = self.random_seed
        started = time.perf_counter()
        sol = solver.solve(model, tee=False, load_solutions=False)
        status = str(sol.solver.termination_condition)
        metadata = {
            "solver_status": str(sol.solver.status),
            "termination_condition": status,
            "solve_wall_time_s": time.perf_counter() - started,
            "variables": sum(1 for _ in model.component_data_objects(pyo.Var)),
            "binary_variables": sum(v.is_binary() for v in model.component_data_objects(pyo.Var)),
            "constraints": sum(
                1 for _ in model.component_data_objects(pyo.Constraint, active=True)
            ),
        }

        # Exact enum matching: 'infeasible' contains 'feasible' as a substring.
        # Time-limit/unknown runs are not published as verified dispatch results.
        if sol.solver.termination_condition != pyo.TerminationCondition.optimal:
            return {
                **metadata,
                "status": status,
                "objective_value": None,
                "power_schedule": None,
                "results_df": None,
            }
        model.solutions.load_from(sol)

        # Extract solution
        power_schedule = np.array([pyo.value(model.p[t]) for t in range(T)])
        online_status = np.array([pyo.value(model.u[t]) for t in range(T)])
        slack_values = np.array([pyo.value(model.demand_slack[t]) for t in range(T)])

        h2_produced_kg_h = power_schedule * self.h2_coeff
        cost_profile = electricity_prices * power_schedule * self.dt
        demand_met_bool = h2_produced_kg_h >= self.hourly_demand_kg - 1e-3

        if enable_grid:
            curtailment_mw = np.array([pyo.value(model.curtail[t]) for t in range(T)])
            grid_import_mw = np.array([pyo.value(model.g_imp[t]) for t in range(T)])
            grid_export_mw = np.array([pyo.value(model.g_exp[t]) for t in range(T)])
            cost_profile = cost_profile + electricity_prices * grid_import_mw * self.dt
            cost_profile = cost_profile - electricity_prices * grid_export_mw * self.dt
        else:
            curtailment_mw = wind_power_mw - power_schedule
            grid_import_mw = np.zeros(T)
            grid_export_mw = np.zeros(T)

        results_df = pd.DataFrame(
            {
                "timestep": np.arange(T),
                "wind_available_mw": wind_power_mw,
                "curtailment_mw": curtailment_mw,
                "power_optimized_mw": power_schedule,
                "online_status": online_status.astype(int),
                "electricity_price": electricity_prices,
                "h2_produced_kg_h": h2_produced_kg_h,
                "h2_demand_kg_h": self.hourly_demand_kg,
                "demand_slack_kg_h": slack_values,
                "demand_met": demand_met_bool,
                "cost_eur": cost_profile,
                "grid_import_mw": grid_import_mw,
                "grid_export_mw": grid_export_mw,
            }
        )
        if enable_storage:
            results_df["storage_level_kg"] = [pyo.value(model.s[t]) for t in range(T)]
            results_df["storage_charge_kg_h"] = [pyo.value(model.h_in[t]) for t in range(T)]
            results_df["storage_discharge_kg_h"] = [pyo.value(model.h_out[t]) for t in range(T)]

        if enable_heat:
            results_df["heat_demand_mw"] = heat_demand_mw
            results_df["waste_heat_recovered_mw"] = [pyo.value(model.q_wh[t]) for t in range(T)]
            results_df["boiler_output_mw"] = [pyo.value(model.q_boiler[t]) for t in range(T)]
            results_df["heat_dumped_mw"] = [pyo.value(model.q_dump[t]) for t in range(T)]
            results_df["heat_slack_mw"] = [pyo.value(model.heat_slack[t]) for t in range(T)]
            if self.heat_storage_capacity_mwh > 0:
                results_df["heat_storage_level_mwh"] = [pyo.value(model.e_hs[t]) for t in range(T)]
                results_df["heat_storage_charge_mw"] = [pyo.value(model.q_hs_in[t]) for t in range(T)]
                results_df["heat_storage_discharge_mw"] = [
                    pyo.value(model.q_hs_out[t]) for t in range(T)
                ]

        return {
            **metadata,
            "status": status,
            "objective_value": pyo.value(model.objective),
            "power_schedule": power_schedule,
            "online_status": online_status,
            "h2_produced_kg_h": h2_produced_kg_h,
            "cost_profile": cost_profile,
            "slack_values": slack_values,
            "results_df": results_df,
            "enable_storage": enable_storage,
            "enable_grid": enable_grid,
            "enable_heat": enable_heat,
        }

    def print_solution_summary(self, result: dict):
        """Print formatted optimization result summary."""
        if result["results_df"] is None:
            print(f"Optimization failed: {result['status']}")
            return

        df = result["results_df"]
        total_slack = df["demand_slack_kg_h"].sum() * self.dt

        print("\n" + "=" * 55)
        print("OPTIMIZATION RESULT SUMMARY")
        print("=" * 55)
        print(f"  Solver status:           {result['status']}")
        print(f"  Objective value:         {result['objective_value']:.2f}")
        print(f"  Hours optimized:         {len(df) * self.dt:g}")
        print(f"  Avg power dispatch:      {df['power_optimized_mw'].mean():.1f} MW")
        print(f"  Electrolyzer online:     {df['online_status'].sum()}/{len(df)} intervals")
        print(
            f"  Total H2 produced:       {df['h2_produced_kg_h'].sum() * self.dt / 1000:.2f} tonnes"
        )
        print(
            f"  Total H2 demand:         {df['h2_demand_kg_h'].sum() * self.dt / 1000:.2f} tonnes"
        )
        print(f"  Unmet demand (slack):    {total_slack:.0f} kg total")
        print(f"  Intervals demand met:    {df['demand_met'].sum()}/{len(df)}")
        print(f"  Total electricity cost:  EUR {df['cost_eur'].sum():.0f}")
        print(f"  Avg electricity price:   EUR {df['electricity_price'].mean():.1f}/MWh")
        print("=" * 55)

    def compare_objectives(
        self,
        wind_power_mw: np.ndarray,
        electricity_prices: np.ndarray,
        carbon_intensity: np.ndarray,
    ) -> tuple:
        """
        Run cost and emissions objectives and compare side by side.
        Demonstrates that cheapest != lowest-carbon dispatch.
        """
        print("  Running cost minimization...")
        r_cost = self.optimize(
            wind_power_mw, electricity_prices, objective="minimize_cost", demand_mode="cumulative"
        )

        print("  Running emissions minimization...")
        r_emis = self.optimize(
            wind_power_mw,
            electricity_prices,
            objective="minimize_emissions",
            demand_mode="cumulative",
            carbon_intensity=carbon_intensity,
        )

        if r_cost["results_df"] is None or r_emis["results_df"] is None:
            print("One or both optimizations failed.")
            return None, r_cost, r_emis

        df_c = r_cost["results_df"]
        df_e = r_emis["results_df"]

        emis_of_cost_opt = (df_c["power_optimized_mw"] * carbon_intensity * self.dt).sum()
        emis_of_emis_opt = (df_e["power_optimized_mw"] * carbon_intensity * self.dt).sum()
        cost_of_emis_opt = (df_e["power_optimized_mw"] * electricity_prices * self.dt).sum()
        cost_of_cost_opt = df_c["cost_eur"].sum()

        comparison = pd.DataFrame(
            {
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
                    f"{df_c['h2_produced_kg_h'].sum() * self.dt:.0f}",
                    f"{df_c['power_optimized_mw'].mean():.1f}",
                    f"{(df_c['power_optimized_mw'] >= self.p_max * 0.99).sum() * self.dt:g}",
                    f"{(df_c['online_status'] == 0).sum() * self.dt:g}",
                ],
                "Emissions-optimal": [
                    f"{cost_of_emis_opt:.0f}",
                    f"{emis_of_emis_opt:.0f}",
                    f"{df_e['h2_produced_kg_h'].sum() * self.dt:.0f}",
                    f"{df_e['power_optimized_mw'].mean():.1f}",
                    f"{(df_e['power_optimized_mw'] >= self.p_max * 0.99).sum() * self.dt:g}",
                    f"{(df_e['online_status'] == 0).sum() * self.dt:g}",
                ],
            }
        )
        return comparison, r_cost, r_emis


# --- Visualisation ------------------------------------------


def plot_optimization_result(
    result: dict, title: str = "Optimized Dispatch", save_path: str = None
):
    """Three-panel: dispatch, H2 vs demand, price vs dispatch."""
    if result["results_df"] is None:
        print("No results to plot.")
        return

    df = result["results_df"]
    hours = df["timestep"].values
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax = axes[0]
    ax.fill_between(
        hours, df["wind_available_mw"], alpha=0.2, color="#2196F3", label="Wind available"
    )
    ax.step(
        hours,
        df["power_optimized_mw"],
        color="#2196F3",
        linewidth=2.0,
        where="post",
        label="Optimized dispatch",
    )
    ax.set_ylabel("Power (MW)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    demand_val = df["h2_demand_kg_h"].iloc[0]
    ax2.step(
        hours,
        df["h2_produced_kg_h"],
        color="#4CAF50",
        linewidth=2.0,
        where="post",
        label="H2 produced (kg/h)",
    )
    ax2.axhline(demand_val, color="orange", linestyle="--", linewidth=1.5, label="Hourly demand")
    ax2.fill_between(
        hours,
        df["h2_produced_kg_h"],
        demand_val,
        where=df["h2_produced_kg_h"] >= demand_val,
        alpha=0.2,
        color="green",
        label="Surplus",
    )
    ax2.fill_between(
        hours,
        df["h2_produced_kg_h"],
        demand_val,
        where=df["h2_produced_kg_h"] < demand_val,
        alpha=0.2,
        color="red",
        label="Deficit",
    )
    ax2.set_ylabel("H2 (kg/hour)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3c = ax3.twinx()
    ax3.bar(hours, df["electricity_price"], alpha=0.4, color="#FF9800", label="Electricity price")
    ax3c.step(
        hours, df["power_optimized_mw"], color="#2196F3", linewidth=1.5, where="post", alpha=0.8
    )
    ax3.set_xlabel("Interval index")
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


def plot_objective_comparison(
    r_cost: dict, r_emis: dict, carbon_intensity: np.ndarray, save_path: str = None
):
    """Side-by-side cost-optimal vs emissions-optimal dispatch."""
    if r_cost["results_df"] is None or r_emis["results_df"] is None:
        return

    df_c = r_cost["results_df"]
    df_e = r_emis["results_df"]
    hours = df_c["timestep"].values
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col, (df, label, color) in enumerate(
        [(df_c, "Cost-optimal", "#2196F3"), (df_e, "Emissions-optimal", "#4CAF50")]
    ):
        ax = axes[0, col]
        ax.fill_between(hours, df["wind_available_mw"], alpha=0.15, color=color)
        ax.step(hours, df["power_optimized_mw"], color=color, linewidth=2.0, where="post")
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

    plt.suptitle("PyNEXUS — Cost vs Emissions Objective Comparison", fontsize=12, fontweight="bold")
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

    print("\nOptimizer parameters:")
    print(f"  Electrolyzer rated:    {opt.p_rated} MW")
    print(f"  Min load:              {opt.p_min} MW")
    print(f"  Max ramp/interval:     {opt.max_ramp_mw} MW per interval")
    print(f"  H2 coefficient:        {opt.h2_coeff:.2f} kg/MWh")
    print(f"  Hourly H2 demand:      {opt.hourly_demand_kg:.0f} kg/h")
    print(f"  Demand penalty:        EUR {opt.DEMAND_PENALTY}/kg unmet")

    np.random.seed(42)
    T = 48

    wind_raw = np.clip(
        10 + 5 * np.sin(np.linspace(0, 4 * np.pi, T)) + np.random.normal(0, 2, T), 0, 30
    )
    wind_48h = OffshoreWindFarm().power_output_mw(wind_raw)

    price_48h = np.clip(
        60
        + 30 * np.sin(np.linspace(-np.pi / 2, 4 * np.pi - np.pi / 2, T))
        + np.random.normal(0, 8, T),
        5,
        200,
    )

    ci_48h = np.clip(
        200 + 100 * np.sin(np.linspace(0, 4 * np.pi, T)) + np.random.normal(0, 20, T), 50, 400
    )

    print("\n[Test 1] Cost minimization — hourly demand (soft constraint):")
    r1 = opt.optimize(wind_48h, price_48h, objective="minimize_cost", demand_mode="hourly")
    opt.print_solution_summary(r1)

    print("\n[Test 2] Cost minimization — cumulative demand:")
    r2 = opt.optimize(wind_48h, price_48h, objective="minimize_cost", demand_mode="cumulative")
    opt.print_solution_summary(r2)

    if r1["objective_value"] and r2["objective_value"]:
        saving = r1["results_df"]["cost_eur"].sum() - r2["results_df"]["cost_eur"].sum()
        print(f"\n  Cost saving (cumulative vs hourly): EUR {saving:.0f}")
        print("  (Temporal flexibility has monetary value)")

    print("\n[Test 3] Cost vs emissions objective comparison:")
    comparison, r_cost, r_emis = opt.compare_objectives(wind_48h, price_48h, ci_48h)
    if comparison is not None:
        print(comparison.to_string(index=False))

    print("\n[Test 4] Plotting cost-optimal dispatch (48h)...")
    plot_optimization_result(
        r2, title="PyNEXUS — MILP Cost-Optimal Dispatch (48h, cumulative demand)"
    )

    if r_cost["results_df"] is not None:
        print("[Test 5] Plotting cost vs emissions comparison...")
        plot_objective_comparison(r_cost, r_emis, ci_48h)

    print("\nDone. Place this file in optimization/dispatch.py")
