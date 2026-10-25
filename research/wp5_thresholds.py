"""WP5: auditable, assumption-free marginal adequacy and settlement thresholds.

The 2023 Stage B electricity prices are SYNTHETIC. These are NOT LCOH,
commercial breakeven prices, measured hydrogen revenue, or net benefits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.attribution import analyse_archive
from research.interventions import registered_cases
from research.run_reference import sha256

ORDER = ["W", "E", "G", "S", "eta"]


def _metric(case, key):
    value = float(case["metrics"][key])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite {key} in {case['scenario']}")
    return value


def calculate_thresholds(summary):
    """Use exactly the registered, complete, verified 2023 WP2 result matrix."""
    if (
        summary.get("status") != "COMPLETE_REGISTERED_MATRIX"
        or summary.get("hours") != 8760
        or summary.get("is_full_year") is not True
        or summary.get("price_data_kind") != "synthetic_daily_sinusoid_NOT_historical_market_data"
    ):
        raise ValueError("Only the complete 2023 WP2 synthetic-price matrix is accepted")
    rows = summary.get("cases", [])
    expected = [name for name, _ in registered_cases(ORDER)]
    if [row.get("scenario") for row in rows] != expected:
        raise ValueError("All 17 registered scenarios are required in original order")
    if any(row.get("status") != "VERIFIED_TWO_STAGE_SOLVE" for row in rows):
        raise ValueError("Every case must be independently verified")
    cases = {row["scenario"]: row for row in rows}
    base, s, e, es = (cases[key] for key in ("baseline", "S", "E", "E+S"))
    demand = _metric(e, "demand_h2_kg")
    if demand <= 0 or any(
        not np.isclose(_metric(c, "demand_h2_kg"), demand, atol=1e-5, rtol=0) for c in (base, s, es)
    ):
        raise ValueError("Compared scenarios must have identical positive annual demand")
    for c in (base, s, e, es):
        if _metric(c, "stage_a_shortfall_kg") < 0:
            raise ValueError("Negative shortfall")
        if _metric(c, "stage_b_shortfall_kg") < _metric(c, "stage_a_shortfall_kg") - 1e-4:
            raise ValueError("Stage B shortfall exceeds physical adequacy accuracy")
    extra = _metric(es, "storage_capacity_kg") - _metric(e, "storage_capacity_kg")
    if extra <= 0:
        raise ValueError("Storage expansion was not registered correctly")
    for key in ("wind_nameplate_mw", "electrolyser_capacity_mw", "grid_capacity_mw"):
        if not np.isclose(_metric(e, key), _metric(es, key), atol=1e-8, rtol=0):
            raise ValueError(f"E to E+S changed asset beyond storage: {key}")
    delta_kg = _metric(e, "stage_b_shortfall_kg") - _metric(es, "stage_b_shortfall_kg")
    delta_grid = _metric(es, "stage_b_grid_settlement_eur_synthetic_prices") - _metric(
        e, "stage_b_grid_settlement_eur_synthetic_prices"
    )
    stage_a_improvement_kg = _metric(e, "stage_a_shortfall_kg") - _metric(
        es, "stage_a_shortfall_kg"
    )
    if delta_kg <= 0 or stage_a_improvement_kg <= 0:
        raise ValueError("No extra delivery: ratio undefined and no economic claim allowed")
    if not np.isclose(delta_kg, stage_a_improvement_kg, atol=1.0, rtol=0):
        raise ValueError("Stage A vs Stage B adequacy differences unexpectedly large")
    no_storage_gain_kg = _metric(base, "stage_a_shortfall_kg") - _metric(s, "stage_a_shortfall_kg")
    return {
        "schema_version": "pynexus.wp5.marginal_thresholds.v1",
        "status": "PHYSICAL_DELIVERY_AND_SYNTHETIC_SETTLEMENT_ONLY",
        "scenarios": ["baseline", "S", "E", "E+S"],
        "year": 2023,
        "price_provenance": "SYNTHETIC_NOT_OBSERVED",
        "storage_alone_adequacy_gain_kg": no_storage_gain_kg,
        "incremental_storage_capacity_kg": extra,
        "extra_h2_delivered_E_plus_S_vs_E_kg": delta_kg,
        "stage_a_incremental_h2_kg": stage_a_improvement_kg,
        "adequacy_gain_E_plus_S_vs_E_percentage_points": 100 * stage_a_improvement_kg / demand,
        "incremental_grid_settlement_eur_SYNTHETIC": delta_grid,
        "incremental_grid_settlement_per_extra_kg_eur_SYNTHETIC": delta_grid / delta_kg,
        "symbolic_value_threshold": (
            "additional_h2_net_value_eur_per_kg >= "
            "(incremental_grid_settlement_eur_SYNTHETIC + "
            "annual_incremental_storage_cost_eur + incremental_other_cost_eur) "
            "/ extra_h2_delivered_kg"
        ),
        "unpriced": [
            "annualised storage investment, operating cost and replacement",
            "additional variable operating costs and market network tariffs",
            "hydrogen value, sales contracts, non-delivery penalties and risk",
            "observed wholesale electricity prices and rerun of dispatch",
        ],
        "not_an_lcoh_or_investment_recommendation": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp2-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Never overwrite research evidence outputs")
    analyse_archive(args.wp2_dir)  # Full hourly, solver and hash verification.
    src = args.wp2_dir / "summary.json"
    result = calculate_thresholds(json.loads(src.read_text(encoding="utf-8")))
    result["wp2_summary_sha256"] = sha256(src)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                k: v
                for k, v in result.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        ]
    ).to_csv(args.output / "thresholds.csv", index=False)
    print("WP5 physical threshold reproduced; SYNTHETIC settlement, no sourced costs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
