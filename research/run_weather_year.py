"""WP4: run registered WP2 physical interventions on consistent ERA5 series, 2021–25."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research.interventions import PROTOCOL as WP2_PROTOCOL
from research.interventions import evaluate_case, source_inputs
from research.run_reference import CONFIG, INPUTS, ROOT, sha256
from research.weather_source import (
    VALID_YEARS,
    convert_weather_to_wind_2023_model,
    fetch_weather,
    frozen_synthetic_signal,
    validate_weather_payload,
)

PROTOCOL = ROOT / "research" / "protocol_wp4.yaml"
CASES = {
    "baseline": (),
    "W": ("W",),
    "E": ("E",),
    "G": ("G",),
    "S": ("S",),
    "eta": ("eta",),
    "E+S": ("E", "S"),
}


def run_year(year, output, *, time_limit=600):
    """Fetch real ERA5-provider hourly values and preserve every study input/output."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("WP4 refuses to replace an existing evidence archive")
    registration = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    wp2 = yaml.safe_load(WP2_PROTOCOL.read_text(encoding="utf-8"))
    if registration["schema_version"] != "pynexus.wp4.weather.v1":
        raise ValueError("Wrong WP4 protocol version")
    if year not in registration["years"] or year not in VALID_YEARS:
        raise ValueError("Year not registered")
    if list(CASES) != registration["registered_scenarios"]:
        raise ValueError("Scenario registration has changed")
    if registration["weather_model"] != "era5":
        raise ValueError("Different atmospheric datasets may not be silently substituted")
    if wp2["order"] != ["W", "E", "G", "S", "eta"]:
        raise ValueError("Unexpected WP2 intervention registration")
    _, base_config, historical_mode = source_inputs(wp2, hours=8760)
    # The annual benchmark is the physical reference, not a weather-year input.
    if sha256(CONFIG) != sha256(ROOT / "outputs" / "annual_2023" / "grid_storage_hourly" / "config.yaml"):
        raise ValueError("Baseline config unexpectedly moved")
    raw, request_url = fetch_weather(year)
    timestamps, speed_100m, meta = validate_weather_payload(raw, year)
    expected_hours = registration["hours_per_year"][str(year)]
    if len(timestamps) != expected_hours:
        raise ValueError("Registered year hour count mismatch")
    if (base_config["location"]["latitude"], base_config["location"]["longitude"]) != (
        registration["latitude"], registration["longitude"]
    ):
        raise ValueError("Site coordinates changed")
    wind_mw = convert_weather_to_wind_2023_model(speed_100m, CONFIG)
    original = pd.read_csv(INPUTS)
    prices = frozen_synthetic_signal(original["electricity_price"].to_numpy(dtype=float), len(wind_mw))
    carbon = frozen_synthetic_signal(original["carbon_intensity"].to_numpy(dtype=float), len(wind_mw))
    frame = pd.DataFrame({
        "wind_available_mw": wind_mw,
        "electricity_price": prices,
        "carbon_intensity": carbon,
    })
    # Use the same equipment and demand parameters, and align the calendar metadata.
    base_config["simulation"]["start_date"] = f"{year}-01-01"
    frame_dates = pd.DataFrame({
        "datetime_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wind_speed_100m_ms": speed_100m,
        **{name: frame[name].to_numpy() for name in frame},
    })
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_weather_api.json").write_bytes(raw)
    frame_dates.to_csv(output / "hourly_weather_and_inputs.csv", index=False)

    comparison = None
    if year == 2023:
        old = original["wind_available_mw"].to_numpy(dtype=float)
        comparison = {
            "reference_kind": "original_CDS_ERA5_2023_input_separate_from_Open_Meteo_API",
            "historical_cds_input_sha256": sha256(INPUTS),
            "historical_cds_input_verified_mode": historical_mode,
            "historical_cds_wind_available_mwh": float(old.sum()),
            "open_meteo_era5_wind_available_mwh": float(wind_mw.sum()),
            "hourly_power_mae_mw": float(np.abs(wind_mw - old).mean()),
            "hourly_power_rmse_mw": float(np.sqrt(np.mean((wind_mw - old) ** 2))),
            "do_not_pool_source_families": True,
        }

    manifest = {
        "schema_version": "pynexus.wp4.year.v1",
        "status": "INCOMPLETE",
        "year": year,
        "hours": len(wind_mw),
        "source": meta,
        "request_url": request_url,
        "original_source_2023_comparison": comparison,
        "hourly_input_csv_sha256": sha256(output / "hourly_weather_and_inputs.csv"),
        "raw_api_response_sha256": sha256(output / "source_weather_api.json"),
        "wp4_protocol_sha256": sha256(PROTOCOL),
        "wp2_protocol_sha256": sha256(WP2_PROTOCOL),
        "base_config_sha256": sha256(CONFIG),
        "code_revision": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
            text=True, check=False,
        ).stdout.strip(),
        "synthetic_price_note": (
            "Archived 2023 SYNTHETIC prices and carbon factors repeated modulo "
            "8760 hours. Stage A is price-independent. Stage B is not a "
            "historical power-market or project-cost assessment."
        ),
        "selected_scenarios": list(CASES),
        "annual_weather_available_mwh": float(wind_mw.sum()),
        "annual_nameplate_capacity_factor": float(
            wind_mw.mean()
            / (base_config["wind_turbine"]["rated_power_mw"]
               * base_config["wind_turbine"]["n_turbines"])
        ),
        "cases": [],
    }
    manifest_path = output / "year_manifest.json"
    for name, active in CASES.items():
        try:
            record = evaluate_case(
                (name, active), base_config, frame, wp2, output / name,
                time_limit=time_limit,
            )
            manifest["cases"].append(record)
            print(
                f"WP4 year={year} case={name} VERIFIED "
                f"shortfall_fraction={record['metrics']['stage_a_shortfall_fraction']:.6f}",
                flush=True,
            )
        except Exception as exc:
            manifest["cases"].append({
                "scenario": name,
                "status": "ERROR_NOT_A_RESULT",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            raise
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    fractions = {
        r["scenario"]: r["metrics"]["stage_a_shortfall_fraction"] for r in manifest["cases"]
    }
    storage_after_e = fractions["E"] - fractions["E+S"]
    interaction = (
        fractions["E"] + fractions["S"] - fractions["baseline"] - fractions["E+S"]
    )
    manifest["storage_conditional_gain_pp"] = 100 * storage_after_e
    manifest["electrolyser_storage_interaction_pp"] = 100 * interaction
    manifest["status"] = "COMPLETE_SEVEN_CASES_YEAR"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"WP4 YEAR {year} COMPLETE 7/7; wind_CF={manifest['annual_nameplate_capacity_factor']:.5f}; "
        f"E+S_interaction_pp={100 * interaction:+.5f}",
        flush=True,
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=int, default=600)
    args = parser.parse_args(argv)
    run_year(args.year, args.output, time_limit=args.time_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
