"""WP1 primary CDS ERA5 2023 wind reconstruction and UTC-hour alignment audit.

Requires original 12 monthly NetCDF files or an authenticated CDS re-request.
Never equates a matching 8760-row count with correct source-hour alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from components.wind_turbine import OffshoreWindFarm
from data.era5 import (
    check_calendar_coverage,
    fetch_era5_year_monthly,
    load_era5_wind,
    month_boundaries,
    wind_speed_at_hub_height_from_100m,
)
from research.interventions import PROTOCOL, source_inputs
from research.run_reference import CONFIG, ROOT, sha256

PROVENANCE = ROOT / "data" / "provenance_records"
EXPECTED_MONTHS = month_boundaries(2023)


def compare_utc_hourly_wind(hourly, archived_wind, *, config_path=CONFIG):
    """Prove timestamp origin and physical conversion equality without a shift."""
    if not isinstance(hourly.index, pd.DatetimeIndex):
        raise ValueError("CDS wind source must have timestamps")
    idx = hourly.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    check_calendar_coverage(idx, "2023-01-01", "2023-12-31", "Europe/Amsterdam")
    expected = pd.date_range("2023-01-01T00:00:00Z", periods=8760, freq="h")
    if not idx.equals(expected):
        raise ValueError("CDS wind timestamps do not match ordered original 2023 UTC calendar")
    base = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    hub = wind_speed_at_hub_height_from_100m(
        hourly["wind_speed_100m_ms"].to_numpy(dtype=float),
        base["wind_turbine"]["hub_height_m"],
    )
    farm = OffshoreWindFarm(str(config_path))
    reconstructed = np.asarray(farm.power_output_mw_from_hub_height(hub), dtype=float)
    original = np.asarray(archived_wind, dtype=float)
    if original.shape != reconstructed.shape or not np.isfinite(original).all():
        raise ValueError("Original archived wind must contain 8760 finite megawatt values")
    difference = np.abs(original - reconstructed)
    if not np.isfinite(difference).all():
        raise ValueError("Nonfinite reconstructed CDS wind")
    if np.any(difference > 1e-6):
        idx_bad = int(np.argmax(difference))
        raise ValueError(
            "Reconstructed CDS raw wind does not match archived 2023 wind; "
            f"first error at UTC hour index {idx_bad}, "
            f"max absolute difference {float(difference.max()):.8g} MW"
        )
    return {
        "hour_count": int(len(expected)),
        "earliest_hour_utc": expected[0].isoformat(),
        "last_hour_utc": expected[-1].isoformat(),
        "max_abs_wind_difference_mw": float(difference.max()),
        "mean_abs_wind_difference_mw": float(difference.mean()),
        "original_wind_mwh": float(original.sum()),
        "reconstructed_wind_mwh": float(reconstructed.sum()),
        "exact_same_source_hour_alignment_verified": True,
        "any_time_shift_or_value_imputation_used": False,
    }


def audit_cds_year(raw_dir, output, *, refetch=False):
    raw_dir, output = Path(raw_dir), Path(output)
    if output.exists():
        raise FileExistsError("Never replace a verified primary-source report")
    base_cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    records = []
    for start, end in EXPECTED_MONTHS:
        path = PROVENANCE / f"era5_{start}_{end}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("source") != "ERA5 reanalysis-era5-single-levels (hourly)"
            or record.get("temporal_range", {}).get("start") != start
            or record.get("temporal_range", {}).get("end") != end
            or record.get("spatial_point", {}).get("grid_cell_center_latitude") != 52.5
            or record.get("spatial_point", {}).get("grid_cell_center_longitude") != 3.5
        ):
            raise ValueError(f"Original CDS provenance incompatible for {start}")
        records.append(record)
    if refetch:
        raw_dir.mkdir(parents=True, exist_ok=True)
        # CDS credentials must already be configured. Never log .cdsapirc,
        # CDSAPI_KEY, private URLs or raw provider exception messages.
        try:
            fetch_era5_year_monthly(
                52.5,
                3.5,
                base_cfg["location"]["era5_grid_resolution"],
                2023,
                cache_dir=raw_dir,
                max_concurrent=2,
            )
        except Exception as exc:
            raise RuntimeError(
                f"CDS recovery failed ({type(exc).__name__}); credentials "
                "and provider response suppressed"
            ) from None
    originals = []
    byte_equal = []
    source_rows = []
    for (start, end), rec in zip(EXPECTED_MONTHS, records):
        candidates = list(raw_dir.glob(f"era5_{start}_{end}_*.nc"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one original CDS monthly NetCDF for {start}..{end}"
            )
        file = candidates[0]
        digest = sha256(file)
        byte_equal.append(digest == rec["file_sha256"])
        frame, spatial = load_era5_wind(file, 52.5, 3.5, "Europe/Amsterdam", start, end)
        if (
            spatial["latitude"] != rec["spatial_point"]["grid_cell_center_latitude"]
            or spatial["longitude"] != rec["spatial_point"]["grid_cell_center_longitude"]
        ):
            raise ValueError("Recovered CDS spatial point differs from original primary record")
        utc = pd.DatetimeIndex(frame.index)
        utc = utc.tz_localize("UTC") if utc.tz is None else utc.tz_convert("UTC")
        frame.index = utc
        if frame.shape[0] != rec["row_count"]["actual"]:
            raise ValueError("Recovered CDS month does not match source row count")
        originals.append(frame)
        source_rows.append(
            {
                "first_utc": utc[0].isoformat(),
                "last_utc": utc[-1].isoformat(),
                "month": start[:7],
                "expected_original_file_sha256": rec["file_sha256"],
                "recovered_file_sha256": digest,
                "byte_identical": byte_equal[-1],
                "hour_count": len(frame),
            }
        )
    data = pd.concat(originals, axis=0)
    wp2 = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    archived, _, _ = source_inputs(wp2, hours=8760)
    physical = compare_utc_hourly_wind(data, archived["wind_available_mw"])
    status = (
        "VERIFIED_ORIGINAL_CDS_BYTES_AND_UTC_WIND_VALUES"
        if all(byte_equal)
        else "VERIFIED_REDELIVERED_CDS_UTC_WIND_VALUES_ORIGINAL_BYTES_DIFFER"
    )
    report = {
        "schema_version": "pynexus.wp1.cds_wind_calendar_and_power.v1",
        "status": status,
        "source": "Copernicus CDS ERA5 single-levels 100m wind vectors",
        "month_count": 12,
        "original_raw_bytes_all_identical": all(byte_equal),
        "source_records": source_rows,
        "reference_configuration_sha256": sha256(CONFIG),
        "frozen_original_input_csv_sha256": sha256(
            ROOT / "outputs/annual_2023/grid_storage_hourly/inputs.csv"
        ),
        **physical,
        "limitations": [
            "Re-delivered NetCDF bytes need not match the original even when hourly wind values agree",
            "A single wind grid cell, generic turbine curve and assumed wake/availability losses are not measurements",
            "One year of matching wind inputs does not establish commercial equipment performance",
        ],
    }
    output.mkdir(parents=True)
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(source_rows).to_csv(output / "monthly_source_hashes.csv", index=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refetch", action="store_true")
    args = parser.parse_args(argv)
    result = audit_cds_year(args.raw_dir, args.output, refetch=args.refetch)
    print(
        json.dumps(
            {
                "status": result["status"],
                "max_abs_wind_difference_mw": result["max_abs_wind_difference_mw"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
