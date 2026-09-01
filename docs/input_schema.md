# Dispatch input boundary

The CLI requires explicit `--synthetic`, `--wind-csv`, or `--input path.csv`,
never an implicit substitution for a unavailable live source. CSV rows must
equal `time_horizon_hours / time_step_hours`, a positive integer.

| Column | Unit | Constraint |
|---|---|---|
| `wind_available_mw` | MW | Finite, non-negative |
| `electricity_price` | EUR/MWh | Finite; negative prices are permitted |
| `carbon_intensity` | kgCO2/MWh | Required for emissions mode; finite, non-negative |
| `heat_demand_mw` | MW-th | Required when `--enable-heat`; finite, non-negative |

Rows are consecutive equal-duration intervals. Timestamp alignment and
external data licensing are the caller's responsibility. `data/era5.py`
implements a real, authenticated ERA5 ingestion path (`pynexus fetch-era5` /
`fetch-era5-year`, then `pynexus solve --wind-csv`) — see
`docs/data_provenance.md`. ENTSO-E (`data/entsoe.py`) is built and
unit-tested but not yet wired into `solve`, same reference. A live-source
adapter must retrieve, cache/version and attribute actual data or fail
clearly — both do.
