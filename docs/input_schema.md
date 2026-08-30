# Dispatch input boundary

The CLI requires either explicit `--synthetic` or `--input path.csv`, never an
implicit substitution for a unavailable live source. CSV rows must equal
`time_horizon_hours / time_step_hours`, a positive integer.

| Column | Unit | Constraint |
|---|---|---|
| `wind_available_mw` | MW | Finite, non-negative |
| `electricity_price` | EUR/MWh | Finite; negative prices are permitted |
| `carbon_intensity` | kgCO2/MWh | Required for emissions mode; finite, non-negative |

Rows are consecutive equal-duration intervals. Timestamp alignment and external
data licensing are the caller's responsibility; there is no authenticated
ENTSO-E/ERA5 ingestion implementation. A live-source adapter must retrieve,
cache/version and attribute actual data or fail clearly.
