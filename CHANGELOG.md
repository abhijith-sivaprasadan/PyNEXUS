# Changelog

## Unreleased

- Added a real ERA5 ingestion path (`data/era5.py`, `pynexus fetch-era5`): CDS request construction, nearest-grid-cell extraction, 100m-to-hub-height shear correction (power-law/log-law), fail-loud calendar-gap checking, and a committed provenance record schema (`data/provenance.py`). Not yet run against live CDS data in this repository.
- Added a pluggable turbine power curve (`wind_turbine.power_curve_model: "tabulated"`) alongside the existing cubic default, and promoted wake/electrical-loss/availability factors to named, sourced, config-driven parameters.
- Corrected timestep scaling for cumulative demand, ramps and shortfall penalties.
- Added known-optimum/failure tests and explicit optimal-termination gating.
- Added a non-interactive CLI, run manifests and independent CSV verification.

- Added package metadata, CI, community files, and explicit MIT licensing.
- Documented the implemented formulation, provenance boundary, assumptions, and verification status.
- Added deterministic component and dispatch tests plus a tiny configuration.
- Qualified unimplemented live-data and annual-run claims.
