# Assumptions and limitations

- The public default is 168 hourly timesteps; no annual result is claimed.
- Wind and market arrays in demonstrations are synthetic and use fixed seeds.
- The pipeline relation is a screening approximation, not a transient pressure-network design model.
- Electrolyser efficiency is simplified and the optimiser uses a linear hydrogen coefficient.
- No electricity-network power flow, hydrogen storage, startup duration, reserve market, stochastic uncertainty, or project finance model is included.
- Configured live-data source names do not prove that live data were downloaded or consumed. `data/era5.py` now implements and unit-tests a real CDS fetch/load/provenance path (see `docs/data_provenance.md`), but no live fetch has been executed in this repository, so this caveat still applies until a committed provenance record from an actual download exists.
- Hub-height wind speed for real ERA5 data is corrected from the 100m components using a power-law (default, offshore shear exponent 0.11) or log-law (open-sea roughness length 0.0002 m) shear model — both are approximations of the true boundary-layer profile at the specific site and date, not a site-specific measured shear profile.
- The turbine power curve defaults to a cubic approximation between cut-in and rated; a tabulated manufacturer curve can now be substituted via `wind_turbine.power_curve_csv`, but no such table is checked in yet, so the default remains the cubic approximation.
- Wake loss, electrical loss, and availability factors (`wind_turbine.wake_loss_fraction`, `electrical_loss_fraction`, `unavailability_fraction` in `config.yaml`) are named, sourced literature/benchmark values, not site-specific measurements.
- The ERA5 capacity-factor sanity check against published North Sea figures (Phase C4) and the 8,760-hour annual run (Phase C5) have not been done.
