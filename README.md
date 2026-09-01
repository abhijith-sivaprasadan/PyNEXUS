# PyNEXUS

A reproducible Pyomo/HiGHS model for hourly dispatch of an offshore-wind, PEM-electrolyser, and hydrogen-pipeline system under physical operating constraints and alternative cost and emissions objectives.

## Scope

The checked-in reference configuration covers 168 hourly timesteps. No 8,760-hour result is claimed. The model includes wind availability, electrolyser minimum/maximum load and ramping, a simplified hydrogen-pipeline capacity constraint, and hydrogen-demand constraints. It is a screening model, not a project-specific engineering design.

Configured ENTSO-E labels describe an intended data source; the present repository does not include an authenticated ENTSO-E download workflow. ERA5 wind is different: `data/era5.py` implements and unit-tests a real Copernicus Climate Data Store fetch, nearest-grid-cell extraction, hub-height shear correction, and a committed provenance record (`pynexus fetch-era5`) — see [`docs/data_provenance.md`](docs/data_provenance.md). No live ERA5 fetch has been run in this repository, so no committed provenance record from a real download exists yet, and the deterministic examples and tests still use synthetic arrays.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
pynexus solve --config configs/tiny_test.yaml --synthetic --output outputs/tiny
pynexus verify --run outputs/tiny
```

For the 168-hour reference, replace the config with `config.yaml` and use a fresh
output directory. `--synthetic` is explicit: no ENTSO-E/ERA5 data are downloaded.
Alternatively supply `--input inputs.csv`; see [input schema](docs/input_schema.md).
The legacy `python -m optimization.dispatch` demonstration remains interactive.

The runner saves the exact config, inputs, dispatch, hashes, environment, solver
settings/status, model size and independent numerical checks. It rejects an
existing output directory to prevent stale results after a failed solve.

`pynexus fetch-era5 --config config.yaml --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
fetches real ERA5 wind from the Copernicus Climate Data Store and writes a
provenance record; it requires the `[data]` extra and a CDS account (see
[`docs/data_provenance.md`](docs/data_provenance.md)) and is independent of `solve`/`verify`.

## Configuration

`config.yaml` remains the backward-compatible weekly default. `configs/tiny_test.yaml` is a compact deterministic test configuration. An annual configuration will only be added after a complete run, recorded solver status, and checked output are available.

## Method and evidence

- `docs/formulation.md` — decision variables, objective, constraints, and units
- `docs/assumptions.md` — implemented scope and limitations
- `docs/data_provenance.md` — source/configuration boundary
- `docs/verification.md` — deterministic test evidence and validation terminology
- `tests/` — fast component and optimisation tests
- [Reproducibility and change evidence](docs/reproducibility.md)

## Contributing

See `CONTRIBUTING.md`. Calculation changes require tests and explicit units. Never change physics, constraints, data, or solver tolerances merely to obtain a preferred result.

## Licence

Code in this repository is available under the MIT License. Third-party dependencies retain their own licences.
