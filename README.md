# PyNEXUS

A reproducible Pyomo/HiGHS model for hourly dispatch of an offshore-wind, PEM-electrolyser, and hydrogen-pipeline system under physical operating constraints and alternative cost and emissions objectives.

## Scope

The checked-in reference configuration covers 168 hourly timesteps. No 8,760-hour result is claimed. The model includes wind availability, electrolyser minimum/maximum load and ramping, a simplified hydrogen-pipeline capacity constraint, and hydrogen-demand constraints. It is a screening model, not a project-specific engineering design.

Configured ENTSO-E and ERA5 labels describe intended data sources; the present repository does not include an authenticated download workflow. The deterministic examples and tests use synthetic arrays.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m optimization.dispatch
```

The module demonstration runs deterministic 48-hour cases and opens plots. For automated verification, use the test suite, which includes hand-checkable dispatch constraints without interactive plotting.

## Configuration

`config.yaml` remains the backward-compatible weekly default. `configs/tiny_test.yaml` is a compact deterministic test configuration. An annual configuration will only be added after a complete run, recorded solver status, and checked output are available.

## Method and evidence

- `docs/formulation.md` — decision variables, objective, constraints, and units
- `docs/assumptions.md` — implemented scope and limitations
- `docs/data_provenance.md` — source/configuration boundary
- `docs/verification.md` — deterministic test evidence and validation terminology
- `tests/` — fast component and optimisation tests

## Contributing

See `CONTRIBUTING.md`. Calculation changes require tests and explicit units. Never change physics, constraints, data, or solver tolerances merely to obtain a preferred result.

## Licence

Code in this repository is available under the MIT License. Third-party dependencies retain their own licences.
