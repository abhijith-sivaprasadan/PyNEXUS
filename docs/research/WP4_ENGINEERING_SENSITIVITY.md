# WP4 engineering and objective sensitivity — registered reproducible screen

**Status: implemented and unit-tested; full-year results NOT yet run.**

The original 2021–2025 WP4 weather screening is retained unchanged. This is
a separate family of isolated 2023 source-locked engineering experiments
designed to investigate the proposal's missing operating-assumption controls.
This registration includes a reference plus **16 one-factor cases** for:

- Minimum electrolyser loading (5% and 20% versus 10% reference).
- Electrolyser ramp rate (20% and 80% nameplate per hour versus 40%).
- Storage hourly charge and discharge limits, changed independently.
- Storage leakage/loss assumption.
- Pipeline flow-rate allowance.
- Daily hydrogen demand, keeping the requested unserved *fraction* distinct
  from absolute kilograms when demand changes.
- Available wind energy multiplier (90%, 110%) with a 150 MW nameplate cap.

**Not literature uncertainty bounds:** all numerical values are transparent
controlled illustrative perturbations, not engineering confidence intervals,
probabilistic sensitivity distributions, or vendor-proven operating regions.
The original full-year annual carbon and electricity-price arrays are
synthetic for this screen. Imported Dutch prices and wind-hour alignment
cannot be claimed. The original CDS wind data's precise UTC alignment still
needs independent source-audit.

Run the 24-hour integration check:

```bash
python -m research.engineering_sensitivity \
  --case slow_ramp --hours 24 --time-limit 180 \
  --output wp4-slow-ramp-smoke
```

To register a genuine full-year result, use `--hours 8760` and a new
output directory; retain stage_a/stage_b CSVs and the manifest. The GitHub
workflow **WP4 registered engineering sensitivity** exposes the same cases
and `annual-8760` mode via manual dispatch on `main`. It preserves the
failure classification and refuses overwrites.

Both stages share exactly the selected modified configuration, source
wind/price arrays and mathematical formulation. The record includes
solver verification, shortfall, settlement (explicitly synthetic), source
hashes and changed configuration. The 24-hour smoke must not be cited as
annual robustness.

The complete research question remains open until every pre-registered
full-year case is executed or classified with a reason, objective-mode
sensitivity is separately run, and results are compared in an archived
synthesis with consistent denominators and clear engineering limitations.
