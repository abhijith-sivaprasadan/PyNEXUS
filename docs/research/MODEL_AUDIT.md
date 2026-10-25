# WP1 model/objective audit — 25 October 2026

Status: code-level audit and first implementation. Not a certification of the historical 8,760-hour results or an independently completed annual re-run.

## Audited current implementation

- Historical `minimize_cost` uses `price * p + price * g_imp - price * g_exp + 1000 * shortfall` for grid-connected hourly cases (plus separate heat terms if enabled).
- This is a **hybrid price-weighted objective**. When interpreted as monetary cash paid for electricity, `price*p` overlaps consumption served through grid imports and is not a pure purchased-energy cost. If interpreted as wind opportunity cost, it requires a clear generator ownership/opportunity-value boundary not currently established.
- `minimize_emissions` contains an electrolyser-price-analogous `carbon*p` plus `carbon*g_imp` term and a shortfall-penalty with inconsistent physical units (penalty weight originally defined for EUR/kg). Therefore the emissions objective also requires a distinct scope redesign before greenhouse-gas results can be reported scientifically.
- `enable_heat` changes the legacy objective by valuing delivered hydrogen and heat. It must not be silently compared with the heat-disabled cost objective as if both measure identical physical/economic quantities.
- Simultaneous grid import/export has no anti-circulation binary; positive/negative price, tariff and emissions accounting all require explicit discussion.
- The legacy grid + storage annual run preserved in `outputs/annual_2023/` used real wind but **synthetic** grid prices and carbon factors, and its saved solver manifest records a dirty working tree.
- `docs/results.md` shows strong penalty dominance: heat recovery value had no effect on electrolyser dispatch under the configured EUR 1000/kg shortfall penalty, which is an objective-design sensitivity, not proof heat recovery lacks value.

## New research-only path (implementation in this branch)

- `objective="minimize_unserved"`: sum of hourly hydrogen shortfall times timestep duration [kg]. No explicit price or carbon weighting.
- `objective="minimize_cash_cost"`: net wholesale grid payments [EUR]; requires an explicit near-optimal shortfall cap sourced from Stage A. It explicitly excludes capital and non-electric OPEX.
- `research/adequacy.py`: two-stage runner, chosen MIP-gap safeguard, independent objective and balance verification. No changes to default objectives.
- `research/physical_bounds.py`: optimistic production-capacity bound ignoring ramps/minimum-load logic; only a **lower bound on shortfall**, never a feasibility certificate.

## Open WP1 gates (not yet claimed complete)

1. Confirm clean CI on all supported platforms, including the historical locked-cost oracle.
2. Pin a full reproducible environment and archive a **clean** annual 2023 research two-stage run. A run that times out is an unresolved gate, not a usable optimization result.
3. Independently audit energy/carbon objective unit semantics before introducing an emissions frontier.
4. Establish price-source, timezone and tariff scope before calling a net grid-settlement result an economic estimate for the Netherlands.
5. Add evidence of achievable solver bound/gap and independently verify alternative seeds/solver tolerances where useful.
6. Investigate import/export degeneracy and consider a physically exclusive-market rule for interpretation of separate flows.

A repository merge or passing CI alone does not satisfy these empirical and economic gates.
