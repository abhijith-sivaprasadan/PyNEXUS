# WP5 observed-price marginal storage evidence and source-based conditional cost

**Status:** full-year E and E+S dispatch archives exist and were verified
previously; this additional tool independently rechecks original **hourly**
CSV bytes and recomputes the marginal values. It does **not** create the
proposed 17-case capital-inclusive cost–adequacy frontier, and it does not
make missing engineering cost quotations appear.

## Reproduction

Use original GitHub Actions runs [E: 40217860564](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40217860564)
and [E+S: 40216993086](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40216993086).
Download their scenario archives, preserving exact bytes, and run:

```bash
python -m research.wp5_observed_storage --E E.zip --ES EplusS.zip \
  --output observed-storage-audit
```

The dedicated `wp5-observed-evidence.yml` CI workflow retrieves the
registered GitHub Actions artifacts, rejects an altered price CSV SHA256,
validates Stage A/B solver status and verification flags, checks all saved
dispatch hashes, requires precisely 8,760 ordered hours, recomputes hydrogen
shortfall and hourly net grid settlement, and refuses mismatched wind or
configuration sources. It stores a new machine-readable `comparison.json`.

The correct observed-price incremental component is approximately
**€1.759 million / 343.793 tonnes = €5.12/kg**. This is **not full
hydrogen cost, levelised cost, financial feasibility, or a vendor quotation**.

## Sourced, *conditional* storage-capital accounting

A further **optional** `--storage-costs path.json` argument supports only
the **20-tonne incremental storage asset**, since W, E, G and all other
equipment quantities are unchanged in the E versus E+S experiment.

Required fields: `schema_version = pynexus.wp5.marginal_storage_costs.v1`;
`currency = EUR`; `price_year = 2023`; `capex_eur_per_kg`;
`fixed_opex_fraction`; `lifetime_years`; `discount_rate`;
`variable_opex_eur_per_additional_kg`; source-register HTTPS citations
for each assumption, and a substantial
`engineering_scope_and_2023_normalisation` explanation.

This optional cost case would compute:

```text
added installed storage cost = 20,000 kg * qualified EUR/kg CAPEX
annualized CAPEX = added installed cost * CRF(r, life)
fixed OPEX = added installed cost * fixed OPEX fraction/year
variable OPEX = extra delivered H2 kg * EUR/kg variable OPEX
incremental annual modelled cost = observed delta wholesale settlement
                                   + annualized CAPEX + fixed OPEX + variable OPEX
break-even incremental H2 value = incremental annual modelled cost
                                  / additional H2 delivered kg
```

No default cost numbers are inserted. Source URLs and disclosure strings
are not themselves independent evidence validation; every vendor/reference
scope, asset pressure, compression hardware, cost-year normalization and
non-overlap must be audited before interpreting such a result commercially.

Costs of grid access, losses, compression, carbon eligibility, capital and
operating investment across all 17 scenarios, O&M/tariffs and sensitivity
to discount/technology life are **not** resolved by this narrow method.
The full WP5 research requirement therefore remains **blocked**, pending
cost evidence and a complete scenario set at matching price/financial scope.

## Core provenance gaps retained

The processed 2023 Dutch price CSV has verified UTC hourly coverage and
an explicitly retrieved missing hour with 72 independent overlapping
controls. Original raw A44 XML is **not archived**. The original CDS-ERA5
2023 wind clock origin is not independently proven from primary source
metadata. Temporary GitHub Actions artifacts require durable independent
long-term archival before expiration.
