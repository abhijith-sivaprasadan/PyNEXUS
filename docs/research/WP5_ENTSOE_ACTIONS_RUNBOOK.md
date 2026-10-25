> **Timestamp correction (2026-10-25):** The original year-spanning client rejected `2023-12-29 23:00 UTC` (2023-12-30 00:00 CET). The original XML's legitimate A03 compressed position `2023-12-30 08:00 UTC` is a separate phenomenon, **not the same missing timestamp**. Both hours match the verified processed 8,760-hour source. The parser's exact reason for omitting the former in the initial large-window response has not been isolated. See [corrected source audit](WP1_2023_A03_PRIMARY_SOURCE_EQUIVALENCE.md).

> **Resolved primary-source A03 finding (2026-10-25):** The originally reported missing `2023-12-30 08:00 UTC` price was not missing market information. The original A44 XML has `curveType=A03` and intentionally omits unchanged-price position 10; ENTSO-E's official constant-block rule assigns the preceding €73.50/MWh price through that position. Original 14 monthly XML documents have been independently expanded and match the accepted processed 8,760-hour 2023 series exactly. [Full source-equality audit](WP1_2023_A03_PRIMARY_SOURCE_EQUIVALENCE.md). **Do not describe this hour as an unresolved provider outage or as interpolation.**

# WP5B — private ENTSO-E token, reproducible Netherlands A44 2023 runbook

**State:** secure *execution workflow* is available; the presence of this file
does **not** mean historical prices have been fetched or market redispatch run.

## 1. Configure the secret outside source control

Open the GitHub repository's **Settings → Secrets and variables → Actions →
New repository secret**. Create a repository secret named **`ENTSOE_API_KEY`**
and set its *value* to your ENTSO-E Transparency Platform API token.

**Never** paste the token into an issue, pull request, repository variable,
source file, notebook, chat, logs or workflow inputs. The workflow reads it only
at runtime through `secrets.ENTSOE_API_KEY`. The workflow is manual-only,
requires the `main` branch, has read-only repository permissions, and refuses
to run if the secret is absent. An upstream ENTSO-E exception is redacted to
its exception class rather than printing a potentially credential-containing
HTTP URL. If a token is accidentally disclosed, rotate it.

## 2. Run the price acquisition first

On the repository's **Actions** page select
**WP5 Netherlands 2023 observed-price acquisition** → **Run workflow**.

Use branch **main** and select:

- `mode = prices-only`
- `scenario = baseline` (ignored for the price-only mode)

The run installs the ENTSO-E optional dependency and executes the existing
`python -m research.wp5_market fetch` path. Its validated evidence artifact is
**`wp5-nl-2023-processed-a44-prices`** with `prices.csv` and `manifest.json`.
The validator requires the NL/A44 zone and the full UTC year 2023
(8,760 hourly timestamps, all finite, no missing/duplicate timestamps).
Negative prices are allowed. It refuses ambiguous non-hourly input; no
interpolation, resampling or silent filling occurs. It reports the processed
CSV SHA-256 and number of negative-price hours.

**Known scientific limits:** the converter relies on the `entsoe-py` provider
adapter and does not archive the original A44 XML; this run is processed-data
provenance only. Its stored archive is a GitHub Actions artifact with 60-day
retention, **not** a DOI-backed permanent research deposit.

## If the ENTSO-E price validation step fails

The first live A44 request in October 2026 returned a price series, but the
strict 8,760-hour UTC calendar gate rejected it with a generic
"Missing/extra or nonhourly records" message. **The source of the mismatch is
not yet established.** Authentication appears to have worked; do not rotate
the token solely because of this failure.

The updated fetch step prints a sanitized JSON diagnosis of the **actual**
provider timestamps and writes
`wp5-nl-2023-rejected-calendar-diagnostics/coverage_diagnostics.json`.
On a failed fetch, GitHub uploads that file under
`wp5-nl-2023-REJECTED-unverified-calendar-diagnostics`.

Inspect `missing_exact_hour_count`, `first_missing_exact_hours_utc`,
`unexpected_in_year_timestamp_count`, and
`successive_interval_minutes_counts`. For example, a 15-minute price series
will expose repeated 15-minute intervals and off-hour timestamps instead of
being silently averaged into hourly observations.

### Evidence-based one-hour requery for the observed December 2023 anomaly

The owner's 2026-10-25 live `prices-only` run, ID `40204281501`, established
that ENTSO-E's year-spanning PandasClient response contains exactly 8,759
2023 UTC hours. Only **2023-12-29 23:00 UTC / 2023-12-30 00:00 CET** is missing;
there are no duplicates or off-grid timestamps, and the adjacent interval
spans 120 minutes. API authorization, data units and general timezone
conversion are working. The underlying provider-vs-parser cause remains
unconfirmed.

The refreshed workflow **does not interpolate that hour**. If and only if
this single registered gap is present, the fetcher requests the official
Netherlands A44 series independently for local dates
`2023-12-29` through `2023-12-31`. It accepts the missing hour only when:

1. The second A44 response contains an explicit, finite hourly observation
   at the missing UTC timestamp.
2. The adjacent control hours and at least 12 overlapping prices are present.
3. **Every** overlapping observation agrees with the year-spanning response
   to within 0.000001 EUR/MWh.
4. The stitched series then independently passes the original strict
   8,760-hour UTC check without resampling.

Successful archives preserve the complete independent query as
`independent_a44_targeted_observations.csv`, its checksum and the exact
recovered price/time and overlap audit in `manifest.json`. The status is
still *processed provider time series*, **not original raw XML**.

If the second response also omits the price, has inconsistent prices, or
cannot be retrieved, the original strict failure remains. Inspect the
uploaded diagnostic before attempting another source interpretation. An
A03 compressed curve may encode repeated values implicitly; no such
source-defined reconstruction has been added without inspection of the
original ENTSO-E XML. A query failure is not evidence that the true market
price is equal to a neighboring price.

If the source is incomplete, source retrieval needs correction or an
independently documented and permitted alternative. If the source truly
uses 15- or 30-minute market time units, a **separate, explicitly
registered transformation** must be specified, justified, provenance-logged,
and tested before any hourly dispatch comparison. Never interpolate missing
market hours or claim that rejected diagnostics establish real-price results.

Re-run the manual workflow with `mode = prices-only` after the diagnostic
fix is merged. Send the workflow run URL or the **sanitized diagnostic file**,
never the ENTSO-E API token, for triage.

## 3. Test or execute physical dispatch under observed prices

Re-run the same workflow on `main` using a new execution mode:

| Mode | Scenario | Meaning |
|---|---|---|
| `smoke-24h` | `baseline` | Verifies the price/data/optimizer wiring; **not** annual evidence |
| `annual-selected` | `baseline` | A complete 8,760-hour baseline run |
| `annual-selected` | `E` | Full-year electrolyser expansion |
| `annual-selected` | `E+S` | Full-year electrolyser-plus-storage case |
| `annual-selected` | `G` | Full-year grid-connection expansion |

Annual runs repeat the price acquisition and hash checking; the selection
never supplies an unverified price file. Stage A minimises unserved hydrogen,
and Stage B minimises observed-price net grid settlement under its Stage A
adequacy cap. A saved run fails if Stage B imports and exports power
simultaneously above the declared tolerance. This is a **detection** safeguard,
not a new anti-circulation MILP formulation.

Artifact names include the selected mode and scenario. Each recorded result
contains verification data, processed-price SHA-256, reference wind input
hash, solver statuses, dispatch and scenario metadata.

The full-year solver time limit is finite; a timeout or infeasible solve is a
**failure**, never a finding. Do not use a 24-hour smoke result as evidence
for annual hydrogen, finance or emissions outcomes.

## 4. Independent scientific gates still open

1. Confirm the original CDS 2023 wind hours correspond to the UTC calendar
   assumed by the price join. Both being 8,760 rows is not proof of alignment.
2. Preserve original ENTSO-E A44 provider XML and hashes if raw-data
   provenance is required for external publication.
3. Independently inspect the price snapshot and solver outputs, and repeat
   the remaining registered scenarios where necessary. This manual workflow
   offers selected configurations; it does **not** assert that all 17 are done.
4. Obtain traceable, scope-matched and common-price-year CAPEX/OPEX for all
   required hardware and credible finance/maintenance assumptions.
5. Validate network tariffs, compression boundaries, stack replacement and
   the price/engineering basis for any 70% → 75% efficiency case.

**Do not publish conditional grid settlement as LCOH, project profit or real
plant operating cost.** Stage A adequacy and Stage B market-price dispatch
are computational model results, not independent hardware validation.

See also `docs/research/WP5_COMPLETION.md` and
`docs/research/WP5_SOURCE_REGISTER.md`.
