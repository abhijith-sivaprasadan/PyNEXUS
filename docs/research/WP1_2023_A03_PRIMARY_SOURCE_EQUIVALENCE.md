# WP1 primary-source equality audit — Dutch 2023 hourly day-ahead prices

**Audit date:** 2026-10-25. **Verdict:** original ENTSO-E NL A44 XML
and the archived processed 2023 hourly prices are **exactly equivalent**,
after applying the official A03 variable-sized-block specification. No
interpolation, numerical smoothing, or unobserved market-price assumptions
were introduced.

This was verified independently by downloading and parsing the source
artifacts, checking every original XML SHA-256, and comparing the resulting
year of UTC hourly prices **hour for hour** with the previously verified
processed 2023 series.

Evidence:
- Original source retrieval: [run 40225702504](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40225702504);
  original [14-document XML artifact](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40225702504/artifacts/12610917985).
- Processed series retrieval: [run 40205807892](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40205807892);
  [2023 price artifact](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40205807892/artifacts/12603687865).
- Machine-readable hash-bound audit: [WP1_NL_2023_A03_EXACT_EQUIVALENCE.json](evidence/WP1_NL_2023_A03_EXACT_EQUIVALENCE.json).
- Re-runnable independent implementation: `research/wp1_raw_a44_exact_audit.py`.
- Official semantics: [ENTSO-E Transparency XML Schema Use v1.0, A03 variable-sized blocks](https://transparency.entsoe.eu/content/static_content/Static%20content/knowledge%20base/entso-e-transparency-xml-schema-use-1-0.pdf).
  A03 publishes a new point only when the value changes; each block
  remains constant through the following position(s) until the next change
  or the declared end of the Period.

## Independently calculated findings

| Evidence check | Result |
|---|---:|
| Original A44 XML files, Dec 2022–Jan 2024 | 14 |
| Individual daily `TimeSeries` in source | 427 |
| A03 periods with at least one omitted, unchanged-value position | 49 |
| Source-defined implicit constant-price positions (all 14 months) | 51 |
| Exact UTC 2023 hours reconstructed | 8,760 |
| Missing, duplicate or off-grid UTC year-hours | 0 |
| Values that differ from the already published processed price series | **0** |
| Maximum price difference | **€0.00/MWh** |
| Negative-price hours | 315 |
| 2023 arithmetic mean price | €95.81777055/MWh |

All original XML entries used `A44`, `A03`, `PT60M`, NL domain
`10YNL----------L`, currency EUR and price unit EUR/MWh. All 14
original XML files matched their manifest-recorded SHA-256 checksums.

### Distinguish the original missing Pandas timestamp from an A03 compressed price

**Correction to the earlier interpretation:** the original rejected year-spanning
PandasClient response was missing **2023-12-29 23:00 UTC**, which corresponds
to **2023-12-30 00:00 CET**, as proved by the exact dated workflow
[40204281501](https://github.com/abhijith-sivaprasadan/PyNEXUS/actions/runs/40204281501).
The A03 compressed position below is instead **2023-12-30 08:00 UTC**.
They are **nine hours apart**, not the same point. The former is an
explicit first-position price in the original day-ahead source, while the
latter is a source-defined constant block price. The *mechanism* causing
the year-spanning client to omit 23:00 UTC is not proven from this A03
observation alone. The processed 8,760-hour price series is still
source-equivalent in all values, so prior dispatch outputs do not change.

The official XML for **local Dutch delivery date 2023-12-30**
(`Period` beginning `2023-12-29T23:00Z`) has price points at positions
9 and 11, but no explicit point at position **10**. This is legitimate
**A03 source encoding**, not a missing market price.

- Source position 9: `2023-12-30 07:00 UTC`, **€73.50/MWh**.
- Source position 10: `2023-12-30 08:00 UTC`, **€73.50/MWh**, as
  defined by the unchanged value of position 9.
- Source position 11: `2023-12-30 09:00 UTC`, **€53.00/MWh**.

The original **A03 block semantics** explain why the XML has 23
`Point` entries instead of 24 for this delivery day. Separately,
the later short-window official A44 retrieval recovered the
**23:00 UTC** value and all 72 overlapping processed prices agreed.
Both facts are independently verified, but one does not establish the
cause of the other. The annual source equivalence result remains
8,760/8,760 identical processed market prices.

This is a significant closure of the WP1/WP5 **primary-market-data
provenance gate**. It does **not** close the CDS ERA5 original hourly wind
reconstruction/alignment gate, prove the mathematical techno-economic
model represents a real electrolyser, source capital cost inputs,
establish LCOH, or create an external long-term archive/DOI.

## Reproduce while GitHub Actions source artifacts are retained

Download both original artifacts, extracting their contents into
`raw-nl-a44/` and `processed-nl-a44/`. Then run:

```bash
python -m research.wp1_raw_a44_exact_audit \
  --raw-dir raw-nl-a44 \
  --processed-dir processed-nl-a44 \
  --output independent-original-xml-equality
```

The command raises an error if *any* original XML hash differs, if an
A03 change position is invalid, if UTC year coverage is incomplete, if
processed input metadata or hashes are wrong, or if **any one** of the
8,760 floating-point prices differs. It writes
`independent-original-xml-equality/report.json` only after success.

**Artifact retention:** source files in GitHub Actions are temporary;
the original XML artifact is set to expire **2026-12-24**. The source
zip and its recorded SHA-256 must be retained in durable independent
research storage (and checked after download). A committed JSON manifest
does not substitute for preserving the actual source bytes.
