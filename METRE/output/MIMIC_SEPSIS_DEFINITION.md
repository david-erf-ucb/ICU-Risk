# MIMIC-IV Sepsis-3 Definition

## Source

The `physionet-data.mimiciv_3_1_derived.sepsis3` table is a derived table in MIMIC-IV. Its definition comes from the [MIMIC Code Repository](https://github.com/MIT-LCP/mimic-code), specifically `mimic-iv/concepts/sepsis/sepsis3.sql`.

## How Sepsis-3 Is Defined

**Sepsis-3** (Singer et al., JAMA 2016) requires:
1. **Suspicion of infection** — antibiotics + culture, or clinical suspicion
2. **Organ dysfunction** — increase in SOFA score ≥ 2 points from baseline

The MIMIC implementation defines sepsis onset **within the ICU** as the earliest time at which:
- SOFA score ≥ 2, **and**
- Suspicion of infection is present

### Key assumptions

- **Baseline SOFA = 0** — The query assumes the patient had no organ dysfunction before ICU admission. This is a limitation: pre-existing chronic organ failure is not accounted for.
- **ICU-only** — SOFA components are derived from ICU data (vitals, labs, interventions). Sepsis onset can only be identified during the ICU stay.
- **Temporal alignment** — The SOFA window must overlap with the suspicion-of-infection event: SOFA `endtime` must be within **48 hours before** to **24 hours after** `suspected_infection_time`.

### Data sources

1. **`mimiciv_3_1_derived.sofa`** — Sequential Organ Failure Assessment (SOFA) scores, computed per 24-hour window. Components: respiration, coagulation, liver, cardiovascular, CNS, renal.
2. **`mimiciv_3_1_derived.suspicion_of_infection`** — Identifies suspected infection from antibiotic administration, culture collection, and positive culture results.

### Logic (simplified)

```
sepsis3 = (SOFA >= 2) AND (suspected_infection = 1)
         AND (SOFA window overlaps suspicion time ± 48h/24h)
```

The query selects the **earliest** such event per stay (`ROW_NUMBER() ... rn_sus = 1`).

## Output Files

| File | Description |
|------|-------------|
| `mimic_sepsis_stays.csv` | Minimal: `stay_id`, `sepsis` (1 for all rows) |
| `mimic_sepsis_full.csv` | Full sepsis3 table: stay_id, SOFA components, suspicion times, etc. |
| `mimic_sepsis_all_stays.csv` | All ICU stays with `sepsis` 0/1 (use `--all_stays`) |

## SOFA Table: One Score per Stay?

**No.** The `sofa` table has **multiple rows per stay** — one per 24-hour window. SOFA is computed over rolling 24-hour windows throughout the ICU stay, so a 5-day stay has ~5 SOFA rows.

The **sepsis3** table, by contrast, has **exactly one row per stay** — it selects the earliest sepsis onset (first time SOFA ≥ 2 and suspicion of infection align). So for sepsis cases, you get one `sofa_score` per stay in sepsis3.

To get SOFA distribution by stay, run:
```bash
python scripts/export_mimic_sepsis.py --sofa
```
This produces:
- `mimic_sofa_by_window.csv` — every SOFA row (stay_id, starttime, endtime, sofa_score, components)
- `mimic_sofa_by_stay.csv` — per-stay summary (n_windows, sofa_min, sofa_max, sofa_mean)
- `mimic_sofa_score_distribution.csv` — count of stays/windows at each SOFA value
- `mimic_sofa_windows_per_stay.csv` — how many 24h windows per stay

## References

- Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016;315(8):801-810.
- [MIMIC Code Repository — sepsis3.sql](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/sepsis/sepsis3.sql)
