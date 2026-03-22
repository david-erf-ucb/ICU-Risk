# METRE Benchmark Results Summary

**Date:** February 2025  
**Data:** MIMIC-IV 3.1 (MEEP parquet pipeline)  
**Reference:** [Liao & Voldman, J Biomed Inform 141 (2023)](https://github.com/weiliao97/METRE)

---

## Methodology

### Data Removed in the METRE Pipeline

**Upstream (MEEP extraction, `extract_database.py`):**
- **Cohort:** ICU stays with age ≥ 18, LOS 24–240h, first ICU stay per hospitalization (`hospstay_seq=1`, `icustay_seq=1`). Default `patient_group='Generic'` (all ICU stays).
- **Outlier removal:** Vital signs outside physiologic ranges (e.g. heart rate > 390, temperature > 47°C) are masked; `mimic_outlier_high.json` and `mimic_outlier_low.json` define thresholds.
- **Imputation:** Z-score normalization (MIMIC mean/std); missing values imputed with stay-level means.
- **Schema gaps (MIMIC-IV 3.1):** Culture data and heparin are not extracted (see `SCHEMA_MIGRATION_NOTES.md`).

**Benchmark compile/training:**
- **Task-specific filters:** Stays excluded if LOS < (observation_window + gap), or if the outcome occurs before the gap (ARF/shock). Stays with NaN `mort_hosp` excluded from hosp_mort.
- **No further aggregation:** Data stays at hourly resolution.

### Models

| Model | Config |
|-------|--------|
| **Logistic Regression** | `class_weight="balanced"`, `max_iter=1000`, L-BFGS solver |
| **Random Forest** | 100 trees, `class_weight="balanced"` |

No hyperparameter tuning (no Bayesian optimization). Training uses **train + dev** combined; 5-fold CV for AUC reporting.

### Train/Dev/Test Split

- **Split:** 70% train, 10% dev, 20% test (seed 41).
- **Unit:** ICU stay (patient-level split; all hours of a stay stay in one split).
- **Training:** Train + dev used for fitting; test held out for evaluation only.
- **Reproducibility:** Same permutation logic as `extract_database` (`list(stay_ids)` from set).

### Input Aggregation

- **Resolution:** Hourly. No coarsening; each hour is one time step.
- **Features:** 184 vital/lab columns (mean, count per hour) + 16 intervention binary flags = **200 features**.
- **Raw shape per stay:** `(200, n_hours)` — one column per hour from ICU admission. `n_hours` = length of stay (24–240h per cohort filter).

### Thresh and Gap (Task Parameters)

| Term | Meaning |
|------|---------|
| **thresh** | Observation window in hours. First `thresh` hours of ICU data used as input. |
| **gap** | Hours between end of observation window and earliest outcome onset. Prevents label leakage. |

**Per-task values:**

| Task | thresh | gap | Min LOS required |
|------|--------|-----|------------------|
| hosp_mort_24h_gap4h | 24 | 4 | 28h |
| ARF_2h_gap4h | 2 | 4 | 6h |
| ARF_6h_gap4h | 6 | 4 | 10h |
| shock_2h_gap4h | 2 | 4 | 6h |
| shock_6h_gap4h | 6 | 4 | 10h |

### Example: How a Stay Becomes Model Input

**Raw stay** (e.g. 72h ICU stay): array of shape `(200, 72)` — 200 features × 72 hours.

**For hosp_mort_24h_gap4h** (thresh=24):
1. Slice to first 24 hours: `(200, 24)`
2. Flatten to vector: `(4800,)` — 200 × 24 = 4,800 features
3. One row in the design matrix

**For ARF_2h_gap4h** (thresh=2):
1. Slice to first 2 hours: `(200, 2)`
2. Flatten to vector: `(400,)` — 200 × 2 = 400 features
3. One row in the design matrix

**For shock_6h_gap4h** (thresh=6):
1. Slice to first 6 hours: `(200, 6)`
2. Flatten to vector: `(1200,)` — 200 × 6 = 1,200 features
3. One row in the design matrix

**Model input:** `(n_samples, 200 × thresh)` — e.g. hosp_mort with 8,736 stays → `(8736, 4800)`. No temporal structure; each stay is a flat feature vector.

### Population Flow

| Stage | Approx. size | Notes |
|-------|--------------|-------|
| MEEP parquet (all stays) | ~45k–55k ICU stays | After extract_database cohort filters |
| Train (70%) | ~31k–38k | |
| Dev (10%) | ~4.5k–5.5k | |
| Test (20%) | ~9k–11k | |
| **Task-specific test n** | 2.5k–8.7k | Per-task filters (LOS, outcome timing) |

### Dependent Variable Definitions

**Hosp_mort (in-hospital mortality)**  
- **Definition:** Patient died before hospital discharge.

- **Source:** `mort_hosp` in static table (`deathtime` between `admittime` and `dischtime`).

- **Label:** Positive = 1, negative = 0.

---

**ARF (acute respiratory failure)**  
- **Definition:** Mechanical ventilation or invasive respiratory support (PEEP) started after the prediction window.

- **Source:**  
  - Vent: `vent` intervention column (index 184) = 1.  
  - PEEP: `peep` vital columns (indices 157, 159) = 1.

- **Label:** Positive if first vent or PEEP onset ≥ (thresh + gap) hours; negative if never, or onset before the gap.

- **Exclusion:** Stays with vent/PEEP before the gap are excluded (no label leakage).

---

**Shock (vasopressor use)**  
- **Definition:** Patient received vasopressors (dopamine, epinephrine, norepinephrine, phenylephrine, vasopressin) after the prediction window.

- **Source:** Intervention columns 186–191 (any ≥ 1).

- **Label:** Positive if first vasopressor use ≥ (thresh + gap) hours; negative if never, or use before the gap.

- **Exclusion:** Stays with vasopressor use before the gap are excluded.

---

### Other Notes

- **Gap:** 4 hours between end of observation window and outcome onset (prevents label leakage). See **Thresh and Gap** above.
- **Cross-validation:** 5-fold CV on train+dev for AUC; final model is fit on full train+dev.
- **eICU cross-database:** MIMIC-trained models evaluated on eICU test set via `evaluate_on_eicu.py` (requires `eICU_compile.npy`).

---

## 0. Task Populations: Why n_total Differs

Each task uses a **different inclusion filter** based on the prediction setup. All tasks share the same underlying test set (20% of stays, seed 41), but only stays that meet task-specific criteria are included.

| Task | n_total | Inclusion logic |
|------|---------|-----------------|
| **hosp_mort_24h_gap4h** | 8,736 | LOS ≥ 28h (24h + 4h gap); valid mort_hosp |
| **ARF_2h_gap4h** | 2,854 | Vent/PEEP onset ≥ 6h or never; uses first 2h |
| **ARF_6h_gap4h** | 2,464 | Vent/PEEP onset ≥ 10h or never; uses first 6h |
| **shock_2h_gap4h** | 7,181 | Vasopressor onset ≥ 6h or never; uses first 2h |
| **shock_6h_gap4h** | 6,927 | Vasopressor onset ≥ 10h or never; uses first 6h |

**Why the differences?**

- **Time-based:** Longer observation windows (6h vs 2h) require longer ICU stays (≥10h vs ≥6h), so fewer stays qualify.
- **Outcome-based:** ARF and shock exclude stays where the event occurs *before* the gap (to avoid label leakage). Hosp_mort has no such exclusion beyond LOS.
- **hosp_mort** has the largest n because it only requires LOS ≥ 28h and a valid mortality label.
- **ARF** has the smallest n because it requires a clear vent/PEEP timeline and excludes early-onset cases.

---

## 1. Models Run

See **Methodology** above. LR and RF with fixed configs; flattened 200×T features.

---

## 2. Results

### Test AUC

| Task | LR | RF |
|------|-----|-----|
| hosp_mort_24h_gap4h | 0.809 | **0.841** |
| ARF_2h_gap4h | 0.661 | **0.693** |
| ARF_6h_gap4h | 0.639 | **0.659** |
| shock_2h_gap4h | 0.674 | **0.700** |
| shock_6h_gap4h | **0.670** | **0.677** |

### Test AP (Average Precision)

| Task | LR | RF |
|------|-----|-----|
| hosp_mort_24h_gap4h | 0.331 | 0.350 |
| ARF_2h_gap4h | 0.564 | 0.586 |
| ARF_6h_gap4h | 0.405 | 0.404 |
| shock_2h_gap4h | 0.183 | 0.196 |
| shock_6h_gap4h | 0.123 | 0.119 |

### Class Balance & Precision/Recall (threshold 0.5)

| Task | n_pos | n_neg | % pos | LR prec | LR rec | RF prec | RF rec |
|------|-------|-------|-------|---------|--------|---------|--------|
| hosp_mort_24h | 787 | 7,949 | 9.0% | 0.23 | 0.68 | 0.67 | 0.04 |
| ARF_2h | 1,053 | 1,801 | 36.9% | 0.52 | 0.52 | 0.64 | 0.31 |
| ARF_6h | 663 | 1,801 | 26.9% | 0.36 | 0.54 | 0.55 | 0.04 |
| shock_2h | 720 | 6,461 | 10.0% | 0.17 | 0.54 | 0.26 | 0.13 |
| shock_6h | 466 | 6,461 | 6.7% | 0.12 | 0.57 | 0.18 | 0.00 |

### eICU Cross-Database Validation (Train MIMIC, Test eICU)

Models trained on MIMIC were evaluated on the eICU test set (20% of eICU stays, same split logic). Run `python training/evaluate_on_eicu.py` after compiling eICU.

| Task | LR (eICU) | RF (eICU) | n (eICU) |
|------|-----------|-----------|----------|
| hosp_mort_24h_gap4h | 0.553 | 0.718 | 22,590 |
| ARF_2h_gap4h | 0.524 | 0.523 | 21,609 |
| ARF_6h_gap4h | 0.526 | 0.519 | 21,535 |
| shock_2h_gap4h | 0.487 | 0.561 | 22,417 |
| shock_6h_gap4h | 0.484 | 0.563 | 22,159 |

### MIMIC vs eICU Comparison

| Task | MIMIC LR | MIMIC RF | eICU LR | eICU RF | Δ LR | Δ RF |
|------|----------|----------|---------|---------|------|------|
| hosp_mort_24h | 0.809 | 0.841 | 0.553 | 0.718 | −0.26 | −0.12 |
| ARF_2h | 0.661 | 0.693 | 0.524 | 0.523 | −0.14 | −0.17 |
| ARF_6h | 0.639 | 0.659 | 0.526 | 0.519 | −0.11 | −0.14 |
| shock_2h | 0.674 | 0.700 | 0.487 | 0.561 | −0.19 | −0.14 |
| shock_6h | 0.670 | 0.677 | 0.484 | 0.563 | −0.19 | −0.11 |

**Interpretation:** Cross-database transfer shows substantial performance drop. LR degrades more than RF (hosp_mort: LR −0.26 vs RF −0.12). ARF and shock fall to near-random (AUC ~0.52) on eICU. Likely causes: different hospitals, documentation practices, outcome definitions, and eICU normalization using MIMIC mean/std. Domain adaptation or eICU-specific training would be needed for deployment.

### Confusion Matrix (threshold 0.5) — MIMIC Test

| Task | Model | TP | FP | TN | FN |
|------|-------|-----|-----|-----|-----|
| hosp_mort_24h | LR | 532 | 1,765 | 6,184 | 255 |
| hosp_mort_24h | RF | 31 | 15 | 7,934 | 756 |
| ARF_2h | LR | 546 | 512 | 1,289 | 507 |
| ARF_2h | RF | 327 | 180 | 1,621 | 726 |
| ARF_6h | LR | 359 | 635 | 1,166 | 304 |
| ARF_6h | RF | 29 | 24 | 1,777 | 634 |
| shock_2h | LR | 389 | 1,932 | 4,529 | 331 |
| shock_2h | RF | 97 | 282 | 6,179 | 623 |
| shock_6h | LR | 267 | 1,994 | 4,467 | 199 |
| shock_6h | RF | 2 | 9 | 6,452 | 464 |

---

## 3. Assessment: Are These Results Good?

**Hosp_mort (AUC 0.81–0.84):** Strong. In-hospital mortality from early ICU data typically reaches AUC 0.80–0.85 in MIMIC. These results are in that range.

**ARF (AUC 0.64–0.69):** Moderate. Predicting respiratory failure from pre-onset vitals is harder than mortality. AUC in the mid-0.60s is reasonable for this task.

**Shock (AUC 0.67–0.70):** Moderate. Similar to ARF; vasopressor onset prediction from prior vitals is challenging.

**vs. original METRE ([weiliao97/METRE](https://github.com/weiliao97/METRE)):**

- Original METRE uses **48h** for mortality (we use 24h), **12h** for ARF (we use 2h/6h), and Bayesian optimization for LR/RF.
- Their README reports AUC for 48h mortality; typical values are ~0.82–0.85 for LR/RF on MIMIC.
- Our hosp_mort AUC (0.81–0.84) is comparable despite a shorter window (24h vs 48h).
- Our ARF/shock setups differ in time windows and definitions, so direct comparison is limited. Our results are in a plausible range for these tasks.

---

## 4. Results Relative to One Another

- **RF generally outperforms LR** on AUC (except shock_6h, where they are close).
- **Hosp_mort is the strongest task** (AUC 0.81–0.84); it has the most data and a well-defined outcome.
- **ARF and shock are similar** (AUC ~0.64–0.70); both predict acute events from prior vitals.
- **Longer windows (6h vs 2h)** do not improve AUC here; ARF_6h and shock_6h are slightly worse or similar to 2h.
- **Precision–recall trade-off:** RF tends toward higher precision and lower recall at 0.5 threshold (especially for rare outcomes), while LR is more recall-oriented. For imbalanced tasks, threshold tuning is important.

---

## 5. Why Models Perform Well or Poorly

**Hosp_mort performs best because:**
- Large sample (8,736 stays).
- Mortality is a clear, well-recorded outcome.
- Early vitals and interventions carry strong prognostic signal.
- Class imbalance (9% positive) is manageable with `class_weight="balanced"`.

**ARF and shock are harder because:**
- Smaller effective samples (2.5k–7k).
- Onset timing is more variable and definition-dependent.
- Pre-onset signal may be weaker than for mortality.
- ARF has more balanced classes (27–37% positive); shock is more imbalanced (7–10%).

**RF vs LR:**
- RF benefits from non-linear interactions and may handle high-dimensional flattened inputs better.
- LR can hit `max_iter` limits; increasing it may help.
- RF’s tendency toward high precision / low recall at 0.5 suggests it is conservative; threshold tuning could improve utility.

---

## 6. Proposals for Future Work

1. **Hyperparameter tuning:** Use Bayesian optimization (as in original METRE) for LR and RF.
2. **Threshold optimization:** Choose thresholds by utility (e.g., F2, cost-sensitive) instead of 0.5.
3. **Temporal models:** Add TCN/LSTM (as in METRE) to exploit time structure instead of flattening.
4. **Longer windows:** Test 48h for mortality and 12h for ARF to match original METRE.
5. **Fairness analysis:** Use `test_predictions.csv` (age, gender, race) for disparate impact and calibration by subgroup.
6. **eICU domain adaptation:** Cross-database validation (train MIMIC, test eICU) shows large AUC drops; train on eICU or use domain adaptation to improve transfer.
7. **Feature importance:** Inspect RF feature importances and LR coefficients for interpretability.
8. **Calibration:** Evaluate probability calibration (e.g., reliability diagrams) for clinical use.

---

## 7. Visuals

**Interactive HTML (open in browser):** `BENCHMARK_FIGURES/benchmark_results.html`  
Charts: AUC by task, class balance, % positive, precision vs recall, confusion matrices (TP/FP/TN/FN).

**ICU Case Load Dashboard:** `dashboard/index.html` — Summary for admins/clinicians: total stays, high-risk counts (mortality, ARF, shock), risk distribution. Run `python training/export_dashboard_data.py` to generate data.

**Static PNGs (optional):** Run `python training/plot_benchmark_results.py` to generate:
- `auc_by_task.png` — AUC by task and model
- `class_balance.png` — Class balance (n_pos, n_neg) by task
- `pct_positive_by_task.png` — % positive class
- `precision_recall_tradeoff.png` — Precision vs recall by task and model  

Requires `matplotlib`.

---

## Addendum: Precision and Recall by Threshold

Precision and recall at thresholds 0.05, 0.10, …, 0.50 (step 0.05). Rows = tasks; columns = thresholds. Generated from `test_predictions.csv` via `python training/generate_precision_recall_tables.py`.

### LR Precision

| Task | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|------|------|------|------|------|------|------|------|------|------|------|
| hosp_mort_24h_gap4h | 0.100 | 0.110 | 0.121 | 0.133 | 0.146 | 0.160 | 0.177 | 0.193 | 0.210 | 0.232 |
| ARF_2h_gap4h | 0.369 | 0.369 | 0.370 | 0.370 | 0.374 | 0.388 | 0.399 | 0.436 | 0.479 | 0.516 |
| ARF_6h_gap4h | 0.270 | 0.270 | 0.270 | 0.272 | 0.275 | 0.282 | 0.293 | 0.312 | 0.341 | 0.361 |
| shock_2h_gap4h | 0.100 | 0.100 | 0.101 | 0.102 | 0.107 | 0.113 | 0.125 | 0.140 | 0.158 | 0.168 |
| shock_6h_gap4h | 0.068 | 0.068 | 0.069 | 0.072 | 0.075 | 0.081 | 0.086 | 0.094 | 0.105 | 0.118 |

### LR Recall

| Task | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|------|------|------|------|------|------|------|------|------|------|------|
| hosp_mort_24h_gap4h | 0.990 | 0.978 | 0.957 | 0.940 | 0.905 | 0.861 | 0.835 | 0.785 | 0.733 | 0.676 |
| ARF_2h_gap4h | 0.999 | 0.999 | 0.998 | 0.993 | 0.982 | 0.961 | 0.890 | 0.783 | 0.645 | 0.519 |
| ARF_6h_gap4h | 0.998 | 0.997 | 0.994 | 0.983 | 0.968 | 0.925 | 0.866 | 0.780 | 0.683 | 0.541 |
| shock_2h_gap4h | 1.000 | 1.000 | 1.000 | 0.990 | 0.982 | 0.932 | 0.857 | 0.764 | 0.669 | 0.540 |
| shock_6h_gap4h | 1.000 | 0.996 | 0.987 | 0.972 | 0.936 | 0.891 | 0.820 | 0.736 | 0.655 | 0.573 |

### RF Precision

| Task | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|------|------|------|------|------|------|------|------|------|------|------|
| hosp_mort_24h_gap4h | 0.140 | 0.205 | 0.270 | 0.329 | 0.378 | 0.421 | 0.514 | 0.583 | 0.642 | 0.674 |
| ARF_2h_gap4h | 0.370 | 0.371 | 0.383 | 0.402 | 0.429 | 0.468 | 0.509 | 0.547 | 0.595 | 0.645 |
| ARF_6h_gap4h | 0.269 | 0.273 | 0.287 | 0.311 | 0.345 | 0.403 | 0.458 | 0.489 | 0.527 | 0.547 |
| shock_2h_gap4h | 0.128 | 0.174 | 0.223 | 0.263 | 0.274 | 0.282 | 0.282 | 0.277 | 0.274 | 0.256 |
| shock_6h_gap4h | 0.093 | 0.132 | 0.144 | 0.206 | 0.214 | 0.154 | 0.200 | 0.188 | 0.214 | 0.182 |

### RF Recall

| Task | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|------|------|------|------|------|------|------|------|------|------|------|
| hosp_mort_24h_gap4h | 0.956 | 0.868 | 0.690 | 0.487 | 0.324 | 0.215 | 0.142 | 0.094 | 0.055 | 0.039 |
| ARF_2h_gap4h | 1.000 | 0.993 | 0.982 | 0.945 | 0.872 | 0.756 | 0.619 | 0.493 | 0.387 | 0.311 |
| ARF_6h_gap4h | 1.000 | 0.995 | 0.964 | 0.851 | 0.679 | 0.495 | 0.323 | 0.166 | 0.074 | 0.044 |
| shock_2h_gap4h | 0.869 | 0.644 | 0.411 | 0.285 | 0.218 | 0.197 | 0.182 | 0.165 | 0.151 | 0.135 |
| shock_6h_gap4h | 0.828 | 0.388 | 0.109 | 0.047 | 0.019 | 0.009 | 0.009 | 0.006 | 0.006 | 0.004 |
