"""
Export per-stay predictions and ground truth for benchmark models.

Loads saved models and compiled data, runs inference on TEST SET ONLY (no train/dev),
and writes a CSV with stay IDs, demographics, ground truth, and predictions for each task.

No retraining - inference only. Test set is the held-out 20% from the 70/10/20 split.
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    log_loss,
)

# Import filter logic and stay-order reconstruction from compile
import sys
sys.path.insert(0, os.path.dirname(__file__))
from compile_meep_to_npy import (
    _load_mimic,
    _get_stay_id_level,
    _build_stay_arrays,
    _split_stays,
    SEED,
    TRAIN_FRAC,
    DEV_FRAC,
    TEST_FRAC,
)


def filter_los(static_data, vitals_data, thresh, gap):
    los = [i.shape[1] for i in vitals_data]
    ind = [i for i in range(len(los)) if los[i] >= (thresh + gap) and not np.isnan(static_data[i, 0])]
    vitals_reduce = [vitals_data[i][:, :thresh] for i in ind]
    static_subset = static_data[ind]
    return static_subset, vitals_reduce, ind


def filter_arf(vital, thresh, gap):
    vital_reduce, target, ind = [], [], []
    for i in range(len(vital)):
        arf_flag = np.where(vital[i][184, :] == 1)[0]
        peep_flag = np.union1d(np.where(vital[i][157, :] == 1)[0], np.where(vital[i][159, :] == 1)[0])
        if len(arf_flag) == 0:
            if len(peep_flag) > 0:
                if peep_flag[0] >= (thresh + gap):
                    vital_reduce.append(vital[i][:, :thresh])
                    target.append(1)
                    ind.append(i)
            else:
                vital_reduce.append(vital[i][:, :thresh])
                target.append(0)
                ind.append(i)
        elif arf_flag[0] >= (thresh + gap):
            if (len(peep_flag) > 0 and peep_flag[0] >= (thresh + gap)) or len(peep_flag) == 0:
                vital_reduce.append(vital[i][:, :thresh])
                target.append(1)
                ind.append(i)
    return vital_reduce, np.asarray(target), ind


def filter_shock(vital, thresh, gap):
    vital_reduce, target, ind = [], [], []
    for i in range(len(vital)):
        shock_flag = np.where(vital[i][186:191].sum(axis=0) >= 1)[0]
        if len(shock_flag) == 0:
            vital_reduce.append(vital[i][:, :thresh])
            target.append(0)
            ind.append(i)
        elif shock_flag[0] >= (thresh + gap):
            vital_reduce.append(vital[i][:, :thresh])
            target.append(1)
            ind.append(i)
    return vital_reduce, np.asarray(target), ind


def flatten_for_sklearn(data_list):
    return np.stack([d.flatten() for d in data_list], axis=0)


def get_test_stay_ids_and_data(input_dir, data_path, database="MIMIC"):
    """Reconstruct test stay IDs and verify they match compiled data."""
    vital, inv, static = _load_mimic(input_dir) if database == "MIMIC" else (
        __import__("compile_meep_to_npy")._load_eicu(input_dir)
    )
    stay_level_name = "stay_id" if database == "MIMIC" else "patientunitstayid"
    stay_ids = set(vital.index.get_level_values(stay_level_name).unique())

    # Same split as compile (must use list(stay_ids) to match compile's permutation order)
    np.random.seed(SEED)
    perm = np.random.permutation(list(stay_ids))
    N = len(perm)
    n_train = int(TRAIN_FRAC * N)
    n_dev = int(DEV_FRAC * N)
    test_stay = set(perm[n_train + n_dev :])

    vital_test = vital[vital.index.get_level_values(stay_level_name).isin(test_stay)]
    inv_test = inv[inv.index.get_level_values(stay_level_name).isin(test_stay)]

    head_list, order = _build_stay_arrays(vital_test, inv_test, database)
    test_stay_ids = order  # same order as test_head in compiled .npy

    # Verify compiled data matches
    data = np.load(data_path, allow_pickle=True).item()
    test_head = data["test_head"]
    assert len(test_head) == len(test_stay_ids), "Mismatch: compiled test size vs reconstructed"
    for i in range(min(5, len(test_head))):
        assert np.allclose(test_head[i], head_list[i]), f"Mismatch at stay {i}"

    return test_stay_ids, data, static


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "MIMIC_compile.npy"))
    parser.add_argument("--input_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--models_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "models"))
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "test_predictions.csv"))
    parser.add_argument("--dict_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "DATA_DICTIONARY.md"))
    parser.add_argument("--metrics_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "test_metrics.csv"))
    parser.add_argument("--population_path", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "benchmarks", "test_population_summary.csv"))
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    input_dir = os.path.abspath(args.input_dir)
    models_dir = os.path.abspath(args.models_dir)
    output_path = os.path.abspath(args.output_path)
    dict_path = os.path.abspath(args.dict_path)
    metrics_path = os.path.abspath(args.metrics_path)
    population_path = os.path.abspath(args.population_path)

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return 1
    if not os.path.exists(models_dir):
        print(f"Error: {models_dir} not found. Run run_benchmarks_lr_rf.py first.")
        return 1

    print("Loading data and reconstructing test stay IDs...")
    test_stay_ids, data, static = get_test_stay_ids_and_data(input_dir, data_path, "MIMIC")

    test_head = data["test_head"]
    s_test = np.stack(data["static_test_filter"], axis=0)

    task_map = {0: "hosp_mort", 1: "ARF", 2: "shock"}
    tasks = [(0, 24, 4), (1, 2, 4), (1, 6, 4), (2, 2, 4), (2, 6, 4)]

    # Build lookup: stay_id -> (subject_id, hadm_id, age, gender, race)
    static_lookup = {}
    for idx in static.index:
        if isinstance(idx, tuple):
            sub_id, hadm_id, stay_id = idx[0], idx[1], idx[2]
        else:
            stay_id = idx
            sub_id, hadm_id = None, None
        row = static.loc[idx]
        static_lookup[stay_id] = {
            "subject_id": sub_id,
            "hadm_id": hadm_id,
            "age": row.get("age", np.nan),
            "gender": row.get("gender", ""),
            "race": row.get("race", ""),
            "admission_type": row.get("admission_type", ""),
            "los_icu": row.get("los_icu", np.nan),
        }

    # Collect all stays that appear in any task's test set
    all_stays = set()
    task_data = {}  # task_name -> (stay_ids, gt, LR_prob, RF_prob)
    metrics_rows = []  # for test_metrics.csv
    population_rows = []  # for test_population_summary.csv

    for target_idx, thresh, gap in tasks:
        task_name = f"{task_map[target_idx]}_{thresh}h_gap{gap}h"
        if target_idx == 0:
            static_subset, test_data, ind = filter_los(s_test, test_head, thresh, gap)
            test_label = static_subset[:, 0]
        elif target_idx == 1:
            test_data, test_label, ind = filter_arf(test_head, thresh, gap)
        else:
            test_data, test_label, ind = filter_shock(test_head, thresh, gap)

        stay_ids_task = [test_stay_ids[i] for i in ind]
        all_stays.update(stay_ids_task)

        # Population summary (class balance)
        n_total = len(test_label)
        n_positive = int(np.sum(test_label))
        n_negative = n_total - n_positive
        pct_positive = 100.0 * n_positive / n_total if n_total > 0 else 0
        population_rows.append({
            "task": task_name,
            "n_total": n_total,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "pct_positive": round(pct_positive, 2),
        })

        X_test = flatten_for_sklearn(test_data)
        lr_model = joblib.load(os.path.join(models_dir, f"LR_{task_name}.joblib"))
        rf_model = joblib.load(os.path.join(models_dir, f"RF_{task_name}.joblib"))
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        rf_prob = rf_model.predict_proba(X_test)[:, 1]

        task_data[task_name] = (stay_ids_task, test_label, lr_prob, rf_prob)

        # Metrics (precision/recall at threshold 0.5; log_loss = binary cross-entropy; confusion matrix)
        def _metrics(y_true, y_prob, model_name):
            y_pred = (y_prob >= 0.5).astype(int)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            eps = 1e-15
            p = np.clip(y_prob, eps, 1 - eps)
            bce = log_loss(y_true, np.column_stack([1 - p, p]), labels=[0, 1])
            return {
                "task": task_name, "model": model_name,
                "auc": roc_auc_score(y_true, y_prob),
                "ap": average_precision_score(y_true, y_prob),
                "precision": prec, "recall": rec, "log_loss": bce,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }

        metrics_rows.append(_metrics(test_label, lr_prob, "LR"))
        metrics_rows.append(_metrics(test_label, rf_prob, "RF"))

        auc_lr = roc_auc_score(test_label, lr_prob)
        auc_rf = roc_auc_score(test_label, rf_prob)
        print(f"{task_name}: LR AUC {auc_lr:.4f}  RF AUC {auc_rf:.4f}  n={len(stay_ids_task)}")

    # Build unified dataframe: one row per stay
    rows = []
    for stay_id in sorted(all_stays):
        info = static_lookup.get(stay_id, {})
        row = {
            "stay_id": stay_id,
            "subject_id": info.get("subject_id"),
            "hadm_id": info.get("hadm_id"),
            "age": info.get("age"),
            "gender": info.get("gender"),
            "race": info.get("race"),
            "admission_type": info.get("admission_type"),
            "los_icu": info.get("los_icu"),
        }
        for task_name in task_data:
            stay_ids_t, gt, lr_p, rf_p = task_data[task_name]
            if stay_id in stay_ids_t:
                idx = stay_ids_t.index(stay_id)
                row[f"{task_name}_gt"] = int(gt[idx])
                row[f"{task_name}_LR_prob"] = float(lr_p[idx])
                row[f"{task_name}_RF_prob"] = float(rf_p[idx])
            else:
                row[f"{task_name}_gt"] = np.nan
                row[f"{task_name}_LR_prob"] = np.nan
                row[f"{task_name}_RF_prob"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path} ({len(df)} rows)")

    # Save metrics (precision, recall, log_loss) per task per model
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # Save test population summary (class balance per DV)
    population_df = pd.DataFrame(population_rows)
    population_df.to_csv(population_path, index=False)
    print(f"Saved population summary: {population_path}")

    # Data dictionary
    dict_content = """# Test Predictions Data Dictionary

Generated by `export_predictions.py`. One row per ICU stay that appears in at least one task's test set.

## ID Columns

| Column | Type | Description |
|--------|------|-------------|
| stay_id | int | ICU stay identifier (primary key) |
| subject_id | int | Patient identifier (MIMIC) |
| hadm_id | int | Hospital admission identifier (MIMIC) |
| age | float | Patient age at admission |
| gender | str | Patient gender |
| race | str | Patient race/ethnicity |

## Task Columns (per task)

For each task, three columns:

| Column pattern | Type | Description |
|----------------|------|-------------|
| {task}_gt | int | Ground truth (0 or 1). NaN if stay not in this task's test set. |
| {task}_LR_prob | float | Logistic Regression predicted probability of positive class. NaN if not in test set. |
| {task}_RF_prob | float | Random Forest predicted probability of positive class. NaN if not in test set. |

## Tasks

- **hosp_mort_24h_gap4h**: In-hospital mortality, using first 24h of data, 4h gap
- **ARF_2h_gap4h**: Acute respiratory failure (4h gap), 2h observation window
- **ARF_6h_gap4h**: Acute respiratory failure (4h gap), 6h observation window
- **shock_2h_gap4h**: Shock (vasopressor use, 4h gap), 2h observation window
- **shock_6h_gap4h**: Shock (vasopressor use, 4h gap), 6h observation window

## Derived Predictions

For precision/recall: threshold probabilities at 0.5 to get binary predictions:
  `pred = (prob >= 0.5).astype(int)`

## Data Leakage

Test set is the held-out 20% from the 70/10/20 split (seed 41). No training or validation data included.

## Related Files

- **test_metrics.csv**: Aggregate metrics (AUC, AP, precision, recall, log_loss, tp, fp, tn, fn) per task per model. Precision/recall use threshold 0.5. Confusion matrix: tp=true positive, fp=false positive, tn=true negative, fn=false negative.
- **test_population_summary.csv**: Test set class balance per task. Columns: task, n_total, n_positive, n_negative, pct_positive.
"""
    with open(dict_path, "w") as f:
        f.write(dict_content)
    print(f"Saved data dictionary: {dict_path}")

    return 0


if __name__ == "__main__":
    exit(main())
