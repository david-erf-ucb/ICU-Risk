"""
Compile MEEP parquet files to .npy format expected by METRE training scripts.

Converts MEEP_MIMIC_vital.parquet, MEEP_MIMIC_inv.parquet, MEEP_MIMIC_static.parquet
into the train_head/dev_head/test_head + static_train_filter/etc. structure.

Usage:
    python compile_meep_to_npy.py --input_dir ../output --output_path ../output/MIMIC_compile.npy
    python compile_meep_to_npy.py --input_dir ../output --output_path ../output/MIMIC_compile.npy --database eICU
"""
import argparse
import json
import os
import numpy as np
import pandas as pd


# Match extract_database split
SEED = 41
TRAIN_FRAC, DEV_FRAC, TEST_FRAC = 0.7, 0.1, 0.2

# Intervention column order (must match extract_database merge order)
INV_COLS = [
    'vent', 'antibiotic', 'dopamine', 'epinephrine', 'norepinephrine',
    'phenylephrine', 'vasopressin', 'dobutamine', 'milrinone', 'heparin',
    'crrt', 'rbc_trans', 'platelets_trans', 'ffp_trans', 'colloid_bolus', 'crystalloid_bolus'
]
# eICU uses different names; map to INV_COLS for consistent feature indices
EICU_TO_MIMIC_INV = {
    'antib': 'antibiotic', 'rbc': 'rbc_trans', 'platelets': 'platelets_trans',
    'ffp': 'ffp_trans', 'colloid': 'colloid_bolus', 'crystalloid': 'crystalloid_bolus',
}


def _load_mimic(input_dir):
    """Load MIMIC MEEP parquets."""
    vital = pd.read_parquet(os.path.join(input_dir, 'MEEP_MIMIC_vital.parquet'))
    inv = pd.read_parquet(os.path.join(input_dir, 'MEEP_MIMIC_inv.parquet'))
    static = pd.read_parquet(os.path.join(input_dir, 'MEEP_MIMIC_static.parquet'))
    return vital, inv, static


def _load_eicu(input_dir):
    """Load eICU MEEP parquets."""
    vital = pd.read_parquet(os.path.join(input_dir, 'MEEP_eICU_vital.parquet'))
    inv = pd.read_parquet(os.path.join(input_dir, 'MEEP_eICU_inv.parquet'))
    static = pd.read_parquet(os.path.join(input_dir, 'MEEP_eICU_static.parquet'))
    return vital, inv, static


def _get_stay_id_level(df, database):
    """Return the index level name for stay_id."""
    if database == 'MIMIC':
        return 2  # subject_id=0, hadm_id=1, stay_id=2
    else:
        return 0  # patientunitstayid


def _build_stay_arrays(vital, inv, database):
    """
    Build list of (n_features, n_hours) arrays, one per stay.
    Features = vital columns (184) + intervention columns (16) = 200.
    """
    stay_level_name = 'stay_id' if database == 'MIMIC' else 'patientunitstayid'

    if database == 'eICU':
        inv = inv.rename(columns=EICU_TO_MIMIC_INV)

    vital_cols = list(vital.columns)
    for c in INV_COLS:
        if c not in inv.columns:
            raise ValueError(f"Missing intervention column: {c}")

    head_list = []
    stay_order = []

    # Merge vital and inv on index for alignment
    merged = vital.merge(inv[INV_COLS], left_index=True, right_index=True, how='outer')

    for stay_id, group in merged.groupby(level=stay_level_name):
        # Reset index to get hours_in as column, sort, ensure one row per hour
        group = group.reset_index()
        if 'hours_in' in group.columns:
            group = group.sort_values('hours_in').drop_duplicates(subset=['hours_in'], keep='first')
        cols_need = vital_cols + INV_COLS
        group = group[[c for c in cols_need if c in group.columns]]
        group = group.fillna(0)

        v_arr = group[vital_cols].values.T.astype(np.float32)  # (184, n_hours)
        i_arr = group[INV_COLS].values.T.astype(np.float32)   # (16, n_hours)
        combined = np.vstack([v_arr, i_arr])  # (200, n_hours)
        head_list.append(combined)
        stay_order.append(stay_id)

    return head_list, stay_order


def _build_static_arrays(static, stay_order, database):
    """Build static_train_filter etc. - list of arrays with mort_hosp as first column."""
    stay_level_name = 'stay_id' if database == 'MIMIC' else 'patientunitstayid'
    mort_col = 'mort_hosp' if 'mort_hosp' in static.columns else 'hosp_mort'

    static_arrays = []
    for stay_id in stay_order:
        try:
            if isinstance(static.index, pd.MultiIndex) and stay_level_name in static.index.names:
                row = static.xs(stay_id, level=stay_level_name)
            else:
                row = static.loc[stay_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            val = row[mort_col] if mort_col in row.index else np.nan
            try:
                val = np.nan if pd.isna(val) else float(val)
            except (TypeError, ValueError):
                val = np.nan
            static_arrays.append(np.array([val], dtype=np.float32))
        except (KeyError, IndexError):
            static_arrays.append(np.array([np.nan], dtype=np.float32))

    return static_arrays


def _split_stays(stay_ids, stay_level_idx):
    """Same 70/10/20 split as extract_database (SEED=41)."""
    np.random.seed(SEED)
    perm = np.random.permutation(list(stay_ids))
    N = len(perm)
    n_train = int(TRAIN_FRAC * N)
    n_dev = int(DEV_FRAC * N)
    n_test = N - n_train - n_dev
    train_stay = set(perm[:n_train])
    dev_stay = set(perm[n_train:n_train + n_dev])
    test_stay = set(perm[n_train + n_dev:])
    return train_stay, dev_stay, test_stay


def compile_mimic(input_dir):
    """Compile MIMIC MEEP parquets to training format."""
    vital, inv, static = _load_mimic(input_dir)
    stay_level = _get_stay_id_level(vital, 'MIMIC')
    stay_ids = set(vital.index.get_level_values(stay_level).unique())

    train_stay, dev_stay, test_stay = _split_stays(stay_ids, stay_level)

    def select_stays(df, stays):
        return df[df.index.get_level_values(stay_level).isin(stays)]

    vital_train = select_stays(vital, train_stay)
    vital_dev = select_stays(vital, dev_stay)
    vital_test = select_stays(vital, test_stay)
    inv_train = select_stays(inv, train_stay)
    inv_dev = select_stays(inv, dev_stay)
    inv_test = select_stays(inv, test_stay)

    def build_split(v, i, s):
        head_list, order = _build_stay_arrays(v, i, 'MIMIC')
        static_list = _build_static_arrays(s, order, 'MIMIC')
        return head_list, static_list

    train_head, static_train_filter = build_split(vital_train, inv_train, static)
    dev_head, static_dev_filter = build_split(vital_dev, inv_dev, static)
    test_head, static_test_filter = build_split(vital_test, inv_test, static)

    return {
        'train_head': train_head,
        'dev_head': dev_head,
        'test_head': test_head,
        'static_train_filter': static_train_filter,
        'static_dev_filter': static_dev_filter,
        'static_test_filter': static_test_filter,
    }


def compile_eicu(input_dir):
    """Compile eICU MEEP parquets to training format."""
    vital, inv, static = _load_eicu(input_dir)
    stay_level = _get_stay_id_level(vital, 'eICU')
    stay_ids = set(vital.index.get_level_values(stay_level).unique())

    train_stay, dev_stay, test_stay = _split_stays(stay_ids, stay_level)

    def select_stays(df, stays):
        return df[df.index.get_level_values(stay_level).isin(stays)]

    vital_train = select_stays(vital, train_stay)
    vital_dev = select_stays(vital, dev_stay)
    vital_test = select_stays(vital, test_stay)
    inv_train = select_stays(inv, train_stay)
    inv_dev = select_stays(inv, dev_stay)
    inv_test = select_stays(inv, test_stay)

    def build_split(v, i, s):
        head_list, order = _build_stay_arrays(v, i, 'eICU')
        static_list = _build_static_arrays(s, order, 'eICU')
        return head_list, static_list

    train_head, static_train_filter = build_split(vital_train, inv_train, static)
    dev_head, static_dev_filter = build_split(vital_dev, inv_dev, static)
    test_head, static_test_filter = build_split(vital_test, inv_test, static)

    return {
        'train_head': train_head,
        'dev_head': dev_head,
        'test_head': test_head,
        'static_train_filter': static_train_filter,
        'static_dev_filter': static_dev_filter,
        'static_test_filter': static_test_filter,
    }


def main():
    parser = argparse.ArgumentParser(description="Compile MEEP parquet to .npy for METRE training")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing MEEP_*_vital.parquet etc.")
    parser.add_argument("--output_path", type=str, required=True, help="Output .npy file path")
    parser.add_argument("--database", type=str, default='MIMIC', choices=['MIMIC', 'eICU'])
    args = parser.parse_args()

    if args.database == 'MIMIC':
        data = compile_mimic(args.input_dir)
    else:
        data = compile_eicu(args.input_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or '.', exist_ok=True)
    np.save(args.output_path, data, allow_pickle=True)
    print(f"Saved {args.output_path}")
    print(f"  train: {len(data['train_head'])} stays, shapes e.g. {data['train_head'][0].shape}")
    print(f"  dev:   {len(data['dev_head'])} stays")
    print(f"  test:  {len(data['test_head'])} stays")


if __name__ == '__main__':
    main()
