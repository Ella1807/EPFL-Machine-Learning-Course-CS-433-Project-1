"""Ablation study for the report: final ridge pipeline, one change at a time.

Run with:  python ablation.py
Each variant re-does the 6-fold preprocessing, so allow a few minutes.
"""

import numpy as np

import preprocessing
from helpers import load_csv_data
from run import load_feature_names, DATA_DIR, N_FOLDS, SEED
from evaluation import prepare_folds, evaluate_model, calculate_metrics

# Final configuration (the one submitted).
RIDGE = {"lambda_": 0.0001}
FINAL_BIAS = -0.25
FINAL_MISSING = 0.90


def run_variant(name, x, y, names, missing=FINAL_MISSING, bias=FINAL_BIAS,
                recode=True, leakage_removal=True):
    """Evaluate ridge with one pipeline component switched off."""
    original = (
        preprocessing.recode,
        preprocessing.convert_units,
        preprocessing.LEAKAGE_FEATURES,
    )

    if not recode:
        preprocessing.recode = lambda x_, name2idx: x_.copy()
        preprocessing.convert_units = lambda x_, name2idx: x_.copy()
    if not leakage_removal:
        preprocessing.LEAKAGE_FEATURES = set()

    try:
        folds = prepare_folds(
            x, y, names,
            n_folds=N_FOLDS, seed=SEED, missing_threshold=missing,
        )
        result = evaluate_model(
            "ridge_regression", folds, {**RIDGE, "bias": bias}, seed=SEED
        )
        n_features = np.mean([f["n_features"] for f in folds])
    finally:
        # Always restore the real preprocessing.
        (preprocessing.recode,
         preprocessing.convert_units,
         preprocessing.LEAKAGE_FEATURES) = original

    m, s = result["mean_metrics"], result["std_metrics"]
    print(
        f"{name:<34}{n_features:>8.1f}"
        f"{m['accuracy']:>10.4f}{m['precision']:>11.4f}{m['recall']:>9.4f}"
        f"{m['f1']:>9.4f} +/- {s['f1']:.4f}",
        flush=True,
    )


def main():
    x, _, y, _, _ = load_csv_data(DATA_DIR)
    names = load_feature_names(DATA_DIR)

    print(
        f"{'Variant':<34}{'#feat':>8}{'Acc':>10}{'Prec':>11}"
        f"{'Recall':>9}{'F1':>9}"
    )
    print("-" * 95)

    # Baseline: always predict "no MICHD".
    base = calculate_metrics(y, -np.ones_like(y))
    print(
        f"{'Always negative':<34}{'-':>8}{base['accuracy']:>10.4f}"
        f"{base['precision']:>11.4f}{base['recall']:>9.4f}{base['f1']:>9.4f}"
    )

    run_variant("Full pipeline (final)", x, y, names)
    run_variant("No threshold shift (bias = 0)", x, y, names, bias=0.0)
    run_variant("No special-code recoding", x, y, names, recode=False)
    run_variant("No leakage-feature removal", x, y, names,
                leakage_removal=False)
    run_variant("Missing threshold 0.80", x, y, names, missing=0.80)


if __name__ == "__main__":
    main()