"""BRFSS/MICHD project: tuning, final training and submission in one script.

Pipeline:
  1. Load data and feature names.
  2. Split + preprocess once with stratified CV, tune every model's
     hyperparameters, rank the models by mean CV F1.
  3. Fit preprocessing on the FULL training set, train the winning model,
     predict the test set and write the submission CSV.

Run with:  python run.py
"""

import csv

import numpy as np

from helpers import load_csv_data, create_csv_submission
from preprocessing import fit_preprocessor, transform_data
from evaluation import (
    evaluate_model,
    prepare_folds,
    score_key,
    calculate_metrics,
    _fit_model,
    _predict,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "dataset"
OUTPUT_PATH = "submission.csv"
N_FOLDS = 5
SEED = 42
MISSING_THRESHOLD = 0.80

MODEL_GRIDS = {

     "least_squares": [
        {"bias": -0.2},
    ],

    "ridge_regression": [
        {"lambda_": 0.0001, "bias": -0.25},
    ],

   "mean_squared_error_gd": [
      {"gamma": 0.01, "max_iters": 500, "bias": -0.25},
    ],

   "mean_squared_error_sgd": [
    {"gamma": 0.0001, "max_iters": 100000, "bias": -0.30},
    ],

       "logistic_regression": [
        {"gamma": 0.01, "max_iters": 500, "bias": -0.35}
        ],

    "reg_logistic_regression": [
        {"lambda_": 0.1, "gamma": 0.01, "max_iters": 500, "bias": -0.35}
    ],

}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_feature_names(data_dir):
    """Read feature names from the x_train.csv header."""
    with open(f"{data_dir}/x_train.csv", newline="") as file:
        header = next(csv.reader(file))

    # The first column contains respondent IDs, not a feature.
    return header[1:]


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune_model(model_name, grid, prepared_folds):
    """Evaluate all configurations of one model and return the best."""
    best_result = None

    print(f"\n{'=' * 85}")
    print(f"Model: {model_name}")
    print(f"Configurations to evaluate: {len(grid)}")
    print(f"{'=' * 85}")

    for params in grid:
        result = evaluate_model(
            model_name=model_name,
            prepared_folds=prepared_folds,
            params=params,
            seed=SEED,
        )
        metrics = result["mean_metrics"]

        print(
            f"Params: {params} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        if (
            best_result is None
            or score_key(metrics) > score_key(best_result["mean_metrics"])
        ):
            best_result = {
                "model": model_name,
                "params": params,
                "mean_metrics": metrics,
                "std_metrics": result["std_metrics"],
                "fold_metrics": result["fold_metrics"],
            }

    print(
        f"\nBest for {model_name}: {best_result['params']} "
        f"(F1 {best_result['mean_metrics']['f1']:.4f}, "
        f"accuracy {best_result['mean_metrics']['accuracy']:.4f})"
    )

    return best_result


def run_tuning(x_train, y_train, feature_names):
    """Tune all models; return (winner_name, winner_params)."""
    # Split and preprocess once; every model/config reuses these folds.
    prepared_folds = prepare_folds(
        x_train,
        y_train,
        feature_names,
        n_folds=N_FOLDS,
        seed=SEED,
        missing_threshold=MISSING_THRESHOLD,
    )
    counts = [fold["n_features"] for fold in prepared_folds]
    print(
        f"Retained features per fold: {counts} "
        f"(mean {sum(counts) / len(counts):.1f})"
    )

    best_results = {
        name: tune_model(name, grid, prepared_folds)
        for name, grid in MODEL_GRIDS.items()
    }

    ranked = sorted(
        best_results.items(),
        key=lambda item: score_key(item[1]["mean_metrics"]),
        reverse=True,
    )

    header = (
        f"{'Model':<30}"
        f"{'Best parameters':<35}"
        f"{'Accuracy':>10}"
        f"{'Precision':>11}"
        f"{'Recall':>9}"
        f"{'F1':>9}"
        f"{'F1 std':>9}"
    )
    print("\n\nFINAL MODEL RANKING")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for name, result in ranked:
        m = result["mean_metrics"]
        print(
            f"{name:<30}"
            f"{str(result['params']):<35}"
            f"{m['accuracy']:>10.4f}"
            f"{m['precision']:>11.4f}"
            f"{m['recall']:>9.4f}"
            f"{m['f1']:>9.4f}"
            f"{result['std_metrics']['f1']:>9.4f}"
        )

    print("\nFold-level F1 scores for the winning configurations:")
    for name, result in ranked:
        scores = [round(f["f1"], 4) for f in result["fold_metrics"]]
        print(f"{name}: {scores}")

    winner_name, winner = ranked[0]
    print(f"\nBest model: {winner_name}")
    print(f"Best parameters: {winner['params']}")
    print(f"Best mean CV F1: {winner['mean_metrics']['f1']:.4f}")

    return winner_name, winner["params"]


# ---------------------------------------------------------------------------
# Final training and submission
# ---------------------------------------------------------------------------
def train_and_submit(model_name, params, x_train, y_train, x_test,
                     test_ids, feature_names):
    """Train on the full training set and write the submission CSV."""
    print(f"\nTraining final model: {model_name} {params}")

    # Fit preprocessing on the full training set only, then apply the same
    # fitted parameters to train and test.
    prep_params = fit_preprocessor(
        x_train, feature_names, missing_threshold=MISSING_THRESHOLD
    )
    x_train_p = transform_data(x_train, feature_names, prep_params)
    x_test_p = transform_data(x_test, feature_names, prep_params)
    print(f"Features after preprocessing: {x_train_p.shape[1]}")

    w = _fit_model(
        model_name=model_name,
        y_train=y_train,
        x_train=x_train_p,
        params=params,
        seed=SEED,
    )

    # Sanity check only: training metrics are optimistic, use CV instead.
    train_metrics = calculate_metrics(y_train, _predict(x_train_p, w))
    print(
        "Training metrics:",
        {k: round(v, 4) for k, v in train_metrics.items()},
    )

    bias = params.get("bias", 0.0)
    # bias = params.get("bias", 0.0) if model_name == "least_squares" else 0.0
    y_pred = _predict(x_test_p, w, bias)
    print(f"Predicted positive rate on test: {(y_pred == 1).mean():.4f}")

    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(f"Wrote {len(y_pred)} predictions to {OUTPUT_PATH}")


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_DIR)
    feature_names = load_feature_names(DATA_DIR)

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Original number of features: {x_train.shape[1]}")
    print(f"Positive target rate: {(y_train == 1).mean():.4f}")

    model_name, params = run_tuning(x_train, y_train, feature_names)

    train_and_submit(
        model_name, params, x_train, y_train, x_test, test_ids,
        feature_names,
    )


if __name__ == "__main__":
    main()