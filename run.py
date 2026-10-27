
"""Run cross-validation experiments for the BRFSS/MICHD project."""

import csv

from helpers import load_csv_data
from evaluation import evaluate_models, METRIC_NAMES


DATA_DIR = "dataset"


def load_feature_names(data_dir):
    """Read feature names from the x_train.csv header."""
    with open(f"{data_dir}/x_train.csv", newline="") as file:
        header = next(csv.reader(file))

    # The first column contains respondent IDs, not a feature.
    return header[1:]


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        DATA_DIR
    )

    feature_names = load_feature_names(DATA_DIR)

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Original number of features: {x_train.shape[1]}")
    print(f"Positive target rate: {(y_train == 1).mean():.4f}")

    results = evaluate_models(
        x=x_train,
        y=y_train,
        feature_names=feature_names,
        n_folds=5,
        max_iters=1000,
        gamma=0.01,
        lambda_=0.01,
        seed=42,
        missing_threshold=0.80,
    )

    # F1 is the primary ranking metric; accuracy breaks ties.
    ranked_models = sorted(
        results.items(),
        key=lambda item: (
            item[1]["mean_metrics"]["f1"],
            item[1]["mean_metrics"]["accuracy"],
        ),
        reverse=True,
    )

    print("\nCross-validation results")
    print("-" * 78)
    print(
        f"{'Model':<30}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>10}"
    )
    print("-" * 78)

    for model_name, result in ranked_models:
        metrics = result["mean_metrics"]

        print(
            f"{model_name:<30}"
            f"{metrics['accuracy']:>12.4f}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>10.4f}"
        )

    print("\nFold-level F1 scores:")
    for model_name, result in ranked_models:
        scores = [
            round(fold["f1"], 4)
            for fold in result["fold_metrics"]
        ]
        print(f"{model_name}: {scores}")

    print("\nMean number of retained features:")
    for model_name, result in ranked_models:
        counts = result["fold_feature_counts"]
        print(f"{model_name}: {sum(counts) / len(counts):.1f}")


if __name__ == "__main__":
    main()
