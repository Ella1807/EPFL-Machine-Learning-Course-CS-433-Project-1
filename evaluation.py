
"""Cross-validation and evaluation metrics for the BRFSS/MICHD project."""

import numpy as np

from preprocessing import fit_preprocessor, transform_data
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)


METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def stratified_k_fold(y, n_folds=5, seed=42):
    """Create stratified train/validation splits."""
    y = np.asarray(y).ravel()

    if n_folds < 2 or n_folds > len(y):
        raise ValueError("n_folds must be between 2 and the number of samples.")

    rng = np.random.default_rng(seed)
    fold_indices = [[] for _ in range(n_folds)]

    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        rng.shuffle(class_indices)

        for i, index in enumerate(class_indices):
            fold_indices[i % n_folds].append(index)

    all_indices = np.arange(len(y))
    folds = []

    for validation_indices in fold_indices:
        validation_indices = np.asarray(validation_indices, dtype=int)
        training_indices = np.setdiff1d(all_indices, validation_indices)

        folds.append((training_indices, validation_indices))

    return folds


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall and F1."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")

    # The positive class is +1; the negative class is -1.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _fit_model(
    model_name,
    y_train,
    x_train,
    max_iters,
    gamma,
    lambda_,
    seed,
):
    """Fit one model and return its weights."""
    n_features = x_train.shape[1]
    initial_w = np.zeros(n_features)

    if model_name == "mean_squared_error_gd":
        w, _ = mean_squared_error_gd(
            y_train, x_train, initial_w, max_iters, gamma
        )

    elif model_name == "mean_squared_error_sgd":
        np.random.seed(seed)
        w, _ = mean_squared_error_sgd(
            y_train, x_train, initial_w, max_iters, gamma
        )

    elif model_name == "least_squares":
        w, _ = least_squares(y_train, x_train)

    elif model_name == "ridge_regression":
        w, _ = ridge_regression(y_train, x_train, lambda_)

    elif model_name == "logistic_regression":
        y01 = (y_train == 1).astype(float)
        w, _ = logistic_regression(
            y01, x_train, initial_w, max_iters, gamma
        )

    elif model_name == "reg_logistic_regression":
        y01 = (y_train == 1).astype(float)
        w, _ = reg_logistic_regression(
            y01, x_train, lambda_, initial_w, max_iters, gamma
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return w


def _predict(model_name, x, w):
    """Convert model scores into labels -1 and +1."""
    scores = x @ w

    # For logistic regression, scores are logits.
    # A score of zero corresponds to probability 0.5.
    return np.where(scores >= 0, 1, -1)


def evaluate_model(
    model_name,
    x,
    y,
    feature_names,
    n_folds=5,
    max_iters=1000,
    gamma=0.01,
    lambda_=0.01,
    seed=42,
    missing_threshold=0.80,
):
    """Evaluate a model using stratified cross-validation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    feature_names = list(feature_names)

    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples.")

    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the columns of x.")

    folds = stratified_k_fold(y, n_folds=n_folds, seed=seed)

    fold_metrics = []
    out_of_fold_predictions = np.empty(len(y), dtype=int)
    fold_feature_counts = []

    for fold_number, (train_idx, val_idx) in enumerate(folds):
        x_fold_train = x[train_idx]
        y_fold_train = y[train_idx]

        x_fold_val = x[val_idx]
        y_fold_val = y[val_idx]

        # Fit preprocessing only on this fold's training data.
        params = fit_preprocessor(
            x_fold_train,
            feature_names,
            missing_threshold=missing_threshold,
        )

        x_fold_train = transform_data(
            x_fold_train, feature_names, params
        )
        x_fold_val = transform_data(
            x_fold_val, feature_names, params
        )

        fold_feature_counts.append(x_fold_train.shape[1])

        w = _fit_model(
            model_name=model_name,
            y_train=y_fold_train,
            x_train=x_fold_train,
            max_iters=max_iters,
            gamma=gamma,
            lambda_=lambda_,
            seed=seed + fold_number,
        )

        predictions = _predict(model_name, x_fold_val, w)
        out_of_fold_predictions[val_idx] = predictions

        fold_metrics.append(
            calculate_metrics(y_fold_val, predictions)
        )

    mean_metrics = {
        metric: float(
            np.mean([fold[metric] for fold in fold_metrics])
        )
        for metric in METRIC_NAMES
    }

    std_metrics = {
        metric: float(
            np.std([fold[metric] for fold in fold_metrics])
        )
        for metric in METRIC_NAMES
    }

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "out_of_fold_predictions": out_of_fold_predictions,
        "fold_feature_counts": fold_feature_counts,
    }


def evaluate_models(
    x,
    y,
    feature_names,
    model_names=None,
    n_folds=5,
    max_iters=1000,
    gamma=0.01,
    lambda_=0.01,
    seed=42,
    missing_threshold=0.80,
):
    """Evaluate multiple models using the same cross-validation splits."""
    if model_names is None:
        model_names = [
            "least_squares",
            "ridge_regression",
            "mean_squared_error_gd",
            "mean_squared_error_sgd",
            "logistic_regression",
            "reg_logistic_regression",
        ]

    results = {}

    for model_name in model_names:
        print(f"Evaluating {model_name}...")

        results[model_name] = evaluate_model(
            model_name=model_name,
            x=x,
            y=y,
            feature_names=feature_names,
            n_folds=n_folds,
            max_iters=max_iters,
            gamma=gamma,
            lambda_=lambda_,
            seed=seed,
            missing_threshold=missing_threshold,
        )

    return results
