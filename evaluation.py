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


def score_key(metrics):
    """Ranking key used everywhere: mean F1, then mean accuracy."""
    return (metrics["f1"], metrics["accuracy"])


def stratified_k_fold(y, n_folds=5, seed=42):
    """Create stratified train/validation index splits."""
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

    folds = []

    for validation_indices in fold_indices:
        validation_indices = np.asarray(validation_indices, dtype=int)

        mask = np.ones(len(y), dtype=bool)
        mask[validation_indices] = False
        training_indices = np.flatnonzero(mask)

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


def prepare_folds(
    x,
    y,
    feature_names,
    n_folds=5,
    seed=42,
    missing_threshold=0.80,
    dtype=np.float64,
):
    """Split and preprocess once; reuse for every hyperparameter config.

    Preprocessing is fit on each fold's training part only, so there is no
    leakage. It does not depend on model hyperparameters, so the result can
    be shared by all models and all configurations.

    Memory note: each fold holds the whole preprocessed dataset (train +
    validation), so the cache is about n_folds times the size of the
    preprocessed matrix. Pass dtype=np.float32 to halve it.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y).ravel()
    feature_names = list(feature_names)

    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional array.")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of samples.")

    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the columns of x.")

    prepared = []

    for train_idx, val_idx in stratified_k_fold(y, n_folds, seed):
        # Fit preprocessing only on this fold's training data.
        params = fit_preprocessor(
            x[train_idx],
            feature_names,
            missing_threshold=missing_threshold,
        )

        x_tr = transform_data(x[train_idx], feature_names, params)
        x_val = transform_data(x[val_idx], feature_names, params)

        prepared.append(
            {
                "x_tr": x_tr.astype(dtype, copy=False),
                "y_tr": y[train_idx],
                "x_val": x_val.astype(dtype, copy=False),
                "y_val": y[val_idx],
                "val_idx": val_idx,
                "n_features": x_tr.shape[1],
            }
        )

    return prepared


def _fit_model(model_name, y_train, x_train, params, seed):
    """Fit one model and return its weights.

    `params` is the hyperparameter dict of one grid entry; each model reads
    only the keys it needs (a missing key raises a KeyError).
    """
    initial_w = np.zeros(x_train.shape[1])

    if model_name == "mean_squared_error_gd":
        w, _ = mean_squared_error_gd(
            y_train, x_train, initial_w,
            params["max_iters"], params["gamma"],
        )

    elif model_name == "mean_squared_error_sgd":
        np.random.seed(seed)
        w, _ = mean_squared_error_sgd(
            y_train, x_train, initial_w,
            params["max_iters"], params["gamma"],
        )

    elif model_name == "least_squares":
        w, _ = least_squares(y_train, x_train)

    elif model_name == "ridge_regression":
        w, _ = ridge_regression(y_train, x_train, params["lambda_"])

    elif model_name == "logistic_regression":
        y01 = (y_train == 1).astype(float)
        w, _ = logistic_regression(
            y01, x_train, initial_w,
            params["max_iters"], params["gamma"],
        )

    elif model_name == "reg_logistic_regression":
        y01 = (y_train == 1).astype(float)
        w, _ = reg_logistic_regression(
            y01, x_train, params["lambda_"], initial_w,
            params["max_iters"], params["gamma"],
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return w


def _predict(x, w):
    """Convert model scores into labels -1 and +1.

    For logistic regression the scores are logits, so a score of zero
    corresponds to probability 0.5.
    """
    return np.where(x @ w >= 0, 1, -1)


def evaluate_model(model_name, prepared_folds, params, seed=42):
    """Evaluate one model/configuration on already-prepared folds.

    Args:
        model_name: Name of the model to fit.
        prepared_folds: Output of prepare_folds().
        params: Hyperparameters for this configuration
            (keys among "gamma", "max_iters", "lambda_").
        seed: Base seed for stochastic methods (offset by fold number).
    """
    fold_metrics = []

    for fold_number, fold in enumerate(prepared_folds):
        w = _fit_model(
            model_name=model_name,
            y_train=fold["y_tr"],
            x_train=fold["x_tr"],
            params=params,
            seed=seed + fold_number,
        )

        predictions = _predict(fold["x_val"], w)
        fold_metrics.append(calculate_metrics(fold["y_val"], predictions))

    mean_metrics = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in METRIC_NAMES
    }

    std_metrics = {
        metric: float(np.std([fold[metric] for fold in fold_metrics]))
        for metric in METRIC_NAMES
    }

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }