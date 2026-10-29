
"""Preprocessing for the BRFSS/MICHD project."""

import numpy as np

from special_codes import recode, convert_units


# Variables that define the target or may leak information about the outcome.
LEAKAGE_FEATURES = {
    "CVDINFR4",
    "CVDCRHD4",
    "_MICHD",
    "HAREHAB1",
}


def fit_preprocessor(x_train, feature_names, missing_threshold=0.80):
    """Learn feature selection, imputation, and scaling from training data.

    Args:
        x_train: Training feature matrix, shape (N, D).
        feature_names: Names of the D columns, excluding the ID column.
        missing_threshold: Remove features with a missing fraction
            strictly greater than this value.

    Returns:
        Dictionary containing all parameters needed by transform_data().
    """
    x_train = np.asarray(x_train, dtype=float)
    feature_names = list(feature_names)

    if x_train.ndim != 2:
        raise ValueError("x_train must be a 2D array.")

    if x_train.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match the number of columns.")

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique.")

    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must be between 0 and 1.")

    # Apply deterministic BRFSS-specific recoding and unit conversion.
    name2idx = {name: j for j, name in enumerate(feature_names)}
    x = recode(x_train, name2idx)
    x = convert_units(x, name2idx)

    # Remove target-defining and potential leakage features.
    eligible_indices = np.array(
        [
            j for j, name in enumerate(feature_names)
            if name not in LEAKAGE_FEATURES
        ],
        dtype=int,
    )

    x = x[:, eligible_indices]
    eligible_names = [feature_names[j] for j in eligible_indices]

    # Remove features with too many missing values.
    missing_fraction = np.mean(np.isnan(x), axis=0)
    keep = missing_fraction <= missing_threshold

    x = x[:, keep]
    kept_indices = eligible_indices[keep]
    kept_names = [
        name for name, selected in zip(eligible_names, keep) if selected
    ]

    if x.shape[1] == 0:
        raise ValueError("No features remain after missing-value filtering.")

    # Remove constant features, ignoring missing values.
    observed_std = np.zeros(x.shape[1])

    for j in range(x.shape[1]):
        observed = x[:, j][~np.isnan(x[:, j])]
        if observed.size > 0:
            observed_std[j] = np.std(observed)

    keep = observed_std > 0

    x = x[:, keep]
    kept_indices = kept_indices[keep]
    kept_names = [
        name for name, selected in zip(kept_names, keep) if selected
    ]

    if x.shape[1] == 0:
        raise ValueError("No features remain after constant-feature removal.")

    # Learn imputation values from training data only.
    imputation_values = np.nanmean(x, axis=0)

    missing_rows, missing_cols = np.where(np.isnan(x))
    x[missing_rows, missing_cols] = imputation_values[missing_cols]

    # Learn standardization parameters from the imputed training data.
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)

    # Avoid division by zero.
    stds[stds == 0] = 1.0

    return {
        "feature_names": feature_names,
        "kept_indices": kept_indices,
        "kept_feature_names": kept_names,
        "imputation_values": imputation_values,
        "means": means,
        "stds": stds,
        "missing_threshold": missing_threshold,
    }


def transform_data(x, feature_names, params):
    """Apply fitted preprocessing parameters to new data.

    The input may be a validation fold or the competition test set.
    No statistics are learned from this input.
    """
    x = np.asarray(x, dtype=float)
    feature_names = list(feature_names)

    if x.ndim != 2:
        raise ValueError("x must be a 2D array.")

    if feature_names != params["feature_names"]:
        raise ValueError("Feature names or column order do not match fit().")

    if x.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match the number of columns.")

    # Apply the same deterministic BRFSS transformations.
    name2idx = {name: j for j, name in enumerate(feature_names)}
    x = recode(x, name2idx)
    x = convert_units(x, name2idx)

    # Keep exactly the features selected during fitting.
    x = x[:, params["kept_indices"]].copy()

    # Impute using training-data means.
    missing_rows, missing_cols = np.where(np.isnan(x))
    x[missing_rows, missing_cols] = params["imputation_values"][missing_cols]

    # Standardize using training-data statistics.
    x = (x - params["means"]) / params["stds"]

    return x
