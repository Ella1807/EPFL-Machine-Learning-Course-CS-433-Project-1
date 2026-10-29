# CS-433 Machine Learning - Project 1
## Prediction of cardiovascular disease (MICHD) from BRFSS survey data

This repository contains our solution for EPFL's CS-433 Machine Learning Project 1 (Fall 2026). The task is to predict whether a respondent has myocardial infarction or coronary heart disease (MICHD), using health and lifestyle variables from the 2015 Behavioral Risk Factor Surveillance System (BRFSS) dataset.

The project includes exploratory data analysis, BRFSS-specific preprocessing, six machine learning algorithms implemented from scratch using NumPy, cross-validation, model comparison, an ablation study, an overfitting/underfitting analysis, and generation of predictions for the AIcrowd competition.

## 1. Requirements

- Python 3
- NumPy (for the prediction pipeline)
- Jupyter Notebook, Matplotlib, and Seaborn (optional, for exploratory analysis and figures only)

Install the main dependency:

```bash
python -m pip install numpy
```

To run the exploratory notebook as well:

```bash
python -m pip install notebook matplotlib seaborn
```

The training and prediction pipeline uses only Python's standard library and NumPy. Visualization libraries are used only for plots in `eda.ipynb`. No external machine learning libraries (such as scikit-learn) or Pandas are used, in accordance with the Project 1 rules.

## 2. Dataset and directory structure

Download the competition data from [AIcrowd - EPFL Machine Learning Project 1](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

Place the three CSV files in a directory named `dataset` at the repository root, keeping their original names:

```text
project1/
|-- dataset/
|   |-- x_train.csv
|   |-- y_train.csv
|   `-- x_test.csv
|-- eda.ipynb
|-- evaluation.py
|-- helpers.py
|-- implementations.py
|-- preprocessing.py
|-- run.py
|-- special_codes.py
`-- README.md
```

The labels in `y_train.csv` are `+1` (MICHD) and `-1` (no MICHD). The data-loading helper excludes the respondent ID column from the feature matrices and retains test IDs for submission.

### File descriptions

| File | Purpose |
| --- | --- |
| `eda.ipynb` | Explore class balance, missingness, feature distributions, associations, redundancy, and train/test differences; produce figures. |
| `helpers.py` | Load the three CSV files and write an AIcrowd-compatible submission. |
| `special_codes.py` | Recode BRFSS-specific responses and convert encoded height/weight measurements. |
| `preprocessing.py` | Fit feature filtering, mean imputation and standardization on training data; apply the same fitted transformations to other data. |
| `implementations.py` | Implement the six required ML methods using NumPy. |
| `evaluation.py` | Create stratified folds, prepare each fold, fit models, predict labels and calculate evaluation metrics. |
| `run.py` | Compare model configurations, select the best by cross-validation, refit on all training data and generate `submission.csv`. |

## 3. Preprocessing

The pipeline in `preprocessing.py` follows these steps, in order:

1. **BRFSS-specific recoding:** `special_codes.py` maps response codes such as "don't know" or "refused" to missing values (`NaN`), and applicable "none" codes to zero. The code also converts `WEIGHT2` to kilograms and `HEIGHT3` to centimeters.
2. **Prevent target leakage:** remove the target-defining or potentially leaking variables `CVDINFR4`, `CVDCRHD4`, `_MICHD`, and `HAREHAB1` when present.
3. **Remove uninformative columns:** discard features with more than 90% missing values (after recoding) and features with zero observed variance.
4. **Impute missing values:** replace remaining `NaN` values by the mean of each retained feature, calculated on training data only.
5. **Standardize features:** subtract the training mean and divide by the training standard deviation for each retained feature.

All learned preprocessing parameters (selected columns, imputation values, means and standard deviations) are fitted on the training portion of a cross-validation fold, then applied unchanged to its validation portion. For the final model, they are fitted on the entire labeled training set and applied unchanged to `x_test.csv`. This avoids using validation or test data to estimate preprocessing parameters.

The EDA notebook investigates additional possible feature removals, including highly correlated variables. These are exploratory diagnostics: the final pipeline does **not** automatically remove correlated features or create additional combined features. The notebook's 80% missingness threshold is also exploratory; the executable pipeline uses **90%**.

## 4. Models and evaluation

The six required methods are implemented in `implementations.py`:

- Linear regression with gradient descent (`mean_squared_error_gd`)
- Linear regression with stochastic gradient descent, using one sample per update (`mean_squared_error_sgd`)
- Least-squares regression (`least_squares`)
- Ridge regression (`ridge_regression`)
- Logistic regression (`logistic_regression`)
- Regularized logistic regression (`reg_logistic_regression`)

### Cross-validation

`evaluation.py` performs **6-fold stratified cross-validation**, using random seed **42**. Preprocessing is fitted independently for each fold, preventing data leakage. Predictions are evaluated using **accuracy, precision, recall and F1 score**, with `+1` as the positive class. The script also reports the standard deviation of fold-level F1 scores.

Candidate models are ranked by **mean validation F1 score**, with mean accuracy used as a tie-breaker. This favors identifying MICHD cases rather than relying solely on accuracy in an imbalanced dataset.

The prediction rule is `+1` when `x @ w + bias >= 0`, and `-1` otherwise. The `bias` values below are prediction-time score offsets, not additional features learned by the six training algorithms. Logistic regression internally converts training labels from `{-1, +1}` to `{0, 1}`.

### Configurations evaluated by `run.py`

| Model | Learning rate (`gamma`) | Iterations | Regularization (`lambda_`) | Prediction bias |
| --- | ---: | ---: | ---: | ---: |
| Least squares | - | - | - | -0.20 |
| Ridge regression | - | - | 0.0001 | -0.25 |
| Gradient descent (MSE) | 0.01 | 500 | - | -0.25 |
| Stochastic gradient descent (MSE) | 0.0001 | 100000 | - | -0.30 |
| Logistic regression | 0.01 | 500 | - | -0.35 |
| Regularized logistic regression | 0.01 | 500 | 0.1 | -0.35 |

These are the currently specified candidate configurations: **one configuration per model**. Additional hyperparameter values can be evaluated by adding entries to the `MODEL_GRIDS` lists in `run.py`. Values were selected by earlier grid searches, and run.py re-evaluates the final candidates.

## 5. Ablation study

We performed an **ablation study** on the best-performing model, **ridge regression**, to measure how much individual choices contributed to its validation performance. Each variant changes one component of the final pipeline at a time. Results are reported as the mean accuracy and the mean +/- standard deviation of F1 across **6 stratified cross-validation folds**.

| Ridge regression variant | Accuracy | F1 score (mean +/- std) |
| --- | ---: | ---: |
| Always predict the negative class (baseline) | 0.9117 | 0.0000 |
| **Final pipeline** | **0.8646** | **0.4173 +/- 0.0076** |
| No decision-threshold shift (`bias = 0`) | 0.6657 | 0.3198 +/- 0.0029 |
| No special-code recoding | 0.8621 | 0.4077 +/- 0.0072 |
| No leakage-feature removal | 0.8646 | 0.4173 +/- 0.0076 |
| Missing-value threshold changed from 0.90 to 0.80 | 0.8647 | 0.4171 +/- 0.0073 |

The **decision-threshold shift** had the largest impact: removing it reduced the F1 score from **0.4173 to 0.3198**. Correctly recoding the special BRFSS response codes also improved F1 (by approximately **0.010**). Removing the potential leakage features or changing the missing-value threshold from 90% to 80% had little or no effect on these results. The always-negative baseline illustrates why accuracy alone is insufficient for this imbalanced task.

Because the prediction bias was selected using the same cross-validation folds used to report performance, the reported F1 estimates may be slightly optimistic. The full ablation results and discussion are provided in the project report.

## 6. Overfitting and underfitting analysis

We investigated potential **overfitting and underfitting** by generating a **learning curve for ridge regression**, the selected model. A diagnostic training split was sampled at **10%, 25%, 50%, 75% and 100%** of its size. At each fraction, we fitted the model and measured F1 on both the corresponding training subset and a **fixed validation set**.

The learning curve shows that training F1 generally decreases as more examples are added, while validation F1 increases slightly and then stabilizes. The gap between training and validation performance is relatively small at the largest training sizes, suggesting **no strong evidence of overfitting** in this diagnostic. The validation curve also approaches a plateau, indicating that simply increasing the training fraction yields limited additional improvement within the tested range.

The learning-curve plot is included in the project report (page 2). **These ablation and learning-curve experiments are additional diagnostics described in the report; the current `run.py` runs the final model comparison and submission pipeline, not the separate diagnostic experiments.**

## 7. Reproducing the predictions

From the repository root, after placing the three data files in `dataset/`, run:

```bash
python run.py
```

The script will:

1. Load the training labels and training/test features.
2. Construct six stratified cross-validation folds, fitting preprocessing separately within each fold.
3. Evaluate all listed model configurations and print their validation metrics and ranking.
4. Select the configuration with the highest mean validation F1 score.
5. Refit preprocessing and the selected model on the full training set.
6. Predict the competition test labels and write **`submission.csv`** in the repository root.

The output has the required two-column format:

```csv
Id,Prediction
123,1
456,-1
```

The two rows above illustrate the format; they are not actual predictions.

The code sets `SEED = 42` to make random splitting and stochastic training reproducible. Exact reproduction of a previously uploaded AIcrowd CSV must still be verified against that submission, using the same input data and project version.

## 8. Exploratory data analysis

To reproduce the exploratory analysis and visualizations, install the optional visualization dependencies and open:

```bash
jupyter notebook eda.ipynb
```

Execute the notebook cells in order. The notebook explores the class distribution, missing values and survey-specific response codes, feature types, feature-target associations, redundant features, and train/test distribution shifts. The figure-generation section can export plots into the `figures/` directory.

**Important:** the notebook supports exploration and reporting; the actual preprocessing and predictions are defined in `preprocessing.py` and `run.py`.

## 9. Submission and reproducibility notes

- Preserve the filenames `x_train.csv`, `y_train.csv`, and `x_test.csv` and the `dataset/` directory name.
- Run the scripts from the repository root so that local imports and relative paths work.
- Use the exact source code and candidate configuration associated with the team's final AIcrowd submission; verify that rerunning `run.py` reproduces the submitted CSV.
- The required six algorithm implementations can be imported directly from `implementations.py`.
- The scientific methodology, experiments and findings are discussed in the accompanying two-page project report.
