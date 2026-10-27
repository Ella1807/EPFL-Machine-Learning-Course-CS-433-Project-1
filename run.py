import numpy as np
from helpers import load_csv_data

# --------------------------------------------------
# 1. Load the data
# --------------------------------------------------
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")

print("Training features:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test features:", x_test.shape)

# --------------------------------------------------
# 2. Create a local validation split
# --------------------------------------------------
# Keep 80% for training and 20% for validation.
# Shuffle the indices reproducibly.
rng = np.random.default_rng(42)
indices = rng.permutation(len(y_train))

split = int(0.8 * len(y_train))
train_idx = indices[:split]
val_idx = indices[split:]

X_train = x_train[train_idx]
y_train_split = y_train[train_idx]

X_val = x_train[val_idx]
y_val = y_train[val_idx]

print("\nTraining split:", X_train.shape)
print("Validation split:", X_val.shape)

# --------------------------------------------------
# 3. Baseline: always predict the majority class
# --------------------------------------------------
majority_label = -1

y_val_pred = np.full(len(y_val), majority_label)

#how often are all predictions correct
accuracy = np.mean(y_val_pred == y_val)

# Positive class is +1
true_positive = np.sum((y_val_pred == 1) & (y_val == 1))
false_positive = np.sum((y_val_pred == 1) & (y_val == -1))
false_negative = np.sum((y_val_pred == -1) & (y_val == 1))

#how reliable are the positive predictions
precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0 else 0.0
)

#how many actual positive cases do we find
recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative > 0 else 0.0
)

#how to balance precision and recall
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0.0
)

print("\nMajority-class baseline")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
