
# import numpy as np
# from helpers import load_csv_data

# x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")


# import numpy as np
# from helpers import load_csv_data

# # 1. Load the data
# x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")

# print("=== Dataset overview ===")
# print("Training features:", x_train.shape)
# print("Training labels:", y_train.shape)
# print("Test features:", x_test.shape)

# # 2. Understand the target distribution
# labels, counts = np.unique(y_train, return_counts=True)

# print("\n=== Target distribution ===")
# for label, count in zip(labels, counts):
#     percentage = 100 * count / len(y_train)
#     print(f"Label {label}: {count:,} ({percentage:.2f}%)")

# # 3. Establish a simple baseline
# # Predict the most frequent training label for everyone.
# majority_label = labels[np.argmax(counts)]
# baseline_predictions = np.full(len(y_train), majority_label)

# baseline_accuracy = np.mean(baseline_predictions == y_train)

# print("\n=== Majority-class baseline ===")
# print("Predicted label:", majority_label)
# print(f"Training accuracy: {baseline_accuracy:.2%}")

# # 4. Check missing values
# print("\n=== Missing values ===")
# print("Training NaNs:", np.isnan(x_train).sum())
# print("Test NaNs:", np.isnan(x_test).sum())
# print("Features:", x_train.shape[1])


# import numpy as np
# from helpers import load_csv_data

# print("Loading datasets...")

# x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")

# print("\nAll datasets loaded successfully!")
# print("Training features:", x_train.shape)
# print("Training labels:", y_train.shape)
# print("Test features:", x_test.shape)

# print("\nTarget distribution:")
# labels, counts = np.unique(y_train, return_counts=True)

# for label, count in zip(labels, counts):
#     print(f"Label {label}: {count} ({count / len(y_train) * 100:.2f}%)")




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

accuracy = np.mean(y_val_pred == y_val)

# Positive class is +1
true_positive = np.sum((y_val_pred == 1) & (y_val == 1))
false_positive = np.sum((y_val_pred == 1) & (y_val == -1))
false_negative = np.sum((y_val_pred == -1) & (y_val == 1))

precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0 else 0.0
)

recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative > 0 else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0 else 0.0
)

print("\nMajority-class baseline")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")






# def main():
#     # 1. Load the dataset
#     x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")

#     # 2. Inspect the data
#     print("Training features:", x_train.shape)
#     print("Training labels:", y_train.shape)
#     print("Test features:", x_test.shape)

#     print("\nTraining IDs:", train_ids.shape)
#     print("Test IDs:", test_ids.shape)

#     print("\nUnique labels and counts:")
#     print(np.unique(y_train, return_counts=True))

#     print("\nNaN values in training data:", np.isnan(x_train).sum())
#     print("NaN values in test data:", np.isnan(x_test).sum())


# if __name__ == "__main__":
#     main()


