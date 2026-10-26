
# ============================================
# CREDIT CARD DEFAULT PREDICTION
# Main analysis and training pipeline
# ============================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)


# ============================================
# 1. LOAD DATA AND BASIC EDA
# ============================================

df = pd.read_csv("Data/UCI_Credit_Card.csv")
target_column = "default.payment.next.month"

print("Original dataset shape:", df.shape)
print(df.head())
print("\nDataset information:")
df.info()

print("\nMissing values:", df.isnull().sum().sum())

print("\nTarget distribution:")
print(df[target_column].value_counts().sort_index())

sns.countplot(x=target_column, data=df)
plt.title("Default Payment Distribution")
plt.xlabel("Default Next Month (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    print(f"\n{col} distribution:")
    print(df[col].value_counts().sort_index())

print("\nNumerical summary:")
print(df[["LIMIT_BAL", "AGE"]].describe())


# ============================================
# 2. DATA CLEANING
# ============================================

# Customer ID is not a predictive feature.
df = df.drop(columns="ID")

# Check duplicate records without deleting them.
print("\nDuplicate rows after removing ID:", df.duplicated().sum())

# Pool rare/unknown categories for interpretability.
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

df = df.rename(columns={target_column: "default"})

for col in ["EDUCATION", "MARRIAGE"]:
    print(f"\n{col} values after pooling:")
    print(df[col].value_counts().sort_index())

# Investigate repeated predictor profiles.
predictor_cols = df.columns.drop("default")
duplicate_mask = df.duplicated(subset=predictor_cols, keep=False)
duplicate_profiles = df.loc[duplicate_mask]

print("\nRows with repeated predictor profiles:", len(duplicate_profiles))

if not duplicate_profiles.empty:
    target_counts = duplicate_profiles.groupby(
        list(predictor_cols), dropna=False
    )["default"].nunique()

    print("Repeated profiles with one target label:", (target_counts == 1).sum())
    print("Repeated profiles with conflicting targets:", (target_counts > 1).sum())

print("Repeated rows across non-ID columns:", df.duplicated().sum())


# ============================================
# 3. FEATURES, TARGET AND DATA SPLIT
# ============================================

X = df.drop(columns="default")
y = df["default"]

print("\nPredictor count:", X.shape[1])

# Stratified 70/15/15 split.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit sizes:")
for name, split in [
    ("Training", X_train),
    ("Validation", X_val),
    ("Final test", X_test)
]:
    print(f"{name}: {len(split)} customers")

print("\nDefault rates by split:")
for name, target in [
    ("Training", y_train),
    ("Validation", y_val),
    ("Final test", y_test)
]:
    print(f"{name}: {target.mean():.4f}")

# Do not use final test outcomes for model selection.


# ============================================
# 4. PREPROCESSING
# ============================================

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

numerical_features = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
     categorical_features)
])

# Fit preprocessing only on training data.
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

print("\nProcessed shapes:")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)


# ============================================
# 5. TRAIN LOGISTIC REGRESSION
# ============================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train_processed, y_train)

print("\nLogistic regression training complete.")


# ============================================
# 6. SELECT THRESHOLD ON VALIDATION DATA
# ============================================

val_probs = model.predict_proba(X_val_processed)[:, 1]
candidate_thresholds = np.arange(0.05, 0.951, 0.01)

threshold_results = [
    {
        "threshold": float(t),
        "f1": f1_score(
            y_val, (val_probs >= t).astype(int), zero_division=0
        )
    }
    for t in candidate_thresholds
]

# Select the threshold with the highest validation F1-score.
best_result = max(threshold_results, key=lambda result: result["f1"])
best_threshold = best_result["threshold"]
y_val_pred = (val_probs >= best_threshold).astype(int)

print("\nThreshold selected using validation data:")
print(f"Threshold: {best_threshold:.3f}")
print(f"Validation F1: {best_result['f1']:.3f}")


# ============================================
# 7. VALIDATION METRICS
# ============================================

metrics = {
    "Accuracy": accuracy_score(y_val, y_val_pred),
    "Precision": precision_score(y_val, y_val_pred, zero_division=0),
    "Recall": recall_score(y_val, y_val_pred, zero_division=0),
    "F1-score": f1_score(y_val, y_val_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_val, val_probs),
    "Average Precision": average_precision_score(y_val, val_probs)
}

print("\nVALIDATION RESULTS")
print("------------------")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")


# ============================================
# 8. VALIDATION CONFUSION MATRIX
# ============================================

ConfusionMatrixDisplay(
    confusion_matrix(y_val, y_val_pred),
    display_labels=["No Default", "Default"]
).plot()

plt.title("Validation Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================
# 9. ROC AND PRECISION-RECALL CURVES
# ============================================

fpr, tpr, _ = roc_curve(y_val, val_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {metrics['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Validation ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

pr_precision, pr_recall, _ = precision_recall_curve(y_val, val_probs)
baseline = y_val.mean()

plt.figure(figsize=(8, 6))
plt.plot(
    pr_recall, pr_precision,
    label=f"Logistic Regression (AP = {metrics['Average Precision']:.3f})"
)
plt.axhline(
    y=baseline, color="black", linestyle="--",
    label=f"Baseline = {baseline:.3f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# 10. COEFFICIENT ANALYSIS
# ============================================

feature_names = preprocessor.get_feature_names_out()

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0]
})

top_positive = coef_df.nlargest(10, "Coefficient")
top_negative = coef_df.nsmallest(10, "Coefficient")

print("\nLargest positive logistic regression coefficients:")
print(top_positive.to_string(index=False))

print("\nLargest negative logistic regression coefficients:")
print(top_negative.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, data, title in [
    (axes[0], top_positive, "Largest Positive Coefficients"),
    (axes[1], top_negative, "Largest Negative Coefficients")
]:
    data = data.sort_values("Coefficient")
    ax.barh(data["Feature"], data["Coefficient"])
    ax.set_title(title)
    ax.set_xlabel("Coefficient (log-odds scale)")

plt.tight_layout()
plt.show()


# ============================================
# 11. SAVE MODEL ARTIFACTS
# ============================================

artifacts = {
    "model": model,
    "preprocessor": preprocessor,
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test,
    "best_threshold": best_threshold,
    "feature_names": feature_names,
    "category_pooling": {
        "EDUCATION": {"pooled_values": [0, 4, 5, 6], "replacement_code": 4},
        "MARRIAGE": {"pooled_values": [0, 3], "replacement_code": 3}
    },
    "split_strategy": {
        "training": 0.70,
        "validation": 0.15,
        "test": 0.15,
        "random_state": 42
    }
}

joblib.dump(artifacts, "model_artifacts.joblib")

print("\nFresh model artifacts saved.")
print("Final test metrics have NOT been calculated.")
