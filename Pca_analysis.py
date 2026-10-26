# ============================================
# PCA ANALYSIS
# Credit Card Default Prediction
# ============================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.decomposition import PCA


# ============================================
# 1. LOAD SAVED ARTIFACTS
# ============================================

artifacts = joblib.load("model_artifacts.joblib")

preprocessor = artifacts["preprocessor"]
model = artifacts["model"]

X_train = artifacts["X_train"]
X_val = artifacts["X_val"]
y_val = artifacts["y_val"]

print("Training customers:", len(X_train))
print("Validation customers:", len(X_val))


# ============================================
# 2. PREPROCESS FEATURES
# ============================================

# The preprocessor was already fitted on training data.
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)


# Convert sparse matrices to dense arrays if necessary.
if sparse.issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()

if sparse.issparse(X_val_processed):
    X_val_processed = X_val_processed.toarray()


# ============================================
# 3. FIT PCA ON TRAINING DATA ONLY
# ============================================

pca = PCA()

X_train_pca = pca.fit_transform(X_train_processed)

# Project validation customers using the same PCA.
X_val_pca = pca.transform(X_val_processed)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)


# ============================================
# 4. EXPLAINED VARIANCE RESULTS
# ============================================

print(
    f"\nPC1 variance explained: "
    f"{explained_variance[0] * 100:.2f}%"
)

print(
    f"PC2 variance explained: "
    f"{explained_variance[1] * 100:.2f}%"
)

print(
    f"First two PCs combined: "
    f"{cumulative_variance[1] * 100:.2f}%"
)

components_90 = (
    np.searchsorted(cumulative_variance, 0.90) + 1
)

print(
    f"Components needed for 90% variance: "
    f"{components_90}"
)


# ============================================
# 5. SCREE PLOT
# ============================================

component_numbers = np.arange(
    1, len(explained_variance) + 1
)

plt.figure(figsize=(10, 5))

plt.bar(
    component_numbers,
    explained_variance * 100
)

plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title("PCA Scree Plot — Training Data")

plt.tight_layout()
plt.show()


# ============================================
# 6. PREDICT VALIDATION RISK GROUPS
# ============================================

# Get default probabilities for validation customers.
validation_probs = model.predict_proba(
    X_val_processed
)[:, 1]

# These cutoffs are defined for this class project.
# They are NOT official banking risk standards.
risk_groups = pd.cut(
    validation_probs,
    bins=[-np.inf, 0.30, 0.60, np.inf],
    labels=["Low", "Medium", "High"],
    right=False
)

risk_df = pd.DataFrame({
    "Actual_Default": np.asarray(y_val),
    "Predicted_Probability": validation_probs,
    "Risk_Group": risk_groups
})

print("\nValidation customers by predicted risk group:")
print(
    risk_df["Risk_Group"]
    .value_counts()
    .reindex(["Low", "Medium", "High"])
)

print("\nActual default rate by predicted risk group:")
print(
    risk_df.groupby(
        "Risk_Group",
        observed=False
    )["Actual_Default"]
    .mean()
    .round(3)
)


# ============================================
# 7. PCA PLOT COLOURED BY PREDICTED RISK
# ============================================

plt.figure(figsize=(10, 7))

for group in ["Low", "Medium", "High"]:

    mask = (
        risk_df["Risk_Group"].to_numpy() == group
    )

    plt.scatter(
        X_val_pca[mask, 0],
        X_val_pca[mask, 1],
        s=15,
        alpha=0.45,
        label=f"{group} Risk"
    )

plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.2f}% variance)"
)

plt.title(
    "Predicted Risk Groups Projected onto PCA Components"
)

plt.legend()
plt.tight_layout()

plt.show()


# ============================================
# 8. PCA PLOT COLOURED BY ACTUAL DEFAULT
# ============================================

actual_default = np.asarray(y_val) == 1

plt.figure(figsize=(10, 7))

plt.scatter(
    X_val_pca[~actual_default, 0],
    X_val_pca[~actual_default, 1],
    s=15,
    alpha=0.35,
    label="Actual: No Default"
)

plt.scatter(
    X_val_pca[actual_default, 0],
    X_val_pca[actual_default, 1],
    s=15,
    alpha=0.45,
    label="Actual: Default"
)

plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.2f}% variance)"
)

plt.title(
    "Actual Default Outcomes Projected onto PCA Components"
)

plt.legend()
plt.tight_layout()

plt.show()