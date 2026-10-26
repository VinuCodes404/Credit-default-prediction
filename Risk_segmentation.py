# ============================================
# RISK SEGMENTATION
# Credit Card Default Prediction
# ============================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# 1. LOAD THE REVISED MODEL ARTIFACTS
# ============================================

artifacts = joblib.load("model_artifacts.joblib")

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]

X_val = artifacts["X_val"]
y_val = artifacts["y_val"]

print("Validation customers:", len(X_val))


# ============================================
# 2. PREDICT DEFAULT PROBABILITIES
# ============================================

X_val_processed = preprocessor.transform(X_val)

default_probs = model.predict_proba(
    X_val_processed
)[:, 1]


# ============================================
# 3. CREATE THE RISK GROUPS
# ============================================

risk_df = pd.DataFrame({
    "Actual_Default": np.asarray(y_val),
    "Predicted_Risk": default_probs
})

# Project-defined risk bands; not banking policy.
risk_df["Risk_Group"] = pd.cut(
    risk_df["Predicted_Risk"],
    bins=[-np.inf, 0.30, 0.60, np.inf],
    labels=["Low", "Medium", "High"],
    right=False
)


# ============================================
# 4. SUMMARIZE THE RISK GROUPS
# ============================================

risk_summary = risk_df.groupby(
    "Risk_Group",
    observed=False
).agg(
    Customers=("Actual_Default", "size"),
    Actual_Defaults=("Actual_Default", "sum"),
    Actual_Default_Rate=("Actual_Default", "mean"),
    Average_Predicted_Risk=("Predicted_Risk", "mean")
)

print("\nVALIDATION RISK SUMMARY")
print(risk_summary.round(3))


# ============================================
# 5. PLOT CUSTOMER COUNTS
# ============================================

risk_summary["Customers"].plot(
    kind="bar",
    figsize=(8, 5),
    title="Validation Customers by Predicted Risk Group"
)

plt.xlabel("Predicted Risk Group")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()


# ============================================
# 6. COMPARE PREDICTED AND ACTUAL RISK
# ============================================

comparison = risk_summary[
    ["Actual_Default_Rate", "Average_Predicted_Risk"]
].copy()

comparison.columns = [
    "Actual Default Rate",
    "Average Predicted Risk"
]

comparison.plot(
    kind="bar",
    figsize=(9, 5),
    title="Predicted vs Actual Default Rate by Risk Group"
)

plt.xlabel("Predicted Risk Group")
plt.ylabel("Rate")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()