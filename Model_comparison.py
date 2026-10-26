# ============================================
# MODEL COMPARISON
# Standard vs Bayesian Logistic Regression
# ============================================

import joblib
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================
# 1. LOAD REVISED ARTIFACTS
# ============================================

artifacts = joblib.load("model_artifacts.joblib")

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]

X_val = artifacts["X_val"]
y_val = np.asarray(artifacts["y_val"])

standard_threshold = artifacts["best_threshold"]

bayesian_results = joblib.load(
    "bayesian_validation_results.joblib"
)

bayesian_probs = np.asarray(
    bayesian_results["validation_probabilities"]
)

bayesian_targets = np.asarray(
    bayesian_results["validation_targets"]
)

bayesian_threshold = bayesian_results[
    "bayesian_best_threshold"
]


# Ensure both models use the same validation targets.
if not np.array_equal(y_val, bayesian_targets):
    raise ValueError(
        "The standard and Bayesian validation targets differ. "
        "Recheck the saved artifacts before comparison."
    )


# ============================================
# 2. STANDARD MODEL VALIDATION PREDICTIONS
# ============================================

X_val_processed = preprocessor.transform(X_val)

standard_probs = model.predict_proba(
    X_val_processed
)[:, 1]

standard_pred = (
    standard_probs >= standard_threshold
).astype(int)

bayesian_pred = (
    bayesian_probs >= bayesian_threshold
).astype(int)


# ============================================
# 3. COMPARE VALIDATION METRICS
# ============================================

def evaluate_model(name, actual, predicted, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "F1-score": f1_score(
            actual, predicted, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(actual, probabilities),
        "Average Precision": average_precision_score(
            actual, probabilities
        )
    }


results = pd.DataFrame([
    evaluate_model(
        "Standard Logistic Regression",
        y_val,
        standard_pred,
        standard_probs
    ),
    evaluate_model(
        "Bayesian Logistic Regression",
        y_val,
        bayesian_pred,
        bayesian_probs
    )
])

print("\nVALIDATION MODEL COMPARISON")
print("===========================")
print("Training customers: 21,000")
print("Validation customers: 4,500")
print("Final test set used: No")

print("\nSelected thresholds:")
print(f"Standard: {standard_threshold:.3f}")
print(f"Bayesian: {bayesian_threshold:.3f}")

print("\nMetrics:")
print(results.round(3).to_string(index=False))


# ============================================
# 4. BAYESIAN COEFFICIENT INTERVAL PLOT
# ============================================

idata = az.from_netcdf("bayesian_trace.nc")

summary = az.summary(
    idata,
    var_names=["beta"],
    ci_prob=0.95,
    round_to=4
)

coef_df = (
    summary
    .rename_axis("Parameter")
    .reset_index()
)

# Keep feature coefficients, excluding the intercept.
coef_df = coef_df[
    coef_df["Parameter"].str.startswith("beta[")
].copy()

coef_df["Feature"] = (
    coef_df["Parameter"]
    .str.removeprefix("beta[")
    .str.removesuffix("]")
)

coef_df = coef_df.sort_values("mean")

# Asymmetric intervals around each posterior mean.
lower_error = coef_df["mean"] - coef_df["eti95_lb"]
upper_error = coef_df["eti95_ub"] - coef_df["mean"]

fig, ax = plt.subplots(figsize=(10, 12))

ax.errorbar(
    coef_df["mean"],
    coef_df["Feature"],
    xerr=np.vstack([lower_error, upper_error]),
    fmt="o",
    capsize=3
)

ax.axvline(
    0,
    color="black",
    linestyle="--",
    label="Zero coefficient"
)

ax.set_xlabel("Posterior Mean Coefficient")
ax.set_ylabel("Feature")
ax.set_title(
    "Bayesian Logistic Regression: "
    "Posterior Means and 95% Credible Intervals"
)

ax.legend()
plt.tight_layout()

plt.savefig(
    "bayesian_coefficient_intervals.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nCoefficient interval plot saved.")
print("Final test set remains untouched.")