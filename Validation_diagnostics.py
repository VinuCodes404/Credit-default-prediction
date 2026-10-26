# ============================================
# VALIDATION DIAGNOSTICS
# Probability Calibration + Permutation Importance
# ============================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    average_precision_score
)
from sklearn.calibration import calibration_curve


# ============================================
# 1. LOAD SAVED ARTIFACTS
# ============================================

artifacts = joblib.load("model_artifacts.joblib")

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]

X_val = artifacts["X_val"]
y_val = np.asarray(artifacts["y_val"])

bayesian_results = joblib.load(
    "bayesian_validation_results.joblib"
)

bayesian_probs = np.asarray(
    bayesian_results["validation_probabilities"]
)

bayesian_targets = np.asarray(
    bayesian_results["validation_targets"]
)

# Make sure both sets of probabilities correspond
# to the same validation customers.
if not np.array_equal(y_val, bayesian_targets):
    raise ValueError(
        "Validation targets differ between the models."
    )

if len(bayesian_probs) != len(y_val):
    raise ValueError(
        "Bayesian probabilities have the wrong length."
    )

print("Validation customers:", len(X_val))


# ============================================
# 2. GET STANDARD MODEL PROBABILITIES
# ============================================

X_val_processed = preprocessor.transform(X_val)

standard_probs = model.predict_proba(
    X_val_processed
)[:, 1]


# ============================================
# 3. PROBABILITY CALIBRATION
# ============================================

standard_brier = brier_score_loss(
    y_val, standard_probs
)

bayesian_brier = brier_score_loss(
    y_val, bayesian_probs
)

# Baseline: assign every customer the validation
# set's overall default rate.
baseline_probability = y_val.mean()

baseline_brier = brier_score_loss(
    y_val,
    np.full(len(y_val), baseline_probability)
)

print("\nBRIER SCORES")
print("------------")
print(f"Standard logistic regression: {standard_brier:.4f}")
print(f"Bayesian logistic regression: {bayesian_brier:.4f}")
print(f"Constant-rate baseline:       {baseline_brier:.4f}")


# ============================================
# 4. DRAW CALIBRATION CURVE
# ============================================

plt.figure(figsize=(8, 6))

# Quantile bins aim to place similar numbers of
# customers in each bin.
for name, probabilities in [
    ("Standard Logistic Regression", standard_probs),
    ("Bayesian Logistic Regression", bayesian_probs)
]:

    observed_rate, predicted_rate = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    plt.plot(
        predicted_rate,
        observed_rate,
        marker="o",
        label=name
    )

# Perfect calibration reference
plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Default Probability")
plt.ylabel("Observed Default Rate")
plt.title("Probability Calibration — Validation Set")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# ============================================
# 5. PERMUTATION IMPORTANCE
# ============================================

# We'll measure performance using Average Precision,
# which is useful for our imbalanced target.
#
# Importances are computed by shuffling each ORIGINAL
# validation feature, then applying the fitted
# preprocessor and model. This gives one importance
# value per original feature, rather than separate
# values for one-hot encoded columns.

baseline_ap = average_precision_score(
    y_val,
    standard_probs
)

print("\nBaseline validation Average Precision:",
      round(baseline_ap, 4))

rng = np.random.default_rng(42)

n_repeats = 10
importance_results = []

for feature in X_val.columns:

    ap_after_shuffle = []

    for repeat in range(n_repeats):

        # Copy validation inputs and shuffle one
        # feature across customers.
        X_permuted = X_val.copy()

        X_permuted[feature] = rng.permutation(
            X_permuted[feature].to_numpy()
        )

        # Use the existing fitted preprocessor.
        X_permuted_processed = preprocessor.transform(
            X_permuted
        )

        shuffled_probs = model.predict_proba(
            X_permuted_processed
        )[:, 1]

        shuffled_ap = average_precision_score(
            y_val,
            shuffled_probs
        )

        ap_after_shuffle.append(shuffled_ap)

    # Positive importance means shuffling the feature
    # reduced Average Precision.
    decreases = baseline_ap - np.asarray(ap_after_shuffle)

    importance_results.append({
        "Feature": feature,
        "Importance_Mean": decreases.mean(),
        "Importance_Std": decreases.std(ddof=1)
    })


# ============================================
# 6. DISPLAY AND SAVE IMPORTANCE RESULTS
# ============================================

importance_df = pd.DataFrame(
    importance_results
).sort_values(
    "Importance_Mean",
    ascending=False
)

print("\nPERMUTATION IMPORTANCE")
print("----------------------")
print(
    importance_df.head(15).round(4).to_string(index=False)
)

importance_df.to_csv(
    "validation_permutation_importance.csv",
    index=False
)


# ============================================
# 7. PLOT TOP 15 FEATURES
# ============================================

top_features = (
    importance_df.head(15)
    .sort_values("Importance_Mean")
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance_Mean"],
    xerr=top_features["Importance_Std"],
    capsize=3
)

plt.axvline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel("Decrease in Average Precision After Shuffling")
plt.ylabel("Original Feature")
plt.title(
    "Permutation Importance — Validation Set"
)

plt.tight_layout()

plt.show()

print("\nCalibration plot saved.")
print("Permutation importance table and plot saved.")
print("Final test set remains untouched.")