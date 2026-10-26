# ============================================
# FINAL TEST EVALUATION
# Credit Card Default Prediction
#
# Run only after all modelling decisions are locked.
# Do not tune the models using these test results.
# ============================================

import joblib
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.calibration import calibration_curve


# ============================================
# 1. LOAD SAVED ARTIFACTS
# ============================================

artifacts = joblib.load("model_artifacts.joblib")

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]

X_train = artifacts["X_train"]
y_train = np.asarray(artifacts["y_train"])

X_val = artifacts["X_val"]
y_val = np.asarray(artifacts["y_val"])

X_test = artifacts["X_test"]
y_test = np.asarray(artifacts["y_test"])

standard_threshold = artifacts["best_threshold"]

# Load Bayesian validation results
bayesian_results = joblib.load(
    "bayesian_validation_results.joblib"
)

bayesian_threshold = bayesian_results[
    "bayesian_best_threshold"
]

bayesian_val_targets = np.asarray(
    bayesian_results["validation_targets"]
)

# Make sure Bayesian results match the current
# validation split before evaluating.
if not np.array_equal(y_val, bayesian_val_targets):
    raise ValueError(
        "Bayesian validation results do not match "
        "the current model artifacts. Rerun Bayesian_Model.py."
    )

if not (
    len(X_train) == len(y_train)
    and len(X_val) == len(y_val)
    and len(X_test) == len(y_test)
):
    raise ValueError(
        "Feature and target lengths do not match."
    )

print("Training customers:", len(X_train))
print("Validation customers:", len(X_val))
print("Final test customers:", len(X_test))

print("\nLocked thresholds:")
print(f"Standard model: {standard_threshold:.3f}")
print(f"Bayesian model: {bayesian_threshold:.3f}")


# ============================================
# 2. PREPROCESS TEST FEATURES
# ============================================

# Use the preprocessor fitted on training data.
X_test_processed = preprocessor.transform(X_test)

if sparse.issparse(X_test_processed):
    X_test_processed = X_test_processed.toarray()

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float64
)


# ============================================
# 3. STANDARD LOGISTIC REGRESSION
# ============================================

standard_probs = model.predict_proba(
    X_test_processed
)[:, 1]

standard_pred = (
    standard_probs >= standard_threshold
).astype(int)


# ============================================
# 4. BAYESIAN LOGISTIC REGRESSION
# ============================================

# Load posterior samples from the revised Bayesian model.
idata = az.from_netcdf("bayesian_trace.nc")

posterior_intercept = (
    idata.posterior["intercept"]
    .values.reshape(-1)
)

n_features = X_test_processed.shape[1]

posterior_beta = (
    idata.posterior["beta"]
    .values.reshape(-1, n_features)
)

if len(posterior_intercept) != len(posterior_beta):
    raise ValueError(
        "Bayesian intercept and coefficient samples "
        "have inconsistent dimensions."
    )

# Select up to 1,000 posterior draws for prediction.
indices = np.linspace(
    0,
    len(posterior_intercept) - 1,
    min(1000, len(posterior_intercept)),
    dtype=int
)

posterior_intercept = posterior_intercept[indices]
posterior_beta = posterior_beta[indices]

# Calculate probabilities for each posterior draw.
logits = (
    posterior_intercept[:, None]
    + posterior_beta @ X_test_processed.T
)

logits = np.clip(logits, -30, 30)

draw_probabilities = 1 / (1 + np.exp(-logits))

# Average probabilities across posterior draws.
bayesian_probs = draw_probabilities.mean(axis=0)

bayesian_pred = (
    bayesian_probs >= bayesian_threshold
).astype(int)


# ============================================
# 5. CALCULATE FINAL TEST METRICS
# ============================================

def evaluate_model(
    name,
    actual,
    predicted,
    probabilities,
    threshold
):
    """Calculate classification and probability metrics."""

    return {
        "Model": name,
        "Threshold": threshold,
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
        ),
        "Brier Score": brier_score_loss(
            actual, probabilities
        )
    }


results = pd.DataFrame([
    evaluate_model(
        "Standard Logistic Regression",
        y_test,
        standard_pred,
        standard_probs,
        standard_threshold
    ),
    evaluate_model(
        "Bayesian Logistic Regression",
        y_test,
        bayesian_pred,
        bayesian_probs,
        bayesian_threshold
    )
])

print("\nFINAL TEST RESULTS")
print("==================")
print(results.round(4).to_string(index=False))

results.to_csv(
    "final_test_metrics.csv",
    index=False
)


# ============================================
# 6. MAJORITY-CLASS BASELINE
# ============================================

# Choose the majority class using TRAINING data.
# Do not select it using test outcomes.

majority_class = int(
    pd.Series(y_train).mode().iloc[0]
)

majority_pred = np.full(
    len(y_test),
    majority_class,
    dtype=int
)

majority_accuracy = accuracy_score(
    y_test,
    majority_pred
)

print("\nMAJORITY-CLASS BASELINE")
print("=======================")
print("Predicted class:", majority_class)
print(f"Test accuracy: {majority_accuracy:.4f}")


# ============================================
# 7. FINAL TEST CONFUSION MATRICES
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models_for_plot = [
    ("Standard Logistic Regression", standard_pred),
    ("Bayesian Logistic Regression", bayesian_pred)
]

for ax, (name, predictions) in zip(
    axes, models_for_plot
):
    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            y_test, predictions
        ),
        display_labels=["No Default", "Default"]
    )

    display.plot(
        ax=ax,
        colorbar=False
    )

    ax.set_title(name)

plt.tight_layout()

plt.savefig(
    "final_test_confusion_matrices.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================
# 8. FINAL TEST CALIBRATION
# ============================================

# Estimate the baseline probability from training data.
training_default_rate = float(np.mean(y_train))

baseline_probs = np.full(
    len(y_test),
    training_default_rate
)

baseline_brier = brier_score_loss(
    y_test,
    baseline_probs
)

print("\nTRAINING-RATE BASELINE BRIER SCORE")
print("==================================")
print(f"{baseline_brier:.4f}")

plt.figure(figsize=(8, 6))

models_for_calibration = [
    ("Standard Logistic Regression", standard_probs),
    ("Bayesian Logistic Regression", bayesian_probs)
]

for name, probabilities in models_for_calibration:

    observed_rate, mean_predicted = calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="quantile"
    )

    plt.plot(
        mean_predicted,
        observed_rate,
        marker="o",
        label=name
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Default Probability")
plt.ylabel("Observed Default Rate")
plt.title("Final Test Probability Calibration")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "final_test_calibration.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================
# 9. SAVE FINAL TEST PREDICTIONS
# ============================================

joblib.dump(
    {
        "y_test": y_test,
        "standard_probabilities": standard_probs,
        "standard_predictions": standard_pred,
        "bayesian_probabilities": bayesian_probs,
        "bayesian_predictions": bayesian_pred,
        "standard_threshold": standard_threshold,
        "bayesian_threshold": bayesian_threshold
    },
    "final_test_predictions.joblib"
)

print("\nFinal test metrics saved to final_test_metrics.csv")
print("Final predictions saved to final_test_predictions.joblib")
print("Final confusion matrices and calibration plot saved.")
print("Do not tune or retrain using these test results.")