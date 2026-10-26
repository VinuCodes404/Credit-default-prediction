import joblib
import arviz as az
import numpy as np
import pymc as pm

from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# 1. LOAD REVISED ARTIFACTS

artifacts = joblib.load("model_artifacts.joblib")

preprocessor = artifacts["preprocessor"]

X_train = artifacts["X_train"]
y_train = artifacts["y_train"]

X_val = artifacts["X_val"]
y_val = artifacts["y_val"]

print("Training customers:", len(X_train))
print("Validation customers:", len(X_val))


# 2. TRANSFORM FEATURES

# The preprocessor is already fitted on training data.
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)

if sparse.issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()

if sparse.issparse(X_val_processed):
    X_val_processed = X_val_processed.toarray()

X_train_processed = np.asarray(
    X_train_processed, dtype=np.float64
)

X_val_processed = np.asarray(
    X_val_processed, dtype=np.float64
)

y_train_array = np.asarray(y_train, dtype=np.int8)
y_val_array = np.asarray(y_val, dtype=np.int8)

feature_names = preprocessor.get_feature_names_out()
n_features = X_train_processed.shape[1]


# 3. DEFINE BAYESIAN MODEL

with pm.Model(coords={"features": feature_names}) as bayes_model:

    # Prior for intercept
    intercept = pm.Normal(
        "intercept",
        mu=0,
        sigma=1.5
    )

    # Prior distributions for coefficients
    beta = pm.Normal(
        "beta",
        mu=0,
        sigma=1,
        dims="features"
    )

    # Weighted sum of features
    logits = intercept + pm.math.dot(
        X_train_processed, beta
    )

    # Convert scores into probabilities
    probability = pm.math.sigmoid(logits)

    # Likelihood: observed customer outcomes
    pm.Bernoulli(
        "observed_default",
        p=probability,
        observed=y_train_array
    )

    # 4. SAMPLE POSTERIOR DISTRIBUTIONS

    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=2,
        random_seed=42,
        nuts={"target_accept": 0.9},
        return_inferencedata=True
    )


# Save the posterior samples
idata.to_netcdf("bayesian_trace.nc")


# 5. CHECK SAMPLING DIAGNOSTICS

summary = az.summary(
    idata,
    var_names=["intercept", "beta"],
    ci_prob=0.95,
    round_to=3
)

print("\nPOSTERIOR COEFFICIENT SUMMARY")
print(summary)

print("\nMaximum R-hat:", round(summary["r_hat"].max(), 3))


# 6. PREDICT VALIDATION PROBABILITIES

posterior_intercept = (
    idata.posterior["intercept"]
    .values.reshape(-1)
)

posterior_beta = (
    idata.posterior["beta"]
    .values.reshape(-1, n_features)
)

# Use up to 1,000 posterior draws for prediction.
indices = np.linspace(
    0,
    len(posterior_intercept) - 1,
    min(1000, len(posterior_intercept)),
    dtype=int
)

posterior_intercept = posterior_intercept[indices]
posterior_beta = posterior_beta[indices]

logits_val = (
    posterior_intercept[:, None]
    + posterior_beta @ X_val_processed.T
)

logits_val = np.clip(logits_val, -30, 30)

probabilities_val = 1 / (1 + np.exp(-logits_val))

# Average predictions across posterior draws.
bayesian_val_probs = probabilities_val.mean(axis=0)


# ============================================
# 7. SELECT THRESHOLD USING VALIDATION DATA
# ============================================

candidate_thresholds = np.arange(0.05, 0.951, 0.01)

threshold_results = []

for threshold in candidate_thresholds:

    val_pred = (
        bayesian_val_probs >= threshold
    ).astype(int)

    score = f1_score(
        y_val_array,
        val_pred,
        zero_division=0
    )

    threshold_results.append({
        "threshold": float(threshold),
        "f1": score
    })

best_result = max(
    threshold_results,
    key=lambda result: result["f1"]
)

bayesian_best_threshold = best_result["threshold"]

print("\nBayesian threshold selected on validation data:")
print("Threshold:", round(bayesian_best_threshold, 3))
print("Validation F1:", round(best_result["f1"], 3))


# 8. VALIDATION EVALUATION

bayesian_val_pred = (
    bayesian_val_probs >= bayesian_best_threshold
).astype(int)

print("\nBAYESIAN VALIDATION RESULTS")
print("---------------------------")

print("Accuracy:",
      round(accuracy_score(
          y_val_array, bayesian_val_pred
      ), 3))

print("Precision:",
      round(precision_score(
          y_val_array, bayesian_val_pred,
          zero_division=0
      ), 3))

print("Recall:",
      round(recall_score(
          y_val_array, bayesian_val_pred,
          zero_division=0
      ), 3))

print("F1-score:",
      round(f1_score(
          y_val_array, bayesian_val_pred,
          zero_division=0
      ), 3))

print("ROC-AUC:",
      round(roc_auc_score(
          y_val_array, bayesian_val_probs
      ), 3))

print("Average Precision:",
      round(average_precision_score(
          y_val_array, bayesian_val_probs
      ), 3))


# 9. SAVE BAYESIAN VALIDATION RESULTS

joblib.dump(
    {
        "bayesian_best_threshold": bayesian_best_threshold,
        "validation_probabilities": bayesian_val_probs,
        "validation_predictions": bayesian_val_pred,
        "validation_targets": y_val_array
    },
    "bayesian_validation_results.joblib"
)

print("\nBayesian posterior saved.")
print("Bayesian validation results saved.")
print("Final test set remains untouched.")