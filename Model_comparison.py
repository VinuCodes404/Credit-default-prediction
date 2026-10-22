import joblib
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# =========================================
# 1. Load existing artifacts
# =========================================

artifacts = joblib.load("model_artifacts.joblib")

full_model = artifacts["model"]
preprocessor = artifacts["preprocessor"]

X_train = artifacts["X_train"]
y_train = artifacts["y_train"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]


# =========================================
# 2. Recreate the same 6,000-row sample
#    used by Bayesian Logistic Regression
# =========================================

X_bayes_raw, _, y_bayes, _ = train_test_split(
    X_train,
    y_train,
    train_size=6000,
    random_state=42,
    stratify=y_train
)

X_bayes = preprocessor.transform(X_bayes_raw)
X_test_processed = preprocessor.transform(X_test)

if sparse.issparse(X_bayes):
    X_bayes = X_bayes.toarray()

if sparse.issparse(X_test_processed):
    X_test_processed = X_test_processed.toarray()

X_bayes = np.asarray(X_bayes, dtype=float)
X_test_processed = np.asarray(X_test_processed, dtype=float)

y_bayes = np.asarray(y_bayes, dtype=int)
y_test = np.asarray(y_test, dtype=int)


# =========================================
# 3. Train standard Logistic Regression
#    on the SAME 6,000 training customers
# =========================================

standard_6k = LogisticRegression(max_iter=1000)

standard_6k.fit(X_bayes, y_bayes)

standard_probs = standard_6k.predict_proba(
    X_test_processed
)[:, 1]

standard_pred = (standard_probs >= 0.5).astype(int)


# =========================================
# 4. Load Bayesian posterior samples
# =========================================

idata = az.from_netcdf("bayesian_trace.nc")

posterior_intercept = (
    idata.posterior["intercept"].values.reshape(-1)
)

n_features = X_bayes.shape[1]

posterior_beta = (
    idata.posterior["beta"].values.reshape(
        -1, n_features
    )
)

# Use up to 1,000 posterior draws for prediction
indices = np.linspace(
    0,
    len(posterior_intercept) - 1,
    min(1000, len(posterior_intercept)),
    dtype=int
)

posterior_intercept = posterior_intercept[indices]
posterior_beta = posterior_beta[indices]

logits = (
    posterior_intercept[:, None]
    + posterior_beta @ X_test_processed.T
)

logits = np.clip(logits, -30, 30)

posterior_probs = 1 / (1 + np.exp(-logits))

bayesian_probs = posterior_probs.mean(axis=0)
bayesian_pred = (bayesian_probs >= 0.5).astype(int)


# =========================================
# 5. Evaluate all three models
# =========================================

full_probs = full_model.predict_proba(
    X_test_processed
)[:, 1]

full_pred = (full_probs >= 0.5).astype(int)


def evaluate_model(name, actual, predicted, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted),
        "Recall": recall_score(actual, predicted),
        "F1-score": f1_score(actual, predicted),
        "ROC-AUC": roc_auc_score(actual, probabilities),
        "Average Precision": average_precision_score(
            actual, probabilities
        )
    }


results = pd.DataFrame([
    evaluate_model(
        "Standard (24k training)",
        y_test, full_pred, full_probs
    ),
    evaluate_model(
        "Standard (6k training)",
        y_test, standard_pred, standard_probs
    ),
    evaluate_model(
        "Bayesian (6k training)",
        y_test, bayesian_pred, bayesian_probs
    )
])

print("\nMODEL COMPARISON")
print(results.round(3).to_string(index=False))


# =========================================
# 6. Bayesian coefficient intervals
# =========================================

summary = az.summary(
    idata,
    var_names=["beta"],
    ci_prob=0.95,
    round_to=4
)

# Extract feature names and posterior estimates
coef_df = summary.reset_index()

coef_df = coef_df.rename(
    columns={
        "index": "Feature",
        "eti95_lb": "Lower",
        "eti95_ub": "Upper"
    }
)

# Compatibility with ArviZ versions that name
# the index column differently
if "Feature" not in coef_df.columns:
    coef_df = coef_df.rename(
        columns={coef_df.columns[0]: "Feature"}
    )

coef_df["Feature"] = coef_df["Feature"].str.replace(
    "beta[", "", regex=False
).str.replace("]", "", regex=False)

coef_df = coef_df.sort_values("mean")

# Draw the posterior mean and 95% intervals
plt.figure(figsize=(10, 12))

xerr = np.vstack([
    coef_df["mean"] - coef_df["Lower"],
    coef_df["Upper"] - coef_df["mean"]
])

plt.errorbar(
    coef_df["mean"],
    coef_df["Feature"],
    xerr=xerr,
    fmt="o",
    capsize=3
)

plt.axvline(0, color="black", linestyle="--")

plt.xlabel("Posterior Coefficient")
plt.ylabel("Feature")
plt.title(
    "Bayesian Logistic Regression: "
    "Posterior Means and 95% Credible Intervals"
)

plt.tight_layout()

plt.savefig(
    "bayesian_coefficient_intervals.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()