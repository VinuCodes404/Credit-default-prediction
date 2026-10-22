import joblib
import arviz as az
import numpy as np
import pymc as pm

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def main():

    # =========================================
    # 1. Load our existing saved artifacts
    # =========================================

    artifacts = joblib.load("model_artifacts.joblib")

    preprocessor = artifacts["preprocessor"]
    X_train = artifacts["X_train"]
    y_train = artifacts["y_train"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    # Use 6,000 training customers for this
    # initial Bayesian experiment.
    X_bayes_raw, _, y_bayes, _ = train_test_split(
        X_train,
        y_train,
        train_size=6000,
        random_state=42,
        stratify=y_train
    )

    # Use the existing fitted preprocessor.
    X_bayes = preprocessor.transform(X_bayes_raw)
    X_test_processed = preprocessor.transform(X_test)

    # Convert sparse matrices if necessary.
    if sparse.issparse(X_bayes):
        X_bayes = X_bayes.toarray()

    if sparse.issparse(X_test_processed):
        X_test_processed = X_test_processed.toarray()

    X_bayes = np.asarray(X_bayes, dtype=np.float64)
    X_test_processed = np.asarray(
        X_test_processed, dtype=np.float64
    )

    y_bayes = np.asarray(y_bayes, dtype=np.int8)
    y_test = np.asarray(y_test, dtype=np.int8)

    feature_names = preprocessor.get_feature_names_out()
    n_features = X_bayes.shape[1]

    print("Bayesian training shape:", X_bayes.shape)
    print("Test shape:", X_test_processed.shape)

    # =========================================
    # MARK: Define the Bayesian logistic model
    # =========================================

    with pm.Model(coords={"features": feature_names}) as bayes_model:

        # Prior for the intercept
        intercept = pm.Normal(
            "intercept",
            mu=0,
            sigma=1.5
        )

        # Priors for feature coefficients
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=1,
            dims="features"
        )

        # Weighted sum of the features
        logits = intercept + pm.math.dot(X_bayes, beta)

        # Convert scores into probabilities
        probability = pm.math.sigmoid(logits)

        # Likelihood: observed default outcomes
        pm.Bernoulli(
            "observed_default",
            p=probability,
            observed=y_bayes
        )

        # =====================================
        # MARK: Estimate posterior distributions
        # =====================================

        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=2,
            random_seed=42,
            nuts={"target_accept": 0.9},
            return_inferencedata=True
        )

    # Save posterior samples for later analysis
    idata.to_netcdf("bayesian_trace.nc")

    # =========================================
    # 4. Examine posterior coefficients
    # =========================================

    summary = az.summary(
        idata,
        var_names=["intercept", "beta"],
        ci_prob=0.95,
        round_to=3
    )

    print("\nPosterior coefficient summary:")
    print(summary)

    print("\nMaximum R-hat:")
    print(summary["r_hat"].max())

    beta_summary = summary.drop(index="intercept")

    print("\nStrongest positive posterior means:")
    print(
        beta_summary.sort_values("mean", ascending=False)
        .head(8)
    )

    print("\nStrongest negative posterior means:")
    print(
        beta_summary.sort_values("mean")
        .head(8)
    )

    # =========================================
    # 5. Predict on the untouched test set
    # =========================================

    posterior_intercept = (
        idata.posterior["intercept"].values.reshape(-1)
    )

    posterior_beta = (
        idata.posterior["beta"].values.reshape(
            -1, n_features
        )
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

    # Calculate a probability for every posterior draw.
    logits_test = (
        posterior_intercept[:, None]
        + posterior_beta @ X_test_processed.T
    )

    logits_test = np.clip(logits_test, -30, 30)

    probabilities = 1 / (1 + np.exp(-logits_test))

    # Average probabilities over the posterior draws.
    bayesian_probs = probabilities.mean(axis=0)

    # Use 0.5 for the initial classification comparison.
    bayesian_pred = (bayesian_probs >= 0.5).astype(int)

    # =========================================
    # MARK: Evaluate the Bayesian model
    # =========================================

    print("\nBayesian Logistic Regression Results")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, bayesian_pred), 3)
    )

    print(
        "Precision:",
        round(precision_score(y_test, bayesian_pred), 3)
    )

    print(
        "Recall:",
        round(recall_score(y_test, bayesian_pred), 3)
    )

    print(
        "F1-score:",
        round(f1_score(y_test, bayesian_pred), 3)
    )

    print(
        "ROC-AUC:",
        round(roc_auc_score(y_test, bayesian_probs), 3)
    )

    print(
        "Average Precision:",
        round(average_precision_score(y_test, bayesian_probs), 3)
    )


if __name__ == "__main__":
    main()