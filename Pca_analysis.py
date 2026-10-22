import joblib
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.decomposition import PCA


# =========================================
# 1. Load saved artifacts
# =========================================

artifacts = joblib.load("model_artifacts.joblib")

preprocessor = artifacts["preprocessor"]

X_train = artifacts["X_train"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]


# =========================================
# 2. Preprocess training and test features
# =========================================

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays if necessary
if sparse.issparse(X_train_processed):
    X_train_processed = X_train_processed.toarray()

if sparse.issparse(X_test_processed):
    X_test_processed = X_test_processed.toarray()


# =========================================
# 3. Fit PCA on training data only
# =========================================

pca = PCA()

X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)


# =========================================
# 4. Examine explained variance
# =========================================

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(
    f"PC1 variance explained: "
    f"{explained_variance[0] * 100:.2f}%"
)

print(
    f"PC2 variance explained: "
    f"{explained_variance[1] * 100:.2f}%"
)

print(
    f"Variance explained by first 2 PCs: "
    f"{cumulative_variance[1] * 100:.2f}%"
)

components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(
    f"Components needed to explain 90% variance: "
    f"{components_90}"
)


# =========================================
# 5. Scree plot
# =========================================

component_numbers = np.arange(1, len(explained_variance) + 1)

plt.figure(figsize=(9, 5))

plt.bar(component_numbers, explained_variance * 100)

plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title("PCA Scree Plot")
plt.xticks(component_numbers)
plt.tight_layout()
plt.show()


# =========================================
# 6. Plot test customers in 2D
# =========================================

no_default = y_test.to_numpy() == 0
default = y_test.to_numpy() == 1

plt.figure(figsize=(9, 6))

plt.scatter(
    X_test_pca[no_default, 0],
    X_test_pca[no_default, 1],
    alpha=0.35,
    s=15,
    label="No Default"
)

plt.scatter(
    X_test_pca[default, 0],
    X_test_pca[default, 1],
    alpha=0.5,
    s=15,
    label="Default"
)

plt.xlabel(
    f"PC1 ({explained_variance[0] * 100:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained_variance[1] * 100:.2f}% variance)"
)

plt.title("PCA: Customers by Actual Default Status")
plt.legend()
plt.tight_layout()
plt.show()