import joblib
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# Load saved artifacts
artifacts = joblib.load("model_artifacts.joblib")

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]

# Transform test data using the existing fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

# Generate updated predictions
y_pred = model.predict(X_test_processed)
default_probs = model.predict_proba(X_test_processed)[:, 1]

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, default_probs))
print("Average Precision:", average_precision_score(y_test, default_probs))