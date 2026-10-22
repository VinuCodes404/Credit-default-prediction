import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import *

df = pd.read_csv("Data/UCI_Credit_Card.csv")

# MARK: EDA
# Exploratory Data Analysis

# Data info
print(df.head())
print("\n")
print(df.shape)
print(df.info())
print(df.isnull().sum())

# Target distribution
print(df["default.payment.next.month"].value_counts())

sns.countplot(x="default.payment.next.month", data=df)
plt.title("Default Payment Distribution")
plt.xlabel("Default Next Month (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
# plt.show()

# Inspect categorical features
print(df["SEX"].value_counts())
print(df["EDUCATION"].value_counts().sort_index())
print(df["MARRIAGE"].value_counts().sort_index())

#Inspect numerical features
# print(df[["LIMIT_BAL", "AGE"]].describe())

# MARK: Data Cleaning
df = df.drop("ID", axis=1)

print(df.duplicated().sum())

# Dealing with unusual categorical values
df["EDUCATION"] = df["EDUCATION"].replace(0, 4)
df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)

# Renaming the target variable for clarity
df = df.rename(columns={
    "default.payment.next.month": "default"
})

# Separating features and target variable
X = df.drop("default", axis=1)
y = df["default"]

# MARK: Data Preprocessing
categorical_features = [
    "SEX",
    "EDUCATION",
    "MARRIAGE"
]

numerical_features = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

# MARK: Splitting data
# stratify=y keeps the default/non-default ratio roughly the same in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MARK: Encoding features
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Comparision
print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

# MARK: Data Modelling
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)

# Few outputs that the model predicted
print("Predicted classes:")
print(y_pred[:10])

print("\nPredicted probabilities:")
print(y_prob[:10])

print("\nDefault probabilities:")
print(y_prob[:10, 1])


# Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("\n")
print(cm)

# MARK: Evaluation Metrics
# Variance Matrix
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# ROC - AUC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Extract the probability of default (class 1)
default_probs = y_prob[:, 1]

# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, default_probs)

# Calculate AUC
roc_auc = roc_auc_score(y_test, default_probs)

print("ROC-AUC:", roc_auc)

# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — Credit Card Default Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# precision - recall curve
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

# Calculate precision-recall values
precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(y_test, default_probs)
)

# Calculate Average Precision
avg_precision = average_precision_score(y_test, default_probs)

# Baseline: proportion of actual defaulters
baseline = y_test.mean()

print("Average Precision:", avg_precision)
print("Baseline precision:", baseline)

# Plot the curve
plt.figure(figsize=(8, 6))

plt.plot(
    recall_values,
    precision_values,
    label=f"Logistic Regression (AP = {avg_precision:.3f})"
)

plt.axhline(
    y=baseline,
    color="black",
    linestyle="--",
    label=f"Baseline = {baseline:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Credit Card Default Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

####################################
# MARK: Experimnenting with threshold
####################################
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

for threshold in [0.5, 0.4, 0.3, 0.2]:
    y_pred_threshold = (default_probs >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

# MARK: Feature Analysis
# Inspecting the model's coefficients
import pandas as pd

# Get names of the processed features
feature_names = preprocessor.get_feature_names_out()

# Get coefficients learned by logistic regression
coefficients = model.coef_[0]

# Create a table
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Strongest positive coefficients
print("Features pushing predictions toward default:")
print(
    coef_df.nlargest(10, "Coefficient")
          .to_string(index=False)
)

# Strongest negative coefficients
print("\nFeatures pushing predictions away from default:")
print(
    coef_df.nsmallest(10, "Coefficient")
          .to_string(index=False)
)

# Visualize the coefficients
import matplotlib.pyplot as plt

top_positive = coef_df.nlargest(10, "Coefficient")
top_negative = coef_df.nsmallest(10, "Coefficient")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].barh(
    top_positive["Feature"],
    top_positive["Coefficient"]
)
axes[0].set_title("Features Pushing Toward Default")
axes[0].set_xlabel("Coefficient")

axes[1].barh(
    top_negative["Feature"],
    top_negative["Coefficient"]
)
axes[1].set_title("Features Pushing Away from Default")
axes[1].set_xlabel("Coefficient")

plt.tight_layout()
plt.show()

# Variable Share
import joblib

# Save the trained model and the fitted preprocessor
joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    },
    "model_artifacts.joblib"
)

print("Model artifacts saved successfully.")