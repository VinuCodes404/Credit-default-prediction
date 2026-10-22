import joblib
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load saved artifacts
artifacts = joblib.load("model_artifacts.joblib")

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]

# 2. Calculate predicted default probabilities
X_test_processed = preprocessor.transform(X_test)
default_probs = model.predict_proba(X_test_processed)[:, 1]

# 3. Create a customer-level results table
risk_df = pd.DataFrame({
    "Actual Default": y_test.to_numpy(),
    "Predicted Default Probability": default_probs
})

# 4. Assign risk groups
risk_df["Risk Group"] = pd.cut(
    risk_df["Predicted Default Probability"],
    bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low", "Medium", "High"]
)

# 5. Summarize each risk group
risk_summary = risk_df.groupby(
    "Risk Group", observed=False
).agg(
    Customers=("Actual Default", "size"),
    Actual_Defaults=("Actual Default", "sum"),
    Actual_Default_Rate=("Actual Default", "mean"),
    Average_Predicted_Risk=("Predicted Default Probability", "mean")
)

print(risk_summary.round(3))

# 6. Visualize customer counts
risk_summary["Customers"].plot(
    kind="bar",
    title="Customers by Predicted Risk Group",
    ylabel="Number of Customers",
    xlabel="Risk Group"
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 7. Visualize actual default rates
risk_summary["Actual_Default_Rate"].mul(100).plot(
    kind="bar",
    title="Actual Default Rate by Predicted Risk Group",
    ylabel="Actual Default Rate (%)",
    xlabel="Risk Group",
    ylim=(0, 100)
)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()