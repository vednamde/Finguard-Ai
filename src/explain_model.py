import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("../models/financial_health_xgb.pkl")
df = pd.read_csv("../data/Engineered_Financial_Statements.csv")

# Prepare input data
X = df.drop(columns=['Financial_Status', 'Company', 'Category', 'Year'])

# SHAP analysis
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Summary plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("../reports/shap_summary.png", bbox_inches="tight")
print("✅ SHAP summary saved to reports/")
