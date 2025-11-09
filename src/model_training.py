import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib   


df = pd.read_csv("../data/Engineered_Financial_Statements.csv")


# Select features and target
X = df.drop(columns=['Financial_Status', 'Company', 'Category', 'Year'])
y = df['Financial_Status']



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Save model
joblib.dump(model, "../models/financial_health_xgb.pkl")
print("✅ Model trained and saved successfully!")