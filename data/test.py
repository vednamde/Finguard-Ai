import pandas as pd

df = pd.read_csv("../data/Engineered_Financial_Statements.csv")

# Take a few random rows
sample = df.sample(5)

# Drop target and identifiers
sample = sample.drop(columns=["Financial_Status", "Company", "Category", "Year"])

# Save sample file for testing
sample.to_csv("../data/test_company.csv", index=False)

print("✅ Test CSV created at: ../data/test_company.csv")
