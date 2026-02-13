import pandas as pd

df = pd.read_csv("data/final_solar_dataset.csv", parse_dates=["timestamp"])

print(df.head())
print(df.isna().sum())
print(df.describe())
print(df.corr(numeric_only=True)["power_output"].sort_values(ascending=False))
