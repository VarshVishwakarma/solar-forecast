import pandas as pd

df = pd.read_csv("data/final_solar_dataset.csv")
df = df.drop(columns=["cloud_cover"])

df.to_csv("data/final_solar_dataset_v2.csv", index=False)
print("Saved: data/final_solar_dataset_v2.csv (cloud_cover removed)")
