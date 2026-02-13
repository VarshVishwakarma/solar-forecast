import pandas as pd
import os

def merge_solar_datasets():
    print("1. Setup paths...")
    # Define paths relative to the project root
    kaggle_path = os.path.join('data', 'kaggle_base.csv')
    nasa_path = os.path.join('data', 'bhopal_hourly.csv')
    output_path = os.path.join('data', 'final_solar_dataset.csv')

    # Verify input files exist
    if not os.path.exists(kaggle_path) or not os.path.exists(nasa_path):
        print("Error: Input files not found in 'data/' folder.")
        return

    print("2. Loading Kaggle dataset...")
    kaggle_df = pd.read_csv(kaggle_path)
    # Convert timestamp to datetime objects
    kaggle_df['timestamp'] = pd.to_datetime(kaggle_df['timestamp'])
    print(f"   Kaggle rows: {len(kaggle_df)}")

    print("3. Loading and processing NASA dataset...")
    # NASA POWER CSVs often have metadata headers. 
    # We scan the file to find the line starting with 'YEAR' to know where data begins.
    header_row = 0
    with open(nasa_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("YEAR,MO,DY,HR"):
                header_row = i
                break
    
    # Read NASA CSV skipping the metadata lines
    nasa_df = pd.read_csv(nasa_path, skiprows=header_row)

    # Create a proper datetime timestamp column
    # We rename columns temporarily to satisfy pd.to_datetime requirements
    nasa_df = nasa_df.rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'})
    nasa_df['timestamp'] = pd.to_datetime(nasa_df[['year', 'month', 'day', 'hour']])

    # Handle potentially missing CLOUD_AMT column (common in some downloads)
    if 'CLOUD_AMT' not in nasa_df.columns:
        print("   [Warning] 'CLOUD_AMT' column not found in NASA file. Creating 'cloud_cover' with NaN values.")
        nasa_df['CLOUD_AMT'] = float('nan')

    # Select and Rename NASA columns
    # We map: T2M -> temperature_nasa, RH2M -> humidity, etc.
    nasa_subset = nasa_df[['timestamp', 'T2M', 'RH2M', 'CLOUD_AMT', 'ALLSKY_SFC_SW_DWN']].copy()
    nasa_subset.columns = ['timestamp', 'temperature_nasa', 'humidity', 'cloud_cover', 'ghi']
    
    print(f"   NASA rows: {len(nasa_subset)}")

    print("4. Merging datasets (Inner Join)...")
    # This will align data on the exact hour. 
    # Note: Kaggle data at :15, :30, :45 minutes will be dropped since NASA is only hourly.
    merged_df = pd.merge(kaggle_df, nasa_subset, on='timestamp', how='inner')

    print("5. Finalizing columns...")
    # We keep 'temperature' from Kaggle and 'humidity', 'cloud_cover', 'ghi' from NASA
    final_cols = ['timestamp', 'temperature', 'humidity', 'cloud_cover', 'ghi', 'power_output']
    final_df = merged_df[final_cols]

    print(f"6. Saving to '{output_path}'...")
    final_df.to_csv(output_path, index=False)

    print("\nSuccess! Final Data Preview:")
    print(final_df.head())
    print(f"\nSaved {len(final_df)} merged rows.")

if __name__ == "__main__":
    merge_solar_datasets()