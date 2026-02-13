import pandas as pd
import os

def process_solar_data():
    print("1. Loading CSV files...")
    
    # Define paths relative to the project root (solar-forecast/)
    # We use os.path.join to handle Windows backslashes automatically
    gen_path = os.path.join('data', 'Plant_1_Generation_Data.csv')
    sensor_path = os.path.join('data', 'Plant_1_Weather_Sensor_Data.csv')
    output_path = os.path.join('data', 'kaggle_base.csv')

    # Verify files exist before trying to load
    if not os.path.exists(gen_path):
        print(f"Error: File not found at {gen_path}")
        print("Make sure you are running this script from the 'solar-forecast' folder.")
        return

    # Load the datasets
    gen_df = pd.read_csv(gen_path)
    sensor_df = pd.read_csv(sensor_path)

    print("2. Converting dates...")
    # Critical Step: Handle different date formats
    # Generation file uses DD-MM-YYYY format (e.g., 15-05-2020)
    gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
    
    # Sensor file uses YYYY-MM-DD format (e.g., 2020-05-15)
    sensor_df['DATE_TIME'] = pd.to_datetime(sensor_df['DATE_TIME'])

    print("3. Aggregating power data...")
    # The generation file has many rows for the same time (one per inverter).
    # We group by time and SUM the AC_POWER to get the total plant output.
    gen_agg = gen_df.groupby('DATE_TIME')['AC_POWER'].sum().reset_index()

    print("4. Processing weather data...")
    # Usually there is only one sensor reading per time, but we group by time
    # and take the mean just to be safe and ensure unique timestamps.
    sensor_agg = sensor_df.groupby('DATE_TIME')[['AMBIENT_TEMPERATURE', 'IRRADIATION']].mean().reset_index()

    print("5. Merging datasets...")
    # Combine both tables where the timestamps match
    merged_df = pd.merge(gen_agg, sensor_agg, on='DATE_TIME', how='inner')

    # Select specific columns and rename them as requested
    final_df = merged_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'IRRADIATION', 'AC_POWER']]
    final_df.columns = ['timestamp', 'temperature', 'ghi_proxy', 'power_output']

    print(f"6. Saving to '{output_path}'...")
    final_df.to_csv(output_path, index=False)
    
    print("\nSuccess! Data Preview:")
    print(final_df.head())
    print(f"\nSaved {len(final_df)} rows to {output_path}")

if __name__ == "__main__":
    process_solar_data()