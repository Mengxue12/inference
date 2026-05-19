
"""
Extracts meter power data from the file with 
name of "$(date -u "+%Y-%m-%dT%H_%M_%SZ").csv"

calculates the power from current and voltage data

transforms the time to timestamp

"""

from datetime import datetime, timezone
import pandas as pd
import argparse
import os


def meter_time_to_utc(meter_time, saved_time, saved_timestamp):

    meter_timestamp = meter_time + (saved_timestamp - saved_time)
    return meter_timestamp

# saved_path = "path-to-data/utc-time"
def extract_meter_power(saved_path):
    if os.path.exists(f"{saved_path}-timestamp.csv"):
        print("extract_meter_power: File already exists, skipping extraction.")
        return pd.read_csv(f"{saved_path}-timestamp.csv", index_col=0)[["meter-platform_power"]]
    power_data_path = f"{saved_path}.xlsx"
    saved_date = saved_path.split("/")[-1]
    power_data = pd.read_excel(power_data_path, index_col=0, skiprows=4)

    # Convert the meter index time to UTC timestamp
    print("extract_meter_power: Converting meter time to UTC...")
    meter_time = power_data.index[-1]
    saved_timestamp = int(datetime.strptime(saved_date, "%Y-%m-%dT%H_%M_%SZ").replace(tzinfo=timezone.utc).timestamp())
    power_data.index = power_data.index.map(lambda x: meter_time_to_utc(x, meter_time, saved_timestamp))
    power_data.index.name = "timestamp"
    # Calculate power from current and voltage
    print("extract_meter_power: Calculating platform power from voltage and current...")
    power_data["meter-platform_power"] = power_data["  Voltage  (V)"] * power_data["  Current  (A)"]
    power_data.to_csv(power_data_path.replace('.xlsx', '-timestamp.csv'))
    return power_data[["meter-platform_power"]]

def parse_args():
    parser = argparse.ArgumentParser(description="Extract meter power data from CSV file.")
    parser.add_argument("--saved_path", type=str, required=True, help="Path to the saved CSV file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    extracted_data = extract_meter_power(args.saved_path)
    print("Extracted data:")
    print(extracted_data)
