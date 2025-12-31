#!/usr/bin/env python3
"""
Dexcom Share API Client - Fetch the last 24 hours of glucose data

SETUP:
1. Ensure your .env file has your Dexcom Share credentials:
   
   DEXCOM_USERNAME=your_dexcom_share_username
   DEXCOM_PASSWORD=your_dexcom_share_password

2. Make sure Dexcom Share is enabled in your Dexcom app with at least one follower

3. Install dependencies:
   pip install -r requirements.txt

USAGE:
   python dexcom_share.py
"""

import csv
import os
from datetime import datetime
from pydexcom import Dexcom
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
USERNAME = os.getenv("DEXCOM_USERNAME")
PASSWORD = os.getenv("DEXCOM_PASSWORD")
OUTPUT_FILE = "share_readings.csv"


def main():
    # Validate credentials
    if not USERNAME or not PASSWORD or USERNAME == "your_dexcom_share_username":
        print("Error: Missing Dexcom Share credentials!")
        print("Edit your .env file and set:")
        print("  DEXCOM_USERNAME=your_dexcom_share_username")
        print("  DEXCOM_PASSWORD=your_dexcom_share_password")
        return 1

    print("Connecting to Dexcom Share...")

    try:
        # Initialize Dexcom client (US region)
        dexcom = Dexcom(username=USERNAME, password=PASSWORD, region="us")

        # Fetch last 24 hours of readings (max 288 readings at 5-min intervals)
        print("Fetching last 24 hours of glucose data...")
        readings = dexcom.get_glucose_readings(minutes=1440, max_count=288)

        if not readings:
            print("No readings found.")
            return 0

        print(f"\nFound {len(readings)} readings:\n")
        print(f"{'Timestamp':<30} {'mg/dL':>6}  {'Trend':<18}")
        print("-" * 58)

        # Collect data for CSV and display
        csv_rows = []
        values = []

        for reading in readings:
            timestamp = reading.datetime.isoformat()
            value = reading.value
            trend_desc = reading.trend_description

            print(f"{timestamp:<30} {value:>6}  {trend_desc:<18}")

            csv_rows.append({
                "timestamp": timestamp,
                "mg_dl": value,
                "trend_description": trend_desc,
            })
            values.append(value)

        # Summary stats
        if values:
            avg = sum(values) / len(values)
            print(f"\n--- Summary ---")
            print(f"Readings: {len(values)}")
            print(f"Average: {avg:.1f} mg/dL")
            print(f"Min: {min(values)} mg/dL")
            print(f"Max: {max(values)} mg/dL")

        # Save to CSV
        with open(OUTPUT_FILE, "w", newline="") as f:
            fieldnames = ["timestamp", "mg_dl", "trend_description"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nSaved {len(csv_rows)} readings to {OUTPUT_FILE}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

