#!/usr/bin/env python3
"""
Glooko API Client - Fetch bolus data from the unofficial Glooko API

Based on the nightscout-connect Glooko implementation:
https://github.com/nightscout/nightscout-connect/tree/main/lib/sources/glooko

NOTE: The Glooko API v2 login no longer works directly (returns 422).
This script uses web form login with CSRF token extraction.
You need to provide your Glooko Patient ID.

To find your Patient ID:
1. Log into https://my.glooko.com (or eu.my.glooko.com for EU)
2. Look at the URL or browser dev tools for a string like "us-east-1-xxxx" or "eu-west-1-xxxx"

SETUP:
1. Ensure your .env file has your Glooko credentials:
   
   GLOOKO_EMAIL=your_glooko_email
   GLOOKO_PASSWORD=your_glooko_password
   GLOOKO_REGION=us  # or "eu"
   GLOOKO_PATIENT_ID=your-patient-id  # e.g., "us-east-1-abc123" 

2. Install dependencies:
   pip install -r requirements.txt

USAGE:
   python glooko_bolus.py [--days N]
   
   --days N: Fetch data from the last N days (default: 7)
"""

import argparse
import csv
import os
import re
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
EMAIL = os.getenv("GLOOKO_EMAIL")
PASSWORD = os.getenv("GLOOKO_PASSWORD")
REGION = os.getenv("GLOOKO_REGION", "us").lower()
PATIENT_ID = os.getenv("GLOOKO_PATIENT_ID")
OUTPUT_FILE = "bolus_data.csv"

# Server configuration
API_SERVERS = {
    "us": "api.glooko.com",
    "eu": "eu.api.glooko.com",
    "default": "api.glooko.com",
}

WEB_SERVERS = {
    "us": "my.glooko.com",
    "eu": "eu.my.glooko.com",
    "default": "my.glooko.com",
}

# API Endpoints
ENDPOINTS = {
    "login_form": "/users/sign_in?locale=en",
    "login_post": "/users/sign_in",
    "normal_boluses": "/api/v2/pumps/normal_boluses",
    "graph_data": "/api/v3/graph/data",
}


def get_headers() -> dict:
    """Get default request headers."""
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Connection": "keep-alive",
    }


def extract_csrf_token(html: str) -> Optional[str]:
    """Extract CSRF token from login form HTML."""
    match = re.search(r'name="authenticity_token" value="([^"]+)"', html)
    return match.group(1) if match else None


class GlookoClient:
    """Client for the unofficial Glooko API using web form login."""

    def __init__(self, email: str, password: str, region: str = "us", patient_id: Optional[str] = None):
        self.email = email
        self.password = password
        self.region = region
        self.patient_id = patient_id
        
        self.api_server = API_SERVERS.get(region, API_SERVERS["default"])
        self.web_server = WEB_SERVERS.get(region, WEB_SERVERS["default"])
        self.api_base = f"https://{self.api_server}"
        self.web_base = f"https://{self.web_server}"
        
        self.session = requests.Session()
        self.session.headers.update(get_headers())

    def login(self) -> bool:
        """Authenticate with Glooko using web form login."""
        if not self.patient_id:
            print("Error: Patient ID is required for web form login.")
            print("Set GLOOKO_PATIENT_ID in your .env file.")
            print("\nTo find your Patient ID:")
            print("1. Log into https://my.glooko.com (or eu.my.glooko.com for EU)")
            print("2. Look at browser dev tools Network tab for requests containing your patient ID")
            print("   It looks like: 'us-east-1-xxxx' or 'eu-west-1-xxxx'")
            return False

        print(f"Authenticating with {self.web_server}...")
        print(f"Email: {self.email[:3]}***{self.email[-10:] if len(self.email) > 13 else ''}")

        try:
            # Step 1: Get login form and extract CSRF token
            print("Step 1: Getting login form and CSRF token...")
            form_url = urljoin(self.web_base, ENDPOINTS["login_form"])
            response = self.session.get(form_url)
            response.raise_for_status()

            csrf_token = extract_csrf_token(response.text)
            if not csrf_token:
                print("Error: Could not extract CSRF token from login form")
                return False
            print("  Found CSRF token")

            # Step 2: POST login form
            print("Step 2: Submitting login form...")
            login_url = urljoin(self.web_base, ENDPOINTS["login_post"])
            
            form_data = {
                "authenticity_token": csrf_token,
                "redirect_to": "",
                "language": "en",
                "user[email]": self.email,
                "user[password]": self.password,
                "commit": "Sign In",
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": form_url,
                "Origin": self.web_base,
            }
            
            response = self.session.post(
                login_url, 
                data=form_data, 
                headers=headers,
                allow_redirects=True
            )
            
            # Check if login was successful by looking at the final URL
            if "/users/sign_in" in response.url:
                print("Error: Login failed - still on sign in page")
                print("Check your email and password.")
                return False
            
            print(f"  Login successful! Redirected to: {response.url}")
            print(f"  Using patient ID: {self.patient_id}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"Error during login: {e}")
            return False

    def fetch_bolus_data(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict]:
        """Fetch normal bolus data for the given date range using Graph API."""
        if not self.patient_id:
            print("Error: Patient ID is required.")
            return []

        # Build Graph API URL with bolus series
        params = {
            "patient": self.patient_id,
            "startDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "endDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "locale": "en",
            "insulinTooltips": "true",
            "filterBgReadings": "true",
            "splitByDay": "false",
        }
        
        # Add bolus and carb-related series
        bolus_series = [
            "deliveredBolus",
            "suggestedBolus",
            "injectionBolus",
            "interruptedBolus",
            "automaticBolus",
            "overrideAboveBolus",
            "overrideBelowBolus",
            "gkInsulinBolus",
            "gkInsulin",
            # Carb series
            "gkCarb",
            "carbNonManual",
        ]

        url = urljoin(self.api_base, ENDPOINTS["graph_data"])
        
        # Build query string with series[]
        query_parts = [f"{k}={v}" for k, v in params.items()]
        for series in bolus_series:
            query_parts.append(f"series[]={series}")
        full_url = f"{url}?{'&'.join(query_parts)}"

        print(f"Fetching bolus data from {start_date.date()} to {end_date.date()}...")

        try:
            headers = {
                "Accept": "application/json, text/plain, */*",
                "Referer": self.web_base + "/",
                "Origin": self.web_base,
            }
            
            response = self.session.get(full_url, headers=headers)
            
            if response.status_code == 401:
                print("Error: Unauthorized - session may have expired")
                return []
            elif response.status_code == 422:
                print("Error: Unprocessable Entity - check patient ID")
                return []
            
            response.raise_for_status()

            data = response.json()
            
            # Extract bolus data from series
            boluses = []
            if "series" in data:
                for series_name, series_data in data["series"].items():
                    if series_data and isinstance(series_data, list):
                        for point in series_data:
                            if point:
                                boluses.append({
                                    "series": series_name,
                                    "timestamp": point.get("timestamp"),
                                    "value": point.get("y"),
                                    "data": point,
                                })
            
            print(f"Retrieved {len(boluses)} bolus records from Graph API")
            return boluses

        except requests.exceptions.RequestException as e:
            print(f"Error fetching bolus data: {e}")
            return []

    def fetch_bolus_data_v2(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict]:
        """Fetch bolus data using v2 API (may not work without proper session)."""
        params = {
            "patient": self.patient_id,
            "startDate": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "endDate": end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }

        url = urljoin(self.api_base, ENDPOINTS["normal_boluses"])
        print(f"Trying v2 API for bolus data...")

        try:
            headers = {
                "Accept": "application/json",
                "Referer": self.web_base + "/",
            }
            
            response = self.session.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            boluses = data.get("normalBoluses", [])
            print(f"Retrieved {len(boluses)} bolus records from v2 API")
            return boluses

        except requests.exceptions.RequestException as e:
            print(f"v2 API failed: {e}")
            return []


def parse_graph_bolus(bolus: dict) -> dict:
    """Parse a bolus record from Graph API into a simplified format."""
    data = bolus.get("data", {})
    return {
        "timestamp": bolus.get("timestamp", ""),
        "series": bolus.get("series", ""),
        "insulin": bolus.get("value") or data.get("insulin") or data.get("y", 0),
        "carbs": data.get("carbs", 0),
        "bg": data.get("bg"),
        "raw_data": str(data),
    }


def parse_v2_bolus(bolus: dict) -> dict:
    """Parse a bolus record from v2 API into a simplified format."""
    return {
        "timestamp": bolus.get("pumpTimestamp", ""),
        "series": "normalBolus",
        "insulin": bolus.get("insulinDelivered", 0),
        "carbs": bolus.get("carbsInput", 0),
        "bg": bolus.get("bgInput"),
        "raw_data": str(bolus),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch bolus data from the Glooko API"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to fetch (default: 7)",
    )
    args = parser.parse_args()

    # Validate credentials
    if not EMAIL or not PASSWORD or EMAIL == "your_glooko_email":
        print("Error: Missing Glooko credentials!")
        print("Edit your .env file and set:")
        print("  GLOOKO_EMAIL=your_email@example.com")
        print("  GLOOKO_PASSWORD=your_password")
        print("  GLOOKO_REGION=us  # or 'eu'")
        print("  GLOOKO_PATIENT_ID=your-patient-id  # Required!")
        return 1

    print("Glooko Bolus Data Fetcher")
    print(f"Region: {REGION}")
    print(f"API Server: {API_SERVERS.get(REGION, API_SERVERS['default'])}")
    print(f"Web Server: {WEB_SERVERS.get(REGION, WEB_SERVERS['default'])}")
    print("-" * 50)

    # Create client and authenticate
    client = GlookoClient(EMAIL, PASSWORD, REGION, PATIENT_ID)
    if not client.login():
        return 1

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    # Try Graph API first
    boluses = client.fetch_bolus_data(start_date, end_date)
    parse_func = parse_graph_bolus
    
    # Fall back to v2 API if Graph API returns nothing
    if not boluses:
        print("Graph API returned no data, trying v2 API...")
        boluses = client.fetch_bolus_data_v2(start_date, end_date)
        parse_func = parse_v2_bolus

    if not boluses:
        print("No bolus data found for the specified date range.")
        return 0

    # Parse and display data
    print(f"\n{'Timestamp':<30} {'Series':<20} {'Insulin':>8} {'Carbs':>6}")
    print("-" * 70)

    parsed_boluses = []
    total_insulin = 0
    total_carbs = 0

    for bolus in boluses:
        parsed = parse_func(bolus)
        parsed_boluses.append(parsed)

        insulin = parsed["insulin"] or 0
        carbs = parsed["carbs"] or 0

        total_insulin += float(insulin) if insulin else 0
        total_carbs += int(carbs) if carbs else 0

        print(
            f"{parsed['timestamp']:<30} {parsed['series']:<20} {str(insulin):>8} {str(carbs):>6}"
        )

    # Summary
    print("-" * 70)
    print(f"\n--- Summary ---")
    print(f"Total bolus events: {len(parsed_boluses)}")
    print(f"Total insulin delivered: {total_insulin:.2f} units")
    print(f"Total carbs logged: {total_carbs}g")
    if len(parsed_boluses) > 0 and total_insulin > 0:
        print(f"Average insulin per bolus: {total_insulin / len(parsed_boluses):.2f} units")

    # Save to CSV
    if parsed_boluses:
        fieldnames = ["timestamp", "series", "insulin", "carbs", "bg", "raw_data"]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, OUTPUT_FILE)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(parsed_boluses)

        print(f"\nSaved {len(parsed_boluses)} records to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
