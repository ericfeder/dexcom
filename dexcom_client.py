#!/usr/bin/env python3
"""
Dexcom API Client - Fetch blood sugar readings from Dexcom G7

SETUP:
1. Create a .env file in this directory with your Dexcom API credentials:
   
   DEXCOM_CLIENT_ID=your_client_id
   DEXCOM_CLIENT_SECRET=your_client_secret

2. Get credentials at: https://developer.dexcom.com/
   - Create an app with redirect URI: https://localhost:8000/callback

3. Install dependencies:
   pip install -r requirements.txt

USAGE:
   python dexcom_client.py              # Last 2 days (default)
   python dexcom_client.py --days 90    # Last 90 days
"""

import argparse
import csv
import json
import os
import ssl
import subprocess
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CLIENT_ID = os.getenv("DEXCOM_CLIENT_ID")
CLIENT_SECRET = os.getenv("DEXCOM_CLIENT_SECRET")
REDIRECT_URI = "https://localhost:8000/callback"
CERT_FILE = "localhost.pem"
KEY_FILE = "localhost-key.pem"
TOKEN_FILE = "tokens.json"

# Dexcom API endpoints - Production (US)
AUTH_URL = "https://api.dexcom.com/v2/oauth2/login"
TOKEN_URL = "https://api.dexcom.com/v2/oauth2/token"
API_BASE = "https://api.dexcom.com/v3"


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback from Dexcom."""
    
    auth_code = None
    
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/callback":
            query = parse_qs(parsed.query)
            if "code" in query:
                OAuthCallbackHandler.auth_code = query["code"][0]
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>")
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                error = query.get("error", ["Unknown error"])[0]
                self.wfile.write(f"<html><body><h1>Authorization failed</h1><p>{error}</p></body></html>".encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


def ensure_ssl_certs():
    """Generate self-signed SSL certificates if they don't exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(script_dir, CERT_FILE)
    key_path = os.path.join(script_dir, KEY_FILE)
    
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path
    
    print("Generating self-signed SSL certificate...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", key_path,
        "-out", cert_path,
        "-days", "365",
        "-nodes",
        "-subj", "/CN=localhost"
    ], check=True, capture_output=True)
    print("SSL certificate generated.")
    
    return cert_path, key_path


def get_authorization_code():
    """Open browser for user authorization and capture the code."""
    cert_path, key_path = ensure_ssl_certs()
    
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "offline_access",
    }
    auth_url = f"{AUTH_URL}?{urlencode(params)}"
    
    print("Opening browser for Dexcom authorization...")
    print(f"If browser doesn't open, visit: {auth_url}")
    print("\nNOTE: Your browser will warn about the self-signed certificate.")
    print("Click 'Advanced' and 'Proceed to localhost' to continue.\n")
    webbrowser.open(auth_url)
    
    # Start local HTTPS server to capture callback
    server = HTTPServer(("localhost", 8000), OAuthCallbackHandler)
    server.timeout = 120  # 2 minute timeout
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_path, key_path)
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
    
    # Handle requests until we get the auth code (handles favicon, etc.)
    while OAuthCallbackHandler.auth_code is None:
        try:
            server.handle_request()
        except ssl.SSLError:
            # Browser may probe the connection, ignore SSL errors
            continue
    
    return OAuthCallbackHandler.auth_code


def exchange_code_for_tokens(code):
    """Exchange authorization code for access and refresh tokens."""
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }
    
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json()


def refresh_access_token(refresh_token):
    """Refresh the access token using the refresh token."""
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json()


def save_tokens(tokens):
    """Save tokens to file."""
    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f)


def load_tokens():
    """Load tokens from file if they exist."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    return None


def get_access_token():
    """Get a valid access token, refreshing or re-authenticating as needed."""
    tokens = load_tokens()
    
    if tokens:
        # Try to use existing tokens, refresh if needed
        try:
            new_tokens = refresh_access_token(tokens["refresh_token"])
            save_tokens(new_tokens)
            return new_tokens["access_token"]
        except requests.exceptions.HTTPError:
            print("Token refresh failed, re-authenticating...")
    
    # Need to authenticate from scratch
    code = get_authorization_code()
    if not code:
        raise Exception("Failed to get authorization code")
    
    tokens = exchange_code_for_tokens(code)
    save_tokens(tokens)
    return tokens["access_token"]


def fetch_egv_data(access_token, start_date, end_date, chunk_days=30):
    """Fetch estimated glucose values (EGV) for the given date range in chunks."""
    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    
    all_records = []
    current_start = start_date
    chunk_num = 1
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        # Format dates as ISO 8601
        start_str = current_start.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = current_end.strftime("%Y-%m-%dT%H:%M:%S")
        
        print(f"  Fetching chunk {chunk_num}: {current_start.date()} to {current_end.date()}...")
        
        params = {
            "startDate": start_str,
            "endDate": end_str,
        }
        
        url = f"{API_BASE}/users/self/egvs"
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        records = data.get("records", [])
        all_records.extend(records)
        print(f"    Got {len(records)} readings")
        
        current_start = current_end
        chunk_num += 1
    
    # Return combined data in same format as API
    return {
        "recordType": "egv",
        "recordVersion": "3.0",
        "records": all_records
    }


def format_trend(trend):
    """Convert trend direction to readable arrow."""
    trend_arrows = {
        "doubleUp": "⬆⬆",
        "singleUp": "⬆",
        "fortyFiveUp": "↗",
        "flat": "→",
        "fortyFiveDown": "↘",
        "singleDown": "⬇",
        "doubleDown": "⬇⬇",
        "notComputable": "?",
        "rateOutOfRange": "?",
    }
    return trend_arrows.get(trend, trend)


def main():
    parser = argparse.ArgumentParser(description="Fetch blood sugar data from Dexcom API")
    parser.add_argument("--days", type=int, default=365, help="Number of days to fetch (default: 365)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted table")
    parser.add_argument("--csv", type=str, metavar="FILE", help="Save readings to CSV file")
    args = parser.parse_args()
    
    # Validate credentials
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Error: Missing Dexcom API credentials!")
        print("Create a .env file with:")
        print("  DEXCOM_CLIENT_ID=your_client_id")
        print("  DEXCOM_CLIENT_SECRET=your_client_secret")
        return 1
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)
    
    print(f"Fetching blood sugar data from {start_date.date()} to {end_date.date()}...")
    
    try:
        access_token = get_access_token()
        data = fetch_egv_data(access_token, start_date, end_date)
        
        if args.csv:
            records = data.get("records", [])
            fieldnames = [
                "recordId", "systemTime", "displayTime", "transmitterId",
                "transmitterTicks", "value", "trend", "trendRate",
                "unit", "rateUnit", "displayDevice", "transmitterGeneration", "displayApp"
            ]
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
            print(f"Saved {len(records)} readings to {args.csv}")
            return 0
        
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            records = data.get("records", [])
            if not records:
                print("No readings found for this period.")
                return 0
            
            print(f"\nFound {len(records)} readings:\n")
            print(f"{'Timestamp':<25} {'mg/dL':>8} {'Trend':>8}")
            print("-" * 45)
            
            for record in records:
                timestamp = record.get("displayTime", record.get("systemTime", "Unknown"))
                value = record.get("value", "N/A")
                trend = format_trend(record.get("trend", ""))
                print(f"{timestamp:<25} {value:>8} {trend:>8}")
            
            # Summary stats
            values = [r["value"] for r in records if "value" in r]
            if values:
                avg = sum(values) / len(values)
                print(f"\n--- Summary ---")
                print(f"Readings: {len(values)}")
                print(f"Average: {avg:.1f} mg/dL")
                print(f"Min: {min(values)} mg/dL")
                print(f"Max: {max(values)} mg/dL")
        
        return 0
        
    except requests.exceptions.HTTPError as e:
        print(f"API Error: {e}")
        if e.response is not None:
            print(f"Response: {e.response.text}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

