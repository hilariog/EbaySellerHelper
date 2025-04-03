#!/usr/bin/env python3
"""
A minimal test script to mimic the request approach (sold vs unsold) 
without any advanced analysis. This lets you verify that 
the request syntax and parameters work correctly.
"""

import requests
import json

# ------------------- BASIC CONFIG -------------------
API_KEY = ""  # Replace with your actual key
EBAY_DOMAIN = "ebay.com"
SEARCH_TERM = "playstation five ps5"  # Example keyword
RESULTS_PER_REQUEST = 0      # Use a valid num for Countdown API (e.g., 60)
# -----------------------------------------------------

def fetch_search_results(sold: bool) -> list:
    """
    Performs a type=search call for either sold or not-sold listings in the last ~90 days.
    Returns a list of items from 'search_results'.
    """
    params = {
        "api_key": API_KEY,
        "ebay_domain": EBAY_DOMAIN,
        "search_term": SEARCH_TERM,
        "type": "search",
        "completed_items": "true", 
        "sold_items": str(sold).lower(),   # 'true' or 'false'
        "num": RESULTS_PER_REQUEST
    }
    resp = requests.get("https://api.countdownapi.com/request", params=params)

    print(f"\n[SOLD={sold}] HTTP Status:", resp.status_code)
    if resp.status_code != 200:
        print("[Error] Could not fetch data:", resp.text)
        return []

    data = resp.json()
    if not data.get("request_info", {}).get("success", False):
        print("[Error] API returned success=False.")
        return []

    results = data.get("search_results", [])
    return results

def main():
    print("=== Testing search requests ===")
    sold_results = fetch_search_results(sold=True)
    unsold_results = fetch_search_results(sold=False)

    print("\n--- SOLD RESULTS ---")
    print(f"Received {len(sold_results)} sold items.")
    if sold_results:
        # Just print a snippet from the first item
        print("Example sold item title:", sold_results[0].get("title", "N/A"))

    print("\n--- UNSOLD (completed but not sold) RESULTS ---")
    print(f"Received {len(unsold_results)} unsold items.")
    if unsold_results:
        print("Example unsold item title:", unsold_results[0].get("title", "N/A"))

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
