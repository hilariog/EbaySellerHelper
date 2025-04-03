"""
Advanced analysis script for eBay sellers provides the following analysis:
Metrics:
-Market quality = num_sold / total
-Market quality by auction and buy now
-Condition success metric comparison
-Seller feedback success comparison
-Brand analysis
-Review rating vs success comparison
-Success metric vs shipping cost
-Plot and correlation
-Success for shipped vs not shipped (local pickup)
-Auction vs buy now success and final price bar plot
-Success metric vs number of images comparison
-Plot and correlation
-Histogram of success metric to show distribution
-Success and price over the last 90 days

This is achieved by building a feature dictionary for all items returned by our requests:
item_data = {
    "title": title,
    "is_sold": is_sold,
    "is_auction": is_auction,
    "buy_it_now": buy_it_now,
    "final_price": final_price,
    "ended_dt": ended_dt,
    "image_count": image_count,
    "shipping_price": shipping_price,
    "time_sold_months": time_sold_months,
    "success_metric": success_val,
    "condition_name": condition_name.lower(),
    "is_new": is_new,
    "is_used": is_used,
    "feedback_score": feedback_score,
    "feedback_percent": feedback_percent,
    "brand": brand_str,
    "avg_review_rating": avg_review_rating
}

These matrics are used in calculation for all of the output metrics and plots

Author: Hilario Gonzalez
Date: 2025-04-01
"""

import requests
import json
import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from collections import defaultdict

# ------------------- CONFIGURATION -------------------
API_KEY = ""  # remove before pushing code
EBAY_DOMAIN = "ebay.com"
SEARCH_TERM = "Playstation five ps5"

# For minimal usage while testing:
RESULTS_PER_REQUEST = 60   # extremely small page number
MAX_SEARCH_RESULTS = 1    # max items to process from each search page (sold or unsold)
MAX_PRODUCT_CALLS = 1     # max product calls total (for advanced data)

#######################################################

def fetch_search_results(sold: bool) -> list:
    """
    Performs a type=search call for either sold or not-sold listings in the last ~90 days
    using a very small page size. Returns a list of items from 'search_results',
    but clipped to MAX_SEARCH_RESULTS to minimize usage.
    """
    params = {
        "api_key": API_KEY,
        "ebay_domain": EBAY_DOMAIN,
        "search_term": SEARCH_TERM,
        "type": "search",
        "completed_items": "true",
        "sold_items": str(sold).lower(),  # "true" or "false"
        "num": RESULTS_PER_REQUEST,
        # no max_page => only page=1 results
    }
    resp = requests.get("https://api.countdownapi.com/request", params=params)

    if resp.status_code != 200:
        print(f"[Error] fetch_search_results: status code={resp.status_code}")
        return []

    data = resp.json()
    # Basic success check
    if not data.get("request_info", {}).get("success", False):
        print("[Error] fetch_search_results: success=False from Countdown API.")
        return []

    results = data.get("search_results", [])
    # Clip to MAX_SEARCH_RESULTS
    return results[:MAX_SEARCH_RESULTS]

def fetch_product_details(epid_or_url: str, product_call_count: list) -> dict:
    """
    Performs a type=product request for an individual listing (by EPID or URL),
    but only if we haven't exceeded MAX_PRODUCT_CALLS. Return empty dict if limit reached.

    We store a single-element list for product_call_count so we can mutate it here.
    """
    if product_call_count[0] >= MAX_PRODUCT_CALLS:
        return {}

    params = {
        "api_key": API_KEY,
        "type": "product",
        "ebay_domain": EBAY_DOMAIN,
        "epid": epid_or_url
    }
    resp = requests.get("https://api.countdownapi.com/request", params=params)
    product_call_count[0] += 1  # increment usage

    if resp.status_code != 200:
        print(f"[Error] Product request failed with status code {resp.status_code}")
        return {}
    
    data = resp.json()
    
    if not data.get("request_info", {}).get("success", False):
        print("[Error] Product request returned success=False from Countdown API.")
        return {}

    # If it's a master page => skip (top picks can blow up the data)
    if data.get("is_master", False):
        print("[Info] Skipping master page to keep usage small.")
        return {}

    return data

def parse_ended_date(search_item: dict):
    """
    Attempt to parse the ended date from a search result item.
    The raw date is often in item["ended"]["date"]["raw"], e.g. "Aug 27, 2021".
    Returns a datetime (UTC) or None if not parsed.
    """
    ended_obj = search_item.get("ended", {})
    date_obj = ended_obj.get("date", {})
    raw_date = date_obj.get("raw")
    if not raw_date:
        return None

    for fmt in ["%b %d, %Y", "%b-%d-%Y", "%b %d %Y"]:
        try:
            dt = datetime.strptime(raw_date, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None

def parse_time_sold(product_data: dict) -> float:
    """
    Use the difference in months between product_data["product"]["last_updated"]
    and auction["end_date"]["utc"] as a stand-in for how long the item was listed.
    """
    if "auction" not in product_data:
        return 0.0

    auction_obj = product_data["auction"]
    end_date_obj = auction_obj.get("end_date", {})
    utc_str = end_date_obj.get("utc")
    if not utc_str:
        return 0.0

    # Parse end_date "utc" as iso
    try:
        end_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    except ValueError:
        print("[Warning] parse_time_sold: couldn't parse end_date.utc.")
        return 0.0

    # Parse product["last_updated"]
    last_updated_str = product_data.get("product", {}).get("last_updated", "")
    if not last_updated_str:
        return 0.0

    try:
        tokens = last_updated_str.split()
        # example: "Dec 25, 2020 17:02:13 PST"
        if len(tokens) < 4:
            return 0.0
        date_time_str = " ".join(tokens[:4])
        dt_updated = datetime.strptime(date_time_str, "%b %d, %Y %H:%M:%S")
        dt_updated = dt_updated.replace(tzinfo=timezone.utc)
    except ValueError:
        print("[Warning] parse_time_sold: couldn't parse product['last_updated'].")
        return 0.0

    # Now compute difference in months
    delta = end_dt - dt_updated
    days = delta.days + (delta.seconds / 86400.0)
    months = days / 30.0
    if months < 0:
        return 0.0
    return months

def compute_success_metric(final_price: float, time_sold_months: float) -> float:
    """
    success = final_price * min(1.25, (1/(time_sold_months + 1) + 1))
    """
    if final_price <= 0:
        return 0.0
    factor = (1.0 / (time_sold_months + 1.0)) + 1.0
    factor = min(1.25, factor)
    return final_price * factor

def pearson_corr(xs, ys):
    """
    Basic Pearson correlation helper.
    """
    if len(xs) < 2:
        return 0.0
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov = sum((xx - mean_x)*(yy - mean_y) for xx, yy in zip(xs, ys))
    var_x = sum((xx - mean_x)**2 for xx in xs)
    var_y = sum((yy - mean_y)**2 for yy in ys)
    if var_x > 0 and var_y > 0:
        return cov / math.sqrt(var_x*var_y)
    return 0.0

def main():
    print("=== Starting advanced EDA with minimal retrieval & advanced tasks ===")
    sold_results = fetch_search_results(sold=True)
    not_sold_results = fetch_search_results(sold=False)

    all_items = []

    # We'll track how many product calls we've made so far
    product_call_count = [0]

    def process_item(search_item: dict, is_sold: bool):
        title = search_item.get("title", "")
        is_auction = search_item.get("is_auction", False)
        buy_it_now = search_item.get("buy_it_now", False)
        
        # Price from search results
        prices_arr = search_item.get("prices", [])
        final_price = prices_arr[0].get("value", 0.0) if prices_arr else 0.0

        ended_dt = parse_ended_date(search_item)
        
        epid = search_item.get("epid", None)
        product_data = {}
        if epid:
            pd = fetch_product_details(epid, product_call_count)
            if pd:
                product_data = pd

        # parse time sold in months
        time_sold_months = parse_time_sold(product_data)

        # glean shipping_price or image_count
        image_count = None
        shipping_price = None
        if product_data and "product" in product_data:
            image_count = product_data["product"].get("image_count", None)
        
        shipping_obj = product_data.get("shipping", {})
        shipping_raw = shipping_obj.get("price", None)
        
        # Convert "Free" to 0.0, or parse float if numeric
        if isinstance(shipping_raw, str):
            if shipping_raw.lower() == "free":
                shipping_price = 0.0
            else:
                try:
                    shipping_price = float(shipping_raw)
                except ValueError:
                    shipping_price = None
        else:
            shipping_price = shipping_raw if isinstance(shipping_raw, (int, float)) else None

        # compute success metric
        success_val = compute_success_metric(final_price, time_sold_months)

        # Data cleaning checks:
        if final_price < 0:
            return
        if success_val < 0:
            return

        # (1) Condition data
        cond_obj = product_data.get("condition", {})
        condition_name = cond_obj.get("name", "unknown")  
        is_new = cond_obj.get("is_new", False)
        is_used = cond_obj.get("is_used", False)

        # (2) Seller data
        seller_obj = product_data.get("seller", {})
        feedback_score = seller_obj.get("feedback_score", 0)
        feedback_percent = seller_obj.get("positive_feedback_percent", 0.0)

        # (3) Keyword / attributes
        brand_str = None
        attributes_arr = product_data.get("product", {}).get("attributes", [])
        for attr in attributes_arr:
            nm = attr.get("name", "").lower()
            val = attr.get("value", "")
            if nm == "brand":
                brand_str = val.lower()

        # (4) Reviews => average rating
        reviews_arr = product_data.get("reviews", [])
        avg_review_rating = 0.0
        if reviews_arr:
            total_r = sum(r.get("rating", 0) for r in reviews_arr)
            avg_review_rating = total_r / len(reviews_arr)

        item_data = {
            "title": title,
            "is_sold": is_sold,
            "is_auction": is_auction,
            "buy_it_now": buy_it_now,
            "final_price": final_price,
            "ended_dt": ended_dt,
            "image_count": image_count,
            "shipping_price": shipping_price,
            "time_sold_months": time_sold_months,
            "success_metric": success_val,
            "condition_name": condition_name.lower(),
            "is_new": is_new,
            "is_used": is_used,
            "feedback_score": feedback_score,
            "feedback_percent": feedback_percent,
            "brand": brand_str,
            "avg_review_rating": avg_review_rating
        }
        all_items.append(item_data)

    # Process a set of sold & not_sold
    for item in sold_results:
        process_item(item, True)
    for item in not_sold_results:
        process_item(item, False)

    print(f"Sold search results => {len(sold_results)} items, Unsold => {len(not_sold_results)}")
    print(f"After product calls => total items: {len(all_items)}, product calls used={product_call_count[0]}")

    if not all_items:
        print("No items to analyze. Exiting.")
        return

    # Basic stats
    num_sold = sum(1 for x in all_items if x["is_sold"])
    num_not_sold = len(all_items) - num_sold
    market_quality = num_sold / len(all_items) if all_items else 0.0

    print(f"Market Quality (#sold / #total) = {market_quality:.3f}")

    # Quick Auction vs. Buy Now
    auction_items = [x for x in all_items if x["is_auction"]]
    buy_now_items = [x for x in all_items if x["buy_it_now"]]
    auction_sold = sum(1 for x in auction_items if x["is_sold"])
    buy_now_sold = sum(1 for x in buy_now_items if x["is_sold"])
    mq_auction = auction_sold / len(auction_items) if auction_items else 0.0
    mq_buynow  = buy_now_sold / len(buy_now_items) if buy_now_items else 0.0

    print(f"Market Quality (auction)={mq_auction:.3f}, (buy now)={mq_buynow:.3f}")

    # Condition data => see if "new" vs. "used"
    cond_new_items = [x for x in all_items if x["condition_name"].startswith("new") or x["is_new"]]
    cond_used_items = [x for x in all_items if x["condition_name"].startswith("pre") or x["is_used"]]

    if cond_new_items:
        avg_succ_new = statistics.mean(x["success_metric"] for x in cond_new_items)
        print(f"Avg Success (NEW) => {avg_succ_new:.3f} with {len(cond_new_items)} items")
    if cond_used_items:
        avg_succ_used = statistics.mean(x["success_metric"] for x in cond_used_items)
        print(f"Avg Success (USED) => {avg_succ_used:.3f} with {len(cond_used_items)} items")

    # Seller feedback => correlation with success
    with_feedback = [x for x in all_items if x["feedback_score"]>0]
    if len(with_feedback)>1:
        def pearson(list_x, list_y):
            return pearson_corr(list_x, list_y)
        seller_corr = pearson(
            [float(x["feedback_score"]) for x in with_feedback],
            [float(x["success_metric"]) for x in with_feedback]
        )
        print(f"Correlation (seller feedback_score vs. success_metric) => {seller_corr:.3f}")

    # Basic brand analysis => see how many items per brand, average success
    brand_map = defaultdict(lambda: {"count": 0, "sum_succ": 0.0})
    for i in all_items:
        if i["brand"]:
            brand_map[i["brand"]]["count"] += 1
            brand_map[i["brand"]]["sum_succ"] += i["success_metric"]

    if brand_map:
        print("\n=== Brand Analysis ===")
        for b, info in brand_map.items():
            c = info["count"]
            if c > 0:
                avg_s = info["sum_succ"] / c
                print(f" Brand={b}, count={c}, avg_success={avg_s:.3f}")

    # Reviews => average rating vs. success => correlation
    with_reviews = [x for x in all_items if x["avg_review_rating"]>0]
    if len(with_reviews) > 1:
        rev_corr = pearson_corr(
            [float(x["avg_review_rating"]) for x in with_reviews],
            [float(x["success_metric"]) for x in with_reviews]
        )
        print(f"Correlation (avg_review_rating vs. success_metric) => {rev_corr:.3f}")

    # Shipping correlation
    sold_with_ship = [x for x in all_items if x["is_sold"] and x["shipping_price"] is not None]
    x_ship = []
    y_succ_ship = []
    for x in sold_with_ship:
        try:
            val = float(x["shipping_price"])
            x_ship.append(val)
            y_succ_ship.append(float(x["success_metric"]))
        except (TypeError, ValueError):
            continue
    if len(x_ship)>1:
        corr_ship = pearson_corr(x_ship, y_succ_ship)
        print(f"Shipping Price vs. Success corr => {corr_ship:.3f}")

    # Success for shipped vs. not shipped
    shipped_items = [it for it in sold_with_ship if it["shipping_price"] and it["shipping_price"]>0]
    local_or_free = [it for it in sold_with_ship if it["shipping_price"]==0 or it["shipping_price"] is None]

    avg_succ_shipped = statistics.mean([s["success_metric"] for s in shipped_items]) if shipped_items else 0.0
    avg_succ_local   = statistics.mean([s["success_metric"] for s in local_or_free]) if local_or_free else 0.0

    print(f"Avg Success (shipped>0) = {avg_succ_shipped:.3f}")
    print(f"Avg Success (free shipping/ local pickup) = {avg_succ_local:.3f}")

    # Auction vs. Buy Now bar plot
    sold_auction = [x for x in all_items if x["is_sold"] and x["is_auction"]]
    sold_buy_now = [x for x in all_items if x["is_sold"] and x["buy_it_now"]]

    avg_succ_auction = statistics.mean([sa["success_metric"] for sa in sold_auction]) if sold_auction else 0.0
    avg_succ_buynow  = statistics.mean([sb["success_metric"] for sb in sold_buy_now]) if sold_buy_now else 0.0
    avg_price_auction = statistics.mean([sa["final_price"] for sa in sold_auction]) if sold_auction else 0.0
    avg_price_buynow  = statistics.mean([sb["final_price"] for sb in sold_buy_now]) if sold_buy_now else 0.0

    print(f"Auction => avg_succ={avg_succ_auction:.2f}, avg_price={avg_price_auction:.2f}")
    print(f"Buy Now => avg_succ={avg_succ_buynow:.2f}, avg_price={avg_price_buynow:.2f}")

    labels = ["Auction","Buy Now"]
    success_means = [avg_succ_auction, avg_succ_buynow]
    price_means   = [avg_price_auction, avg_price_buynow]

    import numpy as np
    xvals = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(xvals - width/2, success_means, width, label='Avg Success')
    plt.bar(xvals + width/2, price_means, width, label='Avg Price')
    plt.xticks(xvals, labels)
    plt.ylabel("Values")
    plt.title("Auction vs. Buy Now: Avg Success & Final Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Success metric vs image_count correlation & plot
    with_imgs = [x for x in all_items if x["image_count"] and x["image_count"]>0]
    if len(with_imgs)>1:
        x_imgs = [float(x["image_count"]) for x in with_imgs]
        y_succ = [float(x["success_metric"]) for x in with_imgs]
        corr_imgs = pearson_corr(x_imgs, y_succ)
        print(f"Correlation (image_count vs. success_metric) => {corr_imgs:.3f}")

        plt.figure()
        plt.scatter(x_imgs, y_succ)
        plt.title("Image Count vs. Success Metric")
        plt.xlabel("Number of Images")
        plt.ylabel("Success Metric")
        plt.tight_layout()
        plt.show()
    else:
        print("[Note] Not enough items with image_count>0 to do correlation/plot.")


    # Basic histogram of success_metric
    all_success = [x["success_metric"] for x in all_items if x["success_metric"]>0]
    if all_success:
        plt.figure()
        plt.hist(all_success, bins=5, edgecolor='black')
        plt.title("Histogram of Success Metrics")
        plt.xlabel("Success Metric")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("[Note] No valid success metrics for histogram.")

    # Weekly average final sale price & success metric
    weekly_data = defaultdict(lambda: {"sum_price":0.0, "sum_succ":0.0, "count":0})
    for i in all_items:
        dt = i["ended_dt"]
        if dt is None:
            continue
        y, w, _ = dt.isocalendar()
        key = (y, w)
        weekly_data[key]["sum_price"] += i["final_price"]
        weekly_data[key]["sum_succ"]  += i["success_metric"]
        weekly_data[key]["count"]     += 1

    sorted_keys = sorted(weekly_data.keys())
    if sorted_keys:
        x_idx = range(len(sorted_keys))
        week_labels = []
        avg_prices  = []
        avg_succ    = []
        for k in sorted_keys:
            c = weekly_data[k]["count"]
            if c == 0:
                continue
            p = weekly_data[k]["sum_price"] / c
            s = weekly_data[k]["sum_succ"]  / c
            week_labels.append(k)  # e.g. (year, week)
            avg_prices.append(p)
            avg_succ.append(s)
        plt.figure(figsize=(8,5))
        plt.plot(x_idx, avg_prices, label="Avg Final Price", marker='o')
        plt.plot(x_idx, avg_succ, label="Avg Success Metric", marker='s')
        plt.xticks(x_idx, [f"{yr}-W{wk}" for (yr,wk) in week_labels], rotation=45)
        plt.title("Weekly Averages (Last 90 Days)")
        plt.xlabel("Year-Week")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("[Note] No ended_dt data for weekly averages.")


if __name__ == "__main__":
    main()
