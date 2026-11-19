"""
data_prep.py — Build an order-level feature table from the raw Olist dataset.

Every feature here is something a hub / ops team would KNOW at the moment the
order is approved (before it ships). Nothing from the future leaks in.

Usage:
    python src/data_prep.py --raw data/raw --out data/orders_features.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build(raw: Path) -> pd.DataFrame:
    r = lambda name: pd.read_csv(raw / name)
    orders = r("olist_orders_dataset.csv")
    items = r("olist_order_items_dataset.csv")
    customers = r("olist_customers_dataset.csv")
    sellers = r("olist_sellers_dataset.csv")
    products = r("olist_products_dataset.csv")
    reviews = r("olist_order_reviews_dataset.csv")
    geo = r("olist_geolocation_dataset.csv")
    cat_tr = r("product_category_name_translation.csv")

    # ---- 1. Keep delivered orders with complete timestamps -----------------
    ts_cols = ["order_purchase_timestamp", "order_approved_at",
               "order_delivered_carrier_date", "order_delivered_customer_date",
               "order_estimated_delivery_date"]
    for c in ts_cols:
        orders[c] = pd.to_datetime(orders[c])
    orders = orders[orders.order_status == "delivered"].dropna(subset=ts_cols)

    # ---- 2. Target: was the order delivered AFTER the promised date? --------
    orders["is_late"] = (orders.order_delivered_customer_date.dt.normalize()
                         > orders.order_estimated_delivery_date).astype(int)
    orders["days_late"] = (orders.order_delivered_customer_date
                           - orders.order_estimated_delivery_date).dt.days

    # ---- 3. Zip-prefix -> lat/lon (mean of points in that prefix) ----------
    geo = geo[(geo.geolocation_lat.between(-34, 6)) & (geo.geolocation_lng.between(-74, -34))]
    zip_ll = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()

    customers = customers.join(zip_ll, on="customer_zip_code_prefix").rename(
        columns={"geolocation_lat": "cust_lat", "geolocation_lng": "cust_lng"})
    sellers = sellers.join(zip_ll, on="seller_zip_code_prefix").rename(
        columns={"geolocation_lat": "sell_lat", "geolocation_lng": "sell_lng"})

    # ---- 4. Item-level -> order-level aggregates ---------------------------
    products = products.merge(cat_tr, on="product_category_name", how="left")
    products["category"] = products.product_category_name_english.fillna("unknown")
    products["volume_cm3"] = (products.product_length_cm * products.product_height_cm
                              * products.product_width_cm)
    it = (items.merge(products[["product_id", "category", "product_weight_g", "volume_cm3"]],
                      on="product_id", how="left")
               .merge(sellers, on="seller_id", how="left"))
    it["shipping_limit_date"] = pd.to_datetime(it.shipping_limit_date)

    # the "main" seller/category of an order = the most expensive item
    main = it.sort_values("price", ascending=False).drop_duplicates("order_id")
    agg = it.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        shipping_limit=("shipping_limit_date", "min"),
    )
    main = main.set_index("order_id")[["seller_id", "seller_state", "seller_city",
                                       "sell_lat", "sell_lng", "category"]]

    df = (orders.merge(customers, on="customer_id", how="left")
                .join(agg, on="order_id").join(main, on="order_id"))

    # ---- 5. Features known at approval time -------------------------------
    df["distance_km"] = haversine_km(df.sell_lat, df.sell_lng, df.cust_lat, df.cust_lng)
    df["same_state"] = (df.seller_state == df.customer_state).astype(int)
    df["promised_days"] = (df.order_estimated_delivery_date - df.order_purchase_timestamp).dt.days
    df["approval_lag_hrs"] = (df.order_approved_at - df.order_purchase_timestamp).dt.total_seconds() / 3600
    df["handling_window_days"] = (df.shipping_limit - df.order_approved_at).dt.total_seconds() / 86400
    df["purchase_dow"] = df.order_purchase_timestamp.dt.dayofweek
    df["purchase_hour"] = df.order_purchase_timestamp.dt.hour
    df["purchase_month"] = df.order_purchase_timestamp.dt.month
    df["freight_ratio"] = df.total_freight / df.total_price.clip(lower=1)
    df["route"] = df.seller_state + "→" + df.customer_state

    # ---- 6. Review score (outcome, used only for business-impact analysis) --
    rv = reviews.drop_duplicates("order_id")[["order_id", "review_score"]]
    df = df.merge(rv, on="order_id", how="left")

    keep = ["order_id", "order_purchase_timestamp", "customer_state", "customer_city",
            "seller_id", "seller_state", "seller_city", "category", "route",
            "n_items", "n_sellers", "total_price", "total_freight", "freight_ratio",
            "total_weight_g", "total_volume_cm3", "distance_km", "same_state",
            "promised_days", "approval_lag_hrs", "handling_window_days",
            "purchase_dow", "purchase_hour", "purchase_month",
            "cust_lat", "cust_lng", "is_late", "days_late", "review_score"]
    return df[keep].dropna(subset=["distance_km", "total_weight_g"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/orders_features.parquet")
    a = ap.parse_args()
    out = build(Path(a.raw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"{len(out):,} orders | late rate {out.is_late.mean():.1%} -> {a.out}")
