"""
train.py — Train the late-delivery risk model and export everything the app needs.

Design notes:
  * TIME-BASED split (train: before May-2018, test: May–Aug 2018). A random split
    leaks "future" information and overstates accuracy.
  * All history features (route transit time, seller late rate) use ONLY orders
    placed earlier — an expanding window shifted by one — so no order ever sees
    its own outcome.
  * The success metric is operational: "If a hub team can only review the
    riskiest X% of orders each day, what share of late orders do they catch?"
  * We compared a 21-feature gradient-boosted model with a 2-signal logistic
    regression. The simple model generalised better to the unseen test period
    (when the platform shortened its promised delivery dates) AND gives reasons
    an ops associate can act on. We ship the simple one.

Usage:
    python src/train.py --data data/orders_features.parquet --out artifacts
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPLIT_DATE = "2018-05-01"
MODEL_FEATURES = ["slack_days", "seller_prior_late_rate"]

# Wide feature set used for the "complex model" comparison
WIDE_CAT = ["customer_state", "seller_state", "category"]
WIDE_NUM = ["n_items", "n_sellers", "total_price", "total_freight", "freight_ratio",
            "total_weight_g", "total_volume_cm3", "distance_km", "same_state",
            "promised_days", "approval_lag_hrs", "handling_window_days",
            "purchase_dow", "purchase_hour", "seller_prior_orders",
            "seller_prior_late_rate", "route_hist_days", "slack_days"]


def expanding_mean(df, key, col, k):
    """Mean of `col` over EARLIER rows in the same group, shrunk toward the
    global mean with k pseudo-observations (so new sellers/routes aren't extreme)."""
    g = df.groupby(key)[col]
    n = g.cumcount()
    prior_sum = g.cumsum() - df[col]
    return (prior_sum + k * df[col].mean()) / (n + k), n


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df["actual_days"] = df.promised_days + df.days_late
    # How long does this seller-state → customer-state lane REALLY take?
    df["route_hist_days"], df["route_prior_orders"] = expanding_mean(df, "route", "actual_days", k=20)
    # Slack = days promised to the customer minus the lane's real transit time.
    df["slack_days"] = df.promised_days - df.route_hist_days
    # How often has THIS seller been late before?
    df["seller_prior_late_rate"], df["seller_prior_orders"] = expanding_mean(df, "seller_id", "is_late", k=10)
    return df


def capture(y, score, frac):
    """Share of all late orders that fall in the top `frac` riskiest orders."""
    n = max(1, int(len(y) * frac))
    return float(y[np.argsort(-score)[:n]].sum() / y.sum())


def main(data, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = add_history_features(pd.read_parquet(data))
    train = df[df.order_purchase_timestamp < SPLIT_DATE]
    test = df[df.order_purchase_timestamp >= SPLIT_DATE]
    ytr, yte = train.is_late.values, test.is_late.values

    # ---- 1. Shipped model: 2-signal logistic regression ---------------------
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(train[MODEL_FEATURES], ytr)
    p = lr.predict_proba(test[MODEL_FEATURES])[:, 1]

    # ---- 2. Comparison: "complex" gradient-boosted model, 21 features -------
    Xw = lambda d: d[WIDE_CAT + WIDE_NUM].astype({c: "category" for c in WIDE_CAT})
    gbm = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=300,
                                         learning_rate=0.05, min_samples_leaf=80,
                                         l2_regularization=1.0, random_state=42)
    gbm.fit(Xw(train), ytr)
    p_gbm = gbm.predict_proba(Xw(test))[:, 1]

    # ---- 3. Baseline: a rule ops could use today (distance ÷ promised days) --
    p_rule = (test.distance_km / test.promised_days.clip(lower=1)).values

    fr = [0.05, 0.10, 0.20, 0.30]
    scores = {"model": p, "gbm": p_gbm, "rule": p_rule}
    metrics = {
        "train_orders": int(len(train)), "test_orders": int(len(test)),
        "train_period": "Sep 2016 – Apr 2018", "test_period": "May 2018 – Aug 2018",
        "test_late_rate": float(yte.mean()),
        "compare": {name: {"roc_auc": float(roc_auc_score(yte, s)),
                           "pr_auc": float(average_precision_score(yte, s)),
                           **{f"capture_at_{int(f*100)}pct": capture(yte, s, f) for f in fr}}
                    for name, s in scores.items()},
        "coefficients": dict(zip(MODEL_FEATURES, lr[-1].coef_[0].round(3).tolist())),
        "review_late": float(df[df.is_late == 1].review_score.mean()),
        "review_ontime": float(df[df.is_late == 0].review_score.mean()),
        "low_review_share_late": float((df[df.is_late == 1].review_score <= 2).mean()),
        "low_review_share_ontime": float((df[df.is_late == 0].review_score <= 2).mean()),
    }
    fracs = np.round(np.linspace(0.01, 1, 100), 2)
    metrics["capture_curve"] = {"frac": fracs.tolist(),
                                **{k: [capture(yte, s, f) for f in fracs] for k, s in scores.items()}}
    # risk thresholds for "High / Medium / Low" bands = top 10% / next 20% of test orders
    metrics["band_thresholds"] = {"high": float(np.quantile(p, 0.90)), "medium": float(np.quantile(p, 0.70))}

    # ---- 4. Lookup tables the app needs to score a NEW order ----------------
    latest_route = df.groupby("route").agg(
        seller_state=("seller_state", "last"), customer_state=("customer_state", "last"),
        route_hist_days=("route_hist_days", "last"), orders=("is_late", "size"),
        late_rate=("is_late", "mean"), avg_km=("distance_km", "mean"),
        typical_promise=("promised_days", "median"))

    # Refit the shipped model on all data for the app
    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
        df[MODEL_FEATURES], df.is_late)
    joblib.dump({"model": final, "features": MODEL_FEATURES,
                 "global_late_rate": float(df.is_late.mean()),
                 "band_thresholds": metrics["band_thresholds"]}, out / "model.joblib")
    latest_route.to_csv(out / "routes.csv")

    # ---- 5. Scored test orders = the "daily hub queue" demo -----------------
    q = test.assign(risk=p)[["order_id", "order_purchase_timestamp", "seller_id", "seller_state",
                             "customer_state", "customer_city", "category", "distance_km",
                             "promised_days", "route_hist_days", "slack_days",
                             "seller_prior_late_rate", "seller_prior_orders", "total_price",
                             "risk", "is_late", "days_late", "review_score"]].copy()
    q["seller_id"] = q.seller_id.str[:8]
    q["order_id"] = q.order_id.str[:10]
    q.to_parquet(out / "scored_orders.parquet", index=False)

    # ---- 6. Aggregates for the insights page --------------------------------
    df["month"] = df.order_purchase_timestamp.dt.to_period("M").astype(str)
    df[df.month >= "2017-01"].groupby("month").agg(
        orders=("is_late", "size"), late_rate=("is_late", "mean"),
        promised_days=("promised_days", "median"), actual_days=("actual_days", "median")
    ).to_csv(out / "monthly.csv")
    df.groupby("customer_state").agg(orders=("is_late", "size"), late_rate=("is_late", "mean"),
                                     avg_review=("review_score", "mean")).to_csv(out / "by_state.csv")
    df.groupby(pd.cut(df.days_late.clip(-30, 30), [-31, -15, -8, -1, 0, 3, 7, 14, 31],
                      labels=["15+ days early", "8–14 early", "1–7 early", "On the day",
                              "1–3 late", "4–7 late", "8–14 late", "15+ days late"]),
               observed=True).review_score.mean().rename("avg_review").to_csv(out / "review_by_delay.csv")
    df.groupby(pd.cut(df.slack_days, [-100, 0, 3, 6, 10, 15, 200],
                      labels=["≤0", "0–3", "3–6", "6–10", "10–15", "15+"]),
               observed=True).is_late.agg(["mean", "size"]).rename(
        columns={"mean": "late_rate", "size": "orders"}).to_csv(out / "late_by_slack.csv")
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)

    show = {k: v for k, v in metrics.items() if k != "capture_curve"}
    print(json.dumps(show, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/orders_features.parquet")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    main(a.data, a.out)
