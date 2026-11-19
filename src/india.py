"""
india.py — DeliveryRadar for Indian line-haul: Delhivery hub-to-hub trip data.

Question: map-route ETAs (OSRM) are what many logistics teams plan with.
How wrong are they on Indian hub lanes, and can lane history fix them?

Data: public Delhivery trip dataset (Sep 12 – Oct 3, 2018; ~145k scan rows,
shared for educational business cases). Place it at data/raw/delhivery.csv.

Unit of analysis: one "leg" = one trip between a source hub and a destination hub.
  actual  = real transit minutes        osrm = map-route estimated minutes
  ratio   = actual / osrm               severe delay = ratio >= 3

Leakage control: the dataset's own time split (train: Sep 12–26, test: Sep 27–Oct 3),
and every history feature uses only legs that STARTED EARLIER.

Usage:
    python src/india.py --raw data/raw/delhivery.csv --out artifacts
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

FEATURES = ["osrm", "osrm_km", "carting", "lane_prior", "lane_n",
            "src_prior", "dst_prior", "hour", "dow"]
SEVERE = 3.0


def build_legs(path):
    d = pd.read_csv(path)
    for c in ["trip_creation_time", "od_start_time", "od_end_time"]:
        d[c] = pd.to_datetime(d[c], format="mixed")
    # rows are cumulative checkpoints within a leg -> max = the full leg
    leg = d.groupby(["trip_uuid", "source_center", "destination_center"]).agg(
        actual=("actual_time", "max"), osrm=("osrm_time", "max"), osrm_km=("osrm_distance", "max"),
        route_type=("route_type", "first"), src=("source_name", "first"), dst=("destination_name", "first"),
        od_start=("od_start_time", "first"), split=("data", "first")).reset_index()
    leg = leg.dropna(subset=["src", "dst"])
    leg["ratio"] = leg.actual / leg.osrm.clip(lower=1)
    leg["lr"] = np.log(leg.ratio)
    leg["src_state"] = leg.src.str.extract(r"\(([^)]+)\)$")[0]
    leg["dst_state"] = leg.dst.str.extract(r"\(([^)]+)\)$")[0]
    leg["src_city"] = leg.src.str.split("_").str[0].str.replace(r" \(.*", "", regex=True)
    leg["dst_city"] = leg.dst.str.split("_").str[0].str.replace(r" \(.*", "", regex=True)
    leg["lane"] = leg.source_center + ">" + leg.destination_center
    leg["carting"] = (leg.route_type == "Carting").astype(int)
    leg["hour"] = leg.od_start.dt.hour
    leg["dow"] = leg.od_start.dt.dayofweek
    leg["severe"] = (leg.ratio >= SEVERE).astype(int)
    return leg.sort_values("od_start").reset_index(drop=True)


def add_priors(leg):
    """Shrunk average log(actual/osrm) over EARLIER legs of the same lane / hub."""
    gm = leg.lr.mean()
    for key, name, k in [("lane", "lane", 5), ("source_center", "src", 10), ("destination_center", "dst", 10)]:
        g = leg.groupby(key)["lr"]
        n = g.cumcount()
        leg[f"{name}_prior"] = (g.cumsum() - leg.lr + k * gm) / (n + k)
        leg[f"{name}_n"] = n
    return leg


def capture(y, s, f):
    n = max(1, int(len(y) * f))
    return float(y[np.argsort(-s)[:n]].sum() / y.sum())


def main(raw, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    leg = add_priors(build_legs(raw))
    tr, te = leg[leg.split == "training"], leg[leg.split == "test"]

    # ---- ETA: predict log(actual/osrm), then ETA = osrm * exp(pred) ----------
    eta = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                        random_state=0).fit(tr[FEATURES], tr.lr)
    global_factor = float(np.exp(tr.lr.median()))
    cands = {
        "Map ETA (OSRM)": te.osrm,
        f"Map ETA × {global_factor:.1f} (one fixed buffer)": te.osrm * global_factor,
        "Map ETA × lane history": te.osrm * np.exp(te.lane_prior),
        "DeliveryRadar ETA model": te.osrm * np.exp(eta.predict(te[FEATURES])),
    }
    eta_cmp = [{"method": k, "within_20pct": float(np.mean(np.abs(v - te.actual) <= 0.2 * te.actual)),
                "median_abs_err_min": float(np.median(np.abs(v - te.actual)))} for k, v in cands.items()]

    # ---- Severe-delay flag ---------------------------------------------------
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                         random_state=0).fit(tr[FEATURES], tr.severe)
    s = clf.predict_proba(te[FEATURES])[:, 1]
    y = te.severe.values
    lane_only = te.lane_prior.values
    fracs = np.round(np.linspace(0.01, 1, 100), 2)
    metrics = {
        "legs": int(len(leg)), "train_legs": int(len(tr)), "test_legs": int(len(te)),
        "hubs": int(pd.concat([leg.source_center, leg.destination_center]).nunique()),
        "states": int(pd.concat([leg.src_state, leg.dst_state]).nunique()),
        "lanes": int(leg.lane.nunique()),
        "median_ratio": float(leg.ratio.median()),
        "share_within_20pct_of_map": float(np.mean(np.abs(leg.actual - leg.osrm) <= 0.2 * leg.actual)),
        "severe_rate_test": float(y.mean()),
        "excess_hours": float((leg.actual - leg.osrm).clip(lower=0).sum() / 60),
        "global_factor": global_factor,
        "eta_compare": eta_cmp,
        "severe": {"roc_auc": float(roc_auc_score(y, s)),
                   "capture_at_10pct": capture(y, s, .10), "capture_at_20pct": capture(y, s, .20),
                   "lane_only_roc_auc": float(roc_auc_score(y, lane_only)),
                   "lane_only_capture_at_20pct": capture(y, lane_only, .20)},
        "capture_curve": {"frac": fracs.tolist(),
                          "model": [capture(y, s, f) for f in fracs],
                          "lane_history": [capture(y, lane_only, f) for f in fracs]},
        "risk_thresholds": {"high": float(np.quantile(s, .80)), "medium": float(np.quantile(s, .60))},
    }

    # ---- Final models on all legs + lookup tables for the app ---------------
    eta_f = HistGradientBoostingRegressor(**eta.get_params()).fit(leg[FEATURES], leg.lr)
    clf_f = HistGradientBoostingClassifier(**clf.get_params()).fit(leg[FEATURES], leg.severe)
    joblib.dump({"eta": eta_f, "severe": clf_f, "features": FEATURES,
                 "thresholds": metrics["risk_thresholds"]}, out / "india_models.joblib", compress=3)

    gm = leg.lr.mean()
    lanes = leg.groupby("lane").agg(
        src=("src", "first"), dst=("dst", "first"), src_city=("src_city", "first"), dst_city=("dst_city", "first"),
        src_state=("src_state", "first"), dst_state=("dst_state", "first"),
        legs=("lr", "size"), lr_sum=("lr", "sum"), median_ratio=("ratio", "median"),
        severe_rate=("severe", "mean"), osrm=("osrm", "median"), osrm_km=("osrm_km", "median"),
        actual=("actual", "median"), carting=("carting", "mean"),
        source_center=("source_center", "first"), destination_center=("destination_center", "first"))
    lanes["lane_prior"] = (lanes.lr_sum + 5 * gm) / (lanes.legs + 5)
    lanes["lane_n"] = lanes.legs
    for key, name in [("source_center", "src"), ("destination_center", "dst")]:
        h = leg.groupby(key).lr.agg(["sum", "size"])
        lanes[f"{name}_prior"] = lanes[key].map((h["sum"] + 10 * gm) / (h["size"] + 10))
    lanes[lanes.legs >= 10].drop(columns="lr_sum").reset_index().to_csv(out / "india_lanes.csv", index=False)

    hubs = leg.groupby("dst").agg(
        state=("dst_state", "first"), inbound_legs=("lr", "size"), median_ratio=("ratio", "median"),
        severe_rate=("severe", "mean"),
        excess_hours=("actual", lambda a: float((a - leg.loc[a.index, "osrm"]).clip(lower=0).sum() / 60)))
    hubs[hubs.inbound_legs >= 50].sort_values("excess_hours", ascending=False).reset_index().to_csv(
        out / "india_hubs.csv", index=False)

    st = leg.groupby("src_state").agg(legs=("lr", "size"), median_ratio=("ratio", "median"),
                                      severe_rate=("severe", "mean"))
    st[st.legs >= 100].sort_values("severe_rate", ascending=False).reset_index().to_csv(
        out / "india_states.csv", index=False)
    leg.groupby(pd.cut(leg.osrm_km, [0, 20, 50, 100, 300, 5000],
                       labels=["<20 km", "20–50", "50–100", "100–300", "300+ km"]), observed=True).agg(
        legs=("lr", "size"), median_ratio=("ratio", "median"), severe_rate=("severe", "mean")
    ).to_csv(out / "india_by_distance.csv")
    json.dump(metrics, open(out / "india_metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "capture_curve"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/delhivery.csv")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    main(a.raw, a.out)
