"""
DeliveryRadar — early-warning for late e-commerce deliveries.
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="DeliveryRadar", page_icon="📦", layout="wide")

BLUE, RED, AMBER, GREEN, GREY = "#1a73e8", "#d93025", "#f29900", "#188038", "#9aa0a6"


@st.cache_resource
def load():
    bundle = joblib.load(ART / "model.joblib")
    m = json.load(open(ART / "metrics.json"))
    q = pd.read_parquet(ART / "scored_orders.parquet")
    routes = pd.read_csv(ART / "routes.csv")
    tables = {n: pd.read_csv(ART / f"{n}.csv") for n in
              ["monthly", "by_state", "review_by_delay", "late_by_slack"]}
    return bundle, m, q, routes, tables


bundle, M, Q, ROUTES, T = load()
HI, MED = bundle["band_thresholds"]["high"], bundle["band_thresholds"]["medium"]
GLOBAL_LATE = bundle["global_late_rate"]


def band(p):
    return "🔴 High" if p >= HI else ("🟠 Medium" if p >= MED else "🟢 Low")


def reasons(slack, seller_rate, seller_orders):
    r = []
    if slack <= 0:
        r.append(f"Promise is at or below this lane's real transit time ({slack:+.1f} days buffer)")
    elif slack < 6:
        r.append(f"Tight promise: only {slack:.1f} days of buffer on this lane")
    if seller_orders < 5:
        r.append("New seller — little delivery history")
    elif seller_rate > 2 * GLOBAL_LATE:
        r.append(f"Seller was late on ~{seller_rate:.0%} of past orders")
    return r or ["No major risk signal"]


ACTIONS = {
    "🔴 High": "Prioritise pickup & first-mile scan today · alert courier partner · send customer a proactive ETA update",
    "🟠 Medium": "Add to hub watch-list · check scan at next checkpoint · escalate if it misses a milestone",
    "🟢 Low": "Standard flow — no action needed",
}

# --------------------------------------------------------------------------- #
st.title("📦 DeliveryRadar")
st.caption("Predict which orders will miss their promised delivery date — **before they ship** — "
           "so hub teams can act on the few that matter. Built on 96k real e-commerce orders "
           "(Olist, Brazil, 2016–18).")

tab1, tab2, tab3, tab4 = st.tabs(["① Why it matters", "② Hub risk queue",
                                  "③ Score an order", "④ How the model works"])

# ---------------- TAB 1: the problem --------------------------------------- #
with tab1:
    c = st.columns(4)
    c[0].metric("Orders analysed", "95,976")
    c[1].metric("Avg rating — on time", f"{M['review_ontime']:.1f} ★")
    c[2].metric("Avg rating — late", f"{M['review_late']:.1f} ★",
                f"{M['review_late'] - M['review_ontime']:.1f} ★", delta_color="inverse")
    c[3].metric("Late orders rated 1–2 ★", f"{M['low_review_share_late']:.0%}",
                f"vs {M['low_review_share_ontime']:.0%} when on time", delta_color="off")

    st.markdown("#### A missed promise costs more than a slow one")
    l, r = st.columns(2)
    rv = T["review_by_delay"]
    fig = px.bar(rv, x="days_late", y="avg_review", text=rv.avg_review.round(1),
                 color=rv.days_late.str.contains("late"),
                 color_discrete_map={True: RED, False: BLUE})
    fig.update_layout(showlegend=False, yaxis_range=[0, 5], xaxis_title="Delivered vs promised date",
                      yaxis_title="Avg review (★)", height=360, margin=dict(t=10))
    l.plotly_chart(fig, width="stretch")
    l.caption("Ratings stay at 4★+ whenever the promise is kept — even if delivery takes long. "
              "They collapse the moment it is broken. **The promise is the product.**")

    mo = T["monthly"]
    fig = px.line(mo, x="month", y="late_rate", markers=True)
    fig.update_traces(line_color=RED)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Late-delivery rate", xaxis_title="",
                      height=360, margin=dict(t=10))
    r.plotly_chart(fig, width="stretch")
    r.caption("Late rate swings from ~3% to ~19% (Nov-17 Black Friday peak, Feb–Mar-18 spike). "
              "Ops teams need a daily, order-level signal — not a monthly report.")

# ---------------- TAB 2: daily hub queue ----------------------------------- #
with tab2:
    st.markdown("#### The daily view a hub lead would open every morning")
    st.caption("Orders from the **held-out test period** (May–Aug 2018), scored by a model that never saw them.")
    days = sorted(Q.order_purchase_timestamp.dt.date.unique())
    a, b = st.columns([1, 2])
    day = a.selectbox("Order date", days, index=days.index(pd.Timestamp("2018-05-14").date()))
    cap = b.slider("Team capacity — % of today's orders the team can personally review", 5, 50, 20, 5)

    d = Q[Q.order_purchase_timestamp.dt.date == day].sort_values("risk", ascending=False).copy()
    n_review = max(1, int(len(d) * cap / 100))
    d["Band"] = d.risk.map(band)
    d["Why flagged"] = [" · ".join(reasons(s, r, n)) for s, r, n in
                        zip(d.slack_days, d.seller_prior_late_rate, d.seller_prior_orders)]
    d["Suggested action"] = d.Band.map(ACTIONS)
    d["risk"] = d.risk * 100

    # whole test period: what capacity buys you
    curve = M["capture_curve"]
    idx = curve["frac"].index(round(cap / 100, 2))
    k = st.columns(4)
    k[0].metric("Orders today", f"{len(d):,}")
    k[1].metric("Team reviews", f"{n_review:,}")
    k[2].metric(f"Late orders caught (test period, top {cap}%)", f"{curve['model'][idx]:.0%}",
                f"vs {cap}% by random checks", delta_color="off")
    k[3].metric("Lift vs random", f"{curve['model'][idx] / (cap / 100):.1f}×")

    show_truth = st.toggle("Reveal what actually happened (hindsight)", value=False)
    cols = ["order_id", "Band", "risk", "seller_state", "customer_state", "customer_city",
            "category", "promised_days", "slack_days", "Why flagged", "Suggested action"]
    if show_truth:
        cols += ["is_late", "days_late", "review_score"]
    view = d.head(n_review)[cols].rename(columns={
        "risk": "Risk", "seller_state": "From", "customer_state": "To", "customer_city": "City",
        "promised_days": "Promised (days)", "slack_days": "Buffer (days)",
        "is_late": "Was late?", "days_late": "Days vs promise", "review_score": "Review ★"})
    st.dataframe(view, hide_index=True, width="stretch", height=420,
                 column_config={"Risk": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=40),
                                "Buffer (days)": st.column_config.NumberColumn(format="%.1f")})
    if show_truth:
        caught, total = int(d.head(n_review).is_late.sum()), int(d.is_late.sum())
        st.info(f"On {day}, {total} of {len(d)} orders were late. Reviewing the top {n_review} "
                f"would have caught **{caught}** of them in advance.")

# ---------------- TAB 3: score a new order -------------------------------- #
with tab3:
    st.markdown("#### Score a new order at the moment it's placed")
    states = sorted(ROUTES.seller_state.unique())
    c1, c2, c3 = st.columns(3)
    o = c1.selectbox("Seller (origin) state", states, index=states.index("SP"))
    dests = sorted(ROUTES[ROUTES.seller_state == o].customer_state.unique())
    dst = c2.selectbox("Customer (destination) state", dests,
                       index=dests.index("RJ") if "RJ" in dests else 0)
    lane = ROUTES[(ROUTES.seller_state == o) & (ROUTES.customer_state == dst)].iloc[0]
    promise = c3.number_input("Promised delivery (days)", 1, 90, int(lane.typical_promise))
    c4, c5 = st.columns(2)
    new_seller = c4.checkbox("New seller (no history)")
    s_rate = GLOBAL_LATE if new_seller else c5.slider(
        "Seller's past late-delivery rate", 0.0, 0.5, 0.07, 0.01, format="%.2f")

    slack = promise - lane.route_hist_days
    x = pd.DataFrame([[slack, s_rate]], columns=bundle["features"])
    p = float(bundle["model"].predict_proba(x)[0, 1])
    bnd = band(p)

    g1, g2 = st.columns([1, 2])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 50]}, "bar": {"color": "#202124"},
               "steps": [{"range": [0, MED * 100], "color": "#e6f4ea"},
                         {"range": [MED * 100, HI * 100], "color": "#fef7e0"},
                         {"range": [HI * 100, 50], "color": "#fce8e6"}]},
        title={"text": f"Late-delivery risk · {bnd}"}))
    fig.update_layout(height=260, margin=dict(t=50, b=0))
    g1.plotly_chart(fig, width="stretch")
    g2.markdown(f"**Lane {o} → {dst}:** {int(lane.orders):,} past orders · ~{lane.avg_km:,.0f} km · "
                f"real transit ≈ **{lane.route_hist_days:.1f} days** · historical late rate {lane.late_rate:.0%}")
    g2.markdown(f"**Buffer:** promised {promise} days − real transit {lane.route_hist_days:.1f} = "
                f"**{slack:+.1f} days**")
    g2.markdown("**Why:** " + " · ".join(reasons(slack, s_rate, 0 if new_seller else 50)))
    g2.markdown(f"**Suggested action:** {ACTIONS[bnd]}")
    st.caption("Try it: drop the promise to the lane's real transit time and watch the risk jump. "
               "The same model can tell the storefront **which promise is safe to show**.")

# ---------------- TAB 4: model ---------------------------------------------- #
with tab4:
    st.markdown("#### Two simple signals beat a 21-feature model")
    cmp = M["compare"]
    st.markdown(
        f"| | ROC-AUC | Late orders caught by reviewing top 10% | …top 20% |\n|---|---|---|---|\n"
        f"| Rule of thumb (distance ÷ promised days) | {cmp['rule']['roc_auc']:.2f} | {cmp['rule']['capture_at_10pct']:.0%} | {cmp['rule']['capture_at_20pct']:.0%} |\n"
        f"| Gradient-boosted model, 21 features | {cmp['gbm']['roc_auc']:.2f} | {cmp['gbm']['capture_at_10pct']:.0%} | {cmp['gbm']['capture_at_20pct']:.0%} |\n"
        f"| **DeliveryRadar — 2 signals, logistic regression** | **{cmp['model']['roc_auc']:.2f}** | **{cmp['model']['capture_at_10pct']:.0%}** | **{cmp['model']['capture_at_20pct']:.0%}** |")
    st.caption(f"Trained on {M['train_orders']:,} orders ({M['train_period']}); "
               f"tested on {M['test_orders']:,} later orders ({M['test_period']}). "
               "Random checking catches 10% / 20%.")

    l, r = st.columns(2)
    cc = pd.DataFrame({"Share of orders reviewed": curve["frac"],
                       "DeliveryRadar": curve["model"], "21-feature GBM": curve["gbm"],
                       "Rule of thumb": curve["rule"], "Random": curve["frac"]})
    fig = px.line(cc, x="Share of orders reviewed", y=["DeliveryRadar", "21-feature GBM", "Rule of thumb", "Random"],
                  color_discrete_sequence=[BLUE, AMBER, GREY, "#dadce0"])
    fig.update_layout(yaxis_title="Share of late orders caught", xaxis_tickformat=".0%",
                      yaxis_tickformat=".0%", legend_title="", height=380, margin=dict(t=10))
    l.plotly_chart(fig, width="stretch")

    ls = T["late_by_slack"]
    fig = px.bar(ls, x="slack_days", y="late_rate", text=(ls.late_rate * 100).round(1).astype(str) + "%")
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(xaxis_title="Buffer = promised days − lane's real transit days",
                      yaxis_tickformat=".0%", yaxis_title="Late rate", height=380, margin=dict(t=10))
    r.plotly_chart(fig, width="stretch")

    st.markdown("""
**Signal 1 — Delivery buffer.** For every seller-state → customer-state lane, compute how many days orders *actually* took
(using only earlier orders). Buffer = promised days − real transit. Orders with ≤0 days of buffer are late **3.8×** as often as those with 15+.

**Signal 2 — Seller reliability.** Each seller's past late rate (earlier orders only), shrunk toward the average for new sellers.

**Why the simple model wins:** in mid-2018 the platform shortened its delivery promises (median ~24 days in early 2018 → 13–19 days by Jul–Aug).
The complex model had memorised patterns from the old regime; the buffer signal adapts automatically, because it compares
each promise with reality. It is also **explainable** — every flag comes with a reason an ops associate can act on.

**Limitations & next steps:** no live courier scan data (would add mid-route re-scoring) · Brazilian data, not Indian ·
seller history uses outcomes that may not be known yet at order time (in production, use only orders whose delivery date has passed) ·
next: A/B test proactive customer messages on High-risk orders and measure review score + "where is my order" ticket volume.
""")

st.divider()
st.caption("Built by Pawan Rathod · Data: Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0) · "
           "Code & PRD on GitHub")
