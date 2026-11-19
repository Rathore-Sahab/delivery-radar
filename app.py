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


@st.cache_resource
def load_india():
    return (joblib.load(ART / "india_models.joblib"), json.load(open(ART / "india_metrics.json")),
            pd.read_csv(ART / "india_lanes.csv"), pd.read_csv(ART / "india_hubs.csv"),
            pd.read_csv(ART / "india_states.csv"), pd.read_csv(ART / "india_by_distance.csv"))


bundle, M, Q, ROUTES, T = load()
IND, IM, ILANES, IHUBS, ISTATES, IDIST = load_india()

BR_STATES = {"AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
             "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
             "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará",
             "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro",
             "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
             "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins"}
br = lambda code: f"{BR_STATES.get(code, code)} ({code})"
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
st.caption("Predict which deliveries will miss their promise — **before they ship** — so hub teams can act on "
           "the few that matter. Tabs ①–④: 96k real e-commerce orders (Olist, **Brazil**, 2016–18). "
           "Tab ⑤: 26k real line-haul legs between **Indian** hubs (Delhivery, 2018).")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① Why it matters", "② Hub risk queue", "③ Score an order",
                                        "④ How the model works", "⑤ 🇮🇳 India: hub lanes"])

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
    st.caption("Orders from the **held-out test period** (May–Aug 2018), scored by a model that never saw them. "
               "From/To are Brazilian state codes (SP = São Paulo, RJ = Rio de Janeiro, MG = Minas Gerais…).")
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
    o = c1.selectbox("Seller (origin) state 🇧🇷", states, index=states.index("SP"), format_func=br)
    dests = sorted(ROUTES[ROUTES.seller_state == o].customer_state.unique())
    dst = c2.selectbox("Customer (destination) state 🇧🇷", dests,
                       index=dests.index("RJ") if "RJ" in dests else 0, format_func=br)
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
    g2.markdown(f"**Lane {br(o)} → {br(dst)}:** {int(lane.orders):,} past orders · ~{lane.avg_km:,.0f} km · "
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

# ---------------- TAB 5: India — Delhivery hub lanes ----------------------- #
with tab5:
    st.markdown("#### Does this work in India? Real line-haul data from Delhivery")
    st.caption(f"{IM['legs']:,} trips between {IM['hubs']:,} hubs in {IM['states']} states and UTs "
               "(Sep 12 – Oct 3, 2018). Each trip is compared with its **map-route ETA** (OSRM), "
               "the estimate many logistics teams plan with.")
    k = st.columns(4)
    k[0].metric("Real time vs map ETA (median)", f"{IM['median_ratio']:.1f}×")
    k[1].metric("Trips within ±20% of map ETA", f"{IM['share_within_20pct_of_map']:.0%}")
    k[2].metric("Truck-hours beyond map plan", f"{IM['excess_hours']:,.0f}")
    k[3].metric("Trips with severe delay (≥3× map ETA)", f"{IM['severe_rate_test']:.0%}")

    st.markdown("##### 1 · Map ETAs are wrong by ~2×, but lane history fixes most of it")
    l, r = st.columns(2)
    ec = pd.DataFrame(IM["eta_compare"])
    fig = px.bar(ec, y="method", x="within_20pct", orientation="h",
                 text=(ec.within_20pct * 100).round(0).astype(int).astype(str) + "%",
                 color=ec.method.str.startswith("DeliveryRadar"), color_discrete_map={True: BLUE, False: GREY})
    fig.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_title="Trips with ETA within ±20% of actual",
                      yaxis_title="", yaxis_autorange="reversed", height=300, margin=dict(t=10, l=10))
    l.plotly_chart(fig, width="stretch")
    me = {e["method"]: e["median_abs_err_min"] for e in IM["eta_compare"]}
    l.caption(f"Median ETA error falls from **{me['Map ETA (OSRM)']:.0f} min** (map) to "
              f"**{me['DeliveryRadar ETA model']:.0f} min**. One fixed 2× buffer gets halfway; each lane's own "
              "history does most of the rest, the same lesson as the Brazil model.")
    dd = IDIST
    fig = px.bar(dd, x="osrm_km", y="severe_rate", text=(dd.severe_rate * 100).round(0).astype(int).astype(str) + "%")
    fig.update_traces(marker_color=RED)
    fig.update_layout(xaxis_title="Trip distance (map route)", yaxis_title="Severe-delay rate",
                      yaxis_tickformat=".0%", height=300, margin=dict(t=10))
    r.plotly_chart(fig, width="stretch")
    r.caption("**The shortest trips are the least reliable.** Under 20 km, about 1 in 3 trips takes 3× the map ETA; "
              "over 300 km it's about 1 in 50. The last mile, not the highway, is where plans break.")

    st.markdown("##### 2 · Where the delays pile up")
    l, r = st.columns([3, 2])
    hv = IHUBS.head(10).assign(hub=lambda d: d.dst.str.replace(r" \(.*", "", regex=True),
                               severe_rate=lambda d: d.severe_rate * 100)
    l.dataframe(hv[["hub", "state", "inbound_legs", "median_ratio", "severe_rate", "excess_hours"]].rename(columns={
        "hub": "Destination hub", "state": "State", "inbound_legs": "Inbound trips",
        "median_ratio": "Real ÷ map ETA", "severe_rate": "Severe-delay rate", "excess_hours": "Hours beyond plan"}),
        hide_index=True, width="stretch", column_config={
            "Real ÷ map ETA": st.column_config.NumberColumn(format="%.1f×"),
            "Severe-delay rate": st.column_config.NumberColumn(format="%.0f%%"),
            "Hours beyond plan": st.column_config.NumberColumn(format="%.0f")})
    l.caption("Top 10 hubs by truck-hours lost against the map plan. Big hubs (Gurgaon, Bangalore) lose the most "
              "hours by volume; Kolkata, Bhiwandi and Muzaffarpur stand out on severity.")
    sv = ISTATES.head(10).sort_values("severe_rate")
    fig = px.bar(sv, y="src_state", x="severe_rate", orientation="h",
                 text=(sv.severe_rate * 100).round(0).astype(int).astype(str) + "%")
    fig.update_traces(marker_color=AMBER)
    fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Severe-delay rate (origin state)", yaxis_title="",
                      height=340, margin=dict(t=10))
    r.plotly_chart(fig, width="stretch")

    st.markdown("##### 3 · Lane checker: a better ETA before the truck leaves")
    srcs = ILANES.groupby("src").legs.sum().sort_values(ascending=False).index.tolist()
    c1, c2, c3 = st.columns([2, 2, 1])
    s_hub = c1.selectbox("Origin hub 🇮🇳", srcs, index=srcs.index("Pune_Tathawde_H (Maharashtra)")
                         if "Pune_Tathawde_H (Maharashtra)" in srcs else 0)
    opts = ILANES[ILANES.src == s_hub].sort_values("legs", ascending=False)
    d_hub = c2.selectbox("Destination hub", opts.dst.tolist())
    hour = c3.slider("Dispatch hour", 0, 23, 10)
    ln = opts[opts.dst == d_hub].iloc[0]
    x = pd.DataFrame([{"osrm": ln.osrm, "osrm_km": ln.osrm_km, "carting": int(ln.carting >= 0.5),
                       "lane_prior": ln.lane_prior, "lane_n": ln.lane_n, "src_prior": ln.src_prior,
                       "dst_prior": ln.dst_prior, "hour": hour, "dow": 2}])[IND["features"]]
    eta = float(ln.osrm * np.exp(IND["eta"].predict(x)[0]))
    risk = float(IND["severe"].predict_proba(x)[0, 1])
    th = IND["thresholds"]
    rb = "🔴 High" if risk >= th["high"] else ("🟠 Medium" if risk >= th["medium"] else "🟢 Low")
    m = st.columns(4)
    m[0].metric("Map ETA", f"{ln.osrm:.0f} min")
    m[1].metric("DeliveryRadar ETA", f"{eta:.0f} min", f"{eta - ln.osrm:+.0f} min vs map", delta_color="inverse")
    m[2].metric("Severe-delay risk", f"{risk:.0%} · {rb}")
    m[3].metric("Past trips on this lane", f"{int(ln.legs)}")
    st.markdown(f"**Lane:** {ln.osrm_km:.0f} km by road · past trips took a median **{ln.median_ratio:.1f}×** the map ETA · "
                f"{ln.severe_rate:.0%} were severely delayed")
    st.caption({"🔴 High": "Action: plan this trip with the DeliveryRadar ETA, not the map ETA · alert the destination hub "
                           "· set customer promises from the adjusted ETA.",
                "🟠 Medium": "Action: add buffer to downstream connections · watch the scan at the destination hub.",
                "🟢 Low": "Action: standard plan."}[rb])

    st.markdown("##### 4 · Model check (held-out week: Sep 27 – Oct 3)")
    sv = IM["severe"]
    st.markdown(
        f"Flagging the riskiest **20%** of trips catches **{sv['capture_at_20pct']:.0%}** of severe delays "
        f"(ROC-AUC {sv['roc_auc']:.2f}). Lane history alone catches {sv['lane_only_capture_at_20pct']:.0%} "
        f"(AUC {sv['lane_only_roc_auc']:.2f}). The test week directly follows the training "
        "period, so conditions are similar and the richer model adds a little on top of the simple lane signal.")
    st.caption("Data: Delhivery trip dataset, publicly shared for educational business case studies. "
               "Line-haul and carting trips between hubs, not doorstep deliveries. Each trip is scored with "
               "only earlier trips' history.")

st.divider()
st.caption("Built by Pawan Rathod · Data: Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0); "
           "Delhivery trip dataset (public, educational) · Code & PRD on GitHub")
