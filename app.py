"""
DeliveryRadar: flags orders likely to miss their promised delivery date,
scored at the time the order is placed.

    streamlit run app.py

Reads the pre-built files in artifacts/. See src/train.py to rebuild them.
"""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="DeliveryRadar", page_icon="▪", layout="wide")

# --- palette --------------------------------------------------------------
INK, MUTED, GRID = "#1d2939", "#667085", "#eaecf0"
NAVY, STEEL = "#1f4e79", "#98a2b3"
HIGH_C, MED_C, LOW_C = "#b42318", "#b54708", "#067647"
LATE_C = "#c4320a"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, .stCaption, button, input, textarea,
[data-testid="stMetric"], [data-baseweb="select"], [data-baseweb="tab"] {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3, h4 { letter-spacing: -0.01em; color: #1d2939; }
[data-testid="stHeader"] { background: transparent; }

.dr-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:2px; }
.dr-title { font-size:1.65rem; font-weight:700; color:#1d2939; letter-spacing:-0.02em; }
.dr-tag { font-size:0.72rem; font-weight:600; color:#1f4e79; background:#eef4fa;
          border:1px solid #d5e3f0; border-radius:4px; padding:2px 7px; }
.dr-sub { color:#475467; font-size:0.95rem; max-width:860px; margin:4px 0 2px; line-height:1.5; }
.dr-meta { color:#667085; font-size:0.8rem; margin-bottom:10px; }

.kpi { border:1px solid #eaecf0; border-radius:8px; padding:14px 16px; background:#fff; height:100%; }
.kpi-label { font-size:0.75rem; color:#667085; font-weight:500; text-transform:uppercase; letter-spacing:0.04em; }
.kpi-value { font-size:1.6rem; font-weight:600; color:#1d2939; margin-top:4px; font-variant-numeric:tabular-nums; }
.kpi-note { font-size:0.8rem; color:#667085; margin-top:2px; }

.section { font-size:1.05rem; font-weight:600; color:#1d2939; margin:18px 0 2px; }
.note { font-size:0.85rem; color:#475467; line-height:1.55; }

.panel { border:1px solid #eaecf0; border-radius:8px; padding:18px 20px; background:#fff; }
.panel-k { font-size:0.75rem; color:#667085; text-transform:uppercase; letter-spacing:0.04em; font-weight:500; }
.risk-num { font-size:2.6rem; font-weight:700; line-height:1.1; font-variant-numeric:tabular-nums; }
.pill { display:inline-block; font-size:0.75rem; font-weight:600; border-radius:999px; padding:2px 10px; margin-left:8px;
        vertical-align:middle; }
.pill-High { color:#b42318; background:#fef3f2; border:1px solid #fecdca; }
.pill-Medium { color:#b54708; background:#fffaeb; border:1px solid #fedf89; }
.pill-Low { color:#067647; background:#ecfdf3; border:1px solid #abefc6; }
.kv { display:flex; justify-content:space-between; border-bottom:1px solid #f2f4f7; padding:7px 0; font-size:0.88rem; }
.kv span:first-child { color:#667085; } .kv span:last-child { color:#1d2939; font-weight:500; font-variant-numeric:tabular-nums; }
.reason { font-size:0.88rem; color:#344054; padding:3px 0; }

.dr-foot { color:#667085; font-size:0.78rem; border-top:1px solid #eaecf0; padding-top:12px; margin-top:28px; }
.dr-foot a { color:#475467; }
[data-testid="stTable"] table { font-size:0.88rem; }
.dr-table { width:100%; border-collapse:collapse; font-size:0.86rem; margin-top:8px; }
.dr-table th { text-align:left; font-weight:500; color:#667085; background:#f9fafb; border-bottom:1px solid #eaecf0; padding:8px 10px; }
.dr-table td { border-bottom:1px solid #f2f4f7; padding:8px 10px; color:#344054; vertical-align:top; line-height:1.45; }
.guide { display:flex; gap:10px; margin:12px 0 6px; flex-wrap:wrap; }
.step { flex:1 1 220px; display:flex; gap:10px; align-items:flex-start; border:1px solid #eaecf0; border-radius:8px;
        padding:10px 12px; font-size:0.85rem; color:#475467; background:#fcfcfd; line-height:1.4; }
.step b { color:#1d2939; font-weight:600; }
.num { flex:0 0 22px; height:22px; border-radius:50%; background:#eef4fa; color:#1f4e79; font-weight:600;
       font-size:0.78rem; display:flex; align-items:center; justify-content:center; }
.summary { font-size:0.95rem; color:#344054; background:#f5f7fa; border-radius:8px; padding:12px 14px; margin:6px 0 12px; line-height:1.5; }
@media (max-width: 640px) {
  .block-container { padding-left:1rem; padding-right:1rem; padding-top:1.2rem; }
  .dr-title { font-size:1.4rem; } .kpi-value { font-size:1.3rem; } .risk-num { font-size:2.1rem; }
  .kpi { margin-bottom:8px; } .dr-table { font-size:0.8rem; }
}
.dr-table td:first-child { font-weight:600; color:#1d2939; white-space:nowrap; }
</style>
""",
    unsafe_allow_html=True,
)


# --- data -----------------------------------------------------------------
@st.cache_resource
def load():
    bundle = joblib.load(ART / "model.joblib")
    metrics = json.load(open(ART / "metrics.json", encoding="utf-8"))
    orders = pd.read_parquet(ART / "scored_orders.parquet")
    routes = pd.read_csv(ART / "routes.csv")
    tables = {n: pd.read_csv(ART / f"{n}.csv")
              for n in ["monthly", "by_state", "review_by_delay", "late_by_slack"]}
    return bundle, metrics, orders, routes, tables


bundle, M, Q, ROUTES, T = load()
HI, MED = bundle["band_thresholds"]["high"], bundle["band_thresholds"]["medium"]
BASE_RATE = bundle["global_late_rate"]
CURVE = M["capture_curve"]
CMP = M["compare"]


STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
    "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais",
    "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte", "RO": "Rondônia",
    "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo",
    "TO": "Tocantins",
}


def state_label(code: str) -> str:
    return f"{STATE_NAMES.get(code, code)} ({code})"


def band(p: float) -> str:
    return "High" if p >= HI else ("Medium" if p >= MED else "Low")


def reasons(slack: float, seller_rate: float, seller_orders: int) -> list[str]:
    out = []
    if slack <= 0:
        out.append(f"Promise is at or under typical lane transit ({slack:+.1f} d)")
    elif slack < 6:
        out.append(f"Thin buffer on this lane ({slack:.1f} d)")
    if seller_orders < 5:
        out.append("New seller, little history")
    elif seller_rate > 2 * BASE_RATE:
        out.append(f"Seller late on {seller_rate:.0%} of past orders")
    return out


ACTIONS = {
    "High": "Priority first-mile pickup, flag to 3PL, WhatsApp/SMS ETA update",
    "Medium": "Hub watch list, check at next scan",
    "Low": "Standard flow",
}


def capture_at(frac: float, key: str = "model") -> float:
    i = int(np.argmin(np.abs(np.array(CURVE["frac"]) - frac)))
    return CURVE[key][i]


def kpi(col, label, value, note=""):
    col.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div>"
                 f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>",
                 unsafe_allow_html=True)


def section(text, note=None):
    st.markdown(f"<div class='section'>{text}</div>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<div class='note'>{note}</div>", unsafe_allow_html=True)


def style(fig, height=340, pct_y=False, pct_x=False):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=12, b=8),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK),
        paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", tickfont_color=MUTED,
                     title_font=dict(size=12, color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont_color=MUTED,
                     title_font=dict(size=12, color=MUTED))
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    if pct_x:
        fig.update_xaxes(tickformat=".0%")
    return fig


def chart(container, fig):
    container.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --- header ---------------------------------------------------------------
st.markdown(
    "<div class='dr-head'><span class='dr-title'>DeliveryRadar</span>"
    "<span class='dr-tag'>v0.1 prototype</span></div>"
    "<div class='dr-sub'>Spots the orders most likely to arrive late, on the day they are placed, "
    "so a hub team knows which ones to check first.</div>"
    f"<div class='dr-meta'>Built for Indian marketplace operations · trained on "
    f"{M['train_orders'] + M['test_orders']:,} real orders from a Brazilian marketplace (Olist), since no "
    "public Indian dataset has promised and actual delivery dates</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='guide'>"
    "<div class='step'><span class='num'>1</span><div><b>Overview</b><br>Why late deliveries hurt ratings</div></div>"
    "<div class='step'><span class='num'>2</span><div><b>Daily risk queue</b><br>Pick a day and see which "
    "orders to check first</div></div>"
    "<div class='step'><span class='num'>3</span><div><b>Check an order</b><br>Try a route and a delivery "
    "promise, get a risk score</div></div>"
    "</div>", unsafe_allow_html=True)

with st.expander("What do the terms mean?"):
    st.markdown(pd.DataFrame([
        ("Promise", "The delivery date shown to the customer at checkout."),
        ("Route (lane)", "Where the order ships from and to. Here: seller's state to customer's state."),
        ("Typical transit", "How many days orders on that route actually took in the past."),
        ("Buffer", "Promised days minus typical transit. A small or negative buffer means the promise is tight."),
        ("Risk", "The model's estimate of the chance this order arrives after the promised date."),
        ("Band", f"High (risk {HI:.0%} or more), Medium ({MED:.0%} to {HI:.0%}), Low (below {MED:.0%})."),
        ("Catch rate", "Of all orders that turned out late, the share that were in the team's queue."),
        ("FC / 3PL", "Fulfilment centre / third-party logistics partner (the courier company)."),
        ("COD / RTO", "Cash on delivery / return to origin, when a parcel goes back to the seller undelivered."),
    ], columns=["Term", "Meaning"]).to_html(index=False, classes="dr-table", border=0), unsafe_allow_html=True)

tab_ctx, tab_queue, tab_score, tab_india, tab_model = st.tabs(
    ["Overview", "Daily risk queue", "Check an order", "India playbook", "How it works"])

# --- Context --------------------------------------------------------------
with tab_ctx:
    k = st.columns(4)
    kpi(k[0], "Late rate (test period)", f"{M['test_late_rate']:.1%}",
        "share delivered after the promised date")
    kpi(k[1], "Avg review, on time", f"{M['review_ontime']:.1f} / 5")
    kpi(k[2], "Avg review, late", f"{M['review_late']:.1f} / 5",
        f"{M['review_ontime'] - M['review_late']:.1f} points lower")
    kpi(k[3], "Late orders rated 1–2", f"{M['low_review_share_late']:.0%}",
        f"vs {M['low_review_share_ontime']:.0%} of on-time orders")

    left, right = st.columns(2, gap="large")
    with left:
        section("Review score by delivery outcome",
                "Customers rate a slow but on-time delivery about the same as a fast one. "
                "The drop starts as soon as the promised date is missed.")
        rv = T["review_by_delay"]
        late = rv.days_late.str.contains("late")
        fig = go.Figure(go.Bar(
            x=rv.days_late.str.replace(" ", "<br>", n=1), y=rv.avg_review, marker_color=np.where(late, LATE_C, STEEL),
            texttemplate="%{y:.1f}", textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>Avg review %{y:.2f}<extra></extra>"))
        style(fig).update_yaxes(range=[0, 5.2], title="Avg review")
        fig.update_xaxes(tickangle=0, tickfont_size=11)
        chart(st, fig)

    with right:
        section("Late-delivery rate by month",
                "The rate moves a lot: under 3% in quiet months, 12% in Nov 2017 and "
                "19% in Mar 2018. A monthly report is too slow to act on.")
        mo = T["monthly"]
        test_start = M["test_period"].split()[0][:3]
        fig = go.Figure(go.Scatter(
            x=mo.month, y=mo.late_rate, mode="lines+markers", line=dict(color=NAVY, width=2),
            marker=dict(size=5), customdata=mo.orders,
            hovertemplate="%{x}<br>Late %{y:.1%}<br>%{customdata:,} orders<extra></extra>"))
        fig.add_vrect(x0="2018-05", x1=mo.month.iloc[-1], fillcolor="#f2f4f7", line_width=0, layer="below")
        fig.add_annotation(x="2018-05", y=1, yref="paper", text="test period", showarrow=False,
                           xanchor="left", font=dict(size=11, color=MUTED))
        style(fig, pct_y=True).update_yaxes(title="Late rate")
        chart(st, fig)

    section("Late rate by customer state")
    bs = T["by_state"].query("orders >= 200").sort_values("late_rate")
    fig = go.Figure(go.Bar(
        x=bs.customer_state, y=bs.late_rate, marker_color=NAVY, customdata=bs.orders,
        hovertemplate="%{x}<br>Late %{y:.1%}<br>%{customdata:,} orders<extra></extra>"))
    fig.add_hline(y=bs.late_rate.mean(), line_dash="dot", line_color=STEEL)
    style(fig, height=260, pct_y=True).update_yaxes(title="Late rate")
    chart(st, fig)
    st.caption("States with at least 200 orders. Most sellers ship from São Paulo, so north-eastern "
               "destinations (AL, MA, SE, CE) sit on the longest lanes. The Indian equivalent is sellers "
               "concentrated in a few metros shipping to the North East, J&K and remote tier-3 pin codes.")

# --- Daily queue ----------------------------------------------------------
with tab_queue:
    days = sorted(Q.order_purchase_timestamp.dt.date.unique())
    c1, c2, c3 = st.columns([1, 1.5, 1.2], gap="medium")
    picked = c1.date_input("Order date", value=date(2018, 5, 14), min_value=days[0], max_value=days[-1],
                           format="DD/MM/YYYY", help="Any day from May to August 2018, the period the model was tested on.")
    day = picked if picked in set(days) else min(days, key=lambda x: abs(x - picked))
    cap = c2.slider("How many orders can your team check?", 5, 50, 20, 5, format="%d%%",
                    help="As a share of the day's orders. The riskiest orders are checked first.")
    bands_sel = c3.pills("Show bands", ["High", "Medium", "Low"], selection_mode="multi",
                         default=["High", "Medium", "Low"])

    d = Q[Q.order_purchase_timestamp.dt.date == day].sort_values("risk", ascending=False).copy()
    n_review = max(1, int(round(len(d) * cap / 100)))
    d["Band"] = d.risk.map(band)
    d["Reason"] = [" · ".join(reasons(s_, r, n)) or "No specific signal"
                   for s_, r, n in zip(d.slack_days, d.seller_prior_late_rate, d.seller_prior_orders)]
    d["Action"] = d.Band.map(ACTIONS)
    d["Lane"] = d.seller_state + " → " + d.customer_state
    d["risk_pct"] = d.risk * 100

    cap_rate = capture_at(cap / 100)
    st.markdown(
        f"<div class='summary'>On <b>{day:%a %d %b %Y}</b>, {len(d):,} orders were placed. At {cap}%, your team "
        f"checks the <b>{n_review:,} riskiest</b>. Over the test period, the riskiest {cap}% of orders contained "
        f"<b>{cap_rate:.0%} of all late deliveries</b>, about {cap_rate / (cap / 100):.1f}× what random "
        "checking finds.</div>", unsafe_allow_html=True)

    t1, t2, _ = st.columns([1, 1, 2])
    show_truth = t1.toggle("Show what actually happened", value=False,
                           help="Reveals each order's real outcome. A hub team wouldn't know this on the day.")
    more_cols = t2.toggle("Show more columns", value=False)

    cols = ["order_id", "Band", "risk_pct", "Lane", "customer_city", "Reason", "Action"]
    if more_cols:
        cols[5:5] = ["category", "promised_days", "slack_days"]
    if show_truth:
        cols += ["is_late", "days_late", "review_score"]
    queue = d.head(n_review)
    queue = queue[queue.Band.isin(bands_sel or [])]
    view = queue[cols].copy()
    if "category" in view:
        view["category"] = view.category.fillna("").str.replace("_", " ")
    view["customer_city"] = view.customer_city.str.title()
    if show_truth:
        view["is_late"] = view.is_late.map({1: "Late", 0: "On time"})

    if view.empty:
        st.info("No orders match the selected bands. Pick at least one band above.")
    else:
        band_colors = {"High": HIGH_C, "Medium": MED_C, "Low": LOW_C}
        styled = view.style.map(lambda v: f"color: {band_colors.get(v, INK)}; font-weight: 600",
                                subset=["Band"])
        if show_truth:
            styled = styled.map(lambda v: f"color: {HIGH_C}; font-weight: 600" if v == "Late" else "",
                                subset=["is_late"])
        st.dataframe(
            styled, hide_index=True, width="stretch", height=min(440, 38 + 35 * len(view)),
            column_config={
                "order_id": st.column_config.TextColumn("Order", width="small"),
                "Band": st.column_config.TextColumn("Band", width="small"),
                "risk_pct": st.column_config.ProgressColumn("Risk", format="%.0f%%", min_value=0, max_value=40,
                                                            width="small"),
                "Lane": st.column_config.TextColumn("Route", width="small", help="Seller state → customer state"),
                "customer_city": "City",
                "category": "Category",
                "promised_days": st.column_config.NumberColumn("Promised (days)", format="%d", width="small"),
                "slack_days": st.column_config.NumberColumn("Buffer (days)", format="%.1f", width="small",
                                                            help="Promised days minus typical transit on this route"),
                "Reason": st.column_config.TextColumn("Why it's flagged", width="large"),
                "Action": st.column_config.TextColumn("What to do", width="medium"),
                "is_late": st.column_config.TextColumn("Outcome", width="small"),
                "days_late": st.column_config.NumberColumn("Days vs promise", format="%+d", width="small"),
                "review_score": st.column_config.NumberColumn("Review", format="%.0f", width="small"),
            },
        )
        st.download_button("Download this queue (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name=f"deliveryradar_queue_{day:%Y%m%d}.csv", mime="text/csv")
    if show_truth:
        caught, total = int(d.head(n_review).is_late.sum()), int(d.is_late.sum())
        st.caption(f"{total} of {len(d)} orders placed that day arrived late, and {caught} of them were in the "
                   "queue. Single days are noisy; the percentage above is the average over the whole test period.")
    else:
        st.caption("These orders are from May–Aug 2018, which the model never saw during training.")

# --- Order scorer ---------------------------------------------------------
with tab_score:
    states = sorted(ROUTES.seller_state.unique())
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        section("Order details", "Pick a route, the delivery promise and how reliable the seller is. "
                                 "The risk on the right updates as you change them.")
        a, b = st.columns(2)
        origin = a.selectbox("Ships from", states, index=states.index("SP"), format_func=state_label)
        dests = sorted(ROUTES[ROUTES.seller_state == origin].customer_state.unique())
        dest = b.selectbox("Ships to", dests, index=dests.index("RJ") if "RJ" in dests else 0,
                           format_func=state_label)
        lane = ROUTES[(ROUTES.seller_state == origin) & (ROUTES.customer_state == dest)].iloc[0]
        promise = st.number_input("Promised delivery (days)", 1, 90, int(lane.typical_promise),
                                  help=f"Typical promise on this lane: {int(lane.typical_promise)} days")
        seller_kind = st.radio("How reliable is the seller?",
                               ["Reliable", "Average", "Often late", "New seller", "Custom"], index=1,
                               horizontal=True,
                               help="Reliable: late on 3% of past orders. Average: 7%. Often late: 20%. "
                                    "New sellers get the network average.")
        new_seller = seller_kind == "New seller"
        if seller_kind == "Custom":
            s_rate = st.slider("Share of the seller's past orders that were late", 0.0, 0.5, 0.07, 0.01,
                               format="%.2f")
        else:
            s_rate = {"Reliable": 0.03, "Average": 0.07, "Often late": 0.20}.get(seller_kind, BASE_RATE)

        st.markdown(
            "<div class='panel' style='margin-top:10px'>"
            f"<div class='panel-k'>Route: {STATE_NAMES.get(origin, origin)} → {STATE_NAMES.get(dest, dest)}</div>"
            f"<div class='kv'><span>Past orders</span><span>{int(lane.orders):,}</span></div>"
            f"<div class='kv'><span>Approx. distance</span><span>{lane.avg_km:,.0f} km</span></div>"
            f"<div class='kv'><span>Typical transit</span><span>{lane.route_hist_days:.1f} days</span></div>"
            f"<div class='kv'><span>Historical late rate</span><span>{lane.late_rate:.0%}</span></div>"
            "</div>", unsafe_allow_html=True)
        if lane.orders < 30:
            st.caption(f"Only {int(lane.orders)} past orders on this lane, so the transit estimate "
                       "leans heavily on the network average.")

    slack = promise - lane.route_hist_days
    X = pd.DataFrame([[slack, s_rate]], columns=bundle["features"])
    p = float(bundle["model"].predict_proba(X)[0, 1])
    bnd = band(p)
    why = reasons(slack, s_rate, 0 if new_seller else 50) or ["No specific risk signal"]

    # risk across possible promises on this lane, same seller
    grid = np.arange(1, 61)
    Xg = pd.DataFrame({bundle["features"][0]: grid - lane.route_hist_days,
                       bundle["features"][1]: s_rate})
    pg = bundle["model"].predict_proba(Xg)[:, 1]
    safe = grid[pg < MED]
    safe_txt = f"{safe.min()} days" if len(safe) else "none within 60 days"

    with right:
        section("Assessment")
        color = {"High": HIGH_C, "Medium": MED_C, "Low": LOW_C}[bnd]
        st.markdown(
            "<div class='panel'>"
            "<div class='panel-k'>Predicted late-delivery risk</div>"
            f"<div><span class='risk-num' style='color:{color}'>{p:.0%}</span>"
            f"<span class='pill pill-{bnd}'>{bnd}</span></div>"
            f"<div class='note' style='margin-top:2px'>Roughly 1 in {max(1, round(1 / p))} orders like this "
            f"arrive late ({p / BASE_RATE:.1f}× the average order).</div>"
            "<div style='height:12px'></div>"
            f"<div class='kv'><span>Buffer (promise − typical transit)</span><span>{promise} − {lane.route_hist_days:.1f} = "
            f"{slack:+.1f} days</span></div>"
            f"<div class='kv'><span>Seller late rate used</span><span>{s_rate:.0%}"
            f"{' (network avg)' if new_seller else ''}</span></div>"
            f"<div class='kv'><span>Shortest promise that stays Low</span><span>{safe_txt}</span></div>"
            f"<div class='kv'><span>Suggested action</span><span>{ACTIONS[bnd]}</span></div>"
            "<div style='height:10px'></div><div class='panel-k'>Why</div>"
            + "".join(f"<div class='reason'>• {r}</div>" for r in why)
            + "</div>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_hrect(y0=HI, y1=max(pg.max(), HI) * 1.05, fillcolor="#fef3f2", line_width=0, layer="below")
        fig.add_hrect(y0=MED, y1=HI, fillcolor="#fffaeb", line_width=0, layer="below")
        fig.add_trace(go.Scatter(x=grid, y=pg, mode="lines", line=dict(color=NAVY, width=2),
                                 hovertemplate="Promise %{x} d<br>Risk %{y:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[promise], y=[p], mode="markers",
                                 marker=dict(size=10, color=color, line=dict(color="white", width=2)),
                                 hovertemplate="Current: %{x} d, %{y:.1%}<extra></extra>"))
        fig.add_vline(x=lane.route_hist_days, line_dash="dot", line_color=STEEL)
        fig.add_annotation(x=lane.route_hist_days, y=1, yref="paper", text="typical transit",
                           showarrow=False, xanchor="left", xshift=4, font=dict(size=11, color=MUTED))
        style(fig, height=260, pct_y=True).update_xaxes(title="Promised delivery (days)")
        fig.update_yaxes(title="Risk", rangemode="tozero")
        st.markdown("<div class='note' style='margin-top:14px'>How the risk changes with a longer or shorter "
                    "promise on this route. The dot is your order; shaded areas are the Medium and High bands.</div>", unsafe_allow_html=True)
        chart(st, fig)

# --- India playbook -------------------------------------------------------
with tab_india:
    section("What carries over, and what doesn't",
            "The model learns two things: how much slack a promise leaves over real lane transit, and how "
            "reliable the seller is. Neither is specific to Brazil. The coefficients, thresholds and lane "
            "transit times are, and would need to be rebuilt from an Indian platform's own order history.")

    mapping = pd.DataFrame([
        ("Lane", "Seller state → customer state (27 states)",
         "Origin pin-code cluster → destination pin-code cluster. State level is too coarse: Pune → rural "
         "Maharashtra and Pune → Mumbai behave very differently."),
        ("Lane transit time", "Typical days on past orders, same lane",
         "Same, but split by 3PL partner and service type (surface vs air), since one lane can have several."),
        ("Promise", "Estimated delivery date shown at checkout",
         "Same. Usually set from pin-code serviceability tables, which is where the buffer gets thin."),
        ("Seller reliability", "Past late share",
         "Past late share plus dispatch-SLA breaches (seller hands over after the cut-off). For FC-fulfilled "
         "orders, use FC dispatch performance instead."),
        ("Not in Olist", "",
         "Festive-sale and monsoon calendar flags, COD vs prepaid, first-mile pickup scan time, address "
         "quality score, 3PL partner."),
    ], columns=["Input", "Olist (this prototype)", "Indian marketplace version"])
    st.markdown(mapping.to_html(index=False, classes="dr-table", border=0), unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        section("Where Indian networks get stressed")
        st.markdown("""
<div class='note'>
<p><b>Festive sales.</b> Volume spikes from September to November around the big sale events and Diwali.
Promises are often held steady while transit stretches, so buffers shrink across the network. Olist shows
the same pattern: the late rate went from 4% in Oct 2017 to 12% in the Black Friday month.</p>
<p><b>Monsoon.</b> June to September disrupts specific regions (Mumbai, Assam, Kerala, coastal Karnataka)
rather than the whole network. Lane-level transit picks this up better than a national average.</p>
<p><b>Remote and tier-3 pin codes.</b> The North East, J&amp;K, Ladakh, Andaman and hill districts have
longer, more variable transit and fewer 3PL options. They are the Indian counterpart of the long north-east
lanes in the Brazilian data.</p>
<p><b>COD and RTO.</b> Cash-on-delivery orders are more likely to be refused at the door and returned to
origin. That is a separate problem from lateness, but a late COD order is a likely RTO, so the two
scores are worth combining in the queue.</p>
</div>""", unsafe_allow_html=True)

    with right:
        section("Actions, in Indian ops terms")
        acts = pd.DataFrame([
            ("High", "Priority first-mile pickup and scan; flag to the 3PL's hub; WhatsApp or SMS ETA update "
                     "before the customer asks; for COD, a confirmation call to reduce RTO."),
            ("Medium", "Add to the sort hub's watch list; check the next scan milestone; escalate to the 3PL "
                       "if a scan is missed."),
            ("Low", "Standard flow."),
        ], columns=["Band", "Action"])
        band_css = {"High": HIGH_C, "Medium": MED_C, "Low": LOW_C}
        acts_html = acts.to_html(index=False, classes="dr-table", border=0)
        for bname, bcol in band_css.items():
            acts_html = acts_html.replace(f"<td>{bname}</td>", f"<td style='color:{bcol}'>{bname}</td>")
        st.markdown(acts_html, unsafe_allow_html=True)

        section("Try it on an Indian lane",
                "Enter the lane's typical transit from your own data. This uses the coefficients learned on "
                "Olist, so treat the output as illustrative.")
        presets = {
            "Bengaluru → Mysuru (intra-state)": (2.0, 4),
            "Delhi NCR → Mumbai (metro to metro)": (3.0, 5),
            "Mumbai → Indore (metro to tier-2)": (4.0, 6),
            "Delhi NCR → Guwahati (metro to North East)": (7.0, 8),
            "Custom": (5.0, 7),
        }
        name = st.selectbox("Lane", list(presets), index=3,
                            help="Transit and promise values are example inputs, not measured data.")
        t0, p0 = presets[name]
        a, b, c = st.columns(3)
        transit = a.number_input("Typical transit (days)", 0.5, 30.0, t0, 0.5, key=f"tr_{name}")
        prom = b.number_input("Promise (days)", 1, 30, p0, key=f"pr_{name}")
        srate = c.number_input("Seller late rate", 0.0, 0.5, round(BASE_RATE, 2), 0.01, format="%.2f",
                               key=f"sr_{name}")
        pi = float(bundle["model"].predict_proba(
            pd.DataFrame([[prom - transit, srate]], columns=bundle["features"]))[0, 1])
        bi = band(pi)
        st.markdown(
            f"<div class='panel'><span class='risk-num' style='color:{band_css[bi]};font-size:2rem'>{pi:.0%}</span>"
            f"<span class='pill pill-{bi}'>{bi}</span>"
            f"<div class='note'>Buffer {prom - transit:+.1f} days · {ACTIONS[bi]}</div></div>",
            unsafe_allow_html=True)

    section("Rolling it out at an Indian marketplace")
    st.markdown("""
<div class='note'>
<ol>
<li><b>Rebuild lane transit</b> from 6–12 months of order history at pin-code-cluster level, split by 3PL.
The window should include at least one festive season and one monsoon.</li>
<li><b>Shadow mode at one mother hub</b> outside the festive period: score every order, act on none, and
check capture at 20% against actual outcomes.</li>
<li><b>Pilot</b> with that hub's team acting on the High band. Track how many flags are actioned and whether
"where is my order" contacts drop for flagged orders.</li>
<li><b>Festive readiness:</b> run the queue during a sale event with extra capacity planned for the High
band, because that is when buffers shrink network-wide.</li>
</ol>
</div>""", unsafe_allow_html=True)

# --- Model & evaluation ---------------------------------------------------
with tab_model:
    section("Held-out comparison",
            f"Trained on {M['train_orders']:,} orders ({M['train_period']}), tested on the next "
            f"{M['test_orders']:,} ({M['test_period']}). Capture = share of all late orders that fall "
            "inside the top X% of orders ranked by score.")
    rows = [("Random sampling", 0.50, 0.10, 0.20, 0.30),
            ("Distance ÷ promised days (rule)", CMP["rule"]["roc_auc"], CMP["rule"]["capture_at_10pct"],
             CMP["rule"]["capture_at_20pct"], CMP["rule"]["capture_at_30pct"]),
            ("Gradient boosting, 21 features", CMP["gbm"]["roc_auc"], CMP["gbm"]["capture_at_10pct"],
             CMP["gbm"]["capture_at_20pct"], CMP["gbm"]["capture_at_30pct"]),
            ("Logistic regression, 2 features (used)", CMP["model"]["roc_auc"],
             CMP["model"]["capture_at_10pct"], CMP["model"]["capture_at_20pct"],
             CMP["model"]["capture_at_30pct"])]
    tbl = pd.DataFrame(rows, columns=["Approach", "ROC-AUC", "Capture @10%", "Capture @20%", "Capture @30%"])
    tbl["ROC-AUC"] = tbl["ROC-AUC"].map("{:.2f}".format)
    for c in tbl.columns[2:]:
        tbl[c] = tbl[c].map("{:.0%}".format)
    st.dataframe(
        tbl.style.apply(lambda r: ["font-weight: 600" if "used" in r.Approach else "" for _ in r], axis=1),
        hide_index=True, width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        section("Capture curve")
        fig = go.Figure()
        for key, name, colr, w in [("model", "Logistic regression (2 features)", NAVY, 2.5),
                                   ("gbm", "Gradient boosting (21 features)", "#7a9cc6", 1.8),
                                   ("rule", "Distance ÷ promise rule", STEEL, 1.8)]:
            fig.add_trace(go.Scatter(x=CURVE["frac"], y=CURVE[key], name=name, mode="lines",
                                     line=dict(color=colr, width=w),
                                     hovertemplate=f"{name}<br>Review %{{x:.0%}} → catch %{{y:.0%}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode="lines",
                                 line=dict(color=GRID, width=1.5, dash="dash"), hoverinfo="skip"))
        style(fig, height=360, pct_x=True, pct_y=True)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.22, x=0, font_size=11))
        fig.update_xaxes(title="Share of orders reviewed")
        fig.update_yaxes(title="Share of late orders caught")
        chart(st, fig)

    with right:
        section("Late rate by delivery buffer")
        ls = T["late_by_slack"]
        fig = go.Figure(go.Bar(
            x=ls.slack_days, y=ls.late_rate, marker_color=NAVY, customdata=ls.orders,
            text=(ls.late_rate * 100).round(1).astype(str) + "%", textposition="outside", cliponaxis=False,
            hovertemplate="Buffer %{x} d<br>Late %{y:.1%}<br>%{customdata:,} orders<extra></extra>"))
        style(fig, height=360, pct_y=True)
        fig.update_xaxes(title="Buffer (promised days − typical lane transit)")
        fig.update_yaxes(title="Late rate")
        chart(st, fig)

    section("How it works")
    st.markdown(
        f"""
<div class='note'>
<p><b>Features.</b> Two inputs, both computed from orders placed <i>before</i> the one being scored.
<b>Buffer</b> is the promised number of days minus the typical transit time on that seller-state to
customer-state lane. <b>Seller late rate</b> is the seller's share of late past orders, shrunk toward the
network average so that new sellers are not scored as 0% or 100%.</p>

<p><b>Why not the bigger model.</b> The gradient-boosted model with 21 features scored lower on the
test months (AUC {CMP['gbm']['roc_auc']:.2f} vs
{CMP['model']['roc_auc']:.2f}). Around mid-2018 the platform shortened its promises: the median went from
about 24 days early in the year to 13–19 days by July–August. The larger model had fitted patterns from the
old regime. Buffer compares each promise with observed transit, so it keeps working when promises change.
It is also easy to explain to the person who has to act on the flag.</p>

<p><b>Why capture and not accuracy.</b> Only {M['test_late_rate']:.1%} of test orders were late, so
predicting "on time" for everything is {1 - M['test_late_rate']:.0%} accurate and useless. What matters is
how many late orders land inside the slice a team can actually check.</p>

<p><b>Known limitations.</b></p>
<ul>
<li>Seller late rate is built from past orders whose outcome might not be known yet on the scoring date.
In production it should only use orders whose promised date has passed, so the reported numbers may be
slightly optimistic.</li>
<li>High-band precision is about 13%: most flagged orders arrive on time anyway. Actions on the High band
need to be cheap.</li>
<li>No courier scan events in the data. The obvious next step is to re-score an order when it misses a
scan milestone.</li>
<li>Brazilian marketplace data. The method should transfer to other networks; the thresholds and
coefficients will not.</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='dr-foot'>Pawan Rathod · Data: "
    "<a href='https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce' target='_blank'>"
    "Olist Brazilian E-Commerce Public Dataset</a> (CC BY-NC-SA 4.0) · "
    "Written up in PRD.md in the repository</div>",
    unsafe_allow_html=True,
)
