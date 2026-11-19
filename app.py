"""
DeliveryRadar: predict which deliveries will miss their promise, before they ship.
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
REPO = "https://github.com/Rathore-Sahab/delivery-radar"
CASE_STUDY = "https://github.com/Rathore-Sahab/google-maps-doorstep-case-study"

st.set_page_config(page_title="DeliveryRadar", page_icon="📦", layout="wide",
                   initial_sidebar_state="auto")

# --------------------------------------------------------------------------- #
#  Design system
# --------------------------------------------------------------------------- #
BLUE, RED, AMBER, GREEN = "#1a73e8", "#d93025", "#e37400", "#188038"
INK, MUTED, LINE, SOFT = "#1f2937", "#6b7280", "#e5e7eb", "#f5f7fb"

st.markdown(f"""
<style>
  [data-testid="stDecoration"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], footer {{ display: none !important; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1240px; }}
  section[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .brand {{ display:flex; align-items:center; gap:10px; margin-bottom: 4px; }}
  .brand-mark {{ width:34px; height:34px; border-radius:9px; background:{BLUE}; color:#fff;
                 display:flex; align-items:center; justify-content:center; font-weight:700; font-size:18px; }}
  .brand-name {{ font-size: 20px; font-weight: 700; color:{INK}; }}
  .brand-tag {{ font-size: 12.5px; color:{MUTED}; margin: 0 0 14px 0; }}
  .eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color:{BLUE}; }}
  .hero {{ font-size: 30px; line-height: 1.2; font-weight: 700; color:{INK}; margin: 4px 0 6px 0; }}
  .lede {{ font-size: 16px; color:{MUTED}; margin-bottom: 22px; max-width: 860px; }}
  .kpi {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:16px 18px; height:100%; }}
  .kpi-label {{ font-size:12.5px; color:{MUTED}; margin-bottom:6px; }}
  .kpi-value {{ font-size:28px; font-weight:700; color:{INK}; line-height:1.1; }}
  .kpi-sub {{ font-size:12.5px; color:{MUTED}; margin-top:6px; }}
  .tone-blue .kpi-value {{ color:{BLUE}; }} .tone-red .kpi-value {{ color:{RED}; }}
  .tone-green .kpi-value {{ color:{GREEN}; }}
  .sec {{ font-size: 18px; font-weight: 700; color:{INK}; margin: 26px 0 2px 0; }}
  .sec-sub {{ font-size: 13.5px; color:{MUTED}; margin-bottom: 10px; }}
  .step {{ background:{SOFT}; border-radius:12px; padding:16px 18px; height:100%; }}
  .step-n {{ font-size:12px; font-weight:700; color:{BLUE}; }}
  .step-t {{ font-size:15px; font-weight:700; color:{INK}; margin:4px 0; }}
  .step-d {{ font-size:13.5px; color:{MUTED}; }}
  .result {{ border:1px solid {LINE}; border-radius:14px; padding:20px 22px; background:#fff; }}
  .pill {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:12.5px; font-weight:600; }}
  .pill-high {{ background:#fce8e6; color:{RED}; }} .pill-med {{ background:#fef3e0; color:{AMBER}; }}
  .pill-low {{ background:#e6f4ea; color:{GREEN}; }}
  .reason {{ font-size:14px; color:{INK}; margin:4px 0; }}
  .note {{ font-size: 12.5px; color:{MUTED}; }}
  .side-foot {{ font-size:12.5px; color:{MUTED}; line-height:1.7; }}
  .side-foot a {{ color:{BLUE}; text-decoration:none; }}
  div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:10px; }}
  .tbl-wrap {{ border:1px solid {LINE}; border-radius:12px; max-height:540px; overflow:auto; }}
  table.q {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  table.q th {{ position:sticky; top:0; background:{SOFT}; color:{MUTED}; font-weight:600; text-align:left;
               padding:10px 12px; border-bottom:1px solid {LINE}; white-space:nowrap; z-index:1; }}
  table.q td {{ padding:10px 12px; border-bottom:1px solid #f0f2f5; vertical-align:top; color:{INK}; }}
  table.q td.num, table.q th.num {{ text-align:right; white-space:nowrap; }}
  table.q tr:hover td {{ background:#fafbfd; }}
  .bar {{ height:6px; background:#eef1f5; border-radius:3px; margin-top:6px; width:80px; }}
  .bar > div {{ height:6px; border-radius:3px; }}
  .muted {{ color:{MUTED}; font-size:12.5px; }}
  .side-label {{ font-size:11px; font-weight:700; letter-spacing:.08em; color:{MUTED}; margin:14px 0 6px 2px; }}
  section[data-testid="stSidebar"] .stButton {{ margin-bottom: -10px; }}
  section[data-testid="stSidebar"] .stButton button > div {{ justify-content: flex-start; width: 100%; }}
  section[data-testid="stSidebar"] .stButton button p {{ text-align: left; }}
  section[data-testid="stSidebar"] .stButton button {{ justify-content: flex-start; text-align: left;
      padding: 8px 12px; border-radius: 8px; min-height: 38px; border: none; box-shadow: none; }}
  section[data-testid="stSidebar"] .stButton button p {{ font-size: 14.5px; }}
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] {{ color: {INK}; background: transparent; }}
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"]:hover {{ background: #eaeef4; color: {INK}; }}
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover,
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:focus {{ background: #e3edfd; color: {BLUE}; }}
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p {{ font-weight: 600; color: {BLUE}; }}
  section[data-testid="stSidebar"] hr {{ margin: 14px 0; }}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone=""):
    col.markdown(f'<div class="kpi tone-{tone}"><div class="kpi-label">{label}</div>'
                 f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sec-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def hero(eyebrow, title, lede):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="hero">{title}</div>'
                f'<div class="lede">{lede}</div>', unsafe_allow_html=True)


def pill(band):
    cls = {"High": "pill-high", "Medium": "pill-med", "Low": "pill-low"}[band]
    return f'<span class="pill {cls}">{band} risk</span>'


def style(fig, h=320, legend=False):
    fig.update_layout(height=h, margin=dict(t=10, b=10, l=10, r=10), plot_bgcolor="#fff",
                      paper_bgcolor="#fff", font=dict(color=INK, size=12), showlegend=legend,
                      legend=dict(orientation="h", y=-0.2, title=""))
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    return fig


def chart(fig, h=320, legend=False):
    st.plotly_chart(style(fig, h, legend), width="stretch", config={"displayModeBar": False})


DOT = {"High": "🔴 High", "Medium": "🟠 Medium", "Low": "🟢 Low"}


def risk_cell(p, band, scale=1.0):
    color = {"High": RED, "Medium": AMBER, "Low": GREEN}[band]
    w = min(100, p * 100 / scale)
    return (f'{pill(band)}<div class="bar"><div style="width:{w:.0f}%;background:{color}"></div></div>'
            f'<div class="muted">{p:.0%} risk</div>')


def html_table(headers, rows, num_cols=()):
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    st.markdown(f'<div class="tbl-wrap"><table class="q"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>',
                unsafe_allow_html=True)


def action_legend(actions):
    st.markdown('<div class="result" style="margin-top:14px;padding:14px 18px">'
                '<div class="note" style="font-weight:600;margin-bottom:6px">Recommended actions</div>'
                + "".join(f'<div class="reason">{pill(b)} &nbsp;{a}</div>' for b, a in actions.items())
                + '</div>', unsafe_allow_html=True)


def hub(name):
    return str(name).split(" (")[0].replace("_", " ")


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
@st.cache_resource
def load():
    br = {
        "model": joblib.load(ART / "model.joblib"),
        "m": json.load(open(ART / "metrics.json")),
        "orders": pd.read_parquet(ART / "scored_orders.parquet"),
        "routes": pd.read_csv(ART / "routes.csv"),
        **{n: pd.read_csv(ART / f"{n}.csv") for n in ["monthly", "review_by_delay", "late_by_slack"]},
    }
    ind = {
        "models": joblib.load(ART / "india_models.joblib"),
        "m": json.load(open(ART / "india_metrics.json")),
        "trips": pd.read_parquet(ART / "india_scored.parquet"),
        "lanes": pd.read_csv(ART / "india_lanes.csv"),
        "hubs": pd.read_csv(ART / "india_hubs.csv"),
        "states": pd.read_csv(ART / "india_states.csv"),
        "dist": pd.read_csv(ART / "india_by_distance.csv"),
    }
    return br, ind


BR, IN = load()

BR_STATES = {"AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá", "BA": "Bahia", "CE": "Ceará",
             "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
             "MG": "Minas Gerais", "MS": "Mato Grosso do Sul", "MT": "Mato Grosso", "PA": "Pará",
             "PB": "Paraíba", "PE": "Pernambuco", "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro",
             "RN": "Rio Grande do Norte", "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul",
             "SC": "Santa Catarina", "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins"}


def br_name(code):
    return f"{BR_STATES.get(code, code)} ({code})"


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
PAGES = ["Overview", "Risk queue", "Checker", "Insights", "About"]

MARKETS = ["India · line-haul", "Brazil · e-commerce"]
st.session_state.setdefault("market", MARKETS[0])
st.session_state.setdefault("page", PAGES[0])


def nav_button(label, key, active):
    if st.button(label, key=key, type="primary" if active else "tertiary", width="stretch"):
        return True
    return False


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">◉</div><div class="brand-name">DeliveryRadar</div></div>'
                '<div class="brand-tag">Late-delivery early warning for logistics teams</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">MARKET</div>', unsafe_allow_html=True)
    for m in MARKETS:
        if nav_button(m, f"mk_{m}", st.session_state.market == m):
            st.session_state.market = m
            st.rerun()
    st.markdown('<div class="side-label">MENU</div>', unsafe_allow_html=True)
    for p in PAGES:
        if nav_button(p, f"pg_{p}", st.session_state.page == p):
            st.session_state.page = p
            st.rerun()
    india = st.session_state.market.startswith("India")
    page = st.session_state.page
    st.divider()
    st.markdown(f'<div class="side-foot">'
                f'{"Delhivery · 26,222 hub-to-hub trips · Sep–Oct 2018" if india else "Olist · 95,976 orders · 2016–2018"}'
                f'<br><a href="{REPO}">Code &amp; PRD on GitHub</a><br><a href="{CASE_STUDY}">Google Maps case study</a>'
                f'<br><br>Built by Pawan Rathod</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  India helpers
# --------------------------------------------------------------------------- #
IM = IN["m"]
I_TH = IN["models"]["thresholds"]


def i_band(p):
    return "High" if p >= I_TH["high"] else ("Medium" if p >= I_TH["medium"] else "Low")


I_ACTION = {"High": "Plan with the DeliveryRadar ETA · alert destination hub · buffer onward connections",
            "Medium": "Watch the arrival scan at the destination hub",
            "Low": "Standard plan"}


def i_reasons(row):
    r = []
    f = row.eta_pred / max(row.osrm, 1)
    if f >= 2.5:
        r.append(f"Lane runs {f:.1f}× map ETA")
    if row.osrm_km < 20:
        r.append("Short leg under 20 km")
    if row.lane_n < 5:
        r.append("Little history on this lane")
    return " · ".join(r) or "No major risk signal"


# --------------------------------------------------------------------------- #
#  Brazil helpers
# --------------------------------------------------------------------------- #
BM = BR["m"]
B_TH = BR["model"]["band_thresholds"]
B_LATE = BR["model"]["global_late_rate"]


def b_band(p):
    return "High" if p >= B_TH["high"] else ("Medium" if p >= B_TH["medium"] else "Low")


B_ACTION = {"High": "Prioritise pickup & first scan · alert courier · proactive ETA message to customer",
            "Medium": "Add to hub watch-list · escalate if it misses a milestone",
            "Low": "Standard flow"}


def b_reasons(slack, seller_rate, seller_orders):
    r = []
    if slack <= 0:
        r.append(f"Promise is at or below this lane's real transit time ({slack:+.1f} days)")
    elif slack < 6:
        r.append(f"Tight promise: {slack:.1f} days of buffer")
    if seller_orders < 5:
        r.append("New seller, little history")
    elif seller_rate > 2 * B_LATE:
        r.append(f"Seller late on ~{seller_rate:.0%} of past orders")
    return " · ".join(r) or "No major risk signal"


# =========================================================================== #
#  PAGES
# =========================================================================== #
def overview():
    if india:
        hero("India · Delhivery line-haul network",
             "Trucks between Indian hubs take 2× the map ETA. DeliveryRadar flags the trips that will slip before they leave.",
             "Planning with map-route estimates breaks downstream promises. DeliveryRadar learns each lane's real "
             "transit time, gives a better ETA, and ranks every trip by its risk of a severe delay.")
        c = st.columns(4)
        kpi(c[0], "Trips analysed", f"{IM['legs']:,}", f"{IM['hubs']:,} hubs · {IM['states']} states & UTs")
        kpi(c[1], "Real time vs map ETA", f"{IM['median_ratio']:.1f}×", "median across all trips", "red")
        kpi(c[2], "ETAs within ±20%", "7% → 60%", "map ETA → DeliveryRadar ETA", "blue")
        kpi(c[3], "Severe delays caught", f"{IM['severe']['capture_at_20pct']:.0%}",
            "by reviewing the riskiest 20% of trips", "green")

        l, r = st.columns(2)
        with l:
            section("ETA accuracy", "Share of trips whose ETA lands within ±20% of the real time (held-out week)")
            ec = pd.DataFrame(IM["eta_compare"])
            ec["label"] = ["Map ETA", "Map ETA × 2", "Map ETA × lane history", "DeliveryRadar"]
            fig = px.bar(ec, y="label", x="within_20pct", orientation="h",
                         text=(ec.within_20pct * 100).round(0).astype(int).astype(str) + "%",
                         color=ec.label.eq("DeliveryRadar"), color_discrete_map={True: BLUE, False: "#c7cdd6"})
            fig.update_layout(xaxis_tickformat=".0%", xaxis_title="", yaxis_title="", yaxis_autorange="reversed")
            chart(fig, 280)
        with r:
            section("Short trips break most often", "Share of trips taking 3× the map ETA or longer, by distance")
            d = IN["dist"]
            fig = px.bar(d, x="osrm_km", y="severe_rate", text=(d.severe_rate * 100).round(0).astype(int).astype(str) + "%")
            fig.update_traces(marker_color=RED)
            fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
            chart(fig, 280)
    else:
        hero("Brazil · Olist e-commerce marketplace",
             "A broken delivery promise costs two stars. DeliveryRadar spots those orders before they ship.",
             "On-time orders average 4.3 stars and late ones 2.3. DeliveryRadar scores every order at checkout so a "
             "small ops team can act on the few that matter.")
        c = st.columns(4)
        kpi(c[0], "Orders analysed", "95,976", "Sep 2016 – Aug 2018")
        kpi(c[1], "Rating: on time vs late", f"{BM['review_ontime']:.1f} vs {BM['review_late']:.1f}★", "average review", "red")
        kpi(c[2], "Late orders rated 1–2★", f"{BM['low_review_share_late']:.0%}",
            f"vs {BM['low_review_share_ontime']:.0%} when on time")
        kpi(c[3], "Late orders caught", f"{BM['compare']['model']['capture_at_20pct']:.0%}",
            "by reviewing the riskiest 20% of orders", "green")
        l, r = st.columns(2)
        with l:
            section("The promise is the product", "Average review by delivery date vs the promised date")
            rv = BR["review_by_delay"].copy()
            rv["days_late"] = rv.days_late.str.replace(" days", "").str.replace(" ", "<br>", n=1)
            fig = px.bar(rv, x="days_late", y="avg_review", text=rv.avg_review.round(1),
                         color=rv.days_late.str.contains("late"), color_discrete_map={True: RED, False: BLUE})
            fig.update_layout(yaxis_range=[0, 5], xaxis_title="", yaxis_title="")
            fig.update_xaxes(tickangle=0)
            chart(fig, 280)
        with r:
            section("Lateness swings month to month", "Share of orders delivered after the promised date")
            mo = BR["monthly"]
            fig = px.line(mo, x="month", y="late_rate", markers=True)
            fig.update_traces(line_color=RED)
            fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
            chart(fig, 280)

    section("How it works")
    s = st.columns(3)
    steps = ([("Learn each lane", "For every hub-to-hub lane, measure how long trips really take, using only earlier trips."),
              ("Score before dispatch", "Each trip gets a better ETA and a severe-delay risk, with the reason behind it."),
              ("Act on the riskiest 20%", "The hub team re-plans, alerts the destination hub and protects customer promises.")]
             if india else
             [("Compare promise to reality", "Buffer = days promised minus the lane's real transit time, from earlier orders."),
              ("Add seller reliability", "Each seller's past late rate, adjusted for sellers with little history."),
              ("Act on the riskiest 20%", "Priority pickup, a courier alert and a proactive message to the customer.")])
    for i, (t, d) in enumerate(steps):
        s[i].markdown(f'<div class="step"><div class="step-n">STEP {i + 1}</div><div class="step-t">{t}</div>'
                      f'<div class="step-d">{d}</div></div>', unsafe_allow_html=True)


def queue():
    if india:
        hero("Risk queue · India", "Today's line-haul trips, riskiest first",
             "Trips from the held-out test week (Sep 27 – Oct 3, 2018), scored by models that never saw them.")
        t = IN["trips"].copy()
        t["day"] = t.od_start.dt.date
        days = [d for d in sorted(t.day.unique()) if pd.Timestamp("2018-09-27").date() <= d <= pd.Timestamp("2018-10-03").date()]
        a, b, c = st.columns([1, 1.4, 1.4])
        day = a.selectbox("Dispatch date", days, index=4, format_func=lambda d: d.strftime("%a, %d %b %Y"))
        states = ["All states"] + sorted(t.src_state.dropna().unique())
        stt = b.selectbox("Origin state", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
        cap = c.slider("Team capacity (% of trips reviewed)", 5, 50, 20, 5)
        d = t[t.day == day]
        if stt != "All states":
            d = d[d.src_state == stt]
        d = d.sort_values("risk", ascending=False)
        n = max(1, int(len(d) * cap / 100))
        top = d.head(n).copy()
        k = st.columns(4)
        kpi(k[0], "Trips dispatched", f"{len(d):,}", day.strftime("%d %b %Y"))
        kpi(k[1], "Flagged for review", f"{n:,}", f"top {cap}% by risk", "blue")
        kpi(k[2], "Severe delays among flagged", f"{int(top.severe.sum())} of {int(d.severe.sum())}",
            "known in hindsight", "green")
        caught = top.severe.sum() / max(d.severe.sum(), 1)
        kpi(k[3], "Lift vs random checks", f"{caught / (cap / 100):.1f}×" if d.severe.sum() else "–",
            f"{caught:.0%} caught vs {cap}% by chance")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        show = st.toggle("Show what actually happened", value=False)
        headers = ["Risk", "Lane", "Map ETA", "Radar ETA", "Why it's flagged"] + (["Actual", "Severe delay?"] if show else [])
        rows = []
        for r in top.itertuples():
            row = [risk_cell(r.risk, i_band(r.risk)),
                   f'{hub(r.src)} → {hub(r.dst)}<div class="muted">{r.osrm_km:.0f} km · {r.route_type}</div>',
                   f"{r.osrm:.0f} min", f"<b>{r.eta_pred:.0f} min</b>", i_reasons(r)]
            if show:
                row += [f"{r.actual:.0f} min", "✔ Yes" if r.severe else "No"]
            rows.append(row)
        html_table(headers, rows, num_cols=(2, 3, 5))
        action_legend(I_ACTION)
    else:
        hero("Risk queue · Brazil", "Today's orders, riskiest first",
             "Orders from the held-out test period (May – Aug 2018), scored at checkout by a model that never saw them.")
        q = BR["orders"].copy()
        q["day"] = q.order_purchase_timestamp.dt.date
        days = sorted(q.day.unique())
        a, c = st.columns([1, 2])
        day = a.selectbox("Order date", days, index=days.index(pd.Timestamp("2018-05-14").date()),
                          format_func=lambda d: d.strftime("%a, %d %b %Y"))
        cap = c.slider("Team capacity (% of orders reviewed)", 5, 50, 20, 5)
        d = q[q.day == day].sort_values("risk", ascending=False)
        n = max(1, int(len(d) * cap / 100))
        top = d.head(n).copy()
        idx = BM["capture_curve"]["frac"].index(round(cap / 100, 2))
        k = st.columns(4)
        kpi(k[0], "Orders placed", f"{len(d):,}", day.strftime("%d %b %Y"))
        kpi(k[1], "Flagged for review", f"{n:,}", f"top {cap}% by risk", "blue")
        kpi(k[2], "Late orders among flagged", f"{int(top.is_late.sum())} of {int(d.is_late.sum())}", "known in hindsight", "green")
        kpi(k[3], "Late orders caught (test period)", f"{BM['capture_curve']['model'][idx]:.0%}",
            f"vs {cap}% by random checks")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        show = st.toggle("Show what actually happened", value=False)
        headers = ["Risk", "Lane", "Category", "Promised", "Buffer", "Why it's flagged"] + \
                  (["Outcome", "Review"] if show else [])
        rows = []
        for r in top.itertuples():
            row = [risk_cell(r.risk, b_band(r.risk), scale=0.4),
                   f"{br_name(r.seller_state)} → {br_name(r.customer_state)}",
                   str(r.category).replace("_", " "), f"{r.promised_days:.0f} days", f"{r.slack_days:+.1f} days",
                   b_reasons(r.slack_days, r.seller_prior_late_rate, r.seller_prior_orders)]
            if show:
                row += [f"Late by {r.days_late:.0f} days" if r.is_late else "On time",
                        "–" if pd.isna(r.review_score) else f"{r.review_score:.0f} ★"]
            rows.append(row)
        html_table(headers, rows, num_cols=(3, 4))
        action_legend(B_ACTION)


def checker():
    if india:
        hero("Lane checker · India", "Get a realistic ETA before the truck leaves",
             "Pick a hub-to-hub lane. DeliveryRadar compares the map ETA with what this lane really takes and "
             "estimates the risk of a severe delay.")
        L = IN["lanes"]
        srcs = L.groupby("src").legs.sum().sort_values(ascending=False).index.tolist()
        a, b = st.columns(2)
        default = "Pune_Tathawde_H (Maharashtra)"
        s_hub = a.selectbox("Origin hub", srcs, index=srcs.index(default) if default in srcs else 0,
                            format_func=lambda h: f"{hub(h)}, {h.split('(')[-1].rstrip(')')}")
        opts = L[L.src == s_hub].sort_values("legs", ascending=False)
        dl = opts.dst.tolist()
        demo = "PNQ Vadgaon Sheri DPC (Maharashtra)"
        d_hub = b.selectbox("Destination hub", dl, index=dl.index(demo) if (s_hub == default and demo in dl) else 0,
                            format_func=lambda h: f"{hub(h)}, {h.split('(')[-1].rstrip(')')}")
        c, e = st.columns(2)
        hour = c.slider("Dispatch hour", 0, 23, 10, format="%d:00")
        dow = e.selectbox("Dispatch day", list(range(7)), index=2,
                          format_func=lambda i: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][i])
        ln = opts[opts.dst == d_hub].iloc[0]
        x = pd.DataFrame([{"osrm": ln.osrm, "osrm_km": ln.osrm_km, "carting": int(ln.carting >= 0.5),
                           "lane_prior": ln.lane_prior, "lane_n": ln.lane_n, "src_prior": ln.src_prior,
                           "dst_prior": ln.dst_prior, "hour": hour, "dow": dow}])[IN["models"]["features"]]
        eta = float(ln.osrm * np.exp(IN["models"]["eta"].predict(x)[0]))
        risk = float(IN["models"]["severe"].predict_proba(x)[0, 1])
        band = i_band(risk)
        reasons = []
        if ln.median_ratio >= 2.5:
            reasons.append(f"Trips on this lane took a median {ln.median_ratio:.1f}× the map ETA")
        if ln.severe_rate >= 0.25:
            reasons.append(f"{ln.severe_rate:.0%} of past trips were severely delayed")
        if ln.osrm_km < 20:
            reasons.append("Short urban leg (<20 km), the least reliable kind of trip")
        if not reasons:
            reasons.append("No major risk signal on this lane")
        st.markdown("")
        k = st.columns(4)
        kpi(k[0], "Map ETA", f"{ln.osrm:.0f} min", f"{ln.osrm_km:.0f} km by road")
        kpi(k[1], "DeliveryRadar ETA", f"{eta:.0f} min", f"{eta - ln.osrm:+.0f} min vs map", "blue")
        kpi(k[2], "Severe-delay risk", f"{risk:.0%}", "chance of taking 3×+ the map ETA",
            {"High": "red", "Medium": "", "Low": "green"}[band])
        kpi(k[3], "Lane history", f"{int(ln.legs)} trips", f"median {ln.median_ratio:.1f}× map ETA")
        st.markdown(f'<div class="result" style="margin-top:16px">{pill(band)}'
                    + "".join(f'<div class="reason">• {r}</div>' for r in reasons)
                    + f'<div class="reason" style="margin-top:10px"><b>Recommended action:</b> {I_ACTION[band]}</div></div>',
                    unsafe_allow_html=True)
    else:
        hero("Order checker · Brazil", "Is this delivery promise safe?",
             "Choose the seller's state, the customer's state and the promised delivery time. DeliveryRadar compares "
             "the promise with the lane's real transit time and the seller's track record.")
        R = BR["routes"]
        states = sorted(R.seller_state.unique())
        a, b, c = st.columns(3)
        o = a.selectbox("Seller state", states, index=states.index("SP"), format_func=br_name)
        dests = sorted(R[R.seller_state == o].customer_state.unique())
        dst = b.selectbox("Customer state", dests, index=dests.index("RJ") if "RJ" in dests else 0, format_func=br_name)
        lane = R[(R.seller_state == o) & (R.customer_state == dst)].iloc[0]
        promise = c.number_input("Promised delivery (days)", 1, 90, int(lane.typical_promise))
        e, f = st.columns(2)
        new_seller = e.toggle("New seller (no history)")
        s_rate = B_LATE if new_seller else f.slider("Seller's past late-delivery rate", 0.0, 0.5, 0.07, 0.01, format="%.2f")
        slack = promise - lane.route_hist_days
        p = float(BR["model"]["model"].predict_proba(pd.DataFrame([[slack, s_rate]], columns=BR["model"]["features"]))[0, 1])
        band = b_band(p)
        st.markdown("")
        k = st.columns(4)
        kpi(k[0], "Real transit on this lane", f"{lane.route_hist_days:.1f} days", f"{int(lane.orders):,} past orders")
        kpi(k[1], "Buffer in the promise", f"{slack:+.1f} days", "promised − real transit", "blue")
        kpi(k[2], "Late-delivery risk", f"{p:.0%}", f"average order: {B_LATE:.0%}",
            {"High": "red", "Medium": "", "Low": "green"}[band])
        kpi(k[3], "Lane late rate", f"{lane.late_rate:.0%}", f"~{lane.avg_km:,.0f} km")
        st.markdown(f'<div class="result" style="margin-top:16px">{pill(band)}'
                    f'<div class="reason">• {b_reasons(slack, s_rate, 0 if new_seller else 50)}</div>'
                    f'<div class="reason" style="margin-top:10px"><b>Recommended action:</b> {B_ACTION[band]}</div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="note" style="margin-top:10px">Try lowering the promise towards the real transit time '
                    'and watch the risk rise. The same model can tell checkout which promise is safe to show.</div>',
                    unsafe_allow_html=True)


def insights():
    if india:
        hero("Insights · India", "Where the network loses time, and how well the model works",
             "Held-out evaluation on the last week of data (Sep 27 – Oct 3, 2018).")
        section("Bottleneck hubs", "Destination hubs ranked by truck-hours lost against the map plan")
        h = IN["hubs"].head(10).assign(Hub=lambda d: d.dst.map(hub), severe_rate=lambda d: d.severe_rate * 100)
        st.dataframe(h[["Hub", "state", "inbound_legs", "median_ratio", "severe_rate", "excess_hours"]].rename(columns={
            "state": "State", "inbound_legs": "Inbound trips", "median_ratio": "Real ÷ map ETA",
            "severe_rate": "Severe delays", "excess_hours": "Hours beyond plan"}), hide_index=True, width="stretch",
            column_config={"Real ÷ map ETA": st.column_config.NumberColumn(format="%.1f×"),
                           "Severe delays": st.column_config.NumberColumn(format="%.0f%%"),
                           "Hours beyond plan": st.column_config.ProgressColumn(
                               format="%.0f", min_value=0, max_value=float(h.excess_hours.max()))})
        l, r = st.columns(2)
        with l:
            section("Severe delays by origin state", "Top 10 states with at least 100 trips")
            sv = IN["states"].head(10).sort_values("severe_rate")
            fig = px.bar(sv, y="src_state", x="severe_rate", orientation="h",
                         text=(sv.severe_rate * 100).round(0).astype(int).astype(str) + "%")
            fig.update_traces(marker_color=AMBER)
            fig.update_layout(xaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
            chart(fig, 360)
        with r:
            section("Catching severe delays", "Share of severe delays caught vs share of trips reviewed")
            cc = IM["capture_curve"]
            df = pd.DataFrame({"Reviewed": cc["frac"], "DeliveryRadar": cc["model"],
                               "Lane history only": cc["lane_history"], "Random": cc["frac"]})
            fig = px.line(df, x="Reviewed", y=["DeliveryRadar", "Lane history only", "Random"],
                          color_discrete_sequence=[BLUE, "#9aa5b1", "#d6dbe1"])
            fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
            chart(fig, 360, legend=True)
        sv = IM["severe"]
        st.markdown(f'<div class="note">Reviewing the riskiest 20% of trips catches <b>{sv["capture_at_20pct"]:.0%}</b> of severe '
                    f'delays (ROC-AUC {sv["roc_auc"]:.2f}). Lane history alone catches {sv["lane_only_capture_at_20pct"]:.0%}: '
                    "a lane's own track record does most of the work.</div>", unsafe_allow_html=True)
    else:
        hero("Insights · Brazil", "Two simple signals beat a 21-feature model",
             f"Trained on {BM['train_orders']:,} orders ({BM['train_period']}), tested on {BM['test_orders']:,} later orders "
             f"({BM['test_period']}).")
        cmp = BM["compare"]
        t = pd.DataFrame([
            ["Rule of thumb (distance ÷ promised days)", cmp["rule"]["roc_auc"], cmp["rule"]["capture_at_10pct"], cmp["rule"]["capture_at_20pct"]],
            ["Gradient-boosted model, 21 features", cmp["gbm"]["roc_auc"], cmp["gbm"]["capture_at_10pct"], cmp["gbm"]["capture_at_20pct"]],
            ["DeliveryRadar: 2 signals, logistic regression", cmp["model"]["roc_auc"], cmp["model"]["capture_at_10pct"], cmp["model"]["capture_at_20pct"]],
        ], columns=["Approach", "ROC-AUC", "Late caught · top 10%", "Late caught · top 20%"])
        for col in ["Late caught · top 10%", "Late caught · top 20%"]:
            t[col] = t[col] * 100
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "ROC-AUC": st.column_config.NumberColumn(format="%.2f"),
            "Late caught · top 10%": st.column_config.NumberColumn(format="%.0f%%"),
            "Late caught · top 20%": st.column_config.NumberColumn(format="%.0f%%")})
        l, r = st.columns(2)
        with l:
            section("Catching late orders", "Share of late orders caught vs share of orders reviewed")
            cc = BM["capture_curve"]
            df = pd.DataFrame({"Reviewed": cc["frac"], "DeliveryRadar": cc["model"], "21-feature model": cc["gbm"],
                               "Rule of thumb": cc["rule"], "Random": cc["frac"]})
            fig = px.line(df, x="Reviewed", y=["DeliveryRadar", "21-feature model", "Rule of thumb", "Random"],
                          color_discrete_sequence=[BLUE, AMBER, "#9aa5b1", "#d6dbe1"])
            fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
            chart(fig, 360, legend=True)
        with r:
            section("Less buffer, more late orders", "Late rate by days of buffer (promised − real lane transit)")
            ls = BR["late_by_slack"]
            fig = px.bar(ls, x="slack_days", y="late_rate", text=(ls.late_rate * 100).round(1).astype(str) + "%")
            fig.update_traces(marker_color=BLUE)
            fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
            chart(fig, 360)
        st.markdown('<div class="note">Why the simple model wins: in mid-2018 the platform shortened its delivery promises. '
                    'The 21-feature model had learned the old pattern; the buffer signal compares each promise with reality, '
                    'so it adapted. It also gives a reason ops teams can act on.</div>', unsafe_allow_html=True)


def about():
    hero("About", "Built to move logistics teams from reacting to delays to preventing them",
         "I built DeliveryRadar after my internship in Flipkart's Issue Resolution Team, where I tracked missing, "
         "damaged and orphan shipments after they had already gone wrong.")
    c = st.columns(3)
    blocks = [("The problem", "Late deliveries cost customer trust: on-time orders average 4.3★, late ones 2.3★. "
                              "Map-route ETAs in India underestimate real line-haul time by about 2×."),
              ("The approach", "Two explainable signals: each lane's real transit time and the seller's or hub's track "
                               "record, all learned from earlier history only, and tested on later periods the models never saw."),
              ("The product", "A daily risk queue, an ETA and lane checker, and network insights, each with a reason "
                              "and a recommended action.")]
    for i, (t, d) in enumerate(blocks):
        c[i].markdown(f'<div class="step"><div class="step-t">{t}</div><div class="step-d">{d}</div></div>',
                      unsafe_allow_html=True)
    section("Data")
    st.markdown("- **India:** Delhivery trip dataset (public, shared for educational business case studies): 26,222 "
                "hub-to-hub line-haul trips, Sep 12 – Oct 3, 2018. Trips between hubs, not doorstep deliveries.\n"
                "- **Brazil:** Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0): 95,976 delivered orders, 2016–2018.")
    section("Links")
    st.markdown(f"- [Source code, PRD and SQL analysis]({REPO})\n"
                f"- [Google Maps “Doorstep” case study]({CASE_STUDY}): last-100-metre delivery, based on interviews with "
                "25 delivery riders in Pune and Chhatrapati Sambhajinagar")
    st.markdown('<div class="note" style="margin-top:18px">Built by Pawan Rathod · MBA, Operations &amp; Supply Chain · '
                'Minor in Data Science &amp; ML, IIT Mandi</div>', unsafe_allow_html=True)


{"Overview": overview, "Risk queue": queue, "Checker": checker, "Insights": insights, "About": about}[page]()
