import os
import json
import anthropic
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL      = "http://uno.ucc.uwm.edu/odata/435"
REFRESH_S     = 20
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

CHANNEL_MAP = {"12": "Wholesale", "14": "Retail"}
LOC_MAP     = {"02": "Central", "02N": "North", "02S": "South", "02W": "West"}
STEPS_PER_ROUND = 20
PROD_WARN_THRESHOLD = 120_000

WAREHOUSE_CAPACITY = {
    "Finished Goods": 250_000,
    "Raw Materials":  250_000,
    "Packaging":      750_000,
}
MAT_TYPE_MAP = {"F": "Finished Goods", "R": "Raw Materials", "P": "Packaging"}

# ── Colour palette ─────────────────────────────────────────────────────────────
COLORS  = px.colors.qualitative.Safe
BG      = "#0f1117"
CARD_BG = "#1a1d27"
ACCENT  = "#4f8ef7"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
YELLOW  = "#f39c12"

STATUS_COLORS = {
    "Completed":   "#2ecc71",
    "In Progress": "#f39c12",
    "Up Next":     "#4f8ef7",
    "Queued":      "#4a4f6a",
}

_VAL_EMPTY = {
    "COMPANY_VALUATION": 0, "BANK_CASH_ACCOUNT": 0, "BANK_LOAN": 0,
    "PROFIT": 0, "COMPANY_RISK_RATE_PCT": 0, "MARKET_RISK_RATE_PCT": 0,
    "CREDIT_RATING": "—", "SIM_ELAPSED_STEPS": 0, "SIM_ROUND": "—", "SIM_DATE": "—",
    "DEBT_LOADING": 0,
}

# ── Data helpers ───────────────────────────────────────────────────────────────
def fetch(entity, auth, base_url=None):
    url = f"{base_url or BASE_URL}/{entity}?$format=json"
    resp = requests.get(url, auth=auth, timeout=30)
    resp.raise_for_status()
    rows = resp.json()["d"]["results"]
    df = pd.DataFrame(rows)
    df.drop(columns=["__metadata"], errors="ignore", inplace=True)
    return df

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def rs_to_elapsed(round_str, step_str):
    try:
        return (int(round_str) - 1) * STEPS_PER_ROUND + int(step_str)
    except Exception:
        return 0

def load_all(auth, base_url=None):
    kw = dict(auth=auth, base_url=base_url)
    sales       = fetch("Sales",                      **kw)
    valuation   = fetch("Company_Valuation",          **kw)
    inv_kpi     = fetch("Current_Inventory_KPI",      **kw)
    market      = fetch("Market",                     **kw)
    carbon      = fetch("Carbon_Emissions",           **kw)
    prod        = fetch("Production",                 **kw)
    prod_orders = fetch("Production_Orders",          **kw)
    inv_hist    = fetch("Inventory",                  **kw)
    pricing     = fetch("Current_Pricing_Conditions", **kw)
    pur_orders  = fetch("Purchase_Orders",            **kw)

    sales       = to_num(sales,       ["QUANTITY","QUANTITY_DELIVERED","NET_PRICE","NET_VALUE","COST","SIM_ELAPSED_STEPS","SIM_PERIOD"])
    valuation   = to_num(valuation,   ["BANK_CASH_ACCOUNT","ACCOUNTS_RECEIVABLE","BANK_LOAN",
                                       "ACCOUNTS_PAYABLE","PROFIT","COMPANY_VALUATION",
                                       "COMPANY_RISK_RATE_PCT","MARKET_RISK_RATE_PCT","SIM_ELAPSED_STEPS","DEBT_LOADING"])
    inv_kpi     = to_num(inv_kpi,     ["CURRENT_INVENTORY","QUANTITY_SOLD","NB_STEPS_AVAILABLE","SIM_ELAPSED_STEPS"])
    market      = to_num(market,      ["QUANTITY","AVERAGE_PRICE","NET_VALUE","SIM_PERIOD"])
    carbon      = to_num(carbon,      ["CO2E_EMISSIONS","TOTAL_CO2E_EMISSIONS","SIM_ELAPSED_STEPS"])
    prod        = to_num(prod,        ["SIM_ELAPSED_STEPS"])
    prod_orders = to_num(prod_orders, ["SIM_ELAPSED_STEPS","TARGET_QUANTITY","CONFIRMED_QUANTITY","SETUP_TIME"])

    if not sales.empty and "NET_VALUE" in sales.columns:
        sales["MARGIN"]  = sales["NET_VALUE"] - sales["COST"]
        sales["CHANNEL"] = sales["DISTRIBUTION_CHANNEL"].map(CHANNEL_MAP).fillna(sales["DISTRIBUTION_CHANNEL"])
        sales["REGION"]  = sales["STORAGE_LOCATION"].map(LOC_MAP).fillna(sales["STORAGE_LOCATION"])
    else:
        for col in ["MARGIN","CHANNEL","REGION","NET_VALUE","COST","QUANTITY","NET_PRICE"]:
            sales[col] = pd.Series(dtype=float)

    if not inv_kpi.empty and "STORAGE_LOCATION" in inv_kpi.columns:
        inv_kpi["REGION"] = inv_kpi["STORAGE_LOCATION"].map(LOC_MAP).fillna(inv_kpi["STORAGE_LOCATION"])

    if not prod.empty and "YIELD" in prod.columns:
        prod["YIELD"] = pd.to_numeric(prod["YIELD"], errors="coerce")
    else:
        prod["YIELD"] = pd.Series(dtype=float)

    if not prod_orders.empty and "SIM_ELAPSED_STEPS" in prod_orders.columns:
        latest_po_step = prod_orders["SIM_ELAPSED_STEPS"].max()
        prod_orders = prod_orders[prod_orders["SIM_ELAPSED_STEPS"] == latest_po_step].copy()
        prod_orders["BEGIN_ELAPSED"] = prod_orders.apply(
            lambda r: rs_to_elapsed(r["BEGIN_ROUND"], r["BEGIN_STEP"]), axis=1)
        prod_orders["END_ELAPSED"]   = prod_orders.apply(
            lambda r: rs_to_elapsed(r["END_ROUND"],   r["END_STEP"]),   axis=1)
        prod_orders["BEGIN_LABEL"]   = prod_orders["BEGIN_ROUND"] + "/" + prod_orders["BEGIN_STEP"]
        prod_orders["END_LABEL"]     = prod_orders["END_ROUND"]   + "/" + prod_orders["END_STEP"]
        prod_orders = prod_orders.sort_values("BEGIN_ELAPSED").reset_index(drop=True)
    else:
        for col in ["BEGIN_ELAPSED","END_ELAPSED","BEGIN_LABEL","END_LABEL",
                    "TARGET_QUANTITY","CONFIRMED_QUANTITY","SETUP_TIME","STATUS",
                    "MATERIAL_DESCRIPTION","PRODUCTION_ORDER"]:
            prod_orders[col] = pd.Series(dtype=object)

    inv_hist = to_num(inv_hist, ["INVENTORY_OPENING_BALANCE", "SIM_ELAPSED_STEPS"])
    if not inv_hist.empty and "MATERIAL_NUMBER" in inv_hist.columns:
        inv_hist["MAT_TYPE"] = inv_hist["MATERIAL_NUMBER"].str[3].map(MAT_TYPE_MAP).fillna("Other")
    else:
        inv_hist["MAT_TYPE"] = pd.Series(dtype=str)

    pricing = to_num(pricing, ["PRICE"])
    pur_orders = to_num(pur_orders, ["QUANTITY", "UNIT_PRICE", "SIM_ELAPSED_STEPS"])

    return sales, valuation, inv_kpi, market, carbon, prod, prod_orders, inv_hist, pricing, pur_orders

def compute_derived(sales, valuation, inv_kpi, carbon, prod, prod_orders):
    latest_val     = valuation.sort_values("SIM_ELAPSED_STEPS").iloc[-1] if not valuation.empty else pd.Series(_VAL_EMPTY)
    total_revenue  = sales["NET_VALUE"].sum() if not sales.empty else 0.0
    total_margin   = sales["MARGIN"].sum()    if not sales.empty else 0.0
    total_co2      = carbon["CO2E_EMISSIONS"].sum() if not carbon.empty else 0.0
    current_elapsed = int(inv_kpi["SIM_ELAPSED_STEPS"].max()) if (not inv_kpi.empty and "SIM_ELAPSED_STEPS" in inv_kpi.columns) else 0
    total_produced  = prod["YIELD"].sum() if not prod.empty else 0.0
    avg_yield_step  = prod.groupby("SIM_ELAPSED_STEPS")["YIELD"].sum().mean() if (not prod.empty and "SIM_ELAPSED_STEPS" in prod.columns) else 0.0

    def classify_order(row, cur):
        if row["CONFIRMED_QUANTITY"] > 0:
            return "Completed"
        if row["BEGIN_ELAPSED"] <= cur <= row["END_ELAPSED"]:
            return "In Progress"
        if row["BEGIN_ELAPSED"] > cur:
            return "Queued"
        return "Completed"

    if not prod_orders.empty and "BEGIN_ELAPSED" in prod_orders.columns:
        prod_orders = prod_orders.copy()
        prod_orders["STATUS"] = prod_orders.apply(lambda r: classify_order(r, current_elapsed), axis=1)
        queued_idx = prod_orders[prod_orders["STATUS"] == "Queued"].index
        if len(queued_idx):
            prod_orders.at[queued_idx[0], "STATUS"] = "Up Next"
    else:
        prod_orders = prod_orders.copy()
        prod_orders["STATUS"] = pd.Series(dtype=str)

    in_progress = prod_orders[prod_orders["STATUS"] == "In Progress"] if "STATUS" in prod_orders.columns else prod_orders.iloc[0:0]
    up_next     = prod_orders[prod_orders["STATUS"] == "Up Next"]     if "STATUS" in prod_orders.columns else prod_orders.iloc[0:0]
    pending     = prod_orders[prod_orders["STATUS"].isin(["Up Next","Queued","In Progress"])] if "STATUS" in prod_orders.columns else prod_orders.iloc[0:0]
    total_to_produce = int(pending["TARGET_QUANTITY"].sum()) if (not pending.empty and "TARGET_QUANTITY" in pending.columns) else 0

    return (latest_val, total_revenue, total_margin, total_co2,
            current_elapsed, total_produced, avg_yield_step,
            in_progress, up_next, pending, total_to_produce, prod_orders)

# ── Initial load ───────────────────────────────────────────────────────────────
_edf = pd.DataFrame()
sales = valuation = inv_kpi = market = carbon = prod = prod_orders = inv_hist = pricing = pur_orders = _edf
latest_val         = pd.Series(_VAL_EMPTY)
total_revenue      = total_margin = total_co2 = 0
total_produced     = avg_yield_step = current_elapsed = 0
in_progress_orders = up_next_orders = pending_orders = pd.DataFrame()
total_to_produce   = 0

# ── UI helpers ─────────────────────────────────────────────────────────────────
def empty_fig(msg="No data yet — waiting for simulation to start"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#8b90a0", size=13))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def safe(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return empty_fig()
    return wrapper

def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c8cdd8",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2d3e", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2d3e", zeroline=False)
    return fig

def kpi_card(title, value, sub=None, color=ACCENT):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1",
                   style={"fontSize":"0.8rem","textTransform":"uppercase","letterSpacing":"0.05em"}),
            html.H4(value, style={"color": color, "fontWeight":"700", "marginBottom":"2px"}),
            html.Small(sub or "", className="text-muted"),
        ])
    ], style={"background": CARD_BG, "border":"1px solid #2a2d3e", "borderRadius":"10px"}, className="h-100")

def chart_card(title, graph_id, fig=None, height="280px"):
    if fig is not None:
        fig = style_fig(fig)
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="mb-3",
                    style={"color":"#8b90a0","textTransform":"uppercase",
                           "letterSpacing":"0.08em","fontSize":"0.75rem"}),
            dcc.Graph(id=graph_id, figure=fig or empty_fig(),
                      config={"displayModeBar": False},
                      style={"height": height} if height else {}),
        ])
    ], style={"background": CARD_BG, "border":"1px solid #2a2d3e", "borderRadius":"10px"})

# ── Chart builders (accept data as arguments) ──────────────────────────────────
@safe
def fig_valuation_over_time(valuation):
    df = valuation.sort_values("SIM_ELAPSED_STEPS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["COMPANY_VALUATION"],
                             mode="lines+markers", name="Company Valuation",
                             line=dict(color=ACCENT, width=2), fill="tozeroy",
                             fillcolor="rgba(79,142,247,0.08)"))
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["PROFIT"],
                             mode="lines+markers", name="Cumulative Profit",
                             line=dict(color=GREEN, width=2, dash="dot")))
    fig.update_layout(hovermode="x unified")
    return fig

@safe
def fig_cash_and_debt(valuation):
    df = valuation.sort_values("SIM_ELAPSED_STEPS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["BANK_CASH_ACCOUNT"],
                             name="Cash", mode="lines", line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["BANK_LOAN"],
                             name="Bank Loan", mode="lines", line=dict(color=RED, width=2)))
    if "DEBT_LOADING" in df.columns:
        fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["DEBT_LOADING"],
                                 name="Net Debt Loading", mode="lines",
                                 line=dict(color=YELLOW, width=2, dash="dash")))
    return fig

@safe
def fig_sales_by_period(sales):
    df = sales.groupby(["SIM_PERIOD","SIM_ELAPSED_STEPS"], as_index=False).agg(
        Revenue=("NET_VALUE","sum"), Margin=("MARGIN","sum"))
    df = df.sort_values("SIM_ELAPSED_STEPS")
    fig = go.Figure()
    fig.add_bar(x=df["SIM_ELAPSED_STEPS"], y=df["Revenue"], name="Revenue",
                marker_color=ACCENT, opacity=0.85)
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["Margin"],
                             name="Margin", mode="lines+markers",
                             line=dict(color=GREEN, width=2), yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                  color="#2ecc71"), barmode="overlay", hovermode="x unified")
    return fig

@safe
def fig_production_over_time(prod):
    prod2 = prod.copy()
    prod2["YIELD"] = pd.to_numeric(prod2["YIELD"], errors="coerce")
    df = prod2.groupby(["SIM_ELAPSED_STEPS","MATERIAL_DESCRIPTION"], as_index=False)["YIELD"].sum()
    fig = px.bar(df, x="SIM_ELAPSED_STEPS", y="YIELD", color="MATERIAL_DESCRIPTION",
                 color_discrete_sequence=COLORS, barmode="stack",
                 labels={"SIM_ELAPSED_STEPS":"Step","YIELD":"Units Produced"})
    return fig

@safe
def fig_revenue_by_product(sales):
    df = sales.groupby("MATERIAL_DESCRIPTION", as_index=False)["NET_VALUE"].sum()
    df = df.sort_values("NET_VALUE", ascending=True)
    fig = px.bar(df, x="NET_VALUE", y="MATERIAL_DESCRIPTION", orientation="h",
                 color="NET_VALUE", color_continuous_scale=["#1e3a5f","#4f8ef7"],
                 labels={"NET_VALUE":"Revenue (EUR)", "MATERIAL_DESCRIPTION":""})
    fig.update_coloraxes(showscale=False)
    return fig

@safe
def fig_sales_by_region(sales):
    df = sales.groupby("REGION", as_index=False)["NET_VALUE"].sum()
    fig = px.pie(df, names="REGION", values="NET_VALUE", hole=0.55,
                 color_discrete_sequence=COLORS)
    fig.update_traces(textposition="outside", textinfo="label+percent")
    return fig

@safe
def fig_channel_split(sales):
    df = sales.groupby("CHANNEL", as_index=False).agg(Revenue=("NET_VALUE","sum"), Margin=("MARGIN","sum"))
    fig = px.bar(df, x="CHANNEL", y=["Revenue","Margin"],
                 barmode="group", color_discrete_map={"Revenue":ACCENT,"Margin":GREEN})
    return fig

@safe
def fig_inventory_kpi(inv_kpi):
    df = inv_kpi.copy()
    if "MATERIAL_NUMBER" in df.columns:
        df["LABEL"] = df["MATERIAL_NUMBER"].str.strip() + " · " + df["MATERIAL_DESCRIPTION"]
    else:
        df["LABEL"] = df["MATERIAL_DESCRIPTION"]
    df = df.groupby(["LABEL","REGION"], as_index=False).agg(
        Stock=("CURRENT_INVENTORY","sum"), DaysAvail=("NB_STEPS_AVAILABLE","mean"))
    fig = px.bar(df, x="LABEL", y="Stock", color="REGION",
                 color_discrete_sequence=COLORS, barmode="stack",
                 labels={"LABEL":"", "Stock":"Current Inventory (units)"})
    fig.update_xaxes(tickangle=-35)
    return fig

@safe
def fig_days_available(inv_kpi):
    df = inv_kpi.copy()
    if "MATERIAL_NUMBER" in df.columns:
        df["LABEL"] = df["MATERIAL_NUMBER"].str.strip() + " · " + df["MATERIAL_DESCRIPTION"]
    else:
        df["LABEL"] = df["MATERIAL_DESCRIPTION"]
    df = df.groupby("LABEL", as_index=False)["NB_STEPS_AVAILABLE"].mean()
    df = df.sort_values("NB_STEPS_AVAILABLE")
    colors = [RED if v < 10 else (YELLOW if v < 20 else GREEN) for v in df["NB_STEPS_AVAILABLE"]]
    fig = px.bar(df, x="NB_STEPS_AVAILABLE", y="LABEL", orientation="h",
                 labels={"NB_STEPS_AVAILABLE":"Avg Steps Stock Available", "LABEL":""})
    fig.update_traces(marker_color=colors)
    fig.add_vline(x=10, line_dash="dash", line_color=RED, annotation_text="Warning")
    return fig

def make_capacity_section(inv_hist):
    """Build warehouse capacity progress bar cards from latest inventory snapshot."""
    if inv_hist.empty or "SIM_ELAPSED_STEPS" not in inv_hist.columns:
        usage = {"Finished Goods": 0, "Raw Materials": 0, "Packaging": 0}
    else:
        latest_step = inv_hist["SIM_ELAPSED_STEPS"].max()
        df = inv_hist[inv_hist["SIM_ELAPSED_STEPS"] == latest_step]
        usage = df.groupby("MAT_TYPE")["INVENTORY_OPENING_BALANCE"].sum().to_dict()

    cards = []
    for label, cap in WAREHOUSE_CAPACITY.items():
        current = int(usage.get(label, 0))
        pct = min(current / cap * 100, 100) if cap else 0
        bar_color = RED if pct >= 90 else (YELLOW if pct >= 70 else GREEN)
        cards.append(dbc.Col(dbc.Card([
            dbc.CardBody([
                html.P(label, className="text-muted mb-1",
                       style={"fontSize":"0.8rem","textTransform":"uppercase","letterSpacing":"0.05em"}),
                html.Div([
                    html.Span(f"{current:,}", style={"color":"#fff","fontWeight":"700","fontSize":"1.1rem"}),
                    html.Span(f" / {cap:,}", style={"color":"#8b90a0","fontSize":"0.9rem"}),
                ], style={"marginBottom":"8px"}),
                dbc.Progress(value=pct, style={"height":"10px","borderRadius":"5px"},
                             color=bar_color if bar_color != RED else "danger",
                             className="mb-1"),
                html.Small(f"{pct:.1f}% used{' — ⚠ Nearly Full!' if pct >= 90 else (' — Getting Full' if pct >= 70 else '')}",
                           style={"color": bar_color}),
            ])
        ], style={"background": CARD_BG, "border": f"1px solid {bar_color}",
                  "borderRadius":"10px"}), md=4))
    return cards

@safe
def fig_inventory_history_by_type(inv_hist):
    """Line chart showing inventory levels over time by warehouse type."""
    df = inv_hist.groupby(["SIM_ELAPSED_STEPS","MAT_TYPE"], as_index=False)["INVENTORY_OPENING_BALANCE"].sum()
    fig = go.Figure()
    type_colors = {"Finished Goods": ACCENT, "Raw Materials": GREEN, "Packaging": YELLOW}
    for mat_type, cap in WAREHOUSE_CAPACITY.items():
        sub = df[df["MAT_TYPE"] == mat_type].sort_values("SIM_ELAPSED_STEPS")
        fig.add_trace(go.Scatter(
            x=sub["SIM_ELAPSED_STEPS"], y=sub["INVENTORY_OPENING_BALANCE"],
            name=mat_type, mode="lines+markers",
            line=dict(color=type_colors.get(mat_type, "#8b90a0"), width=2),
        ))
        fig.add_hline(y=cap, line_dash="dot",
                      line_color=type_colors.get(mat_type, "#8b90a0"),
                      opacity=0.4,
                      annotation_text=f"{mat_type} cap ({cap:,})",
                      annotation_font_color=type_colors.get(mat_type, "#8b90a0"),
                      annotation_position="right")
    fig.update_layout(hovermode="x unified", yaxis_title="Units in Stock")
    return fig

@safe
def fig_price_heatmap(pricing, market):
    """Heatmap: our price vs market average. Red=too low, Green=close, Orange=too high."""
    # Get latest period market averages by product + channel
    mkt = to_num(market.copy(), ["AVERAGE_PRICE", "SIM_PERIOD"])
    latest_period = mkt["SIM_PERIOD"].max()
    mkt_avg = (mkt[mkt["SIM_PERIOD"] == latest_period]
               .groupby(["MATERIAL_DESCRIPTION","DISTRIBUTION_CHANNEL"], as_index=False)
               ["AVERAGE_PRICE"].mean())

    # Our prices with channel code
    our = pricing.copy()
    our = to_num(our, ["PRICE"])

    # Merge on product + channel code
    merged = our.merge(mkt_avg, on=["MATERIAL_DESCRIPTION","DISTRIBUTION_CHANNEL"], how="inner")
    if merged.empty:
        return empty_fig("No market comparison data yet")

    # % diff: positive = our price higher than market, negative = lower
    merged["PCT_DIFF"] = ((merged["PRICE"] - merged["AVERAGE_PRICE"]) / merged["AVERAGE_PRICE"] * 100).round(1)
    merged["LABEL"] = merged.apply(
        lambda r: f"Us: €{r['PRICE']:.2f}<br>Mkt: €{r['AVERAGE_PRICE']:.2f}<br>{r['PCT_DIFF']:+.1f}%", axis=1)

    # Pivot for heatmap
    z_df   = merged.pivot(index="MATERIAL_DESCRIPTION", columns="DC_NAME", values="PCT_DIFF")
    txt_df = merged.pivot(index="MATERIAL_DESCRIPTION", columns="DC_NAME", values="LABEL")

    # Cell labels: blank for NaN cells so we don't show stray "%"
    import numpy as np
    cell_text = pd.DataFrame(
        [[f"{v:+.1f}%" if pd.notna(v) else "" for v in row] for row in z_df.values],
        index=z_df.index, columns=z_df.columns,
    )
    # Hover text: replace NaN with "No data"
    hover_text = txt_df.fillna("No data")

    CLAMP = 20
    colorscale = [
        [0.0,  "#e74c3c"],
        [0.5,  "#2ecc71"],
        [1.0,  "#f39c12"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_df.values,
        x=z_df.columns.tolist(),
        y=z_df.index.tolist(),
        customdata=hover_text.values,
        hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
        text=cell_text.values,
        texttemplate="%{text}",
        colorscale=colorscale,
        zmin=-CLAMP, zmid=0, zmax=CLAMP,
        colorbar=dict(
            title=dict(text="% vs Market", font=dict(color="#c8cdd8")),
            tickvals=[-20, -10, 0, 10, 20],
            ticktext=["-20% (Low)", "-10%", "At Market", "+10%", "+20% (High)"],
            tickfont=dict(color="#c8cdd8"),
        ),
    ))
    # Dynamic height: 60px per row, min 360
    row_height = max(360, len(z_df) * 62 + 120)
    fig.update_layout(
        xaxis_title="Distribution Channel",
        yaxis_title="",
        height=row_height,
        xaxis=dict(side="top"),
    )
    return fig

@safe
def fig_price_vs_market_bar(pricing, market):
    """Bar chart comparing our price vs market avg per product for latest period."""
    mkt = to_num(market.copy(), ["AVERAGE_PRICE", "SIM_PERIOD"])
    latest_period = mkt["SIM_PERIOD"].max()
    mkt_avg = (mkt[mkt["SIM_PERIOD"] == latest_period]
               .groupby(["MATERIAL_DESCRIPTION","DISTRIBUTION_CHANNEL"], as_index=False)
               ["AVERAGE_PRICE"].mean())

    our = to_num(pricing.copy(), ["PRICE"])
    merged = our.merge(mkt_avg, on=["MATERIAL_DESCRIPTION","DISTRIBUTION_CHANNEL"], how="inner")
    if merged.empty:
        return empty_fig("No market comparison data yet")

    merged["PCT_DIFF"] = ((merged["PRICE"] - merged["AVERAGE_PRICE"]) / merged["AVERAGE_PRICE"] * 100).round(1)
    merged = merged.sort_values(["DC_NAME","MATERIAL_DESCRIPTION"])
    bar_colors = [RED if v < -5 else (YELLOW if v > 10 else GREEN) for v in merged["PCT_DIFF"]]

    fig = go.Figure()
    fig.add_bar(name="Our Price", x=merged["MATERIAL_DESCRIPTION"],
                y=merged["PRICE"], marker_color=bar_colors,
                text=merged["DC_NAME"], hovertemplate="%{x}<br>%{text}<br>Our: €%{y:.2f}<extra></extra>")
    fig.add_trace(go.Scatter(name="Market Avg", x=merged["MATERIAL_DESCRIPTION"],
                             y=merged["AVERAGE_PRICE"], mode="markers",
                             marker=dict(color="#fff", size=8, symbol="diamond"),
                             hovertemplate="%{x}<br>Market Avg: €%{y:.2f}<extra></extra>"))
    fig.update_layout(hovermode="x unified", xaxis_tickangle=-35,
                      yaxis_title="Price (EUR)", barmode="group")
    return fig

@safe
def fig_market_share(sales, market):
    our  = sales.groupby(["SIM_PERIOD","MATERIAL_DESCRIPTION"], as_index=False)["QUANTITY"].sum()
    our.rename(columns={"QUANTITY":"OurQty"}, inplace=True)
    mkt  = market.groupby(["SIM_PERIOD","MATERIAL_DESCRIPTION"], as_index=False)["QUANTITY"].sum()
    mkt.rename(columns={"QUANTITY":"MktQty"}, inplace=True)
    merged = our.merge(mkt, on=["SIM_PERIOD","MATERIAL_DESCRIPTION"], how="inner")
    merged["Share%"] = (merged["OurQty"] / merged["MktQty"] * 100).round(1)
    fig = px.line(merged, x="SIM_PERIOD", y="Share%", color="MATERIAL_DESCRIPTION",
                  markers=True, color_discrete_sequence=COLORS,
                  labels={"SIM_PERIOD":"Period","Share%":"Market Share (%)"})
    fig.add_hline(y=25, line_dash="dot", line_color="gray", annotation_text="25% benchmark")
    return fig

@safe
def fig_carbon_by_type(carbon):
    df = carbon.groupby("TYPE", as_index=False)["CO2E_EMISSIONS"].sum()
    fig = px.pie(df, names="TYPE", values="CO2E_EMISSIONS", hole=0.5,
                 color_discrete_sequence=["#27ae60","#e67e22","#e74c3c","#3498db"])
    fig.update_traces(textinfo="label+percent+value")
    return fig

@safe
def fig_carbon_over_time(carbon):
    df = carbon.groupby("SIM_ELAPSED_STEPS", as_index=False)["CO2E_EMISSIONS"].sum()
    df["Cumulative"] = df["CO2E_EMISSIONS"].cumsum()
    fig = go.Figure()
    fig.add_bar(x=df["SIM_ELAPSED_STEPS"], y=df["CO2E_EMISSIONS"], name="Step CO₂e",
                marker_color="#e67e22", opacity=0.7)
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["Cumulative"],
                             name="Cumulative", mode="lines", yaxis="y2",
                             line=dict(color=RED, width=2)))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, color=RED))
    return fig

@safe
def fig_prod_gantt(prod_orders, current_elapsed):
    fig = go.Figure()
    for _, row in prod_orders.iloc[::-1].iterrows():
        begin  = row["BEGIN_ELAPSED"]
        dur    = max(row["END_ELAPSED"] - row["BEGIN_ELAPSED"] + 1, 1)
        color  = STATUS_COLORS.get(row["STATUS"], "#4a4f6a")
        border = "rgba(255,255,255,0.25)" if row["STATUS"] in ("Up Next","In Progress") else "rgba(0,0,0,0)"
        tip = (f"<b>{row['MATERIAL_DESCRIPTION']}</b><br>"
               f"Order: {row['PRODUCTION_ORDER']}<br>"
               f"Target: {int(row['TARGET_QUANTITY']):,} units<br>"
               f"Confirmed: {int(row['CONFIRMED_QUANTITY']):,} units<br>"
               f"Start: Round {row['BEGIN_LABEL']} → End: Round {row['END_LABEL']}<br>"
               f"Setup time: {int(row['SETUP_TIME'])} step(s)<br>"
               f"Status: <b>{row['STATUS']}</b>")
        fig.add_trace(go.Bar(
            x=[dur], y=[row["MATERIAL_DESCRIPTION"]],
            base=[begin], orientation="h",
            marker=dict(color=color, line=dict(color=border, width=2)),
            name=row["STATUS"],
            text=f"#{row['PRODUCTION_ORDER']}", textposition="inside",
            hovertemplate=tip + "<extra></extra>",
            showlegend=False,
        ))
    fig.add_vline(x=current_elapsed, line_color="#e74c3c", line_dash="dash",
                  annotation_text=f"Now (Step {current_elapsed})",
                  annotation_font_color="#e74c3c", annotation_position="top right")
    for status, color in STATUS_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(color=color, size=10, symbol="square"),
                                 name=status, showlegend=True))
    fig.update_layout(barmode="overlay", hovermode="closest",
                      xaxis_title="Elapsed Simulation Step", yaxis_title="",
                      height=360,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    return fig

@safe
def fig_to_be_produced_by_product(pending_orders):
    df = (pending_orders.groupby("MATERIAL_DESCRIPTION", as_index=False)["TARGET_QUANTITY"].sum()
                        .sort_values("TARGET_QUANTITY"))
    colors = [RED if v <= PROD_WARN_THRESHOLD else (YELLOW if v <= PROD_WARN_THRESHOLD * 2 else GREEN)
              for v in df["TARGET_QUANTITY"]]
    fig = px.bar(df, x="TARGET_QUANTITY", y="MATERIAL_DESCRIPTION", orientation="h",
                 labels={"TARGET_QUANTITY": "Units To Be Produced", "MATERIAL_DESCRIPTION": ""})
    fig.update_traces(marker_color=colors)
    fig.add_vline(x=PROD_WARN_THRESHOLD, line_dash="dash", line_color=RED,
                  annotation_text="120k Warning", annotation_font_color=RED,
                  annotation_position="top right")
    return fig

@safe
def fig_yield_over_time_detail(prod):
    df = prod.groupby(["SIM_ELAPSED_STEPS","MATERIAL_DESCRIPTION"], as_index=False)["YIELD"].sum()
    fig = px.bar(df, x="SIM_ELAPSED_STEPS", y="YIELD", color="MATERIAL_DESCRIPTION",
                 color_discrete_sequence=COLORS, barmode="stack",
                 labels={"SIM_ELAPSED_STEPS":"Step","YIELD":"Units Produced"})
    fig.update_layout(hovermode="x unified")
    return fig

@safe
def fig_actual_yield_by_product(prod):
    df = prod.groupby("MATERIAL_DESCRIPTION", as_index=False)["YIELD"].sum()
    df = df.sort_values("YIELD", ascending=True)
    fig = px.bar(df, x="YIELD", y="MATERIAL_DESCRIPTION", orientation="h",
                 color="YIELD", color_continuous_scale=["#1e3a5f","#2ecc71"],
                 labels={"YIELD":"Total Units Produced","MATERIAL_DESCRIPTION":""})
    fig.update_coloraxes(showscale=False)
    return fig

@safe
def fig_setup_time(prod_orders):
    df = prod_orders[prod_orders["SETUP_TIME"] > 0].copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No setup time data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#8b90a0"))
        return fig
    fig = px.bar(df, x="BEGIN_LABEL", y="SETUP_TIME", color="MATERIAL_DESCRIPTION",
                 color_discrete_sequence=COLORS,
                 labels={"BEGIN_LABEL":"Starts at Round/Step","SETUP_TIME":"Setup Steps Lost"})
    fig.add_hline(y=3, line_dash="dot", line_color=YELLOW,
                  annotation_text="Typical 3-step changeover")
    return fig

# ── Production detail components ───────────────────────────────────────────────
def _info_block(label, value, value_color=ACCENT, sub=None):
    return html.Div([
        html.Div(label, style={"fontSize":"0.7rem","textTransform":"uppercase",
                               "color":"#8b90a0","letterSpacing":"0.06em","marginBottom":"2px"}),
        html.Div(value, style={"fontSize":"1.1rem","fontWeight":"700","color":value_color}),
        html.Div(sub or "", style={"fontSize":"0.78rem","color":"#8b90a0","marginTop":"2px"}),
    ], style={"padding":"10px 14px","borderRight":"1px solid #2a2d3e","flex":"1","minWidth":"120px"})

def current_in_production_card(in_progress_orders):
    if in_progress_orders.empty:
        body = html.Div("No order currently in production.", className="text-muted",
                        style={"padding":"14px","fontStyle":"italic"})
    else:
        r = in_progress_orders.iloc[0]
        body = html.Div([
            html.Div(r["MATERIAL_DESCRIPTION"],
                     style={"fontSize":"1rem","fontWeight":"700","color":YELLOW,
                            "marginBottom":"8px","padding":"10px 14px 0"}),
            html.Div([
                _info_block("Order #",     r["PRODUCTION_ORDER"]),
                _info_block("Target",      f"{int(r['TARGET_QUANTITY']):,} units"),
                _info_block("Confirmed",   f"{int(r['CONFIRMED_QUANTITY']):,} units",
                            GREEN if r['CONFIRMED_QUANTITY'] > 0 else "#8b90a0"),
                _info_block("Start → End", f"{r['BEGIN_LABEL']} → {r['END_LABEL']}"),
                _info_block("Setup",       f"{int(r['SETUP_TIME'])} step(s)",
                            YELLOW if r['SETUP_TIME'] > 0 else GREEN),
            ], style={"display":"flex","flexWrap":"wrap","padding":"0 4px 10px"}),
        ])
    border_color = YELLOW if not in_progress_orders.empty else '#2a2d3e'
    return [
        html.H6("Currently In Production",
                style={"color":"#8b90a0","textTransform":"uppercase",
                       "letterSpacing":"0.08em","fontSize":"0.75rem","marginBottom":"10px"}),
        body,
    ], border_color

def up_next_card(up_next_orders, current_elapsed):
    if up_next_orders.empty:
        body = html.Div("No upcoming orders planned.", className="text-muted",
                        style={"padding":"14px","fontStyle":"italic"})
    else:
        r = up_next_orders.iloc[0]
        steps_away = r["BEGIN_ELAPSED"] - current_elapsed
        body = html.Div([
            html.Div(r["MATERIAL_DESCRIPTION"],
                     style={"fontSize":"1rem","fontWeight":"700","color":ACCENT,
                            "marginBottom":"8px","padding":"10px 14px 0"}),
            html.Div([
                _info_block("Order #",     r["PRODUCTION_ORDER"]),
                _info_block("Target",      f"{int(r['TARGET_QUANTITY']):,} units"),
                _info_block("Starts",      f"Round {r['BEGIN_LABEL']}",
                            GREEN if steps_away <= 2 else ACCENT,
                            f"{steps_away} step(s) away"),
                _info_block("Ends",        f"Round {r['END_LABEL']}"),
                _info_block("Setup cost",  f"{int(r['SETUP_TIME'])} step(s)",
                            YELLOW if r['SETUP_TIME'] > 0 else GREEN),
            ], style={"display":"flex","flexWrap":"wrap","padding":"0 4px 10px"}),
        ])
    return [
        html.H6("Up Next",
                style={"color":"#8b90a0","textTransform":"uppercase",
                       "letterSpacing":"0.08em","fontSize":"0.75rem","marginBottom":"10px"}),
        body,
    ]

def prod_order_table(order_df):
    if order_df.empty:
        return html.P("None", className="text-muted", style={"padding":"10px"})
    rows = []
    for _, r in order_df.iterrows():
        status_color = STATUS_COLORS.get(r["STATUS"], "#8b90a0")
        rows.append(html.Tr([
            html.Td(r["PRODUCTION_ORDER"],          style={"color":"#8b90a0","fontSize":"0.8rem","padding":"6px 8px"}),
            html.Td(r["MATERIAL_DESCRIPTION"],      style={"color":"#c8cdd8","padding":"6px 8px","fontWeight":"500"}),
            html.Td(f"{int(r['TARGET_QUANTITY']):,}", style={"color":ACCENT,"textAlign":"right","padding":"6px 8px"}),
            html.Td(r["BEGIN_LABEL"],               style={"color":"#8b90a0","textAlign":"center","padding":"6px 8px"}),
            html.Td(r["END_LABEL"],                 style={"color":"#8b90a0","textAlign":"center","padding":"6px 8px"}),
            html.Td(f"{int(r['SETUP_TIME'])}",       style={"color":YELLOW if r['SETUP_TIME']>0 else "#8b90a0","textAlign":"center","padding":"6px 8px"}),
            html.Td(r["STATUS"],                    style={"color":status_color,"fontWeight":"600","padding":"6px 8px"}),
        ]))
    header = html.Tr([
        html.Th(h, style={"color":"#8b90a0","fontSize":"0.72rem","textTransform":"uppercase",
                           "padding":"6px 8px","borderBottom":"1px solid #2a2d3e","textAlign":a})
        for h, a in [("Order","left"),("Product","left"),("Qty","right"),
                     ("Start","center"),("End","center"),("Setup","center"),("Status","left")]
    ])
    return html.Table([html.Thead(header), html.Tbody(rows)],
                      style={"width":"100%","borderCollapse":"collapse"})

# ── "Make" helpers for callback-updatable rows ─────────────────────────────────
def make_top_kpi_row(latest_val, total_revenue, total_margin):
    val_color = GREEN if latest_val["COMPANY_VALUATION"] >= 0 else RED
    return [
        dbc.Col(kpi_card("Company Valuation",
                         f"€{latest_val['COMPANY_VALUATION']:,.0f}",
                         f"Risk Rate: {latest_val['COMPANY_RISK_RATE_PCT']}%",
                         val_color), md=3),
        dbc.Col(kpi_card("Cash in Bank",
                         f"€{latest_val['BANK_CASH_ACCOUNT']:,.0f}",
                         f"Loan: €{latest_val['BANK_LOAN']:,.0f}",
                         GREEN if latest_val['BANK_CASH_ACCOUNT'] > 0 else RED), md=3),
        dbc.Col(kpi_card("Total Revenue",
                         f"€{total_revenue:,.0f}",
                         f"Margin: €{total_margin:,.0f}  ({total_margin/total_revenue*100:.1f}%)" if total_revenue else "No sales data yet"),
                md=3),
        dbc.Col(kpi_card("Credit Rating",
                         str(latest_val.get("CREDIT_RATING","—")),
                         f"Market Risk: {latest_val['MARKET_RISK_RATE_PCT']}%",
                         GREEN if "A" in str(latest_val.get("CREDIT_RATING","")) else YELLOW), md=3),
    ]

def make_header_info(latest_val):
    return [
        html.Span("● LIVE", style={"color":GREEN,"fontWeight":"600","marginRight":"8px"}),
        html.Span(f"Step {int(latest_val['SIM_ELAPSED_STEPS'])}  |  Round {latest_val.get('SIM_ROUND','01')}  |  {latest_val.get('SIM_DATE','')}",
                  style={"color":"#8b90a0"}),
    ]

def make_prod_kpi_row(total_to_produce, pending_orders, in_progress_orders, up_next_orders, total_produced, avg_yield_step):
    return [
        dbc.Col(kpi_card(
            "Total To Be Produced",
            f"{total_to_produce:,} units",
            f"{'⚠ Below 120k threshold!' if total_to_produce <= PROD_WARN_THRESHOLD else f'{len(pending_orders)} orders queued'}",
            RED if total_to_produce <= PROD_WARN_THRESHOLD else (YELLOW if total_to_produce <= PROD_WARN_THRESHOLD*2 else GREEN),
        ), md=3),
        dbc.Col(kpi_card(
            "Currently In Production",
            f"{len(in_progress_orders)} order(s)",
            in_progress_orders.iloc[0]["MATERIAL_DESCRIPTION"] if not in_progress_orders.empty else "Machine is idle",
            YELLOW if not in_progress_orders.empty else "#8b90a0",
        ), md=3),
        dbc.Col(kpi_card(
            "Total Produced (All Time)",
            f"{total_produced:,.0f} units",
            f"Avg {avg_yield_step:,.0f} units/step",
            GREEN,
        ), md=3),
        dbc.Col(kpi_card(
            "Orders Remaining",
            f"{len(pending_orders)}",
            f"Next: {up_next_orders.iloc[0]['MATERIAL_DESCRIPTION'] if not up_next_orders.empty else '—'}",
            ACCENT,
        ), md=3),
    ]

def make_notifications(pur_orders, ip_orders, inv_kpi):
    """Build notification badges: pending POs, in-progress production, low stock."""
    badges = []

    # Pending purchase orders (not yet delivered)
    if not pur_orders.empty and "STATUS" in pur_orders.columns:
        pending_po = pur_orders[pur_orders["STATUS"].str.lower() != "delivered"]
        for _, row in pending_po.iterrows():
            qty = f"{float(row['QUANTITY']):,.0f} {row.get('UNIT','')}"
            due = row.get("GOODS_RECEIPT_DATE", "?")
            badges.append(dbc.Badge(
                f"📦 PO {row['PURCHASING_ORDER']} · {row['MATERIAL_DESCRIPTION']} · {qty} · due {due}",
                color="warning", className="me-2 mb-1 p-2", style={"fontSize":"0.78rem","fontWeight":"500"},
            ))

    # In-progress production orders
    if not ip_orders.empty:
        for _, row in ip_orders.iterrows():
            end = row.get("END_LABEL", f"step {row.get('END_ELAPSED','?')}")
            badges.append(dbc.Badge(
                f"⚙ IN PRODUCTION · {row['MATERIAL_DESCRIPTION']} · {int(row.get('TARGET_QUANTITY',0)):,} units · ends {end}",
                color="info", className="me-2 mb-1 p-2", style={"fontSize":"0.78rem","fontWeight":"500"},
            ))

    # Low stock alerts
    if not inv_kpi.empty and "NB_STEPS_AVAILABLE" in inv_kpi.columns and "MATERIAL_NUMBER" in inv_kpi.columns:
        low = (inv_kpi[inv_kpi["NB_STEPS_AVAILABLE"] < 10]
               .groupby(["MATERIAL_NUMBER","MATERIAL_DESCRIPTION"])["NB_STEPS_AVAILABLE"]
               .mean().reset_index())
        for _, row in low.iterrows():
            steps = row["NB_STEPS_AVAILABLE"]
            clr = "danger" if steps < 5 else "warning"
            badges.append(dbc.Badge(
                f"⚠ LOW STOCK · {row['MATERIAL_NUMBER']} {row['MATERIAL_DESCRIPTION']} · {steps:.0f} steps left",
                color=clr, className="me-2 mb-1 p-2", style={"fontSize":"0.78rem","fontWeight":"500"},
            ))

    if not badges:
        badges = [dbc.Badge(
            "✓ All orders on track · No alerts",
            color="success", className="me-2 p-2", style={"fontSize":"0.78rem"},
        )]
    return badges

def make_data_snapshot(lv, tr, tm, ik, ip_orders, pr, mkt):
    """Build a compact JSON-serialisable summary of current game state for AI context."""
    snap = {
        "step":      int(lv.get("SIM_ELAPSED_STEPS", 0)),
        "round":     str(lv.get("SIM_ROUND", "?")),
        "date":      str(lv.get("SIM_DATE", "")),
        "cash":      float(lv.get("BANK_CASH_ACCOUNT", 0)),
        "loan":      float(lv.get("BANK_LOAN", 0)),
        "valuation": float(lv.get("COMPANY_VALUATION", 0)),
        "profit":    float(lv.get("PROFIT", 0)),
        "revenue":   float(tr),
        "margin":    float(tm),
        "credit":    str(lv.get("CREDIT_RATING", "—")),
        "inventory": [],
        "pricing":   [],
        "in_production": None,
        "pending_po": [],
    }
    # Inventory
    if not ik.empty and "MATERIAL_DESCRIPTION" in ik.columns:
        grp = ik.groupby(["MATERIAL_NUMBER","MATERIAL_DESCRIPTION"], as_index=False).agg(
            stock=("CURRENT_INVENTORY","sum"), days=("NB_STEPS_AVAILABLE","mean"))
        for _, r in grp.iterrows():
            snap["inventory"].append({
                "id": str(r["MATERIAL_NUMBER"]).strip(),
                "name": r["MATERIAL_DESCRIPTION"],
                "stock": int(r["stock"]),
                "days": round(float(r["days"]), 1),
            })
    # Pricing vs market
    if not pr.empty and not mkt.empty:
        mkt2 = to_num(mkt.copy(), ["AVERAGE_PRICE","SIM_PERIOD"])
        latest = mkt2["SIM_PERIOD"].max()
        mkt_avg = (mkt2[mkt2["SIM_PERIOD"]==latest]
                   .groupby(["MATERIAL_DESCRIPTION","DISTRIBUTION_CHANNEL"], as_index=False)
                   ["AVERAGE_PRICE"].mean())
        merged = to_num(pr.copy(), ["PRICE"]).merge(
            mkt_avg, on=["MATERIAL_DESCRIPTION","DISTRIBUTION_CHANNEL"], how="inner")
        for _, r in merged.iterrows():
            pct = ((r["PRICE"] - r["AVERAGE_PRICE"]) / r["AVERAGE_PRICE"] * 100) if r["AVERAGE_PRICE"] else 0
            snap["pricing"].append({
                "product":   r["MATERIAL_DESCRIPTION"],
                "channel":   r.get("DC_NAME", r["DISTRIBUTION_CHANNEL"]),
                "our_price": round(float(r["PRICE"]), 2),
                "mkt_price": round(float(r["AVERAGE_PRICE"]), 2),
                "pct":       round(pct, 1),
            })
    # In-progress production
    if not ip_orders.empty:
        row = ip_orders.iloc[0]
        snap["in_production"] = (f"{row['MATERIAL_DESCRIPTION']} — "
                                 f"{int(row.get('TARGET_QUANTITY',0)):,} units, "
                                 f"ends step {row.get('END_LABEL','?')}")
    return snap


def build_ai_context(snapshot):
    """Turn snapshot dict into a readable context block for the system prompt."""
    if not snapshot:
        return "No live data available yet."
    s = snapshot
    lines = [
        f"=== ERPsim Live Data — Step {s['step']}, Round {s['round']} ({s['date']}) ===",
        f"Cash: €{s['cash']:,.0f}  |  Loan: €{s['loan']:,.0f}  |  Valuation: €{s['valuation']:,.0f}  |  Profit: €{s['profit']:,.0f}",
        f"Total Revenue: €{s['revenue']:,.0f}  |  Margin: €{s['margin']:,.0f}  |  Credit: {s['credit']}",
        "",
        "INVENTORY (stock / steps of stock remaining):",
    ]
    for i in s.get("inventory", []):
        warn = " ⚠ LOW" if i["days"] < 10 else ""
        lines.append(f"  {i['id']} {i['name']}: {i['stock']:,} units, {i['days']} steps{warn}")
    lines.append("")
    lines.append("PRICING vs MARKET AVERAGE:")
    for p in s.get("pricing", []):
        arrow = "↓ too low" if p["pct"] < -5 else ("↑ high" if p["pct"] > 5 else "≈ at market")
        lines.append(f"  {p['product']} [{p['channel']}]: €{p['our_price']:.2f} vs mkt €{p['mkt_price']:.2f} ({p['pct']:+.1f}% {arrow})")
    if s.get("in_production"):
        lines.append(f"\nIN PRODUCTION: {s['in_production']}")
    return "\n".join(lines)

def make_sustain_kpi_row(carbon, total_co2):
    last_co2 = (carbon[carbon['SIM_ELAPSED_STEPS']==carbon['SIM_ELAPSED_STEPS'].max()]['CO2E_EMISSIONS'].sum()
                if (not carbon.empty and 'SIM_ELAPSED_STEPS' in carbon.columns) else 0)
    return [
        dbc.Col(kpi_card("Total CO₂e Emissions", f"{total_co2:,.0f} kg", "All scopes combined", "#e67e22"), md=6),
        dbc.Col(kpi_card("Emissions This Period", f"{last_co2:,.0f} kg", "Latest step", "#e67e22"), md=6),
    ]

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                title="ERPsim Dashboard")
server = app.server

_LOGIN_OVERLAY = html.Div(id="login-overlay", style={
    "position":"fixed","top":0,"left":0,"width":"100%","height":"100%",
    "backgroundColor":"#0f1117","zIndex":9999,
    "display":"flex","alignItems":"center","justifyContent":"center",
}, children=[
    dbc.Card(style={"width":"420px","background":"#1a1d27","border":"1px solid #2a2d3e","borderRadius":"12px"},
    children=[
        dbc.CardBody([
            html.H4("ERPsim Dashboard", className="text-white text-center mb-1",
                    style={"fontWeight":"700"}),
            html.P("Connect to your SAP OData server", className="text-muted text-center mb-4",
                   style={"fontSize":"0.85rem"}),
            dbc.Label("OData URL", style={"color":"#8b90a0","fontSize":"0.82rem"}),
            dbc.Input(id="input-url", value=BASE_URL, type="text",
                      className="mb-3",
                      style={"background":"#0f1117","color":"#fff","border":"1px solid #2a2d3e",
                             "fontSize":"0.82rem"}),
            dbc.Label("Username", style={"color":"#8b90a0","fontSize":"0.82rem"}),
            dbc.Input(id="input-username", placeholder="e.g. A_1", type="text",
                      className="mb-3",
                      style={"background":"#0f1117","color":"#fff","border":"1px solid #2a2d3e"}),
            dbc.Label("Password", style={"color":"#8b90a0","fontSize":"0.82rem"}),
            dbc.Input(id="input-password", placeholder="Password", type="password",
                      className="mb-3",
                      style={"background":"#0f1117","color":"#fff","border":"1px solid #2a2d3e"}),
            dbc.Button("Connect", id="login-btn", color="primary", className="w-100 mb-2"),
            html.Div(id="login-status", className="text-center",
                     style={"color":RED,"fontSize":"0.83rem","minHeight":"20px"}),
        ])
    ])
])

app.layout = dbc.Container(fluid=True,
    style={"backgroundColor": BG, "minHeight":"100vh", "padding":"20px"}, children=[

    dcc.Store(id="auth-store",     storage_type="session"),
    dcc.Store(id="data-snapshot",  storage_type="memory"),
    dcc.Store(id="chat-history",   storage_type="memory", data=[]),
    dcc.Interval(id="refresh", interval=REFRESH_S * 1000, n_intervals=0),
    _LOGIN_OVERLAY,

    # ── Floating AI chat panel ──────────────────────────────────────────────────
    html.Div(style={"position":"fixed","bottom":"24px","right":"24px","zIndex":4000}, children=[
        # Toggle button
        dbc.Button("✦ Ask AI", id="chat-toggle", color="primary", size="sm",
                   style={"borderRadius":"20px","fontWeight":"600","boxShadow":"0 2px 12px rgba(79,142,247,0.4)"}),
        # Chat window
        html.Div(id="chat-window", style={"display":"none"}, children=[
            dbc.Card(style={
                "width":"380px","background":"#1a1d27","border":"1px solid #2a2d3e",
                "borderRadius":"12px","marginBottom":"10px",
                "boxShadow":"0 4px 24px rgba(0,0,0,0.5)",
            }, children=[
                # Header
                dbc.CardHeader(html.Div([
                    html.Span("✦ ERPsim AI Analyst", style={"color":"#fff","fontWeight":"600","fontSize":"0.9rem"}),
                    dbc.Button("✕", id="chat-close", color="link", size="sm",
                               style={"color":"#8b90a0","float":"right","padding":"0","lineHeight":"1"}),
                ]), style={"background":"#1a1d27","border":"none","paddingBottom":"8px"}),
                # Message history
                html.Div(id="chat-messages", style={
                    "height":"340px","overflowY":"auto","padding":"12px",
                    "display":"flex","flexDirection":"column","gap":"8px",
                }, children=[
                    html.Div("Hi! I have your live ERPsim data. Ask me anything — pricing strategy, inventory risks, production decisions.",
                             style={"background":"#2a2d3e","color":"#c8cdd8","borderRadius":"8px",
                                    "padding":"10px 12px","fontSize":"0.83rem","maxWidth":"90%"}),
                ]),
                # Input row
                dbc.CardFooter(html.Div([
                    dbc.Input(id="chat-input", placeholder="Ask about your data...", type="text",
                              debounce=False,
                              style={"background":"#0f1117","color":"#fff","border":"1px solid #2a2d3e",
                                     "borderRadius":"8px","fontSize":"0.83rem","flex":"1"}),
                    dbc.Button("→", id="chat-send", color="primary", size="sm",
                               style={"borderRadius":"8px","marginLeft":"6px","fontWeight":"700"}),
                ], style={"display":"flex","alignItems":"center"}),
                style={"background":"#1a1d27","border":"none","paddingTop":"8px"}),
            ]),
        ]),
    ]),

    # Main dashboard (hidden until logged in)
    html.Div(id="main-content", style={"display":"none"}, children=[

    # Header
    dbc.Row(className="mb-3", children=[
        dbc.Col([
            html.H2("ERPsim Dashboard",
                    style={"color":"#fff","fontWeight":"700","marginBottom":"2px"}),
            html.Div(id="header-subtitle",
                     children=html.Small("SAP Client 435 · UWM", className="text-muted")),
        ], width=8),
        dbc.Col([
            html.Div([
                html.Div(id="header-info", children=make_header_info(latest_val),
                         className="mb-1"),
                dbc.Button("Switch Team", id="logout-btn", size="sm", color="secondary",
                           outline=True, style={"fontSize":"0.72rem","padding":"2px 10px"}),
            ], className="text-end mt-1"),
        ], width=4),
    ]),

    # Top KPI row
    dbc.Row(id="top-kpi-row", className="mb-3 g-3",
            children=make_top_kpi_row(latest_val, total_revenue, total_margin)),

    # Notifications bar
    dbc.Row(className="mb-3", children=[
        dbc.Col(html.Div([
            html.Small("ORDERS & ALERTS  ",
                       style={"color":"#8b90a0","fontWeight":"600","letterSpacing":"0.05em",
                              "fontSize":"0.72rem","marginRight":"6px","verticalAlign":"middle"}),
            html.Span(id="notifications-bar",
                      children=make_notifications(pur_orders, in_progress_orders, inv_kpi),
                      style={"display":"inline"}),
        ], style={"padding":"10px 16px","background":CARD_BG,
                  "borderRadius":"8px","border":"1px solid #2a2d3e"}))
    ]),

    # Tabs
    dbc.Tabs(style={"borderBottom":"1px solid #2a2d3e"}, children=[

        # Overview
        dbc.Tab(label="Overview", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Company Valuation & Profit Over Time", "g-valuation",
                                   fig_valuation_over_time(valuation)), md=8),
                dbc.Col(chart_card("Cash vs Bank Loan", "g-cash",
                                   fig_cash_and_debt(valuation)), md=4),
            ]),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Revenue & Margin Per Step", "g-sales-period",
                                   fig_sales_by_period(sales)), md=8),
                dbc.Col(chart_card("Production Output Per Step", "g-prod-time",
                                   fig_production_over_time(prod)), md=4),
            ]),
        ]),

        # Sales
        dbc.Tab(label="Sales", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Revenue by Product", "g-revenue-product",
                                   fig_revenue_by_product(sales)), md=6),
                dbc.Col(chart_card("Sales Split by Region", "g-sales-region",
                                   fig_sales_by_region(sales)), md=3),
                dbc.Col(chart_card("Wholesale vs Retail", "g-channel",
                                   fig_channel_split(sales)), md=3),
            ]),
        ]),

        # Inventory
        dbc.Tab(label="Inventory", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            # Warehouse Capacity
            dbc.Row(className="mt-3 mb-1", children=[
                dbc.Col(html.H6("Warehouse Capacity",
                                style={"color":"#8b90a0","textTransform":"uppercase",
                                       "letterSpacing":"0.08em","fontSize":"0.75rem"}))
            ]),
            dbc.Row(id="capacity-row", className="g-3",
                    children=make_capacity_section(inv_hist)),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Warehouse Levels Over Time", "g-inv-hist",
                                   fig_inventory_history_by_type(inv_hist)), md=12),
            ]),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Current Stock by Product & Region", "g-inv-kpi",
                                   fig_inventory_kpi(inv_kpi)), md=7),
                dbc.Col(chart_card("Days of Stock Available (avg steps)", "g-days-avail",
                                   fig_days_available(inv_kpi)), md=5),
            ]),
        ]),

        # Production
        dbc.Tab(label="Production", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(id="prod-kpi-row", className="mt-3 g-3",
                    children=make_prod_kpi_row(total_to_produce, pending_orders,
                                               in_progress_orders, up_next_orders,
                                               total_produced, avg_yield_step)),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(dbc.Card([
                    dbc.CardBody(id="div-current-prod",
                                 children=current_in_production_card(in_progress_orders)[0])
                ], id="card-current-prod",
                   style={"background": CARD_BG,
                          "border": f"1px solid {current_in_production_card(in_progress_orders)[1]}",
                          "borderRadius":"10px", "height":"100%"}), md=6),
                dbc.Col(dbc.Card([
                    dbc.CardBody(id="div-up-next",
                                 children=up_next_card(up_next_orders, current_elapsed))
                ], style={"background": CARD_BG, "border": f"1px solid {ACCENT}",
                          "borderRadius":"10px", "height":"100%"}), md=6),
            ]),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Production Schedule (Gantt)", "g-prod-gantt",
                                   fig_prod_gantt(prod_orders, current_elapsed)), md=12),
            ]),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Full Production Queue",
                                style={"color":"#8b90a0","textTransform":"uppercase",
                                       "letterSpacing":"0.08em","fontSize":"0.75rem","marginBottom":"12px"}),
                        html.Div(id="div-prod-queue", children=prod_order_table(prod_orders)),
                    ])
                ], style={"background": CARD_BG, "border":"1px solid #2a2d3e", "borderRadius":"10px"}), md=8),
                dbc.Col(chart_card("Setup Time Per Order (Steps Lost)", "g-setup-time",
                                   fig_setup_time(prod_orders)), md=4),
            ]),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Planned Qty Per Product (120k Warning)", "g-to-produce",
                                   fig_to_be_produced_by_product(pending_orders)), md=5),
                dbc.Col(chart_card("Actual Yield Per Step", "g-yield-time",
                                   fig_yield_over_time_detail(prod)), md=4),
                dbc.Col(chart_card("Total Units Produced by Product", "g-yield-product",
                                   fig_actual_yield_by_product(prod)), md=3),
            ]),
        ]),

        # Pricing
        dbc.Tab(label="Pricing", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Our Price vs Market Average — Heatmap", "g-price-heatmap",
                                   fig_price_heatmap(pricing, market), height=None), md=12),
            ]),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Our Price vs Market Average — By Product", "g-price-bar",
                                   fig_price_vs_market_bar(pricing, market)), md=12),
            ]),
        ]),

        # Market
        dbc.Tab(label="Market", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Market Share % by Product Over Periods", "g-market-share",
                                   fig_market_share(sales, market)), md=12),
            ]),
        ]),

        # Sustainability
        dbc.Tab(label="Sustainability", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col([
                    dbc.Row(id="sustain-kpi-row", className="g-3 mb-3",
                            children=make_sustain_kpi_row(carbon, total_co2)),
                    chart_card("CO₂e Over Time (step + cumulative)", "g-carbon-time",
                               fig_carbon_over_time(carbon)),
                ], md=6),
                dbc.Col(chart_card("Emissions by Type", "g-carbon-type",
                                   fig_carbon_by_type(carbon)), md=6),
            ]),
        ]),
    ]),

    html.Hr(style={"borderColor":"#2a2d3e","marginTop":"30px"}),
    html.P(f"Auto-refreshes every {REFRESH_S}s  ·  Data: SAP NetWeaver OData · Client 435",
           className="text-center text-muted", style={"fontSize":"0.75rem"}),

    ]),  # end main-content
])

# ── Login callback ─────────────────────────────────────────────────────────────
@callback(
    Output("auth-store",    "data"),
    Output("login-status",  "children"),
    Input("login-btn",      "n_clicks"),
    State("input-url",      "value"),
    State("input-username", "value"),
    State("input-password", "value"),
    prevent_initial_call=True,
)
def do_login(n_clicks, base_url, username, password):
    if not base_url or not username or not password:
        return no_update, "Please fill in all fields."
    base_url  = base_url.rstrip("/")
    username  = username.strip()
    password  = password.strip()
    try:
        r = requests.get(f"{base_url}/Company_Valuation", auth=(username, password),
                         params={"$format":"json","$top":"1"}, timeout=10)
        if r.status_code == 200:
            team = username.split("_")[0].upper()
            return {"base_url": base_url, "username": username,
                    "password": password, "team": team}, ""
        return no_update, f"Login failed — check credentials (HTTP {r.status_code})."
    except Exception as e:
        return no_update, f"Connection error: {e}"

@callback(
    Output("auth-store", "data", allow_duplicate=True),
    Input("logout-btn", "n_clicks"),
    prevent_initial_call=True,
)
def do_logout(n):
    return None

@callback(
    Output("login-overlay", "style"),
    Output("main-content",  "style"),
    Output("header-subtitle", "children"),
    Input("auth-store", "data"),
)
def toggle_auth(auth_data):
    hidden  = {"display":"none"}
    overlay = {"position":"fixed","top":0,"left":0,"width":"100%","height":"100%",
                "backgroundColor":"#0f1117","zIndex":9999,
                "display":"flex","alignItems":"center","justifyContent":"center"}
    if auth_data:
        team = auth_data.get("team","?")
        subtitle = html.Small(f"Team {team} · SAP Client 435 · UWM", className="text-muted")
        return hidden, {"display":"block"}, subtitle
    return overlay, hidden, no_update

# ── Live refresh callback ──────────────────────────────────────────────────────
@callback(
    # Graphs
    Output("g-valuation",      "figure"),
    Output("g-cash",           "figure"),
    Output("g-sales-period",   "figure"),
    Output("g-prod-time",      "figure"),
    Output("g-revenue-product","figure"),
    Output("g-sales-region",   "figure"),
    Output("g-channel",        "figure"),
    Output("g-inv-kpi",        "figure"),
    Output("g-days-avail",     "figure"),
    Output("g-prod-gantt",     "figure"),
    Output("g-setup-time",     "figure"),
    Output("g-to-produce",     "figure"),
    Output("g-yield-time",     "figure"),
    Output("g-yield-product",  "figure"),
    Output("g-price-heatmap",  "figure"),
    Output("g-price-bar",      "figure"),
    Output("g-market-share",   "figure"),
    Output("g-carbon-type",    "figure"),
    Output("g-carbon-time",    "figure"),
    Output("g-inv-hist",       "figure"),
    # KPI rows
    Output("header-info",      "children"),
    Output("top-kpi-row",      "children"),
    Output("prod-kpi-row",     "children"),
    Output("sustain-kpi-row",  "children"),
    # Production detail
    Output("div-current-prod", "children"),
    Output("div-up-next",      "children"),
    Output("div-prod-queue",   "children"),
    Output("capacity-row",        "children"),
    Output("notifications-bar",   "children"),
    Output("data-snapshot",       "data"),
    Input("refresh", "n_intervals"),
    State("auth-store", "data"),
)
def refresh_all(n, auth_data):
    if not auth_data:
        return tuple([no_update] * 30)
    auth     = (auth_data["username"], auth_data["password"])
    base_url = auth_data.get("base_url", BASE_URL)
    # Reload all data from OData
    s, v, ik, mkt, c, p, po, ih, pr, pur = load_all(auth, base_url)
    (lv, tr, tm, tco2, ce, tp, ays,
     ip_orders, un_orders, pend, ttp, po) = compute_derived(s, v, ik, c, p, po)

    card_body, border_color = current_in_production_card(ip_orders)

    def sf(fig):
        return style_fig(fig)

    return (
        sf(fig_valuation_over_time(v)),
        sf(fig_cash_and_debt(v)),
        sf(fig_sales_by_period(s)),
        sf(fig_production_over_time(p)),
        sf(fig_revenue_by_product(s)),
        sf(fig_sales_by_region(s)),
        sf(fig_channel_split(s)),
        sf(fig_inventory_kpi(ik)),
        sf(fig_days_available(ik)),
        sf(fig_prod_gantt(po, ce)),
        sf(fig_setup_time(po)),
        sf(fig_to_be_produced_by_product(pend)),
        sf(fig_yield_over_time_detail(p)),
        sf(fig_actual_yield_by_product(p)),
        sf(fig_price_heatmap(pr, mkt)),
        sf(fig_price_vs_market_bar(pr, mkt)),
        sf(fig_market_share(s, mkt)),
        sf(fig_carbon_by_type(c)),
        sf(fig_carbon_over_time(c)),
        sf(fig_inventory_history_by_type(ih)),
        make_header_info(lv),
        make_top_kpi_row(lv, tr, tm),
        make_prod_kpi_row(ttp, pend, ip_orders, un_orders, tp, ays),
        make_sustain_kpi_row(c, tco2),
        card_body,
        up_next_card(un_orders, ce),
        prod_order_table(po),
        make_capacity_section(ih),
        make_notifications(pur, ip_orders, ik),
        make_data_snapshot(lv, tr, tm, ik, ip_orders, pr, mkt),
    )

# ── Chat callbacks ─────────────────────────────────────────────────────────────
@callback(
    Output("chat-window", "style"),
    Input("chat-toggle",  "n_clicks"),
    Input("chat-close",   "n_clicks"),
    State("chat-window",  "style"),
    prevent_initial_call=True,
)
def toggle_chat(open_clicks, close_clicks, current_style):
    from dash import ctx
    if ctx.triggered_id == "chat-close":
        return {"display": "none"}
    visible = current_style and current_style.get("display") != "none"
    return {"display": "none"} if visible else {"display": "block"}

@callback(
    Output("chat-messages", "children"),
    Output("chat-history",  "data"),
    Output("chat-input",    "value"),
    Input("chat-send",      "n_clicks"),
    State("chat-input",     "value"),
    State("chat-history",   "data"),
    State("data-snapshot",  "data"),
    prevent_initial_call=True,
)
def send_message(n_clicks, user_text, history, snapshot):
    if not user_text or not user_text.strip():
        return no_update, no_update, no_update

    if not ANTHROPIC_KEY:
        err_bubble = html.Div("⚠ ANTHROPIC_API_KEY not set.",
                              style={"background":"#e74c3c22","color":RED,"borderRadius":"8px",
                                     "padding":"10px 12px","fontSize":"0.83rem","maxWidth":"90%",
                                     "alignSelf":"flex-start"})
        return [err_bubble], history, ""

    history = history or []
    history.append({"role": "user", "content": user_text.strip()})

    system_prompt = (
        "You are an expert ERPsim business analyst and coach. "
        "ERPsim is a SAP business simulation where teams manage a muesli company — "
        "buying raw materials, scheduling production, setting prices, and managing cash. "
        "You have access to the team's live data below. Give concise, actionable advice. "
        "Focus on what to do THIS step. Be direct — the simulation moves fast.\n\n"
        + build_ai_context(snapshot)
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system_prompt,
            messages=history,
        )
        reply = response.content[0].text
    except Exception as e:
        reply = f"Error calling Claude API: {e}"

    history.append({"role": "assistant", "content": reply})

    # Rebuild message bubbles from full history
    greeting = html.Div(
        "Hi! I have your live ERPsim data. Ask me anything — pricing strategy, inventory risks, production decisions.",
        style={"background":"#2a2d3e","color":"#c8cdd8","borderRadius":"8px",
               "padding":"10px 12px","fontSize":"0.83rem","maxWidth":"90%","alignSelf":"flex-start"})
    bubbles = [greeting]
    for msg in history:
        is_user = msg["role"] == "user"
        bubbles.append(html.Div(msg["content"], style={
            "background":   ACCENT + "33" if is_user else "#2a2d3e",
            "color":        "#fff" if is_user else "#c8cdd8",
            "borderRadius": "8px",
            "padding":      "10px 12px",
            "fontSize":     "0.83rem",
            "maxWidth":     "85%",
            "alignSelf":    "flex-end" if is_user else "flex-start",
        }))

    return bubbles, history, ""

if __name__ == "__main__":
    app.run(debug=False, port=8050)
