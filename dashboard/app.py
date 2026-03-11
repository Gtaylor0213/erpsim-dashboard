import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL  = "http://uno.ucc.uwm.edu/odata/435"
AUTH      = ("A_1", "legoman99")
COMPANY   = "AA"
REFRESH_S = 60          # auto-refresh interval (seconds)

CHANNEL_MAP = {"12": "Wholesale", "14": "Retail"}
LOC_MAP     = {"02": "Central", "02N": "North", "02S": "South", "02W": "West"}

# ── Data helpers ──────────────────────────────────────────────────────────────
def fetch(entity: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{entity}?$format=json"
    resp = requests.get(url, auth=AUTH, timeout=30)
    resp.raise_for_status()
    rows = resp.json()["d"]["results"]
    df = pd.DataFrame(rows)
    df.drop(columns=["__metadata"], errors="ignore", inplace=True)
    return df

def to_num(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

STEPS_PER_ROUND = 20   # ERPsim simulation steps per round

def rs_to_elapsed(round_str, step_str) -> int:
    """Convert '04/02' style round/step to absolute elapsed step number."""
    try:
        return (int(round_str) - 1) * STEPS_PER_ROUND + int(step_str)
    except Exception:
        return 0

def load_all():
    sales       = fetch("Sales")
    valuation   = fetch("Company_Valuation")
    inv_kpi     = fetch("Current_Inventory_KPI")
    market      = fetch("Market")
    carbon      = fetch("Carbon_Emissions")
    prod        = fetch("Production")
    prod_orders = fetch("Production_Orders")

    sales       = to_num(sales,       ["QUANTITY","QUANTITY_DELIVERED","NET_PRICE","NET_VALUE","COST","SIM_ELAPSED_STEPS","SIM_PERIOD"])
    valuation   = to_num(valuation,   ["BANK_CASH_ACCOUNT","ACCOUNTS_RECEIVABLE","BANK_LOAN",
                                       "ACCOUNTS_PAYABLE","PROFIT","COMPANY_VALUATION",
                                       "COMPANY_RISK_RATE_PCT","MARKET_RISK_RATE_PCT","SIM_ELAPSED_STEPS"])
    inv_kpi     = to_num(inv_kpi,     ["CURRENT_INVENTORY","QUANTITY_SOLD","NB_STEPS_AVAILABLE","SIM_ELAPSED_STEPS"])
    market      = to_num(market,      ["QUANTITY","AVERAGE_PRICE","NET_VALUE","SIM_PERIOD"])
    carbon      = to_num(carbon,      ["CO2E_EMISSIONS","TOTAL_CO2E_EMISSIONS","SIM_ELAPSED_STEPS"])
    prod        = to_num(prod,        ["SIM_ELAPSED_STEPS"])
    prod_orders = to_num(prod_orders, ["SIM_ELAPSED_STEPS","TARGET_QUANTITY","CONFIRMED_QUANTITY","SETUP_TIME"])

    # derived columns (guard against empty datasets between sim rounds)
    if not sales.empty and "NET_VALUE" in sales.columns:
        sales["MARGIN"]  = sales["NET_VALUE"] - sales["COST"]
        sales["CHANNEL"] = sales["DISTRIBUTION_CHANNEL"].map(CHANNEL_MAP).fillna(sales["DISTRIBUTION_CHANNEL"])
        sales["REGION"]  = sales["STORAGE_LOCATION"].map(LOC_MAP).fillna(sales["STORAGE_LOCATION"])
    else:
        for col in ["MARGIN","CHANNEL","REGION","NET_VALUE","COST","QUANTITY","NET_PRICE"]:
            sales[col] = pd.Series(dtype=float)
    if not inv_kpi.empty and "STORAGE_LOCATION" in inv_kpi.columns:
        inv_kpi["REGION"] = inv_kpi["STORAGE_LOCATION"].map(LOC_MAP).fillna(inv_kpi["STORAGE_LOCATION"])

    # ── Production Orders: keep only the latest planning snapshot ──────────
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

    return sales, valuation, inv_kpi, market, carbon, prod, prod_orders

# ── Load data ─────────────────────────────────────────────────────────────────
sales, valuation, inv_kpi, market, carbon, prod, prod_orders = load_all()

# ── Latest snapshot values ────────────────────────────────────────────────────
_VAL_EMPTY = {"COMPANY_VALUATION":0,"BANK_CASH_ACCOUNT":0,"BANK_LOAN":0,
              "PROFIT":0,"COMPANY_RISK_RATE_PCT":0,"MARKET_RISK_RATE_PCT":0,
              "CREDIT_RATING":"—","SIM_ELAPSED_STEPS":0,"SIM_ROUND":"—","SIM_DATE":"—"}
latest_val   = valuation.sort_values("SIM_ELAPSED_STEPS").iloc[-1] if not valuation.empty else pd.Series(_VAL_EMPTY)
total_revenue = sales["NET_VALUE"].sum() if not sales.empty else 0.0
total_margin  = sales["MARGIN"].sum()  if not sales.empty else 0.0
total_qty     = sales["QUANTITY"].sum() if not sales.empty else 0.0
total_co2     = carbon["CO2E_EMISSIONS"].sum() if not carbon.empty else 0.0

# ── Production KPIs ───────────────────────────────────────────────────────────
PROD_WARN_THRESHOLD = 120_000
current_elapsed = int(inv_kpi["SIM_ELAPSED_STEPS"].max()) if (not inv_kpi.empty and "SIM_ELAPSED_STEPS" in inv_kpi.columns) else 0
prod["YIELD"]   = pd.to_numeric(prod["YIELD"], errors="coerce") if (not prod.empty and "YIELD" in prod.columns) else pd.Series(dtype=float)
total_produced  = prod["YIELD"].sum() if not prod.empty else 0.0
avg_yield_step  = prod.groupby("SIM_ELAPSED_STEPS")["YIELD"].sum().mean() if (not prod.empty and "SIM_ELAPSED_STEPS" in prod.columns) else 0.0

# Classify each order
def classify_order(row, cur):
    if row["CONFIRMED_QUANTITY"] > 0:
        return "Completed"
    if row["BEGIN_ELAPSED"] <= cur <= row["END_ELAPSED"]:
        return "In Progress"
    if row["BEGIN_ELAPSED"] > cur:
        return "Queued"
    return "Completed"

if not prod_orders.empty and "BEGIN_ELAPSED" in prod_orders.columns:
    prod_orders["STATUS"] = prod_orders.apply(lambda r: classify_order(r, current_elapsed), axis=1)
    queued_idx = prod_orders[prod_orders["STATUS"] == "Queued"].index
    if len(queued_idx):
        prod_orders.at[queued_idx[0], "STATUS"] = "Up Next"
else:
    prod_orders["STATUS"] = pd.Series(dtype=str)

in_progress_orders = prod_orders[prod_orders["STATUS"] == "In Progress"] if "STATUS" in prod_orders.columns else prod_orders.iloc[0:0]
up_next_orders     = prod_orders[prod_orders["STATUS"] == "Up Next"]     if "STATUS" in prod_orders.columns else prod_orders.iloc[0:0]
queued_orders      = prod_orders[prod_orders["STATUS"] == "Queued"]      if "STATUS" in prod_orders.columns else prod_orders.iloc[0:0]
pending_orders     = prod_orders[prod_orders["STATUS"].isin(["Up Next","Queued","In Progress"])] if "STATUS" in prod_orders.columns else prod_orders.iloc[0:0]
total_to_produce   = int(pending_orders["TARGET_QUANTITY"].sum()) if (not pending_orders.empty and "TARGET_QUANTITY" in pending_orders.columns) else 0

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS  = px.colors.qualitative.Safe
BG      = "#0f1117"
CARD_BG = "#1a1d27"
ACCENT  = "#4f8ef7"
GREEN   = "#2ecc71"
RED     = "#e74c3c"
YELLOW  = "#f39c12"

def kpi_card(title, value, sub=None, color=ACCENT):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize":"0.8rem","textTransform":"uppercase","letterSpacing":"0.05em"}),
            html.H4(value, style={"color": color, "fontWeight":"700", "marginBottom":"2px"}),
            html.Small(sub or "", className="text-muted"),
        ])
    ], style={"background": CARD_BG, "border":"1px solid #2a2d3e", "borderRadius":"10px"}, className="h-100")

def empty_fig(msg="No data yet — waiting for simulation to start"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#8b90a0", size=13))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def safe(fn):
    """Decorator: return empty_fig if the chart function raises (e.g. empty data)."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return empty_fig()
    return wrapper

def chart_card(title, fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#c8cdd8",
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#2a2d3e", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2d3e", zeroline=False)
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="mb-3", style={"color":"#8b90a0","textTransform":"uppercase","letterSpacing":"0.08em","fontSize":"0.75rem"}),
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height":"280px"}),
        ])
    ], style={"background": CARD_BG, "border":"1px solid #2a2d3e", "borderRadius":"10px"})

# ── Chart builders ────────────────────────────────────────────────────────────
@safe
def fig_valuation_over_time():
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
def fig_cash_and_debt():
    df = valuation.sort_values("SIM_ELAPSED_STEPS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["BANK_CASH_ACCOUNT"],
                             name="Cash", mode="lines", line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["BANK_LOAN"],
                             name="Bank Loan", mode="lines", line=dict(color=RED, width=2)))
    fig.add_trace(go.Scatter(x=df["SIM_ELAPSED_STEPS"], y=df["DEBT_LOADING"],
                             name="Net Debt Loading", mode="lines",
                             line=dict(color=YELLOW, width=2, dash="dash")))
    return fig

@safe
def fig_sales_by_period():
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
def fig_revenue_by_product():
    df = sales.groupby("MATERIAL_DESCRIPTION", as_index=False)["NET_VALUE"].sum()
    df = df.sort_values("NET_VALUE", ascending=True)
    fig = px.bar(df, x="NET_VALUE", y="MATERIAL_DESCRIPTION", orientation="h",
                 color="NET_VALUE", color_continuous_scale=["#1e3a5f","#4f8ef7"],
                 labels={"NET_VALUE":"Revenue (EUR)", "MATERIAL_DESCRIPTION":""})
    fig.update_coloraxes(showscale=False)
    return fig

@safe
def fig_sales_by_region():
    df = sales.groupby("REGION", as_index=False)["NET_VALUE"].sum()
    fig = px.pie(df, names="REGION", values="NET_VALUE", hole=0.55,
                 color_discrete_sequence=COLORS)
    fig.update_traces(textposition="outside", textinfo="label+percent")
    return fig

@safe
def fig_channel_split():
    df = sales.groupby("CHANNEL", as_index=False).agg(Revenue=("NET_VALUE","sum"), Margin=("MARGIN","sum"))
    fig = px.bar(df, x="CHANNEL", y=["Revenue","Margin"],
                 barmode="group", color_discrete_map={"Revenue":ACCENT,"Margin":GREEN})
    return fig

@safe
def fig_inventory_kpi():
    df = inv_kpi.groupby(["MATERIAL_DESCRIPTION","REGION"], as_index=False).agg(
        Stock=("CURRENT_INVENTORY","sum"), DaysAvail=("NB_STEPS_AVAILABLE","mean"))
    fig = px.bar(df, x="MATERIAL_DESCRIPTION", y="Stock", color="REGION",
                 color_discrete_sequence=COLORS, barmode="stack",
                 labels={"MATERIAL_DESCRIPTION":"", "Stock":"Current Inventory (units)"})
    fig.update_xaxes(tickangle=-35)
    return fig

@safe
def fig_days_available():
    df = inv_kpi.groupby("MATERIAL_DESCRIPTION", as_index=False)["NB_STEPS_AVAILABLE"].mean()
    df = df.sort_values("NB_STEPS_AVAILABLE")
    colors = [RED if v < 10 else (YELLOW if v < 20 else GREEN) for v in df["NB_STEPS_AVAILABLE"]]
    fig = px.bar(df, x="NB_STEPS_AVAILABLE", y="MATERIAL_DESCRIPTION", orientation="h",
                 labels={"NB_STEPS_AVAILABLE":"Avg Steps Stock Available", "MATERIAL_DESCRIPTION":""})
    fig.update_traces(marker_color=colors)
    fig.add_vline(x=10, line_dash="dash", line_color=RED, annotation_text="Warning")
    return fig

@safe
def fig_market_share():
    # company sales vs total market by product/period
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
def fig_carbon_by_type():
    df = carbon.groupby("TYPE", as_index=False)["CO2E_EMISSIONS"].sum()
    fig = px.pie(df, names="TYPE", values="CO2E_EMISSIONS", hole=0.5,
                 color_discrete_sequence=["#27ae60","#e67e22","#e74c3c","#3498db"],
                 title="")
    fig.update_traces(textinfo="label+percent+value")
    return fig

@safe
def fig_carbon_over_time():
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

# ── Production Tab Charts ─────────────────────────────────────────────────────

STATUS_COLORS = {
    "Completed":  "#2ecc71",
    "In Progress": "#f39c12",
    "Up Next":    "#4f8ef7",
    "Queued":     "#4a4f6a",
}

@safe
def fig_prod_gantt():
    """Horizontal Gantt chart of all production orders."""
    fig = go.Figure()
    # Draw in reverse order so first order appears at top
    for _, row in prod_orders.iloc[::-1].iterrows():
        begin = row["BEGIN_ELAPSED"]
        dur   = max(row["END_ELAPSED"] - row["BEGIN_ELAPSED"] + 1, 1)
        color = STATUS_COLORS.get(row["STATUS"], "#4a4f6a")
        border= "rgba(255,255,255,0.25)" if row["STATUS"] in ("Up Next","In Progress") else "rgba(0,0,0,0)"
        label = f"#{row['PRODUCTION_ORDER']}"
        tip   = (f"<b>{row['MATERIAL_DESCRIPTION']}</b><br>"
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
            text=label, textposition="inside",
            hovertemplate=tip + "<extra></extra>",
            showlegend=False,
        ))

    # Current-step reference line
    fig.add_vline(x=current_elapsed, line_color="#e74c3c", line_dash="dash",
                  annotation_text=f"Now (Step {current_elapsed})",
                  annotation_font_color="#e74c3c", annotation_position="top right")

    # Warning threshold marker (120k label on x-axis context doesn't apply to Gantt time,
    # so we add a legend manually via invisible scatter)
    for status, color in STATUS_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(color=color, size=10, symbol="square"),
                                 name=status, showlegend=True))

    fig.update_layout(
        barmode="overlay", hovermode="closest",
        xaxis_title="Elapsed Simulation Step",
        yaxis_title="",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig

@safe
def fig_to_be_produced_by_product():
    """Horizontal bar: planned qty per product, colored by 120k warning threshold."""
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
def fig_actual_yield_by_product():
    """Total historical yield per product (bar)."""
    df = prod.groupby("MATERIAL_DESCRIPTION", as_index=False)["YIELD"].sum()
    df = df.sort_values("YIELD", ascending=True)
    fig = px.bar(df, x="YIELD", y="MATERIAL_DESCRIPTION", orientation="h",
                 color="YIELD", color_continuous_scale=["#1e3a5f", "#2ecc71"],
                 labels={"YIELD": "Total Units Produced", "MATERIAL_DESCRIPTION": ""})
    fig.update_coloraxes(showscale=False)
    return fig

@safe
def fig_yield_over_time_detail():
    """Stacked bar of actual yield per step (detail view for Production tab)."""
    df = prod.groupby(["SIM_ELAPSED_STEPS", "MATERIAL_DESCRIPTION"], as_index=False)["YIELD"].sum()
    fig = px.bar(df, x="SIM_ELAPSED_STEPS", y="YIELD", color="MATERIAL_DESCRIPTION",
                 color_discrete_sequence=COLORS, barmode="stack",
                 labels={"SIM_ELAPSED_STEPS": "Step", "YIELD": "Units Produced"})
    fig.update_layout(hovermode="x unified")
    return fig

@safe
def fig_setup_time():
    """Bar showing setup time cost (lost steps) per planned order."""
    df = prod_orders[prod_orders["SETUP_TIME"] > 0].copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No setup time data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#8b90a0"))
        return fig
    fig = px.bar(df, x="BEGIN_LABEL", y="SETUP_TIME", color="MATERIAL_DESCRIPTION",
                 color_discrete_sequence=COLORS,
                 labels={"BEGIN_LABEL": "Starts at Round/Step", "SETUP_TIME": "Setup Steps Lost"})
    fig.add_hline(y=3, line_dash="dot", line_color=YELLOW, annotation_text="Typical 3-step changeover")
    return fig

def prod_order_table(order_df):
    """Build a styled HTML table for a set of production orders."""
    if order_df.empty:
        return html.P("None", className="text-muted", style={"padding": "10px"})
    rows = []
    for _, r in order_df.iterrows():
        status_color = STATUS_COLORS.get(r["STATUS"], "#8b90a0")
        rows.append(html.Tr([
            html.Td(r["PRODUCTION_ORDER"],       style={"color":"#8b90a0","fontSize":"0.8rem","padding":"6px 8px"}),
            html.Td(r["MATERIAL_DESCRIPTION"],   style={"color":"#c8cdd8","padding":"6px 8px","fontWeight":"500"}),
            html.Td(f"{int(r['TARGET_QUANTITY']):,}", style={"color":ACCENT,"textAlign":"right","padding":"6px 8px"}),
            html.Td(r["BEGIN_LABEL"],            style={"color":"#8b90a0","textAlign":"center","padding":"6px 8px"}),
            html.Td(r["END_LABEL"],              style={"color":"#8b90a0","textAlign":"center","padding":"6px 8px"}),
            html.Td(f"{int(r['SETUP_TIME'])}",   style={"color":YELLOW if r['SETUP_TIME']>0 else "#8b90a0","textAlign":"center","padding":"6px 8px"}),
            html.Td(r["STATUS"],                 style={"color":status_color,"fontWeight":"600","padding":"6px 8px"}),
        ]))
    header = html.Tr([
        html.Th(h, style={"color":"#8b90a0","fontSize":"0.72rem","textTransform":"uppercase",
                           "padding":"6px 8px","borderBottom":"1px solid #2a2d3e","textAlign":a})
        for h, a in [("Order","left"),("Product","left"),("Qty","right"),
                      ("Start","center"),("End","center"),("Setup","center"),("Status","left")]
    ])
    return html.Table([html.Thead(header), html.Tbody(rows)],
                      style={"width":"100%","borderCollapse":"collapse"})

def _info_block(label, value, value_color=ACCENT, sub=None):
    return html.Div([
        html.Div(label, style={"fontSize":"0.7rem","textTransform":"uppercase",
                               "color":"#8b90a0","letterSpacing":"0.06em","marginBottom":"2px"}),
        html.Div(value, style={"fontSize":"1.1rem","fontWeight":"700","color":value_color}),
        html.Div(sub or "", style={"fontSize":"0.78rem","color":"#8b90a0","marginTop":"2px"}),
    ], style={"padding":"10px 14px","borderRight":"1px solid #2a2d3e","flex":"1","minWidth":"120px"})

def current_in_production_card():
    if in_progress_orders.empty:
        body = html.Div([
            html.Div("No order currently in production.", className="text-muted",
                     style={"padding":"14px","fontStyle":"italic"}),
        ])
    else:
        r = in_progress_orders.iloc[0]
        body = html.Div([
            html.Div(r["MATERIAL_DESCRIPTION"],
                     style={"fontSize":"1rem","fontWeight":"700","color":YELLOW,"marginBottom":"8px","padding":"10px 14px 0"}),
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
    return dbc.Card([
        dbc.CardBody([
            html.H6("Currently In Production",
                    style={"color":"#8b90a0","textTransform":"uppercase",
                           "letterSpacing":"0.08em","fontSize":"0.75rem","marginBottom":"10px"}),
            body,
        ])
    ], style={"background": CARD_BG, "border": f"1px solid {YELLOW if not in_progress_orders.empty else '#2a2d3e'}",
              "borderRadius":"10px", "height":"100%"})

def up_next_card():
    if up_next_orders.empty:
        body = html.Div("No upcoming orders planned.", className="text-muted",
                        style={"padding":"14px","fontStyle":"italic"})
    else:
        r = up_next_orders.iloc[0]
        steps_away = r["BEGIN_ELAPSED"] - current_elapsed
        body = html.Div([
            html.Div(r["MATERIAL_DESCRIPTION"],
                     style={"fontSize":"1rem","fontWeight":"700","color":ACCENT,"marginBottom":"8px","padding":"10px 14px 0"}),
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
    return dbc.Card([
        dbc.CardBody([
            html.H6("Up Next",
                    style={"color":"#8b90a0","textTransform":"uppercase",
                           "letterSpacing":"0.08em","fontSize":"0.75rem","marginBottom":"10px"}),
            body,
        ])
    ], style={"background": CARD_BG, "border": f"1px solid {ACCENT}",
              "borderRadius":"10px", "height":"100%"})

@safe
def fig_production_over_time():
    prod2 = prod.copy()
    prod2["YIELD"] = pd.to_numeric(prod2["YIELD"], errors="coerce")
    df = prod2.groupby(["SIM_ELAPSED_STEPS","MATERIAL_DESCRIPTION"], as_index=False)["YIELD"].sum()
    fig = px.bar(df, x="SIM_ELAPSED_STEPS", y="YIELD", color="MATERIAL_DESCRIPTION",
                 color_discrete_sequence=COLORS, barmode="stack",
                 labels={"SIM_ELAPSED_STEPS":"Step","YIELD":"Units Produced"})
    return fig

# ── Layout ────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                title="ERPsim Dashboard · Team AA")

val_color = GREEN if latest_val["COMPANY_VALUATION"] >= 0 else RED

app.layout = dbc.Container(fluid=True, style={"backgroundColor": BG, "minHeight":"100vh", "padding":"20px"}, children=[

    dcc.Interval(id="refresh", interval=REFRESH_S * 1000, n_intervals=0),

    # ── Header ──────────────────────────────────────────────────────────────
    dbc.Row(className="mb-3", children=[
        dbc.Col([
            html.H2("ERPsim Dashboard", style={"color":"#fff","fontWeight":"700","marginBottom":"2px"}),
            html.Small(f"Team AA · SAP Client 435 · UWM", className="text-muted"),
        ], width=8),
        dbc.Col([
            html.Div([
                html.Span("● LIVE", style={"color":GREEN,"fontWeight":"600","marginRight":"8px"}),
                html.Span(f"Step {int(latest_val['SIM_ELAPSED_STEPS'])}  |  Round {latest_val.get('SIM_ROUND','01')}  |  {latest_val.get('SIM_DATE','')}",
                          style={"color":"#8b90a0"}),
            ], className="text-end mt-2"),
        ], width=4),
    ]),

    # ── KPI Cards ────────────────────────────────────────────────────────────
    dbc.Row(className="mb-3 g-3", children=[
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
    ]),

    # ── Tabs ─────────────────────────────────────────────────────────────────
    dbc.Tabs(style={"borderBottom":"1px solid #2a2d3e"}, children=[

        # ── Tab 1: Overview ─────────────────────────────────────────────────
        dbc.Tab(label="Overview", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Company Valuation & Profit Over Time", fig_valuation_over_time()), md=8),
                dbc.Col(chart_card("Cash vs Bank Loan", fig_cash_and_debt()), md=4),
            ]),
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Revenue & Margin Per Step", fig_sales_by_period()), md=8),
                dbc.Col(chart_card("Production Output Per Step", fig_production_over_time()), md=4),
            ]),
        ]),

        # ── Tab 2: Sales ─────────────────────────────────────────────────────
        dbc.Tab(label="Sales", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Revenue by Product", fig_revenue_by_product()), md=6),
                dbc.Col(chart_card("Sales Split by Region", fig_sales_by_region()), md=3),
                dbc.Col(chart_card("Wholesale vs Retail", fig_channel_split()), md=3),
            ]),
        ]),

        # ── Tab 3: Inventory ──────────────────────────────────────────────────
        dbc.Tab(label="Inventory", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Current Stock by Product & Region", fig_inventory_kpi()), md=7),
                dbc.Col(chart_card("Days of Stock Available (avg steps)", fig_days_available()), md=5),
            ]),
        ]),

        # ── Tab 4: Production ────────────────────────────────────────────────
        dbc.Tab(label="Production", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[

            # KPI row
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(kpi_card(
                    "Total To Be Produced",
                    f"{total_to_produce:,} units",
                    f"{'⚠ Below 120k threshold!' if total_to_produce <= PROD_WARN_THRESHOLD else f'{len(pending_orders)} orders queued'}",
                    RED if total_to_produce <= PROD_WARN_THRESHOLD else (YELLOW if total_to_produce <= PROD_WARN_THRESHOLD*2 else GREEN),
                ), md=3),
                dbc.Col(kpi_card(
                    "Currently In Production",
                    str(len(in_progress_orders)) + " order(s)",
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
            ]),

            # Current in production + Up Next cards
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(current_in_production_card(), md=6),
                dbc.Col(up_next_card(), md=6),
            ]),

            # Gantt chart
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Production Schedule (Gantt)", fig_prod_gantt()), md=12),
            ]),

            # Queue table
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Full Production Queue",
                                style={"color":"#8b90a0","textTransform":"uppercase",
                                       "letterSpacing":"0.08em","fontSize":"0.75rem","marginBottom":"12px"}),
                        prod_order_table(prod_orders),
                    ])
                ], style={"background": CARD_BG, "border":"1px solid #2a2d3e", "borderRadius":"10px"}), md=8),
                dbc.Col(chart_card("Setup Time Per Order (Steps Lost)", fig_setup_time()), md=4),
            ]),

            # Bottom charts
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Planned Qty Per Product (120k Warning)", fig_to_be_produced_by_product()), md=5),
                dbc.Col(chart_card("Actual Yield Per Step", fig_yield_over_time_detail()), md=4),
                dbc.Col(chart_card("Total Units Produced by Product", fig_actual_yield_by_product()), md=3),
            ]),
        ]),

        # ── Tab 6: Market ──────────────────────────────────────────────────
        dbc.Tab(label="Market", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col(chart_card("Market Share % by Product Over Periods", fig_market_share()), md=12),
            ]),
        ]),

        # ── Tab 5: Sustainability ─────────────────────────────────────────
        dbc.Tab(label="Sustainability", tab_style={"color":"#8b90a0"}, active_tab_style={"color":"#fff"}, children=[
            dbc.Row(className="mt-3 g-3", children=[
                dbc.Col([
                    dbc.Row(className="g-3 mb-3", children=[
                        dbc.Col(kpi_card("Total CO₂e Emissions",
                                         f"{total_co2:,.0f} kg",
                                         "All scopes combined", "#e67e22"), md=6),
                        dbc.Col(kpi_card("Emissions This Period",
                                         f"{carbon[carbon['SIM_ELAPSED_STEPS']==carbon['SIM_ELAPSED_STEPS'].max()]['CO2E_EMISSIONS'].sum():,.0f} kg" if (not carbon.empty and 'SIM_ELAPSED_STEPS' in carbon.columns) else '0 kg',
                                         "Latest step", "#e67e22"), md=6),
                    ]),
                    chart_card("CO₂e Over Time (step + cumulative)", fig_carbon_over_time()),
                ], md=6),
                dbc.Col(chart_card("Emissions by Type", fig_carbon_by_type()), md=6),
            ]),
        ]),

    ]),

    # Footer
    html.Hr(style={"borderColor":"#2a2d3e","marginTop":"30px"}),
    html.P(f"Auto-refreshes every {REFRESH_S}s  ·  Data: SAP NetWeaver OData · Client 435",
           className="text-center text-muted", style={"fontSize":"0.75rem"}),
])

# ── Callback: live refresh ────────────────────────────────────────────────────
# (Full page reload via browser refresh; for true live update extend callbacks)

if __name__ == "__main__":
    app.run(debug=False, port=8050)
