"""
Macro Indicators vs Stock Performance & Volatility
------------------------------------------------
An interactive Streamlit tool that lets you explore how macroeconomic
indicators (pulled live from FRED) relate to a stock's price, returns,
and volatility (pulled live from Yahoo Finance).

Run with:
    streamlit run app.py

You need a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro vs Stocks",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------------------------------
# Reference list of common FRED macro series
# --------------------------------------------------------------------------
MACRO_SERIES = {
    "CPI (Inflation, Index)": "CPIAUCSL",
    "Core CPI (ex food & energy)": "CPILFESL",
    "Fed Funds Rate": "FEDFUNDS",
    "10-Year Treasury Yield": "DGS10",
    "2-Year Treasury Yield": "DGS2",
    "Unemployment Rate": "UNRATE",
    "Real GDP": "GDPC1",
    "M2 Money Supply": "M2SL",
    "Consumer Sentiment (Michigan)": "UMCSENT",
    "Industrial Production Index": "INDPRO",
    "PCE Price Index": "PCEPI",
    "10Y-2Y Treasury Spread": "T10Y2Y",
    "VIX (Volatility Index)": "VIXCLS",
    "Retail Sales": "RSAFS",
    "Housing Starts": "HOUST",
}

FREQ_MAP = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}

# --------------------------------------------------------------------------
# Cached data fetchers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_macro(series_id: str, api_key: str, start: str, end: str) -> pd.Series:
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s.dropna()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stock(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def resample_series(s: pd.Series, freq: str, how: str = "last") -> pd.Series:
    if how == "last":
        return s.resample(freq).last().dropna()
    return s.resample(freq).mean().dropna()


def annualized_vol(returns: pd.Series, window: int, periods_per_year: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

api_key = st.sidebar.text_input(
    "FRED API Key",
    type="password",
    help="Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
)

ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()

macro_label = st.sidebar.selectbox("Macro Indicator", list(MACRO_SERIES.keys()), index=2)
macro_series_id = MACRO_SERIES[macro_label]

col_a, col_b = st.sidebar.columns(2)
start_date = col_a.date_input("Start", value=pd.Timestamp.today() - pd.DateOffset(years=10))
end_date = col_b.date_input("End", value=pd.Timestamp.today())

freq_label = st.sidebar.selectbox("Analysis Frequency", list(FREQ_MAP.keys()), index=2)
freq = FREQ_MAP[freq_label]
periods_per_year = {"D": 252, "W": 52, "ME": 12, "QE": 4}[freq]

vol_window = st.sidebar.slider("Rolling Volatility Window (periods)", 3, 36, 12)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Analysis", use_container_width=True, type="primary")

st.title("📈 Macro Indicators vs Stock Performance & Volatility")
st.caption(
    "Explore the relationship between a macroeconomic time series (FRED) and a stock's "
    "price, returns, and volatility (Yahoo Finance)."
)

if not api_key:
    st.info("👈 Enter your free FRED API key in the sidebar to get started.")
    st.stop()

if not run_btn:
    st.info("👈 Set your parameters and click **Run Analysis**.")
    st.stop()

# --------------------------------------------------------------------------
# Fetch data
# --------------------------------------------------------------------------
try:
    with st.spinner("Fetching macro data from FRED..."):
        macro_raw = fetch_macro(macro_series_id, api_key, str(start_date), str(end_date))
except Exception as e:
    st.error(f"Failed to fetch FRED data: {e}")
    st.stop()

try:
    with st.spinner(f"Fetching {ticker} price data from Yahoo Finance..."):
        stock_raw = fetch_stock(ticker, str(start_date), str(end_date))
except Exception as e:
    st.error(f"Failed to fetch stock data: {e}")
    st.stop()

if macro_raw.empty:
    st.error("No macro data returned. Check the series or date range.")
    st.stop()
if stock_raw.empty or "Close" not in stock_raw:
    st.error("No stock data returned. Check the ticker or date range.")
    st.stop()

# --------------------------------------------------------------------------
# Align & transform
# --------------------------------------------------------------------------
macro_rs = resample_series(macro_raw, freq, how="last")
close_rs = resample_series(stock_raw["Close"], freq, how="last")

df = pd.concat([close_rs.rename("price"), macro_rs.rename("macro")], axis=1).dropna()
df["stock_return"] = df["price"].pct_change()
df["macro_change"] = df["macro"].pct_change()
df["macro_diff"] = df["macro"].diff()
df["volatility"] = annualized_vol(df["stock_return"], vol_window, periods_per_year)
df = df.dropna(subset=["stock_return"])

if len(df) < 5:
    st.warning("Not enough overlapping data points after alignment. Try a wider date range or lower frequency.")
    st.stop()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Time Series", "🔗 Correlation", "📉 Volatility", "🧮 Regression"]
)

# ---- Tab 1: dual-axis time series ----
with tab1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df.index, y=df["price"], name=f"{ticker} Price", line=dict(color="#2563eb")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["macro"], name=macro_label, line=dict(color="#f97316")),
        secondary_y=True,
    )
    fig.update_layout(
        title=f"{ticker} Price vs {macro_label}",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
        height=500,
    )
    fig.update_yaxes(title_text=f"{ticker} Price ($)", secondary_y=False)
    fig.update_yaxes(title_text=macro_label, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{ticker} Total Return", f"{(df['price'].iloc[-1] / df['price'].iloc[0] - 1) * 100:.1f}%")
    c2.metric("Avg Period Return", f"{df['stock_return'].mean() * 100:.2f}%")
    c3.metric(f"{macro_label} — Latest", f"{df['macro'].iloc[-1]:.2f}")
    c4.metric(f"{macro_label} — Change (period)", f"{df['macro'].iloc[-1] - df['macro'].iloc[0]:.2f}")

# ---- Tab 2: correlation ----
with tab2:
    corr_level = df["price"].corr(df["macro"])
    corr_change = df["stock_return"].corr(df["macro_change"])
    corr_diff = df["stock_return"].corr(df["macro_diff"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlation (levels)", f"{corr_level:.3f}")
    c2.metric("Correlation (% changes)", f"{corr_change:.3f}")
    c3.metric("Correlation (return vs Δmacro)", f"{corr_diff:.3f}")

    scatter_fig = px.scatter(
        df, x="macro_change", y="stock_return", trendline="ols",
        labels={"macro_change": f"{macro_label} % change", "stock_return": f"{ticker} return"},
        title=f"{ticker} Return vs {macro_label} % Change",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    roll_window = min(max(6, vol_window), max(6, len(df) // 2))
    rolling_corr = df["price"].rolling(roll_window).corr(df["macro"])
    rc_fig = go.Figure()
    rc_fig.add_trace(go.Scatter(x=df.index, y=rolling_corr, name="Rolling Correlation", line=dict(color="#059669")))
    rc_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    rc_fig.update_layout(title=f"Rolling {roll_window}-Period Correlation (Price Level)", height=350)
    st.plotly_chart(rc_fig, use_container_width=True)

# ---- Tab 3: volatility ----
with tab3:
    fig_v = make_subplots(specs=[[{"secondary_y": True}]])
    fig_v.add_trace(
        go.Scatter(x=df.index, y=df["volatility"], name="Annualized Volatility", line=dict(color="#dc2626")),
        secondary_y=False,
    )
    fig_v.add_trace(
        go.Scatter(x=df.index, y=df["macro"], name=macro_label, line=dict(color="#f97316")),
        secondary_y=True,
    )
    fig_v.update_layout(
        title=f"{ticker} Rolling Volatility vs {macro_label}",
        hovermode="x unified",
        height=500,
    )
    fig_v.update_yaxes(title_text="Annualized Volatility", secondary_y=False)
    fig_v.update_yaxes(title_text=macro_label, secondary_y=True)
    st.plotly_chart(fig_v, use_container_width=True)

    vol_corr = df["volatility"].corr(df["macro"])
    st.metric("Correlation: Volatility vs Macro Level", f"{vol_corr:.3f}")

    st.caption(
        "Volatility is the rolling annualized standard deviation of period returns "
        f"using a {vol_window}-period window."
    )

# ---- Tab 4: regression ----
with tab4:
    st.subheader("Simple Linear Regression: Stock Return ~ Macro Change")
    reg_df = df[["stock_return", "macro_change"]].dropna()
    if len(reg_df) < 5:
        st.warning("Not enough data points for regression.")
    else:
        X = sm.add_constant(reg_df["macro_change"])
        y = reg_df["stock_return"]
        model = sm.OLS(y, X).fit()

        c1, c2, c3 = st.columns(3)
        c1.metric("Beta (sensitivity)", f"{model.params['macro_change']:.3f}")
        c2.metric("R²", f"{model.rsquared:.3f}")
        c3.metric("P-value (macro_change)", f"{model.pvalues['macro_change']:.4f}")

        with st.expander("Full regression summary"):
            st.text(model.summary())

    st.markdown("---")
    st.subheader("Raw Data")
    st.dataframe(df.round(4), use_container_width=True)
    st.download_button(
        "⬇️ Download data as CSV",
        df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_{macro_series_id}_{freq}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "Data: FRED (Federal Reserve Economic Data) and Yahoo Finance. "
    "For research/educational purposes only — not investment advice."
)
