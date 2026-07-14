# Macro vs Stocks — Interactive Streamlit Tool

Explore how macroeconomic indicators (CPI, Fed Funds Rate, Treasury yields,
unemployment, GDP, etc. — live from FRED) relate to a stock's price, returns,
and rolling volatility (live from Yahoo Finance).

## 1. Get a free FRED API key
Sign up here (instant, free): https://fred.stlouisfed.org/docs/api/api_key.html

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Run the app
```bash
streamlit run app.py
```
This opens the tool in your browser at `http://localhost:8501`.

## What it does
- **Time Series tab** — dual-axis chart of stock price vs the chosen macro series
- **Correlation tab** — level correlation, % change correlation, scatter + OLS trendline, rolling correlation over time
- **Volatility tab** — rolling annualized volatility vs the macro series
- **Regression tab** — simple OLS of stock return on macro % change (beta, R², p-value), plus raw data table + CSV export

## Included macro indicators (FRED series)
CPI, Core CPI, Fed Funds Rate, 10Y & 2Y Treasury yields, 10Y-2Y spread,
Unemployment Rate, Real GDP, M2 Money Supply, Consumer Sentiment,
Industrial Production, PCE Price Index, VIX, Retail Sales, Housing Starts.

You can add more by editing the `MACRO_SERIES` dict at the top of `app.py`
with any FRED series ID (browse series at https://fred.stlouisfed.org).

## Next steps / ideas to extend
- Add a **Risk Modeling** tab: VaR (historical/parametric), max drawdown, Sharpe/Sortino ratios
- Multi-stock comparison (portfolio-level macro sensitivity)
- Multiple macro indicators combined in a multi-factor regression
- Deploy for free on [Streamlit Community Cloud](https://streamlit.io/cloud) to get a shareable URL
