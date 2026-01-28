# Beta & Correlation Analytics - Streamlit App

A comprehensive asset analytics dashboard for analyzing beta, correlation, Sharpe ratios, and relative performance across a large asset universe.

## Features

- **Large Asset Universe**: S&P 500 + Nasdaq-100 constituents + ETFs + ADRs (~600+ tickers)
- **Multi-Period Analysis**: 1W, 2W, 1M, 2M, 1Q, 6M, 1Y, 2Y lookback periods
- **Five Analysis Tabs**:
  1. **Overview**: Top/Bottom Sharpe, Diversifiers, Multi-period correlations
  2. **Correlation Matrix**: Interactive heatmap with highest/lowest correlation tables
  3. **Beta Analysis**: Beta vs Correlation scatter + multi-period comparison
  4. **Sharpe Ratios**: Bar chart + Rolling Sharpe with ticker selector
  5. **Relative Performance**: Normalized price chart + Top/Bottom performers

- **Configurable Parameters**:
  - Risk-free rate (annual %)
  - Return type (simple/log)
  - Lookback period
  - Benchmark (SPY, QQQ, IWM, etc.)

- **Download Buttons**: Export all tables to CSV

## Formulas

All metrics use the same formulas as the original Dash app:

- **Daily RF (simple)**: `(1 + RF_annual)^(1/252) - 1`
- **Daily RF (log)**: `ln(1 + daily_simple_rf)`
- **Sharpe**: `mean(R - RF) / std(R - RF) * sqrt(252)`
- **Beta**: `Cov(R_asset, R_bench) / Var(R_bench)` [pairwise dropna, ddof=1]
- **Min observations**: `max(5, ceil(0.8 * period_days))`


## Data Sources

- **S&P 500 / Nasdaq-100 Constituents**: Wikipedia (with fallback to hardcoded lists)
- **Market Data**: yfinance (Yahoo Finance)

## Notes

- Initial data load may take 1-2 minutes for tickers (chunked downloads)
- Data is cached for 1 hour using `@st.cache_data`
