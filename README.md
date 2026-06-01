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

## Data Sources

- **S&P 500 / Nasdaq-100 Constituents**: Wikipedia first, committed static CSVs when live Wikipedia is unavailable, and small emergency lists only if both fail
- **Primary Market Data**: yfinance / Yahoo Finance adjusted close
- **Fallback Market Data**: Stooq daily close through `stooq-market-data-fallback`, used only for symbols still missing after Yahoo retries

The dashboard keeps Yahoo adjusted close as the preferred research basis. If Stooq recovery is used, those symbols are labeled separately because Stooq daily close is not Yahoo adjusted close.

## Data Diagnostics

The sidebar reports raw universe size, final unique universe size, Wikipedia/static/emergency constituent sources, Yahoo loaded and failed counts, Stooq recovered counts, final loaded count, and unavailable tickers. Detailed failed ticker samples are kept inside an expander.


