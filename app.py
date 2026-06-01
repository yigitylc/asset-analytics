"""
BETA & CORRELATION ANALYTICS - Streamlit App

Features:
- Beta, correlation, Sharpe, relative performance across a large asset universe
- S&P 500 + Nasdaq-100 constituents + ETFs + ADRs
- Multi-period analysis with configurable lookback
- Robust yfinance handling with chunked downloads

Formulas:
- Daily RF (simple): (1 + RF_annual)^(1/252) - 1
- Daily RF (log): ln(1 + daily_simple_rf)
- Sharpe: mean(R - RF) / std(R - RF) * sqrt(252)
- Beta: Cov(R_asset, R_bench) / Var(R_bench)  [pairwise dropna, ddof=1]
- Min obs: max(5, ceil(0.8 * period_days))
"""


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import requests
import re
import time
from dataclasses import dataclass, field
from io import StringIO
from math import ceil
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set

warnings.filterwarnings('ignore')

try:
    from stooq_market_data_fallback import get_price_history
except ImportError:
    get_price_history = None

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Beta & Correlation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration."""
    YEARS_OF_DATA = 5
    CHUNK_SIZE = 50
    MAX_FFILL_DAYS = 2

    # Caching - disabled for Streamlit Cloud (no persistent filesystem)
    ENABLE_CACHE = False
    CACHE_DIR = Path("./cache")
    CACHE_FILE = "price_data.parquet"

    # Universe expansion
    INCLUDE_SP500 = True
    INCLUDE_NASDAQ100 = True

    # Download settings
    DATA_INTERVAL = "1d"
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0
    USE_THREADING = False  # Disabled for large universes
    LARGE_UNIVERSE_THRESHOLD = 500

    # Min obs rule
    MIN_OBS_FLOOR = 5
    MIN_OBS_RATIO = 0.8

    # Periods
    PERIODS = [5, 10, 21, 42, 63, 126, 252, 504]
    PERIOD_LABELS = {
        5: '1W', 10: '2W', 21: '1M', 42: '2M',
        63: '1Q', 126: '6M', 252: '1Y', 504: '2Y'
    }

    EXCLUDED_TICKERS = {'Q'}


def get_min_obs(period_days: int) -> int:
    """Calculate minimum observations for a given period."""
    return max(Config.MIN_OBS_FLOOR, ceil(Config.MIN_OBS_RATIO * period_days))


# =============================================================================
# TICKER UNIVERSE
# =============================================================================

BASE_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLY", "XLP", "XLF", "XLV", "XLI", "XLB", "XLK", "XBI", "SMH", "IGV","XLC", "XLU",
    "XME", "GDX", "URA", "XOP", "XHB", "XLRE", "XRT",
    "IBIT", "BIL",
    "EFA", "VGK", "VEA", "EEM", "EUFN",
    "EWA", "EWC", "EWQ", "EWG", "EWI", "EWJ", "EWP", "EWU", "EWL",
    "FXI", "KWEB", "CQQQ","EWZ", "EWW", "EWS", "EWY", "EWT", "INDA", "EWH", "EZA",
    "SHY", "IEI", "IEF", "TLT", "TIP", "STIP", "IGSB", "HYG", "EMB", "BNDX", "BWX", "HYEM", "IAGG",
    "DBC", "GSG", "USO", "UNG", "GLD", "SLV", "DBA",
    "VNQ", "REM",
    "UUP", "VIXY",
    "TUR",
    "AAPL", "BABA", "BIDU", "BILI", "UNH", "PANW", "PLTR", "SNOW", "TSLA", "NVDA", "MU", "JD"
]

ADR_TICKERS = [
    "TSM", "AZN", "ASML", "SMFG", "HSBC", "TM", "NVS", "NVO", "SAP", "SHEL",
    "MUFG", "SAN", "BHP", "HDB", "TTE", "RIO", "BBVA", "UL", "BUD", "SONY",
    "BTI", "ARM", "SNY", "MFG", "GSK","GDS","VNET","KC","XPEV","XNET","TME","ZHU", "TCOM","ATGL","BP", "BCS", "PBR", "PBR-A", "ING",
    "LYG", "NTES", "PKX", "SE", "INFY", "VALE", "RELX", "EQNR", "ARGX", "BIDU",
    "IBN", "TAK", "GFI", "BABA", "DEO", "HLN", "ABEV", "VIPS", "BBDO", "BBD",
    "ASX", "MT", "TCOM", "PUK", "HMC", "NOK", "CUK", "PDD", "TEVA", "ERIC",
    "KB", "RYAAY", "VOD", "IX", "CHT", "WDS", "BSBR", "UMC", "E",
]

DOT_TO_DASH = {
    "BRK.B": "BRK-B", "BRK.A": "BRK-A",
    "BF.B": "BF-B", "BF.A": "BF-A",
    "PBR.A": "PBR-A", "EBR.B": "EBR-B",
}

STATIC_UNIVERSE_DIR = Path(__file__).resolve().parent / "data" / "universe"
SP500_STATIC_CSV = STATIC_UNIVERSE_DIR / "sp500_constituents.csv"
NASDAQ100_STATIC_CSV = STATIC_UNIVERSE_DIR / "nasdaq100_constituents.csv"

SP500_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY", "PEP",
    "KO", "AVGO", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT", "DHR", "BAC",
    "PFE", "CRM", "ADBE", "CMCSA", "NKE", "DIS", "VZ", "NFLX", "INTC", "WFC", "TXN",
    "PM", "NEE", "RTX", "BMY", "UPS", "QCOM", "HON", "ORCL", "IBM", "AMGN", "UNP",
    "LOW", "SPGI", "MS", "GS", "ELV", "CAT", "DE", "BLK", "INTU", "AXP", "AMD",
]

NASDAQ100_FALLBACK = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
    "ASML", "PEP", "AZN", "CSCO", "ADBE", "NFLX", "AMD", "CMCSA", "TXN", "INTC",
    "QCOM", "TMUS", "AMGN", "INTU", "HON", "ISRG", "AMAT", "BKNG", "SBUX", "ADI",
]


def sanitize_ticker(t: str) -> str:
    """Convert ticker to Yahoo Finance format."""
    t = (t or "").strip().upper()
    if t in DOT_TO_DASH:
        return DOT_TO_DASH[t]
    return t.replace(".", "-")


def unique_order(seq: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        x_upper = x.upper()
        if x_upper not in seen:
            seen.add(x_upper)
            out.append(x_upper)
    return out


def _read_html_with_headers(url: str) -> List[pd.DataFrame]:
    """Read HTML tables with proper User-Agent."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def _load_static_universe_csv(path: Path, min_count: int) -> List[str]:
    """Load a committed static constituent list."""
    df = pd.read_csv(path)
    if 'symbol' not in df.columns:
        raise ValueError(f"{path} is missing a symbol column.")

    tickers = [sanitize_ticker(t) for t in df['symbol'].dropna().astype(str).tolist()]
    tickers = [ticker for ticker in tickers if ticker]
    tickers = unique_order(tickers)
    if len(tickers) < min_count:
        raise ValueError(f"{path} has only {len(tickers)} symbols; expected at least {min_count}.")
    return tickers


def fetch_sp500_constituents() -> Tuple[List[str], Dict[str, Any]]:
    """Fetch S&P 500 constituents with source metadata."""
    live_error = ""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = _read_html_with_headers(url)
        for df in tables:
            for col in ['Symbol', 'Ticker symbol', 'Ticker']:
                if col in df.columns:
                    tickers = df[col].dropna().astype(str).tolist()
                    result = [sanitize_ticker(t) for t in tickers]
                    if len(result) > 400:
                        return result, {'source': 'wikipedia', 'error': ''}
    except Exception as exc:
        live_error = str(exc)
    if not live_error:
        live_error = 'Wikipedia table did not contain enough S&P 500 symbols.'

    try:
        return _load_static_universe_csv(SP500_STATIC_CSV, 400), {
            'source': 'static_csv',
            'error': live_error,
        }
    except Exception as static_exc:
        return list(SP500_FALLBACK), {
            'source': 'emergency_fallback',
            'error': f"{live_error}; static CSV failed: {static_exc}",
        }


def fetch_nasdaq100_constituents() -> Tuple[List[str], Dict[str, Any]]:
    """Fetch Nasdaq-100 constituents with source metadata."""
    live_error = ""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = _read_html_with_headers(url)
        for df in tables:
            for col in ['Ticker', 'Ticker symbol', 'Symbol']:
                if col in df.columns:
                    tickers = df[col].dropna().astype(str).tolist()
                    result = [sanitize_ticker(t) for t in tickers]
                    if len(result) > 90:
                        return result, {'source': 'wikipedia', 'error': ''}
    except Exception as exc:
        live_error = str(exc)
    if not live_error:
        live_error = 'Wikipedia table did not contain enough Nasdaq-100 symbols.'

    try:
        return _load_static_universe_csv(NASDAQ100_STATIC_CSV, 90), {
            'source': 'static_csv',
            'error': live_error,
        }
    except Exception as static_exc:
        return list(NASDAQ100_FALLBACK), {
            'source': 'emergency_fallback',
            'error': f"{live_error}; static CSV failed: {static_exc}",
        }


def build_universe_uncached() -> Tuple[List[str], Dict[str, Any]]:
    """Build complete ticker universe."""
    breakdown = {
        'base_count': len(BASE_TICKERS),
        'adr_count': len(ADR_TICKERS),
        'sp500_count': 0,
        'sp500_source': 'disabled',
        'sp500_error': '',
        'nasdaq100_count': 0,
        'nasdaq100_source': 'disabled',
        'nasdaq100_error': '',
        'total_raw': 0,
        'duplicates_removed': 0,
        'final_unique': 0
    }

    universe = list(BASE_TICKERS)
    universe.extend(ADR_TICKERS)

    if Config.INCLUDE_SP500:
        sp500, sp500_meta = fetch_sp500_constituents()
        universe.extend(sp500)
        breakdown['sp500_count'] = len(sp500)
        breakdown['sp500_source'] = sp500_meta.get('source', 'unknown')
        breakdown['sp500_error'] = sp500_meta.get('error', '')

    if Config.INCLUDE_NASDAQ100:
        ndx, ndx_meta = fetch_nasdaq100_constituents()
        universe.extend(ndx)
        breakdown['nasdaq100_count'] = len(ndx)
        breakdown['nasdaq100_source'] = ndx_meta.get('source', 'unknown')
        breakdown['nasdaq100_error'] = ndx_meta.get('error', '')

    breakdown['total_raw'] = len(universe)

    unique = unique_order(universe)
    excluded_count = len([t for t in unique if t in Config.EXCLUDED_TICKERS])
    unique = [t for t in unique if t not in Config.EXCLUDED_TICKERS]

    breakdown['duplicates_removed'] = breakdown['total_raw'] - len(unique) - excluded_count
    breakdown['excluded_count'] = excluded_count
    breakdown['final_unique'] = len(unique)

    return unique, breakdown


@st.cache_data(ttl=3600, show_spinner=False)
def build_universe() -> Tuple[List[str], Dict[str, Any]]:
    """Build complete ticker universe with Streamlit caching."""
    return build_universe_uncached()



# DATA FETCHING


@dataclass
class YahooDownloadResult:
    """Structured result for Yahoo's primary adjusted-close fetch."""
    data: pd.DataFrame
    requested_tickers: List[str] = field(default_factory=list)
    loaded_tickers: List[str] = field(default_factory=list)
    missing_tickers: List[str] = field(default_factory=list)
    all_nan_tickers: List[str] = field(default_factory=list)
    retried_tickers: List[str] = field(default_factory=list)
    recovered_tickers: List[str] = field(default_factory=list)
    failed_tickers: List[str] = field(default_factory=list)
    error_by_ticker: Dict[str, str] = field(default_factory=dict)


@dataclass
class StooqRecoveryResult:
    """Structured result for daily Stooq close recovery."""
    data: pd.DataFrame
    metadata_by_ticker: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recovered_tickers: List[str] = field(default_factory=list)
    failed_tickers: List[str] = field(default_factory=list)
    error_by_ticker: Dict[str, str] = field(default_factory=dict)


def _ticker_key(ticker: str) -> str:
    """Normalize symbols for comparing requested and returned columns."""
    return sanitize_ticker(str(ticker)).upper()


def _ordered_unique(seq: List[str]) -> List[str]:
    """Deduplicate symbols while preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for item in seq:
        key = _ticker_key(item)
        if key not in seen:
            seen.add(key)
            out.append(str(item).upper())
    return out


def format_price_basis_label(price_basis: str) -> str:
    """Human-readable label for mixed data-source diagnostics."""
    labels = {
        'yfinance_adjusted_close': 'Yahoo adjusted close',
        'stooq_close': 'Stooq daily close (not Yahoo adjusted close)',
    }
    return labels.get(price_basis, price_basis)


def _retry_delay(attempt: int) -> None:
    if attempt > 0:
        delay = Config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
        time.sleep(delay)


def _extract_close_frame(data: pd.DataFrame, requested: List[str]) -> pd.DataFrame:
    """Extract Yahoo close columns from either single or multi-ticker frames."""
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' not in data.columns.get_level_values(0):
            return pd.DataFrame()
        close = data['Close']
    else:
        if 'Close' not in data.columns:
            return pd.DataFrame()
        close = data[['Close']]
        if len(requested) == 1:
            close.columns = [requested[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=requested[0] if requested else close.name)

    close = close.copy()
    close.columns = [sanitize_ticker(str(col)).upper() for col in close.columns]
    return close


def _align_close_frame(close: pd.DataFrame, requested: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return close columns in requested ticker order and list missing symbols."""
    if close.empty:
        return pd.DataFrame(index=close.index), list(requested)

    col_by_key = {_ticker_key(col): col for col in close.columns}
    aligned: Dict[str, pd.Series] = {}
    missing: List[str] = []

    for ticker in requested:
        key = _ticker_key(ticker)
        if key in col_by_key:
            aligned[ticker] = pd.to_numeric(close[col_by_key[key]], errors='coerce')
        else:
            missing.append(ticker)

    if not aligned:
        return pd.DataFrame(index=close.index), missing
    return pd.DataFrame(aligned), missing


def _download_yahoo_request(
    tickers: List[str],
    start: dt.datetime,
    end: dt.datetime,
) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    """Download one Yahoo request with retries and explicit missing detection."""
    last_error = ""

    for attempt in range(Config.MAX_RETRIES):
        try:
            _retry_delay(attempt)
            data = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=Config.USE_THREADING
            )
        except Exception as exc:
            last_error = str(exc)
            continue

        close = _extract_close_frame(data, tickers)
        aligned, missing = _align_close_frame(close, tickers)
        all_nan = [col for col in aligned.columns if aligned[col].isna().all()]
        valid = aligned.drop(columns=all_nan, errors='ignore')
        still_missing = _ordered_unique(missing + all_nan)

        if valid.empty and attempt < Config.MAX_RETRIES - 1:
            last_error = "Yahoo returned an empty or all-NaN response."
            continue

        if valid.empty:
            return pd.DataFrame(), list(tickers), all_nan, last_error or "Yahoo returned no usable close data."

        return valid, still_missing, all_nan, ""

    return pd.DataFrame(), list(tickers), [], last_error or "Yahoo download failed."


def download_chunked(
    tickers: List[str],
    start: dt.datetime,
    end: dt.datetime,
    chunk_size: int = 50,
    progress_bar=None
) -> YahooDownloadResult:
    """Download Yahoo adjusted close in chunks, then retry missing symbols singly."""
    all_data: List[pd.DataFrame] = []
    initial_missing: List[str] = []
    all_nan_tickers: List[str] = []
    error_by_ticker: Dict[str, str] = {}
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        chunk_num = i // chunk_size + 1

        if progress_bar:
            progress_bar.progress(chunk_num / total_chunks, f"Downloading chunk {chunk_num}/{total_chunks}...")

        close, missing, all_nan, error = _download_yahoo_request(chunk, start, end)
        if not close.empty:
            all_data.append(close)
        initial_missing.extend(missing)
        all_nan_tickers.extend(all_nan)
        if error:
            for ticker in missing:
                error_by_ticker[ticker] = error

    retry_tickers = _ordered_unique(initial_missing)
    retry_data: List[pd.DataFrame] = []
    recovered_tickers: List[str] = []
    final_failed: List[str] = []

    for ticker in retry_tickers:
        close, missing, all_nan, error = _download_yahoo_request([ticker], start, end)
        all_nan_tickers.extend(all_nan)
        if not close.empty and ticker in close.columns:
            retry_data.append(close[[ticker]])
            recovered_tickers.append(ticker)
            error_by_ticker.pop(ticker, None)
        else:
            final_failed.append(ticker)
            error_by_ticker[ticker] = error or "Yahoo returned no usable close data."

    frames = all_data + retry_data
    combined = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined = combined.sort_index()
    if hasattr(combined.index, 'tz') and combined.index.tz is not None:
        combined.index = combined.index.tz_localize(None)

    loaded_tickers = list(combined.columns)
    return YahooDownloadResult(
        data=combined,
        requested_tickers=list(tickers),
        loaded_tickers=loaded_tickers,
        missing_tickers=retry_tickers,
        all_nan_tickers=_ordered_unique(all_nan_tickers),
        retried_tickers=retry_tickers,
        recovered_tickers=recovered_tickers,
        failed_tickers=_ordered_unique(final_failed),
        error_by_ticker=error_by_ticker,
    )


def recover_missing_with_stooq(
    missing_tickers: List[str],
    start: dt.datetime,
    end: dt.datetime,
    interval: str = Config.DATA_INTERVAL,
) -> StooqRecoveryResult:
    """Recover residual missing symbols with Stooq daily close only."""
    if not missing_tickers:
        return StooqRecoveryResult(data=pd.DataFrame())

    if interval != "1d":
        return StooqRecoveryResult(
            data=pd.DataFrame(),
            failed_tickers=list(missing_tickers),
            error_by_ticker={ticker: "Stooq fallback supports daily data only." for ticker in missing_tickers},
        )

    if get_price_history is None:
        return StooqRecoveryResult(
            data=pd.DataFrame(),
            failed_tickers=list(missing_tickers),
            error_by_ticker={ticker: "stooq-market-data-fallback is not installed." for ticker in missing_tickers},
        )

    recovered_frames: List[pd.DataFrame] = []
    metadata_by_ticker: Dict[str, Dict[str, Any]] = {}
    recovered_tickers: List[str] = []
    failed_tickers: List[str] = []
    error_by_ticker: Dict[str, str] = {}
    start_s = start.strftime('%Y-%m-%d')
    end_s = end.strftime('%Y-%m-%d')

    for ticker in missing_tickers:
        try:
            history = get_price_history(
                ticker,
                start=start_s,
                end=end_s,
                interval="1d",
                preferred_provider="stooq",
                fallback=False,
            )
            data = history.data
            close_col = 'close' if 'close' in data.columns else 'Close' if 'Close' in data.columns else None
            if close_col is None:
                raise ValueError("Stooq response did not include a close column.")

            series = pd.to_numeric(data[close_col], errors='coerce').rename(ticker)
            series = series.dropna()
            if series.empty:
                raise ValueError("Stooq returned no usable close data.")
            if hasattr(series.index, 'tz') and series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            recovered_frames.append(series.to_frame())
            recovered_tickers.append(ticker)
            metadata_by_ticker[ticker] = {
                'provider': getattr(history, 'provider', 'stooq'),
                'provider_symbol': getattr(history, 'provider_symbol', ticker),
                'price_basis': getattr(history, 'price_basis', 'stooq_close'),
                'data_source': 'Stooq daily close',
            }
        except Exception as exc:
            failed_tickers.append(ticker)
            error_by_ticker[ticker] = str(exc)

    recovered = pd.concat(recovered_frames, axis=1).sort_index() if recovered_frames else pd.DataFrame()
    return StooqRecoveryResult(
        data=recovered,
        metadata_by_ticker=metadata_by_ticker,
        recovered_tickers=recovered_tickers,
        failed_tickers=failed_tickers,
        error_by_ticker=error_by_ticker,
    )


def fetch_price_data_uncached(
    tickers: List[str],
    interval: str = Config.DATA_INTERVAL,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fetch price data and return two frames.
    cl_price_raw (no fill) for stats, cl_price_plot (ffill) for charts.
    """
    end = dt.datetime.today()
    start = end - dt.timedelta(days=365 * Config.YEARS_OF_DATA)

    yahoo_result = download_chunked(tickers, start, end, Config.CHUNK_SIZE)
    raw_data = yahoo_result.data
    stooq_result = recover_missing_with_stooq(yahoo_result.failed_tickers, start, end, interval)

    if not stooq_result.data.empty:
        raw_data = pd.concat([raw_data, stooq_result.data], axis=1) if not raw_data.empty else stooq_result.data.copy()
        raw_data = raw_data.loc[:, ~raw_data.columns.duplicated()]

    all_nan_after_recovery = list(raw_data.columns[raw_data.isna().all()]) if not raw_data.empty else []
    raw_data = raw_data.dropna(how='all', axis=1)
    raw_data = raw_data.dropna(how='all', axis=0)

    final_columns = list(raw_data.columns)
    provider_by_ticker = {ticker: 'yfinance' for ticker in yahoo_result.loaded_tickers if ticker in final_columns}
    provider_symbol_by_ticker = {ticker: ticker for ticker in yahoo_result.loaded_tickers if ticker in final_columns}
    price_basis_by_ticker = {ticker: 'yfinance_adjusted_close' for ticker in yahoo_result.loaded_tickers if ticker in final_columns}
    data_source_by_ticker = {ticker: 'Yahoo adjusted close' for ticker in yahoo_result.loaded_tickers if ticker in final_columns}

    for ticker, meta in stooq_result.metadata_by_ticker.items():
        if ticker in final_columns:
            provider_by_ticker[ticker] = meta.get('provider', 'stooq')
            provider_symbol_by_ticker[ticker] = meta.get('provider_symbol', ticker)
            price_basis_by_ticker[ticker] = meta.get('price_basis', 'stooq_close')
            data_source_by_ticker[ticker] = meta.get('data_source', 'Stooq daily close')

    final_failed = [
        ticker for ticker in _ordered_unique(stooq_result.failed_tickers + all_nan_after_recovery)
        if ticker not in final_columns
    ]

    source_breakdown = {
        'Yahoo adjusted close': sum(1 for source in data_source_by_ticker.values() if source == 'Yahoo adjusted close'),
        'Stooq daily close': sum(1 for source in data_source_by_ticker.values() if source == 'Stooq daily close'),
    }

    if raw_data.empty:
        fetch_stats = {
            'pre_fetch_universe_count': len(tickers),
            'yahoo_loaded_count': 0,
            'yahoo_missing_count': len(yahoo_result.missing_tickers),
            'yahoo_retried_count': len(yahoo_result.retried_tickers),
            'yahoo_recovered_count': len(yahoo_result.recovered_tickers),
            'yahoo_failed_count': len(yahoo_result.failed_tickers),
            'yahoo_all_nan_count': len(yahoo_result.all_nan_tickers),
            'stooq_recovered_count': 0,
            'stooq_failed_count': len(stooq_result.failed_tickers),
            'failed_unavailable_count': len(final_failed),
            'final_ticker_count': 0,
            'final_failed_tickers': final_failed,
            'trading_days': 0,
            'data_completeness_pct': 0,
            'date_start': 'N/A',
            'date_end': 'N/A',
        }
        return pd.DataFrame(), pd.DataFrame(), fetch_stats

    cl_price_raw = raw_data.copy()
    cl_price_plot = raw_data.ffill(limit=Config.MAX_FFILL_DAYS)

    total_cells = cl_price_raw.shape[0] * cl_price_raw.shape[1]
    nan_cells = cl_price_raw.isna().sum().sum()
    completeness = (1 - nan_cells / total_cells) * 100 if total_cells > 0 else 0

    fetch_stats = {
        'pre_fetch_universe_count': len(tickers),
        'yahoo_loaded_count': len([t for t in yahoo_result.loaded_tickers if t in final_columns]),
        'yahoo_missing_count': len(yahoo_result.missing_tickers),
        'yahoo_retried_count': len(yahoo_result.retried_tickers),
        'yahoo_recovered_count': len(yahoo_result.recovered_tickers),
        'yahoo_failed_count': len(yahoo_result.failed_tickers),
        'yahoo_all_nan_count': len(yahoo_result.all_nan_tickers),
        'yahoo_loaded_tickers': [t for t in yahoo_result.loaded_tickers if t in final_columns],
        'yahoo_missing_tickers': yahoo_result.missing_tickers,
        'yahoo_retried_tickers': yahoo_result.retried_tickers,
        'yahoo_recovered_tickers': yahoo_result.recovered_tickers,
        'yahoo_failed_tickers': yahoo_result.failed_tickers,
        'yahoo_all_nan_tickers': yahoo_result.all_nan_tickers,
        'stooq_recovered_count': len(stooq_result.recovered_tickers),
        'stooq_failed_count': len(stooq_result.failed_tickers),
        'stooq_recovered_tickers': stooq_result.recovered_tickers,
        'stooq_failed_tickers': stooq_result.failed_tickers,
        'failed_unavailable_count': len(final_failed),
        'final_failed_tickers': final_failed,
        'final_ticker_count': len(cl_price_raw.columns),
        'trading_days': len(cl_price_raw),
        'data_completeness_pct': completeness,
        'date_start': cl_price_raw.index[0].strftime('%Y-%m-%d') if len(cl_price_raw) > 0 else 'N/A',
        'date_end': cl_price_raw.index[-1].strftime('%Y-%m-%d') if len(cl_price_raw) > 0 else 'N/A',
        'provider_by_ticker': provider_by_ticker,
        'provider_symbol_by_ticker': provider_symbol_by_ticker,
        'price_basis_by_ticker': price_basis_by_ticker,
        'data_source_by_ticker': data_source_by_ticker,
        'data_source_breakdown': source_breakdown,
        'yahoo_error_by_ticker': yahoo_result.error_by_ticker,
        'stooq_error_by_ticker': stooq_result.error_by_ticker,
    }

    return cl_price_raw, cl_price_plot, fetch_stats


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Fetch price data with Streamlit caching."""
    return fetch_price_data_uncached(tickers, Config.DATA_INTERVAL)


@st.cache_data(ttl=3600, show_spinner=False)
def calculate_returns(prices_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate returns from RAW prices only."""
    simple_returns = prices_raw.pct_change()
    log_returns = np.log(prices_raw / prices_raw.shift(1))
    return simple_returns, log_returns



# METRICS FUNCTIONS


def get_daily_rf(annual_rf: float, return_type: str) -> float:
    """Convert annual RF to daily."""
    daily_simple_rf = (1 + annual_rf) ** (1 / 252) - 1
    if return_type == 'log':
        return np.log(1 + daily_simple_rf)
    return daily_simple_rf


def is_numeric(val: Any) -> bool:
    """Robust numeric check."""
    return isinstance(val, (int, float, np.integer, np.floating))


def calculate_sharpe_ratio(
    returns: pd.DataFrame,
    annual_rf: float,
    period_days: int,
    return_type: str,
    annualize: bool = True
) -> pd.Series:
    """Calculate Sharpe Ratio."""
    daily_rf = get_daily_rf(annual_rf, return_type)
    min_obs = get_min_obs(period_days)

    def calc_single_sharpe(r: pd.Series) -> float:
        r_clean = r.tail(period_days).dropna()
        if len(r_clean) < min_obs:
            return np.nan

        excess = r_clean - daily_rf
        mean_excess = excess.mean()
        std_excess = excess.std(ddof=1)

        if std_excess == 0 or np.isnan(std_excess):
            return np.nan

        sharpe = mean_excess / std_excess
        if annualize:
            sharpe *= np.sqrt(252)
        return sharpe

    return returns.apply(calc_single_sharpe)


def calculate_beta(
    returns: pd.DataFrame,
    benchmark_col: str,
    period_days: int
) -> pd.Series:
    """Calculate beta with pairwise dropna."""
    min_obs = get_min_obs(period_days)

    def calc_single_beta(asset_returns: pd.Series) -> float:
        asset = asset_returns.tail(period_days)
        bench = returns[benchmark_col].tail(period_days)

        combined = pd.concat([asset, bench], axis=1).dropna()
        if len(combined) < min_obs:
            return np.nan

        asset_clean = combined.iloc[:, 0]
        bench_clean = combined.iloc[:, 1]

        cov = asset_clean.cov(bench_clean)
        var = bench_clean.var(ddof=1)

        if var == 0 or np.isnan(var):
            return np.nan

        return cov / var

    return returns.apply(calc_single_beta)


def calculate_correlation(
    returns: pd.DataFrame,
    benchmark_col: str,
    period_days: int
) -> pd.Series:
    """Calculate correlation with pairwise dropna."""
    min_obs = get_min_obs(period_days)

    def calc_single_corr(asset_returns: pd.Series) -> float:
        asset = asset_returns.tail(period_days)
        bench = returns[benchmark_col].tail(period_days)

        combined = pd.concat([asset, bench], axis=1).dropna()
        if len(combined) < min_obs:
            return np.nan

        return combined.iloc[:, 0].corr(combined.iloc[:, 1])

    return returns.apply(calc_single_corr)


def calculate_correlation_matrix(
    returns: pd.DataFrame,
    period_days: int
) -> pd.DataFrame:
    """Calculate correlation matrix."""
    subset = returns.tail(period_days)
    min_obs = get_min_obs(period_days)
    return subset.corr(method='pearson', min_periods=min_obs)


def calculate_volatility(
    returns: pd.DataFrame,
    period_days: int,
    annualize: bool = True
) -> pd.Series:
    """Calculate volatility per asset."""
    min_obs = get_min_obs(period_days)

    def calc_vol(r: pd.Series) -> float:
        r_clean = r.tail(period_days).dropna()
        if len(r_clean) < min_obs:
            return np.nan
        vol = r_clean.std(ddof=1)
        if annualize:
            vol *= np.sqrt(252)
        return vol

    return returns.apply(calc_vol)


def calculate_total_return(
    prices_raw: pd.DataFrame,
    period_days: int
) -> pd.Series:
    """Calculate total return using RAW prices."""
    min_obs = get_min_obs(period_days)

    result = {}
    for col in prices_raw.columns:
        p = prices_raw[col].tail(period_days).dropna()
        if len(p) < min_obs:
            result[col] = np.nan
            continue
        result[col] = (p.iloc[-1] / p.iloc[0]) - 1

    return pd.Series(result)


def calculate_max_drawdown(
    prices_raw: pd.DataFrame
) -> pd.Series:
    """Calculate max drawdown per asset."""
    if isinstance(prices_raw, pd.Series):
        prices_raw = prices_raw.to_frame()

    result = {}
    for col in prices_raw.columns:
        p = prices_raw[col].dropna()
        if len(p) < 2:
            result[col] = np.nan
            continue

        rolling_max = p.cummax()
        drawdown = (p - rolling_max) / rolling_max
        result[col] = drawdown.min()

    return pd.Series(result)


def calculate_rolling_sharpe(
    returns: pd.DataFrame,
    annual_rf: float,
    return_type: str,
    window: int = 252
) -> pd.DataFrame:
    """Calculate rolling Sharpe ratio."""
    daily_rf = get_daily_rf(annual_rf, return_type)
    min_periods = get_min_obs(window)

    excess_returns = returns - daily_rf

    rolling_mean = excess_returns.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = excess_returns.rolling(window=window, min_periods=min_periods).std(ddof=1)

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    return rolling_sharpe



# VISUALIZATION HELPERS


def create_multiperiod_beta_chart(returns: pd.DataFrame, benchmark: str) -> go.Figure:
    """Create multi-period beta comparison chart."""
    fig = make_subplots(rows=2, cols=4, subplot_titles=[f'{Config.PERIOD_LABELS[p]} Beta' for p in Config.PERIODS[:8]])

    for i, p in enumerate(Config.PERIODS[:8]):
        if p > len(returns):
            continue
        betas = calculate_beta(returns, benchmark, p)
        corrs = calculate_correlation(returns, benchmark, p)

        valid_idx = betas.dropna().index.intersection(corrs.dropna().index)

        row = i // 4 + 1
        col = i % 4 + 1

        fig.add_trace(go.Scatter(
            x=betas[valid_idx], y=corrs[valid_idx],
            mode='markers', marker=dict(size=6, opacity=0.6),
            text=valid_idx, hovertemplate='%{text}<br>Beta=%{x:.2f}<br>Corr=%{y:.2f}<extra></extra>',
            showlegend=False
        ), row=row, col=col)

        fig.add_vline(x=1, line_dash="dash", line_color="gray", row=row, col=col)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

    fig.update_layout(height=500, margin=dict(t=50, b=20))

    return fig



# MAIN APPLICATION


def main():
    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------
    st.title("Beta & Correlation Analytics")
    st.markdown("**Beta, correlation, Sharpe, and relative performance across an asset universe.**")

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # -------------------------------------------------------------------------
    st.sidebar.header("Parameters")

    rf_rate_pct = st.sidebar.number_input(
        "Risk-Free Rate (Annual %)",
        min_value=0.0, max_value=20.0, value=4.5, step=0.1
    )
    rf_rate = rf_rate_pct / 100

    return_type = st.sidebar.selectbox(
        "Return Type",
        options=['simple', 'log'],
        format_func=lambda x: 'Simple Returns' if x == 'simple' else 'Log Returns'
    )

    period_options = {p: f"{Config.PERIOD_LABELS[p]} ({p}d)" for p in Config.PERIODS}
    period = st.sidebar.selectbox(
        "Lookback Period",
        options=list(period_options.keys()),
        index=6,  # 252 = 1Y
        format_func=lambda x: period_options[x]
    )

    # Show daily RF
    daily_rf = get_daily_rf(rf_rate, return_type)
    st.sidebar.caption(f"Daily RF: {daily_rf * 100:.4f}% ({return_type})")

    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------
    with st.spinner("Building universe and fetching data..."):
        tickers, universe_breakdown = build_universe()
        cl_price_raw, cl_price_plot, fetch_stats = fetch_price_data(tickers)

        if cl_price_raw.empty:
            st.error("Failed to load price data. Please refresh.")
            return

        simple_returns, log_returns = calculate_returns(cl_price_raw)

    available_tickers = list(cl_price_raw.columns)

    if (
        universe_breakdown.get('sp500_source') == 'static_csv'
        or universe_breakdown.get('nasdaq100_source') == 'static_csv'
    ):
        st.warning("Using committed static universe because live Wikipedia fetch failed.")

    if (
        universe_breakdown.get('sp500_source') == 'emergency_fallback'
        or universe_breakdown.get('nasdaq100_source') == 'emergency_fallback'
    ):
        st.warning("Using reduced emergency universe; coverage is incomplete.")

    if fetch_stats.get('stooq_recovered_count', 0) > 0:
        st.warning("Some symbols use Stooq daily close fallback. This is not Yahoo adjusted close.")

    # Benchmark selector (in sidebar, after data is loaded)
    benchmark_options = ['SPY', 'QQQ', 'IWM', 'DIA', 'EFA', 'EEM']
    benchmark_options = [b for b in benchmark_options if b in available_tickers]
    if not benchmark_options:
        benchmark_options = available_tickers[:5]

    benchmark = st.sidebar.selectbox("Benchmark", options=benchmark_options, index=0)

    # Data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Universe Info**")
    st.sidebar.caption(f"Raw universe: {universe_breakdown.get('total_raw', 0)}")
    st.sidebar.caption(f"Final unique universe: {universe_breakdown.get('final_unique', 0)}")
    st.sidebar.caption(
        f"S&P 500: {universe_breakdown.get('sp500_count', 0)} "
        f"({universe_breakdown.get('sp500_source', 'unknown')})"
    )
    st.sidebar.caption(
        f"Nasdaq-100: {universe_breakdown.get('nasdaq100_count', 0)} "
        f"({universe_breakdown.get('nasdaq100_source', 'unknown')})"
    )
    st.sidebar.caption(f"Final loaded: {fetch_stats['final_ticker_count']}")
    st.sidebar.caption(f"Trading days: {fetch_stats['trading_days']}")
    st.sidebar.caption(f"Date range: {fetch_stats['date_start']} to {fetch_stats['date_end']}")
    st.sidebar.caption(f"Completeness: {fetch_stats['data_completeness_pct']:.1f}%")

    source_breakdown = fetch_stats.get('data_source_breakdown', {})
    with st.sidebar.expander("Data diagnostics", expanded=False):
        st.caption(f"Base tickers: {universe_breakdown.get('base_count', 0)}")
        st.caption(f"ADR tickers: {universe_breakdown.get('adr_count', 0)}")
        st.caption(f"Duplicates removed: {universe_breakdown.get('duplicates_removed', 0)}")
        st.caption(f"Excluded tickers: {universe_breakdown.get('excluded_count', 0)}")
        st.caption(f"Yahoo loaded: {fetch_stats.get('yahoo_loaded_count', 0)}")
        st.caption(f"Yahoo initial missing/retried: {fetch_stats.get('yahoo_missing_count', 0)}")
        st.caption(f"Yahoo recovered on retry: {fetch_stats.get('yahoo_recovered_count', 0)}")
        st.caption(f"Yahoo failed/missing: {fetch_stats.get('yahoo_failed_count', 0)}")
        st.caption(f"Yahoo all-NaN dropped: {fetch_stats.get('yahoo_all_nan_count', 0)}")
        st.caption(f"Stooq recovered: {fetch_stats.get('stooq_recovered_count', 0)}")
        st.caption(f"Failed/unavailable: {fetch_stats.get('failed_unavailable_count', 0)}")
        st.caption(f"{format_price_basis_label('yfinance_adjusted_close')}: {source_breakdown.get('Yahoo adjusted close', 0)}")
        st.caption(f"{format_price_basis_label('stooq_close')}: {source_breakdown.get('Stooq daily close', 0)}")

        failed = fetch_stats.get('final_failed_tickers', [])
        if failed:
            st.caption("Failed sample: " + ", ".join(failed[:50]))

    # Select returns based on type
    returns = log_returns if return_type == 'log' else simple_returns
    return_label = 'Log' if return_type == 'log' else 'Simple'

    # -------------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Correlation Matrix", "Beta Analysis", "Sharpe Ratios", "Relative Performance"
    ])

    
    # TAB 1: OVERVIEW
    
    with tab1:
        sharpe_all = calculate_sharpe_ratio(returns, rf_rate, period, return_type)
        betas = calculate_beta(returns, benchmark, period)
        correlations = calculate_correlation(returns, benchmark, period)
        volatility = calculate_volatility(returns, period)
        total_ret = calculate_total_return(cl_price_raw, period)

        summary_df = pd.DataFrame({
            'Sharpe': sharpe_all,
            'Beta': betas,
            'Correlation': correlations,
            'Volatility': volatility,
            'Return': total_ret
        }).round(4)
        summary_df = summary_df.dropna(how='all')
        summary_df = summary_df.sort_values('Sharpe', ascending=False)

        top_sharpe = summary_df.nlargest(10, 'Sharpe')
        bottom_sharpe = summary_df.nsmallest(10, 'Sharpe')
        diversifiers = summary_df.nsmallest(10, 'Correlation')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(f"Top 10 Sharpe ({Config.PERIOD_LABELS[period]})")
            st.dataframe(
                top_sharpe[['Sharpe', 'Beta', 'Return', 'Volatility']].style.format("{:.3f}"),
                use_container_width=True,
                height=400
            )
            csv = top_sharpe.to_csv()
            st.download_button("Download CSV", csv, "top_sharpe.csv", "text/csv", key="dl_top_sharpe")

        with col2:
            st.subheader(f"Bottom 10 Sharpe ({Config.PERIOD_LABELS[period]})")
            st.dataframe(
                bottom_sharpe[['Sharpe', 'Beta', 'Return', 'Volatility']].style.format("{:.3f}"),
                use_container_width=True,
                height=400
            )
            csv = bottom_sharpe.to_csv()
            st.download_button("Download CSV", csv, "bottom_sharpe.csv", "text/csv", key="dl_bot_sharpe")

        with col3:
            st.subheader(f"Top 10 Diversifiers (Low Corr to {benchmark})")
            st.dataframe(
                diversifiers[['Correlation', 'Beta', 'Sharpe']].style.format("{:.3f}"),
                use_container_width=True,
                height=400
            )
            csv = diversifiers.to_csv()
            st.download_button("Download CSV", csv, "diversifiers.csv", "text/csv", key="dl_div")

        # Multi-period correlations
        st.subheader(f"Multi-Period Correlations to {benchmark}")
        corr_data = {}
        for p in Config.PERIODS:
            if p <= len(returns):
                corr_data[Config.PERIOD_LABELS[p]] = calculate_correlation(returns, benchmark, p)

        corr_df = pd.DataFrame(corr_data)
        sort_col = '1Y' if '1Y' in corr_df.columns else corr_df.columns[-1]
        corr_df = corr_df.sort_values(sort_col, ascending=False)
        corr_df_display = corr_df.drop(benchmark, errors='ignore').head(20)

        st.dataframe(
            corr_df_display.style.format("{:.3f}"),
            use_container_width=True
        )

    
    # TAB 2: CORRELATION MATRIX
    
    with tab2:
        corr_matrix = calculate_correlation_matrix(returns, period)

        # Order with benchmark first
        if benchmark in corr_matrix.columns:
            ordered = [benchmark] + [t for t in corr_matrix.columns if t != benchmark]
            corr_matrix = corr_matrix.reindex(columns=ordered, index=ordered)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            hovertemplate='%{y} <-> %{x}<br>Corr = %{z:.3f}<extra></extra>',
            colorbar=dict(title='Correlation')
        ))

        fig.update_layout(
            title=f'{return_label} Returns Correlation Matrix | {Config.PERIOD_LABELS[period]} | RF={rf_rate_pct:.1f}%',
            height=800,
            xaxis=dict(tickangle=-45, tickfont=dict(size=7)),
            yaxis=dict(tickfont=dict(size=7), autorange='reversed'),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Highest/Lowest correlation tables
        bench_corr = corr_matrix[benchmark].drop(benchmark, errors='ignore')
        highest = bench_corr.nlargest(10)
        lowest = bench_corr.nsmallest(10)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Highest Correlation to {benchmark}")
            st.dataframe(highest.to_frame('Correlation').style.format("{:.4f}"), use_container_width=True)

        with col2:
            st.subheader(f"Lowest Correlation to {benchmark}")
            st.dataframe(lowest.to_frame('Correlation').style.format("{:.4f}"), use_container_width=True)

    
    # TAB 3: BETA ANALYSIS
    
    with tab3:
        betas = calculate_beta(returns, benchmark, period)
        correlations = calculate_correlation(returns, benchmark, period)

        df = pd.DataFrame({'Beta': betas, 'Correlation': correlations}).dropna()

        top_beta = df.nlargest(5, 'Beta').index
        bottom_beta = df.nsmallest(5, 'Beta').index

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Beta'], y=df['Correlation'],
            mode='markers+text',
            marker=dict(size=8, color='lightgray', opacity=0.6),
            text=df.index, textposition='top center', textfont=dict(size=7),
            name='All Assets', hovertemplate='%{text}<br>Beta=%{x:.2f}<br>Corr=%{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=df.loc[top_beta, 'Beta'], y=df.loc[top_beta, 'Correlation'],
            mode='markers+text',
            marker=dict(size=12, color='green', symbol='star'),
            text=top_beta, textposition='top center', textfont=dict(size=9, color='green'),
            name='Top 5 Beta'
        ))

        fig.add_trace(go.Scatter(
            x=df.loc[bottom_beta, 'Beta'], y=df.loc[bottom_beta, 'Correlation'],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='star'),
            text=bottom_beta, textposition='top center', textfont=dict(size=9, color='red'),
            name='Bottom 5 Beta'
        ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=f'Beta vs Correlation to {benchmark} | {return_label} Returns | {Config.PERIOD_LABELS[period]}',
            xaxis_title=f'Beta to {benchmark}',
            yaxis_title=f'Correlation to {benchmark}',
            height=600,
            legend=dict(x=0.02, y=0.98)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Multi-period beta comparison
        st.subheader("Multi-Period Beta Comparison")
        fig_multi = create_multiperiod_beta_chart(returns, benchmark)
        st.plotly_chart(fig_multi, use_container_width=True)

        # Beta table with download
        st.subheader("Beta & Correlation Table")
        beta_corr_df = df.sort_values('Beta', ascending=False)
        st.dataframe(beta_corr_df.style.format("{:.3f}"), use_container_width=True, height=400)
        csv = beta_corr_df.to_csv()
        st.download_button("Download Beta/Corr CSV", csv, "beta_correlation.csv", "text/csv", key="dl_beta")

    
    # TAB 4: SHARPE RATIOS
    
    with tab4:
        sharpe = calculate_sharpe_ratio(returns, rf_rate, period, return_type).dropna()
        sharpe_df = sharpe.sort_values(ascending=False).reset_index()
        sharpe_df.columns = ['Asset', 'Sharpe']

        colors_bar = ['green' if s > 0 else 'red' for s in sharpe_df['Sharpe']]

        fig_bar = go.Figure(data=[go.Bar(
            x=sharpe_df['Asset'],
            y=sharpe_df['Sharpe'],
            marker_color=colors_bar,
            text=sharpe_df['Sharpe'].round(2),
            textposition='outside',
            textfont=dict(size=7),
            hovertemplate='%{x}<br>Sharpe: %{y:.3f}<extra></extra>'
        )])

        fig_bar.update_layout(
            title=f'Sharpe Ratios ({return_label}) | {Config.PERIOD_LABELS[period]} | RF={rf_rate_pct:.1f}%',
            xaxis_title='Assets',
            yaxis_title='Annualized Sharpe Ratio',
            height=500,
            xaxis=dict(tickangle=-45, tickfont=dict(size=7))
        )
        fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_bar.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.3, annotation_text="SR=1", annotation_position="right")
        fig_bar.add_hline(y=2, line_dash="dot", line_color="green", opacity=0.3, annotation_text="SR=2", annotation_position="right")

        st.plotly_chart(fig_bar, use_container_width=True)

        # Rolling Sharpe with ticker selector
        st.subheader("Rolling Sharpe Ratio")

        rolling_sharpe_default = ['SPY', 'QQQ']
        rolling_sharpe_default = [t for t in rolling_sharpe_default if t in available_tickers]

        selected_tickers = st.multiselect(
            "Select tickers to display:",
            options=sorted(available_tickers),
            default=rolling_sharpe_default,
            key="rolling_sharpe_tickers"
        )

        rolling_window = min(252, period)
        rolling_sharpe = calculate_rolling_sharpe(returns, rf_rate, return_type, window=rolling_window)

        fig_rolling = go.Figure()

        if selected_tickers:
            for ticker in selected_tickers:
                if ticker in rolling_sharpe.columns:
                    fig_rolling.add_trace(go.Scatter(
                        x=rolling_sharpe.index,
                        y=rolling_sharpe[ticker],
                        name=ticker,
                        mode='lines'
                    ))

        fig_rolling.update_layout(
            title=f'Rolling {rolling_window}-Day Sharpe Ratio | RF={rf_rate_pct:.1f}%',
            xaxis_title='Date',
            yaxis_title='Sharpe Ratio',
            height=400,
            legend=dict(orientation='h', y=-0.15)
        )
        fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_rolling.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.5)
        fig_rolling.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.5)

        st.plotly_chart(fig_rolling, use_container_width=True)

        # Sharpe table with download
        st.subheader("Sharpe Ratio Table")
        st.dataframe(sharpe_df.set_index('Asset').style.format("{:.3f}"), use_container_width=True, height=400)
        csv = sharpe_df.to_csv(index=False)
        st.download_button("Download Sharpe CSV", csv, "sharpe_ratios.csv", "text/csv", key="dl_sharpe")

    
    # TAB 5: RELATIVE PERFORMANCE
    
    with tab5:
        # Use ffilled prices for plotting
        prices_period = cl_price_plot.tail(period)

        norm_prices = pd.DataFrame(index=prices_period.index)
        for col in prices_period.columns:
            p = prices_period[col].dropna()
            if len(p) > 0:
                norm_prices[col] = prices_period[col] / p.iloc[0] * 100

        fig_perf = go.Figure()

        key_assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'NVDA', 'TSLA', 'IWM', 'EEM']
        for ticker in key_assets:
            if ticker in norm_prices.columns:
                fig_perf.add_trace(go.Scatter(
                    x=norm_prices.index, y=norm_prices[ticker],
                    name=ticker, mode='lines',
                    visible=True if ticker in ['SPY', 'QQQ', 'TLT'] else 'legendonly'
                ))

        for ticker in norm_prices.columns:
            if ticker not in key_assets:
                fig_perf.add_trace(go.Scatter(
                    x=norm_prices.index, y=norm_prices[ticker],
                    name=ticker, mode='lines', visible='legendonly'
                ))

        fig_perf.add_hline(y=100, line_dash="dash", line_color="gray")

        fig_perf.update_layout(
            title=f'Relative Performance (Normalized to 100) | {Config.PERIOD_LABELS[period]}',
            xaxis_title='Date',
            yaxis_title='Indexed Price (Start = 100)',
            height=600,
            legend=dict(orientation='h', y=-0.2, x=0)
        )

        st.plotly_chart(fig_perf, use_container_width=True)

        # Performance summary tables
        total_ret = calculate_total_return(cl_price_raw, period) * 100
        volatility = calculate_volatility(returns, period) * 100
        max_dd = calculate_max_drawdown(cl_price_raw.tail(period)) * 100

        perf_summary = pd.DataFrame({
            'Total Return (%)': total_ret,
            'Volatility (%)': volatility,
            'Max Drawdown (%)': max_dd
        }).round(2)
        perf_summary = perf_summary.dropna(subset=['Total Return (%)'])
        perf_summary = perf_summary.sort_values('Total Return (%)', ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 10 Performers")
            st.dataframe(
                perf_summary.head(10).style.format("{:.2f}"),
                use_container_width=True
            )

        with col2:
            st.subheader("Bottom 10 Performers")
            st.dataframe(
                perf_summary.tail(10).iloc[::-1].style.format("{:.2f}"),
                use_container_width=True
            )

        # Full table with download
        st.subheader("Full Performance Table")
        st.dataframe(perf_summary.style.format("{:.2f}"), use_container_width=True, height=400)
        csv = perf_summary.to_csv()
        st.download_button("Download Performance CSV", csv, "performance.csv", "text/csv", key="dl_perf")


if __name__ == "__main__":
    main()
