import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd

import app


def yahoo_close_frame(values_by_ticker):
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    data = {("Close", ticker): values for ticker, values in values_by_ticker.items()}
    return pd.DataFrame(data, index=index)


def stooq_history(symbol, values=(10.0, 11.0, 12.0)):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return SimpleNamespace(
        data=pd.DataFrame({"close": values}, index=index),
        provider="stooq",
        provider_symbol=f"{symbol.lower()}.us",
        price_basis="stooq_close",
    )


def requested_list(tickers):
    return list(tickers) if isinstance(tickers, (list, tuple)) else [tickers]


def write_static_csv(path, prefix, count):
    symbols = [f"{prefix}{i:03d}" for i in range(count)]
    pd.DataFrame({"symbol": symbols, "source": "test"}).to_csv(path, index=False)
    return symbols


def test_wikipedia_success_source_diagnostics(monkeypatch):
    sp500_df = pd.DataFrame({"Symbol": [f"T{i}" for i in range(401)]})
    nasdaq_df = pd.DataFrame({"Ticker": [f"N{i}" for i in range(91)]})

    monkeypatch.setattr(app, "_read_html_with_headers", lambda url: [sp500_df])
    tickers, meta = app.fetch_sp500_constituents()
    assert len(tickers) == 401
    assert meta["source"] == "wikipedia"

    monkeypatch.setattr(app, "_read_html_with_headers", lambda url: [nasdaq_df])
    tickers, meta = app.fetch_nasdaq100_constituents()
    assert len(tickers) == 91
    assert meta["source"] == "wikipedia"


def test_wikipedia_failure_uses_static_csv_source(monkeypatch, tmp_path):
    sp500_csv = tmp_path / "sp500_constituents.csv"
    nasdaq_csv = tmp_path / "nasdaq100_constituents.csv"
    sp500_symbols = write_static_csv(sp500_csv, "SP", 401)
    nasdaq_symbols = write_static_csv(nasdaq_csv, "ND", 91)

    def raise_fetch(_url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(app, "_read_html_with_headers", raise_fetch)
    monkeypatch.setattr(app, "SP500_STATIC_CSV", sp500_csv)
    monkeypatch.setattr(app, "NASDAQ100_STATIC_CSV", nasdaq_csv)

    tickers, meta = app.fetch_sp500_constituents()
    assert tickers == sp500_symbols
    assert meta["source"] == "static_csv"
    assert "blocked" in meta["error"]

    tickers, meta = app.fetch_nasdaq100_constituents()
    assert tickers == nasdaq_symbols
    assert meta["source"] == "static_csv"
    assert "blocked" in meta["error"]


def test_missing_or_broken_static_csv_uses_emergency_fallback(monkeypatch, tmp_path):
    broken_csv = tmp_path / "broken_sp500.csv"
    pd.DataFrame({"ticker": ["AAA"]}).to_csv(broken_csv, index=False)

    def raise_fetch(_url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(app, "_read_html_with_headers", raise_fetch)
    monkeypatch.setattr(app, "SP500_STATIC_CSV", broken_csv)
    monkeypatch.setattr(app, "NASDAQ100_STATIC_CSV", tmp_path / "missing_nasdaq.csv")

    tickers, meta = app.fetch_sp500_constituents()
    assert tickers == app.SP500_FALLBACK
    assert meta["source"] == "emergency_fallback"
    assert "static CSV failed" in meta["error"]

    tickers, meta = app.fetch_nasdaq100_constituents()
    assert tickers == app.NASDAQ100_FALLBACK
    assert meta["source"] == "emergency_fallback"
    assert "static CSV failed" in meta["error"]


def test_build_universe_preserves_counts_and_sources(monkeypatch):
    monkeypatch.setattr(app, "BASE_TICKERS", ["AAA", "BBB"])
    monkeypatch.setattr(app, "ADR_TICKERS", ["BBB", "CCC"])
    monkeypatch.setattr(app.Config, "EXCLUDED_TICKERS", {"CCC"})
    monkeypatch.setattr(
        app,
        "fetch_sp500_constituents",
        lambda: (["AAA", "DDD"], {"source": "wikipedia", "error": ""}),
    )
    monkeypatch.setattr(
        app,
        "fetch_nasdaq100_constituents",
        lambda: (["EEE"], {"source": "static_csv", "error": "test"}),
    )

    tickers, breakdown = app.build_universe_uncached()

    assert tickers == ["AAA", "BBB", "DDD", "EEE"]
    assert breakdown["base_count"] == 2
    assert breakdown["adr_count"] == 2
    assert breakdown["sp500_count"] == 2
    assert breakdown["sp500_source"] == "wikipedia"
    assert breakdown["nasdaq100_count"] == 1
    assert breakdown["nasdaq100_source"] == "static_csv"
    assert breakdown["total_raw"] == 7
    assert breakdown["duplicates_removed"] == 2
    assert breakdown["excluded_count"] == 1
    assert breakdown["final_unique"] == 4


def test_build_universe_uses_full_static_csvs_when_live_fetch_fails(monkeypatch, tmp_path):
    sp500_csv = tmp_path / "sp500_constituents.csv"
    nasdaq_csv = tmp_path / "nasdaq100_constituents.csv"
    write_static_csv(sp500_csv, "SP", 401)
    write_static_csv(nasdaq_csv, "ND", 91)

    def raise_fetch(_url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(app, "_read_html_with_headers", raise_fetch)
    monkeypatch.setattr(app, "SP500_STATIC_CSV", sp500_csv)
    monkeypatch.setattr(app, "NASDAQ100_STATIC_CSV", nasdaq_csv)
    monkeypatch.setattr(app, "BASE_TICKERS", ["SPY", "QQQ"])
    monkeypatch.setattr(app, "ADR_TICKERS", ["TSM", "ASML"])
    monkeypatch.setattr(app.Config, "EXCLUDED_TICKERS", set())

    tickers, breakdown = app.build_universe_uncached()

    assert "SPY" in tickers
    assert "QQQ" in tickers
    assert "TSM" in tickers
    assert "ASML" in tickers
    assert breakdown["sp500_source"] == "static_csv"
    assert breakdown["nasdaq100_source"] == "static_csv"
    assert breakdown["sp500_count"] == 401
    assert breakdown["nasdaq100_count"] == 91
    assert breakdown["final_unique"] == 496


def test_static_universe_warning_names_each_fallback_and_its_error():
    assert app.static_universe_warning(
        {"sp500_source": "wikipedia", "nasdaq100_source": "wikipedia"}
    ) is None

    message = app.static_universe_warning({
        "sp500_source": "wikipedia",
        "sp500_error": "",
        "nasdaq100_source": "static_csv",
        "nasdaq100_error": "Wikipedia table did not contain enough Nasdaq-100 symbols.",
    })
    assert message.startswith("Using committed static universe because live Wikipedia fetch failed.")
    assert "Nasdaq-100: Wikipedia table did not contain enough Nasdaq-100 symbols." in message
    assert "S&P 500" not in message


def test_dot_to_dash_mapping_still_works():
    assert app.sanitize_ticker("BRK.B") == "BRK-B"
    assert app.sanitize_ticker("PBR.A") == "PBR-A"


def test_partial_yfinance_chunk_detects_missing_tickers(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(app.Config, "MAX_RETRIES", 2)

    def fake_download(tickers, **_kwargs):
        requested = requested_list(tickers)
        if requested == ["AAPL", "MSFT"]:
            return yahoo_close_frame({"AAPL": [1.0, 2.0, 3.0]})
        return pd.DataFrame()

    monkeypatch.setattr(app.yf, "download", fake_download)

    result = app.download_chunked(
        ["AAPL", "MSFT"],
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
        chunk_size=2,
    )

    assert "AAPL" in result.loaded_tickers
    assert "MSFT" in result.missing_tickers
    assert "MSFT" in result.failed_tickers


def test_empty_yfinance_chunk_is_retried(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(app.Config, "MAX_RETRIES", 2)
    calls = {"count": 0}

    def fake_download(tickers, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return pd.DataFrame()
        return yahoo_close_frame({"AAPL": [1.0, 2.0, 3.0]})

    monkeypatch.setattr(app.yf, "download", fake_download)

    result = app.download_chunked(
        ["AAPL"],
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
        chunk_size=1,
    )

    assert calls["count"] == 2
    assert result.loaded_tickers == ["AAPL"]
    assert result.failed_tickers == []


def test_all_nan_yfinance_columns_are_reported(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(app.Config, "MAX_RETRIES", 2)
    monkeypatch.setattr(
        app.yf,
        "download",
        lambda tickers, **_kwargs: yahoo_close_frame({"AAPL": [np.nan, np.nan, np.nan]}),
    )

    result = app.download_chunked(
        ["AAPL"],
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
        chunk_size=1,
    )

    assert "AAPL" in result.all_nan_tickers
    assert "AAPL" in result.failed_tickers
    assert result.data.empty


def test_individual_yfinance_retry_recovers_missing_symbol(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(app.Config, "MAX_RETRIES", 2)

    def fake_download(tickers, **_kwargs):
        requested = requested_list(tickers)
        if requested == ["AAPL", "MSFT"]:
            return yahoo_close_frame({"AAPL": [1.0, 2.0, 3.0]})
        if requested == ["MSFT"]:
            return yahoo_close_frame({"MSFT": [4.0, 5.0, 6.0]})
        return pd.DataFrame()

    monkeypatch.setattr(app.yf, "download", fake_download)

    result = app.download_chunked(
        ["AAPL", "MSFT"],
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
        chunk_size=2,
    )

    assert result.recovered_tickers == ["MSFT"]
    assert result.failed_tickers == []
    assert list(result.data.columns) == ["AAPL", "MSFT"]


def test_stooq_fallback_bridge_merges_close_series(monkeypatch):
    monkeypatch.setattr(app, "get_price_history", lambda symbol, **_kwargs: stooq_history(symbol))

    result = app.recover_missing_with_stooq(
        ["NVDA"],
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
    )

    assert list(result.data.columns) == ["NVDA"]
    assert result.recovered_tickers == ["NVDA"]
    assert result.metadata_by_ticker["NVDA"]["provider"] == "stooq"
    assert result.metadata_by_ticker["NVDA"]["provider_symbol"] == "nvda.us"
    assert result.metadata_by_ticker["NVDA"]["price_basis"] == "stooq_close"


def test_stooq_fallback_is_daily_only_and_not_called(monkeypatch):
    called = {"value": False}

    def fake_history(symbol, **_kwargs):
        called["value"] = True
        return stooq_history(symbol)

    monkeypatch.setattr(app, "get_price_history", fake_history)

    result = app.recover_missing_with_stooq(
        ["NVDA"],
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 4),
        interval="1h",
    )

    assert called["value"] is False
    assert result.data.empty
    assert result.failed_tickers == ["NVDA"]


def test_stooq_not_called_when_yfinance_retry_recovers(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(app.Config, "MAX_RETRIES", 2)

    def fake_download(tickers, **_kwargs):
        requested = requested_list(tickers)
        if requested == ["AAPL", "MSFT"]:
            return yahoo_close_frame({"AAPL": [1.0, 2.0, 3.0]})
        if requested == ["MSFT"]:
            return yahoo_close_frame({"MSFT": [4.0, 5.0, 6.0]})
        return pd.DataFrame()

    def fail_if_called(symbol, **_kwargs):
        raise AssertionError(f"Stooq should not be called for {symbol}")

    monkeypatch.setattr(app.yf, "download", fake_download)
    monkeypatch.setattr(app, "get_price_history", fail_if_called)

    raw, _plot, stats = app.fetch_price_data_uncached(["AAPL", "MSFT"])

    assert list(raw.columns) == ["AAPL", "MSFT"]
    assert stats["yahoo_loaded_count"] == 2
    assert stats["yahoo_failed_count"] == 0
    assert stats["stooq_recovered_count"] == 0


def test_fetch_stats_reports_yahoo_and_stooq_counts(monkeypatch):
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(app.Config, "MAX_RETRIES", 2)

    def fake_download(tickers, **_kwargs):
        requested = requested_list(tickers)
        if requested == ["AAPL", "MSFT"]:
            return yahoo_close_frame({"AAPL": [1.0, 2.0, 3.0]})
        return pd.DataFrame()

    monkeypatch.setattr(app.yf, "download", fake_download)
    monkeypatch.setattr(app, "get_price_history", lambda symbol, **_kwargs: stooq_history(symbol))

    raw, _plot, stats = app.fetch_price_data_uncached(["AAPL", "MSFT"])

    assert list(raw.columns) == ["AAPL", "MSFT"]
    assert stats["yahoo_loaded_count"] == 1
    assert stats["yahoo_failed_count"] == 1
    assert stats["stooq_recovered_count"] == 1
    assert stats["final_ticker_count"] == 2
    assert stats["provider_by_ticker"]["AAPL"] == "yfinance"
    assert stats["provider_by_ticker"]["MSFT"] == "stooq"
    assert stats["price_basis_by_ticker"]["MSFT"] == "stooq_close"


def test_price_basis_display_labels_stooq_close_clearly():
    label = app.format_price_basis_label("stooq_close")
    assert "Stooq daily close" in label
    assert "not Yahoo adjusted close" in label
