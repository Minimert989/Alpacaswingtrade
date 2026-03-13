import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DataQualityError(Exception):
    pass


class DataLoader:
    def __init__(self, max_retries=3, cache_stale_hours=18):
        self.max_retries       = max_retries
        self.cache_stale_hours = cache_stale_hours

    def fetch(self, tickers, start, end, interval="1d"):
        return {t: self._load_with_cache(t, start, end, interval)
                for t in tickers}

    def fetch_market_context(self, start, end):
        return self.fetch(
            ["^VIX", "SPY", "XLK", "XLF", "XLE", "XLV", "XLY"],
            start, end, "1d"
        )

    def fetch_1h(self, tickers, start, end):
        # yfinance hard-limits 1h data to the last 730 days
        earliest_1h = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
        start_1h    = max(start, earliest_1h)
        return {t: self._fetch_1h_chunked(t, start_1h, end) for t in tickers}

    # ── 내부 ────────────────────────────────────────────────────

    def _cache_path(self, ticker, interval):
        safe = ticker.replace("^", "hat_")
        return CACHE_DIR / f"{safe}_{interval}.parquet"

    def _is_stale(self, path):
        if not path.exists():
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime > timedelta(hours=self.cache_stale_hours)

    def _load_with_cache(self, ticker, start, end, interval):
        path = self._cache_path(ticker, interval)
        if not self._is_stale(path):
            df = pd.read_parquet(path)
            if (df.index.min() <= pd.Timestamp(start) and
                    df.index.max() >= pd.Timestamp(end) - timedelta(days=2)):
                return df
        df = self._fetch_with_retry(ticker, start, end, interval)
        df = self._validate(df, ticker)
        df.to_parquet(path)
        return df

    def _fetch_with_retry(self, ticker, start, end, interval):
        for attempt in range(self.max_retries):
            try:
                df = yf.download(
                    ticker, start=start, end=end,
                    interval=interval, progress=False, auto_adjust=True,
                )
                if df.empty:
                    raise ValueError(f"Empty response: {ticker}")
                df.index = pd.to_datetime(df.index)
                # Flatten MultiIndex columns if present (yfinance v0.2+)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df
            except Exception as e:
                wait = 2 ** attempt
                log.warning(f"[{ticker}] retry {attempt+1}/{self.max_retries} "
                            f"in {wait}s ({e})")
                time.sleep(wait)
        raise DataQualityError(
            f"Failed to fetch {ticker} after {self.max_retries} retries"
        )

    def _fetch_1h_chunked(self, ticker, start, end):
        chunks, cur, end_ts = [], pd.Timestamp(start), pd.Timestamp(end)
        while cur < end_ts:
            chunk_end = min(cur + timedelta(days=59), end_ts)
            try:
                df = self._fetch_with_retry(
                    ticker,
                    cur.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                    "1h",
                )
                chunks.append(df)
            except DataQualityError as e:
                log.warning(f"1h chunk failed for {ticker}: {e}")
            cur = chunk_end + timedelta(days=1)
        if not chunks:
            raise DataQualityError(f"No 1h data for {ticker}")
        return pd.concat(chunks).drop_duplicates().sort_index()

    def _validate(self, df, ticker):
        price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
        df = df[(df[price_cols] > 0).all(axis=1)]
        if "Volume" in df.columns and not ticker.startswith("^"):
            df = df[df["Volume"] > 0]
        # Z-score spike 제거: rolling 252일 기준으로 국소 이상치만 제거
        if "Close" in df.columns:
            rolling_mean = df["Close"].rolling(252, min_periods=20).mean()
            rolling_std  = df["Close"].rolling(252, min_periods=20).std()
            z            = (df["Close"] - rolling_mean) / (rolling_std + 1e-9)
            df           = df[z.abs() <= 6]
        nan_ratio = df.isnull().mean().max()
        if nan_ratio > 0.05:
            log.warning(f"[{ticker}] NaN ratio={nan_ratio:.2%}. Dropping NaN rows.")
        df = df.dropna()
        if df.empty:
            raise DataQualityError(f"[{ticker}] Empty after validation")
        return df
