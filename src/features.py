"""
features_df: 원본 + 파생 전체 (primary model, reporting, labeling 용)
X_model:     MetaModel 학습/추론용 숫자 서브셋
split_inputs()로 항상 명시적으로 분리.
barrier 컬럼은 labeling.attach_barrier_to_features()가 붙임.
운영(main.py) 시에는 barrier 컬럼 없음 — 정상.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf          # earnings calendar 조회

MODEL_FEATURE_COLS = [
    "ma_cross_20_60", "ema_slope", "adx_14", "aroon_osc_14",
    "rsi_14", "macd_hist", "cci_20", "roc_10", "stoch_k", "stoch_d",
    "atr_14", "bb_width", "bb_pos", "natr_14",
    "obv_slope", "mfi_14", "vwap_dev",
    "vix", "spy_ret_1d", "sector_rel",
    "rsi_1h", "ma_cross_1h",
    "hv_20", "hv_ratio", "vix_regime", "adx_regime",
    "gap_open", "overnight_ret",
    "cs_momentum_rank", "cs_vol_rank", "cs_rsi_rank",
    "breadth_pct",   # fraction of universe above SMA200 (market health)
    "primary_signal",
]

PRIMARY_REQUIRED_COLS = [
    "adx_14", "ma_cross_20_60", "close", "sma_200",
    "spy_price", "spy_ma60", "spy_ma200", "earnings_proximity",
    "cs_momentum_rank", "breadth_pct",
]

SECTOR_MAP = {
    # Tech / Semis
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK",
    "META": "XLK", "AMD": "XLK", "AVGO": "XLK", "ORCL": "XLK",
    "CRM": "XLK", "ADBE": "XLK", "MU": "XLK", "QCOM": "XLK",
    "NOW": "XLK", "PANW": "XLK",
    # Consumer Discretionary / Streaming
    "AMZN": "XLY", "TSLA": "XLY", "NFLX": "XLY",
    "HD": "XLY", "WMT": "XLY", "COST": "XLY", "MCD": "XLY", "PG": "XLY",
    # Financials
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BAC": "XLF", "GS": "XLF",
    # Healthcare
    "UNH": "XLV", "LLY": "XLV",
    # Energy
    "XOM": "XLE",
}


def _col(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Return first column whose name starts with `prefix` (case-sensitive)."""
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise KeyError(f"No column starting with '{prefix}' in {list(df.columns)}")
    return df[matches[0]]


def engineer(ohlcv_dict, context_dict, ohlcv_1h_dict, config):
    spy     = context_dict["SPY"]["Close"].rename("spy_price")
    vix     = context_dict["^VIX"]["Close"].rename("vix")
    sectors = {k: context_dict[k]["Close"]
               for k in ["XLK", "XLF", "XLE", "XLV", "XLY"]}
    result = {}
    for ticker, df in ohlcv_dict.items():
        result[ticker] = _engineer_single(
            ticker, df, ohlcv_1h_dict.get(ticker),
            spy, vix, sectors, config,
        )
    return _add_cross_sectional(result)


def split_inputs(features_df: pd.DataFrame):
    """
    primary model → features_df 그대로
    meta model    → X_model (MODEL_FEATURE_COLS)
    barrier_* 컬럼은 X_model에 포함하지 않음.
    """
    missing = [c for c in MODEL_FEATURE_COLS if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing model feature cols: {missing}")
    return features_df, features_df[MODEL_FEATURE_COLS].copy()


def _engineer_single(ticker, df, df_1h, spy, vix, sectors, config):
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["close"]  = c
    f["volume"] = v

    # 추세
    sma20  = ta.sma(c, 20)
    sma60  = ta.sma(c, 60)
    sma200 = ta.sma(c, 200)
    f["sma_200"]        = sma200
    f["ma_cross_20_60"] = sma20 - sma60
    f["ema_slope"]      = ta.ema(c, 10).diff(3)

    adx_df          = ta.adx(h, l, c, 14)
    f["adx_14"]     = _col(adx_df, "ADX_")

    aroon_df        = ta.aroon(h, l, 14)
    f["aroon_osc_14"] = _col(aroon_df, "AROONU") - _col(aroon_df, "AROOND")

    # 모멘텀
    f["rsi_14"]    = ta.rsi(c, 14)
    macd_df        = ta.macd(c)
    f["macd_hist"] = _col(macd_df, "MACDh_")
    f["cci_20"]    = ta.cci(h, l, c, 20)
    f["roc_10"]    = ta.roc(c, 10)
    stoch_df       = ta.stoch(h, l, c)
    f["stoch_k"]   = _col(stoch_df, "STOCHk_")
    f["stoch_d"]   = _col(stoch_df, "STOCHd_")

    # 변동성
    f["atr_14"]  = ta.atr(h, l, c, 14)
    bb_df        = ta.bbands(c, 20)
    bb_u         = _col(bb_df, "BBU_")
    bb_m         = _col(bb_df, "BBM_")
    bb_l         = _col(bb_df, "BBL_")
    f["bb_width"] = (bb_u - bb_l) / (bb_m + 1e-9)
    f["bb_pos"]   = (c - bb_l) / (bb_u - bb_l + 1e-9)
    f["natr_14"]  = ta.natr(h, l, c, 14)

    # 거래량
    f["obv_slope"]   = ta.obv(c, v).diff(5)
    f["mfi_14"]      = ta.mfi(h, l, c, v, 14)
    vwap             = (c * v).cumsum() / v.cumsum()
    f["vwap_dev"]    = (c - vwap) / (vwap + 1e-9)
    f["avg_vol_20d"] = v.rolling(20).mean()

    # 레짐
    log_ret         = np.log(c / c.shift(1))
    f["hv_20"]      = log_ret.rolling(20).std() * np.sqrt(252)
    f["hv_60"]      = log_ret.rolling(60).std() * np.sqrt(252)
    f["hv_ratio"]   = f["hv_20"] / (f["hv_60"] + 1e-9)
    f["adx_regime"] = (f["adx_14"] >= 20).astype(int)

    # 이벤트
    f["gap_open"]      = (o - c.shift(1)) / (c.shift(1) + 1e-9)
    f["overnight_ret"] = f["gap_open"]
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is not None and not cal.empty:
            next_earnings = pd.Timestamp(cal.columns[0])
            days_to_earn  = (next_earnings - pd.Timestamp("today")).days
            f["earnings_proximity"] = days_to_earn <= config["earnings_blackout_days"]
        else:
            f["earnings_proximity"] = False
    except Exception:
        f["earnings_proximity"] = False

    # 시장 컨텍스트
    spy_a           = spy.reindex(f.index, method="ffill")
    f["spy_price"]  = spy_a
    f["spy_ma60"]   = spy_a.rolling(60).mean()
    f["spy_ma200"]  = spy_a.rolling(200).mean()
    f["spy_ret_1d"] = spy_a.pct_change()
    f["vix"]        = vix.reindex(f.index, method="ffill")
    f["vix_regime"] = pd.cut(
        f["vix"], bins=[0, 15, 25, 999], labels=[0, 1, 2]
    ).astype(float)

    # 섹터 상대강도
    sec_key         = SECTOR_MAP.get(ticker, "XLK")
    sec_ret         = sectors[sec_key].pct_change().reindex(f.index, method="ffill")
    f["sector_rel"] = c.pct_change() - sec_ret

    # 멀티 타임프레임
    if df_1h is not None:
        rsi_1h         = ta.rsi(df_1h["Close"], 14).resample("1D").last()
        c1h            = df_1h["Close"]
        mac_1h         = (ta.sma(c1h, 20) - ta.sma(c1h, 60)).resample("1D").last()
        # strip tz so it aligns with the tz-naive daily index
        rsi_1h.index   = rsi_1h.index.tz_convert(None)
        mac_1h.index   = mac_1h.index.tz_convert(None)
        f["rsi_1h"]      = rsi_1h.reindex(f.index, method="ffill")
        f["ma_cross_1h"] = mac_1h.reindex(f.index, method="ffill")
    else:
        f["rsi_1h"]      = np.nan
        f["ma_cross_1h"] = np.nan

    # ADV / cost model
    f["adv_20d"] = (c * v).rolling(20).mean()
    f["hv_20d"]  = f["hv_20"]
    f["ticker"]  = ticker

    return f.dropna(subset=["rsi_14", "adx_14", "atr_14"])


def _add_cross_sectional(result_dict):
    tickers = list(result_dict.keys())
    hv_mat  = pd.DataFrame({t: result_dict[t]["hv_20"]  for t in tickers})
    rsi_mat = pd.DataFrame({t: result_dict[t]["rsi_14"] for t in tickers})
    mom_mat = pd.DataFrame({
        t: result_dict[t]["sector_rel"].rolling(20).sum()
        for t in tickers
    })

    # Market breadth: fraction of universe with close > SMA200 each day.
    # Choppy/bear markets show breadth < 0.4; strong bull markets > 0.6.
    above_sma = pd.DataFrame({
        t: (result_dict[t]["close"] > result_dict[t]["sma_200"]).astype(float)
        for t in tickers
    })
    breadth_series = above_sma.mean(axis=1)   # 0.0 – 1.0

    for ticker in tickers:
        f = result_dict[ticker]
        f["cs_momentum_rank"] = mom_mat.rank(axis=1, pct=True)[ticker]
        f["cs_vol_rank"]      = (1 - hv_mat.rank(axis=1, pct=True))[ticker]
        f["cs_rsi_rank"]      = rsi_mat.rank(axis=1, pct=True)[ticker]
        f["breadth_pct"]      = breadth_series.reindex(f.index, method="ffill")
        result_dict[ticker]   = f
    return result_dict
