import numpy as np
import logging

log = logging.getLogger(__name__)


def is_bear_market(spy_price, spy_ma200, breadth_pct, config):
    """
    True when the market is in a bear regime:
      - SPY is below its 200-day moving average, AND
      - fewer than breadth_min_long of the universe is above SMA200
    Both conditions must hold to avoid false positives on brief dips.
    """
    threshold = config.get("breadth_min_long", 0.35)
    return (spy_price < spy_ma200) and (breadth_pct < threshold)


def generate_signal(meta_model, calibrator, X_live,
                    primary_signal, features_row, threshold,
                    config=None):
    """
    Returns (signal, prob) after applying all pre-filters.

    Filter chain:
      1. PrimaryModel pass-through check
      2. Earnings blackout
      3. Volume filter (>= 0.5x 20d avg)
      4. VIX hard stop (VIX > vix_stop_threshold → no trade)
      5. MetaModel prob threshold (with VIX caution bump)

    config keys used:
      threshold_short, vix_stop_threshold (30), vix_caution_threshold (20)
    """
    if primary_signal == 0:
        return 0, None

    if features_row.get("earnings_proximity", False):
        return 0, None

    # volume filter: 오늘 거래량 vs 20일 평균 거래량
    volume_today = features_row.get("volume", 0)
    avg_vol_20d  = features_row.get("avg_vol_20d", 1)
    if volume_today < avg_vol_20d * 0.5:
        log.debug(f"Volume filter: {volume_today:.0f} < {avg_vol_20d*0.5:.0f}")
        return 0, None

    # VIX hard stop: extreme panic → ML model unreliable, flat is best
    vix_val  = features_row.get("vix", 0)
    vix_stop = config.get("vix_stop_threshold", 30) if config else 30
    if vix_val > vix_stop:
        log.debug(f"VIX stop: vix={vix_val:.1f} > threshold={vix_stop}")
        return 0, None

    raw  = meta_model.predict_proba_positive(X_live.reshape(1, -1))
    prob = float(raw[0])  # raw ensemble prob (calibrator retrain pending)

    # VIX caution zone: raise threshold slightly but keep trading
    vix_caution = config.get("vix_caution_threshold", 20) if config else 20
    vix_bump    = 0.03 if vix_val > vix_caution else 0.0

    # Dual threshold: longs use standard threshold (+VIX bump),
    # shorts use threshold_short (MetaModel less calibrated for shorts).
    if primary_signal == -1 and config is not None:
        adj_thresh = config.get("threshold_short", threshold - 0.15) + vix_bump
    else:
        adj_thresh = threshold + vix_bump

    if prob >= adj_thresh:
        return primary_signal, prob
    return 0, None


def select_cs_signals(candidates, config):
    """
    Cross-sectional selection: from all of today's passing signal candidates,
    pick the top-N longs and top-N shorts by MetaModel prob.

    This is the core of the L/S equity strategy — rather than taking every
    signal that clears the threshold, we rank and select only the highest-
    conviction names each day, creating a portfolio that's:
      • Long the strongest outliers (highest prob)
      • Short the weakest outliers (highest short prob)

    candidates: list of (ticker, signal, prob, feat_row, X_row)
    returns:    filtered list, top cs_long_n longs + top cs_short_n shorts
    """
    cs_long_n  = config.get("cs_long_n", 3)
    cs_short_n = config.get("cs_short_n", 3)

    longs  = sorted([c for c in candidates if c[1] == +1], key=lambda x: -x[2])
    shorts = sorted([c for c in candidates if c[1] == -1], key=lambda x: -x[2])

    selected = longs[:cs_long_n] + shorts[:cs_short_n]

    if selected:
        n_l = sum(1 for c in selected if c[1] == +1)
        n_s = sum(1 for c in selected if c[1] == -1)
        log.info(
            f"[CS-Select] {n_l}L / {n_s}S  "
            f"(from {len(longs)} long / {len(shorts)} short candidates)"
        )
    return selected


def position_size(signal, prob, capital, atr, price,
                  portfolio_state, config):
    if signal == 0:
        return 0

    if portfolio_state["daily_pnl"] < \
            -config["daily_loss_limit_pct"] * capital:
        log.warning("[RISK] Daily loss limit hit. No new entries.")
        return 0

    if portfolio_state["total_exposure"] >= \
            config["max_total_exposure"] * capital:
        log.warning("[RISK] Max total exposure reached.")
        return 0

    b     = config["tp"] / config["sl"]
    kelly = (prob * b - (1 - prob)) / b
    kelly = max(kelly, 0)
    frac  = min(kelly * config["kelly_fraction"],
                config["max_position_pct"])
    dollar_risk = capital * frac
    shares      = int(dollar_risk / (atr * 2 + 1e-9))
    return max(shares, 0)


def count_active_positions(portfolio_state) -> int:
    return sum(
        1 for p in portfolio_state["positions"].values()
        if p.get("qty", 0) != 0
    )


def check_concurrent_limit(portfolio_state, max_concurrent) -> bool:
    return count_active_positions(portfolio_state) < max_concurrent
