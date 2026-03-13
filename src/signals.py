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
    features_row에 반드시 포함돼야 하는 키:
      earnings_proximity, volume, avg_vol_20d, vix_regime

    config (optional): if provided, uses threshold_short for signal=-1
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

    raw  = meta_model.predict_proba_positive(X_live.reshape(1, -1))
    prob = float(raw[0])  # raw ensemble prob (calibrator retrain pending)

    vix_regime = features_row.get("vix_regime", 1)

    # Dual threshold: longs use standard threshold (+ VIX adjustment),
    # shorts use threshold_short (MetaModel less well-calibrated for shorts).
    if primary_signal == -1 and config is not None:
        # In high VIX, shorts face mean-reversion risk → tighten slightly
        adj_thresh = config.get("threshold_short", threshold - 0.15)
        adj_thresh += 0.03 if vix_regime == 2 else 0.0
    else:
        adj_thresh = threshold + (0.05 if vix_regime == 2 else 0.0)

    if prob >= adj_thresh:
        return primary_signal, prob
    return 0, None


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
