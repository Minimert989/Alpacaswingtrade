import numpy as np
import logging

log = logging.getLogger(__name__)


def generate_signal(meta_model, calibrator, X_live,
                    primary_signal, features_row, threshold):
    """
    features_row에 반드시 포함돼야 하는 키:
      earnings_proximity, volume, avg_vol_20d, vix_regime
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
