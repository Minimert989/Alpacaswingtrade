import yaml
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression

from data_loader import DataLoader
from features import engineer, split_inputs
from labeling import attach_barrier_to_features   # 운영엔 미사용, import만
from primary_model import PrimaryModel
from meta_model import MetaModel, calibrate
from signals import generate_signal, position_size, is_bear_market
from executor import AlpacaExecutor
from alpaca.trading.enums import OrderSide, TimeInForce

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run():
    config   = load_config()
    loader   = DataLoader()
    primary  = PrimaryModel(config)
    executor = AlpacaExecutor(config)

    calibrator = joblib.load("models/production/calibrator.pkl")
    meta_model = MetaModel()
    meta_model.load("models/production/model.pkl")

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    ohlcv    = loader.fetch(config["tickers"], start, end, "1d")
    context  = loader.fetch_market_context(start, end)
    ohlcv_1h = loader.fetch_1h(config["tickers"], start, end)

    feat_dict = engineer(ohlcv, context, ohlcv_1h, config)

    portfolio_state = executor.get_portfolio_state()
    capital         = portfolio_state["capital"]

    # ── Bear market detection ───────────────────────────────────────────────
    # Use the most recent row from any ticker to get today's SPY + breadth values.
    # All tickers share the same market context, so any recent row works.
    _sample_feat = next(
        (feat_dict[t] for t in config["tickers"] if t in feat_dict and not feat_dict[t].empty),
        None,
    )
    bear_mode = False
    if _sample_feat is not None:
        today_ctx = _sample_feat.iloc[-1]
        spy_p     = float(today_ctx.get("spy_price", float("inf")))
        spy_m     = float(today_ctx.get("spy_ma200", 0))
        breadth   = float(today_ctx.get("breadth_pct", 1.0))
        bear_mode = is_bear_market(spy_p, spy_m, breadth, config)
        log.info(f"[REGIME] SPY={spy_p:.1f}  MA200={spy_m:.1f}  "
                 f"breadth={breadth:.1%}  → {'BEAR 🐻' if bear_mode else 'BULL 🐂'}")

    # ── SH inverse ETF management ───────────────────────────────────────────
    bear_etf = config.get("bear_etf", "SH")
    bear_etf_alloc = config.get("bear_etf_alloc", 0.25)
    bear_etf_exit_buf = config.get("bear_etf_exit_buffer", 0.01)

    positions = portfolio_state.get("positions", {})
    in_sh = bear_etf in positions and positions[bear_etf].get("qty", 0) > 0

    if bear_mode and not in_sh:
        # Enter SH: allocate bear_etf_alloc fraction of capital
        sh_price = context[bear_etf]["Close"].iloc[-1]
        sh_dollars = capital * bear_etf_alloc
        sh_qty = int(sh_dollars / sh_price)
        if sh_qty > 0:
            try:
                executor.api.submit_order(
                    symbol=bear_etf, qty=sh_qty, side="buy",
                    type="market", time_in_force="day",
                )
                log.info(f"[BEAR] Entered {bear_etf}: {sh_qty} shares @ ~${sh_price:.2f}")
            except Exception as e:
                log.warning(f"[BEAR] SH entry failed: {e}")

    elif not bear_mode and in_sh:
        # Confirm bull re-entry with exit buffer before closing SH
        if _sample_feat is not None:
            spy_above_buf = spy_p > spy_m * (1 + bear_etf_exit_buf)
        else:
            spy_above_buf = True
        if spy_above_buf:
            sh_qty_held = positions[bear_etf].get("qty", 0)
            try:
                executor.api.submit_order(
                    symbol=bear_etf, qty=sh_qty_held, side="sell",
                    type="market", time_in_force="day",
                )
                log.info(f"[BEAR] Exited {bear_etf}: sold {sh_qty_held} shares (bull confirmed)")
            except Exception as e:
                log.warning(f"[BEAR] SH exit failed: {e}")

    # ── Individual stock signals ────────────────────────────────────────────
    log_rows = []
    for ticker in config["tickers"]:
        features_df = feat_dict.get(ticker)
        if features_df is None or features_df.empty:
            continue

        # primary signal 생성 후 피처에 추가
        primary_signals               = primary.predict(features_df)
        features_df["primary_signal"] = primary_signals

        # 오늘 행 (운영 — barrier 컬럼 없음, 정상)
        feat_today = features_df.iloc[[-1]]
        psig_today = int(primary_signals.iloc[-1])
        feat_row   = feat_today.iloc[0].to_dict()

        # In bear mode: only allow short signals; in bull mode: only allow longs
        if bear_mode and psig_today == 1:
            continue   # no new longs in bear market
        if not bear_mode and psig_today == -1:
            continue   # no new shorts in bull market

        _, X_today = split_inputs(feat_today)
        signal, prob = generate_signal(
            meta_model, calibrator,
            X_today.values[0], psig_today,
            feat_row, config["threshold"],
            config=config,              # enables threshold_short for signal=-1
        )

        if signal == 0:
            continue

        atr_today   = feat_row["atr_14"]
        price_today = feat_row["close"]

        qty = position_size(
            signal, prob, capital, atr_today,
            price_today, portfolio_state, config,
        )
        if qty == 0:
            continue

        side  = OrderSide.BUY if signal == 1 else OrderSide.SELL
        order = executor.submit_order(
            ticker, qty, side, price_today, portfolio_state,
        )
        log_rows.append({
            "date":        datetime.today().date(),
            "ticker":      ticker,
            "signal":      signal,
            "prob":        round(prob, 4),
            "qty":         qty,
            "entry_price": price_today,
            "bear_mode":   bear_mode,
            "order_id":    order.id if order else None,
        })

    if log_rows:
        log_path = Path("outputs/trade_log.csv")
        pd.DataFrame(log_rows).to_csv(
            log_path, mode="a",
            header=not log_path.exists(), index=False,
        )


if __name__ == "__main__":
    run()
