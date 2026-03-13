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
from signals import generate_signal, position_size
from executor import AlpacaExecutor
from alpaca.trading.enums import OrderSide

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

    log_rows = []
    for ticker in config["tickers"]:
        features_df = feat_dict.get(ticker)
        if features_df is None or features_df.empty:
            continue

        # primary signal 생성 후 피처에 추가
        primary_signals              = primary.predict(features_df)
        features_df["primary_signal"] = primary_signals

        # 오늘 행 (운영 — barrier 컬럼 없음, 정상)
        feat_today = features_df.iloc[[-1]]
        psig_today = int(primary_signals.iloc[-1])
        feat_row   = feat_today.iloc[0].to_dict()

        _, X_today = split_inputs(feat_today)
        signal, prob = generate_signal(
            meta_model, calibrator,
            X_today.values[0], psig_today,
            feat_row, config["threshold"],
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
