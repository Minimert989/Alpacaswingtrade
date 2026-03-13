import uuid
import time
import logging
from datetime import datetime, timedelta
import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from signals import check_concurrent_limit

log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")


class AlpacaExecutor:
    def __init__(self, config):
        self.client = TradingClient(
            config["alpaca_key"], config["alpaca_secret"],
            paper=config["paper"],
        )
        self.config = config

    def submit_order(self, symbol, qty, side, entry_price, portfolio_state):
        if qty <= 0:
            return None
        if not self._market_time_ok():
            log.info(f"[EXEC] Market buffer. Skipping {symbol}.")
            return None
        if not check_concurrent_limit(
            portfolio_state, self.config["max_concurrent_positions"]
        ):
            log.warning(f"[EXEC] Max concurrent positions. Skipping {symbol}.")
            return None

        existing = portfolio_state["positions"].get(symbol, {})
        if existing.get("qty", 0) != 0:
            if existing.get("side") == side:
                log.info(f"[EXEC] Already {side} {symbol}. Skipping.")
                return None
            log.info(f"[EXEC] Opposite signal {symbol}. Closing first.")
            self.close_position(symbol)
            time.sleep(1)

        sign     = 1 if side == OrderSide.BUY else -1
        tp_price = round(entry_price * (1 + sign * self.config["tp"]), 2)
        sl_price = round(entry_price * (1 - sign * self.config["sl"]), 2)
        return self._submit_bracket(symbol, qty, side, tp_price, sl_price)

    def close_position(self, symbol):
        try:
            self.client.close_position(symbol)
            log.info(f"[EXEC] Closed {symbol}")
        except Exception as e:
            log.error(f"[EXEC] close failed {symbol}: {e}")

    def get_portfolio_state(self):
        positions = {}
        try:
            for p in self.client.get_all_positions():
                positions[p.symbol] = {
                    "qty":   float(p.qty),
                    "side":  "buy" if float(p.qty) > 0 else "sell",
                    "value": float(p.market_value),
                }
        except Exception as e:
            log.error(f"[EXEC] get_positions failed: {e}")
        acct           = self.client.get_account()
        total_exposure = sum(abs(p["value"]) for p in positions.values())
        daily_pnl      = float(acct.equity) - float(acct.last_equity)
        return {
            "positions":      positions,
            "capital":        float(acct.portfolio_value),
            "total_exposure": total_exposure,
            "daily_pnl":      daily_pnl,
        }

    def _market_time_ok(self):
        """timedelta 기반 계산 — 분 단위 오버플로 방지"""
        now_et = datetime.now(ET)
        market_open  = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        buf_open  = market_open  + timedelta(minutes=self.config["market_open_buffer_min"])
        buf_close = market_close - timedelta(minutes=self.config["market_close_buffer_min"])
        return buf_open <= now_et <= buf_close

    def _submit_bracket(self, symbol, qty, side, tp_price, sl_price):
        client_id = str(uuid.uuid4())
        req = MarketOrderRequest(
            symbol=symbol, qty=qty, side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
            client_order_id=client_id,
        )
        for attempt in range(3):
            try:
                order = self.client.submit_order(req)
                log.info(
                    f"[EXEC] {symbol} {side} qty={qty} "
                    f"tp={tp_price} sl={sl_price}"
                )
                self._check_fill(order.id)
                return order
            except Exception as e:
                log.warning(f"[EXEC] submit retry {attempt+1}/3 ({e})")
                time.sleep(2 ** attempt)
        log.error(f"[EXEC] Failed to submit {symbol}")
        return None

    def _check_fill(self, order_id, wait_sec=300):
        time.sleep(wait_sec)
        try:
            order = self.client.get_order_by_id(order_id)
            if order.status not in ("filled", "partially_filled"):
                log.warning(f"[EXEC] Unfilled {order_id}. Cancelling.")
                self.client.cancel_order_by_id(order_id)
        except Exception as e:
            log.error(f"[EXEC] fill check failed: {e}")
