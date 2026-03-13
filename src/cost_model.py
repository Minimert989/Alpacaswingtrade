import numpy as np
import logging

log = logging.getLogger(__name__)

HIGH_VOL_MULTIPLIER = {
    "TSLA": 1.8, "NVDA": 1.6, "META": 1.3, "AMZN": 1.2,
}
EXIT_SLIP_MULTIPLIER = {
    "tp": 0.8, "sl": 2.0, "time": 1.0,
}


class TransactionCostModel:
    def __init__(self, commission=0.001, slippage_model="sqrt",
                 conservative=True):
        self.commission     = commission
        self.slippage_model = slippage_model
        self.conservative   = conservative

    def _base_slip(self, order_size_usd, adv_usd, volatility, ticker=None):
        part = order_size_usd / (adv_usd + 1e-9)
        if   self.slippage_model == "sqrt":   slip = 0.1 * np.sqrt(part) * volatility
        elif self.slippage_model == "linear": slip = 0.1 * part * volatility
        else:                                 slip = 0.0005
        slip = max(slip, 0.0002)
        slip *= HIGH_VOL_MULTIPLIER.get(ticker, 1.0)
        if self.conservative:
            slip *= 1.2
        return slip

    def estimate_entry_slip(self, sz, adv, vol, ticker=None,
                             time_of_day="normal"):
        slip = self._base_slip(sz, adv, vol, ticker)
        return slip * 1.5 if time_of_day in ("open", "close") else slip

    def estimate_exit_slip(self, sz, adv, vol, ticker=None,
                            exit_type="sl"):
        slip = self._base_slip(sz, adv, vol, ticker)
        return slip * EXIT_SLIP_MULTIPLIER.get(exit_type, 1.0)

    def total_cost(self, sz, adv, vol, ticker=None,
                   time_of_day="normal", exit_type="sl"):
        return (self.commission
                + self.estimate_entry_slip(sz, adv, vol, ticker, time_of_day)
                + self.estimate_exit_slip(sz, adv, vol, ticker, exit_type))

    def adjust_returns(self, raw_returns, order_sizes, adv, volatility,
                       tickers=None, exit_types=None, time_of_day="normal"):
        n          = len(raw_returns)
        tickers    = tickers    or [None] * n
        exit_types = exit_types or ["sl"] * n
        costs = np.array([
            self.total_cost(sz, a, v, t, time_of_day, et)
            for sz, a, v, t, et
            in zip(order_sizes, adv, volatility, tickers, exit_types)
        ])
        net  = raw_returns - costs
        drag = costs.mean() / (abs(raw_returns.mean()) + 1e-9)
        if drag > 0.5:
            log.warning(
                f"[COST SEVERE] drag={drag:.2f}x gross={raw_returns.mean():.4f} "
                f"cost={costs.mean():.4f} net={net.mean():.4f}"
            )
        return net
