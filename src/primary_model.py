import pandas as pd
import numpy as np


class PrimaryModel:
    def __init__(self, config):
        self.adx_threshold  = config["primary_adx_threshold"]
        self.cs_min_rank    = config["cs_momentum_min_rank"]
        # SMA200 distance buffer: require close > SMA200 * (1 + buffer) for longs
        # avoids range-bound stocks oscillating around SMA200
        self.sma200_buffer  = config.get("sma200_buffer", 0.02)
        # Market breadth: fraction of universe above SMA200
        # low breadth = choppy/bear market → suppress longs
        self.breadth_min_long = config.get("breadth_min_long", 0.35)

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """입력: features_df. 출력: {+1, -1, 0}"""
        f   = features_df
        sig = pd.Series(0, index=f.index)

        trending    = f["adx_14"]    > self.adx_threshold
        # SMA200 buffer: must be meaningfully above SMA200, not just touching it
        above_ma200 = f["close"]     > f["sma_200"] * (1 + self.sma200_buffer)
        bull_market = f["spy_price"] > f["spy_ma200"]   # relaxed: MA200 vs MA60
        no_earnings = ~f["earnings_proximity"].astype(bool)
        cs_eligible = f["cs_momentum_rank"] >= self.cs_min_rank
        # Breadth filter: require healthy market participation for longs
        broad_market_long  = f["breadth_pct"] >= self.breadth_min_long
        broad_market_short = f["breadth_pct"] <= (1 - self.breadth_min_long)

        long_cond = (
            trending & above_ma200 & bull_market &
            no_earnings & cs_eligible &
            broad_market_long &
            (f["ma_cross_20_60"] > 0)
        )
        # Short: below SMA200 with buffer, bear market, weak breadth
        below_ma200 = f["close"] < f["sma_200"] * (1 - self.sma200_buffer)
        short_cond = (
            trending & below_ma200 & ~bull_market &
            no_earnings &
            (f["cs_momentum_rank"] <= (1 - self.cs_min_rank)) &
            broad_market_short &
            (f["ma_cross_20_60"] < 0)
        )
        sig[long_cond]  = 1
        sig[short_cond] = -1
        return sig
