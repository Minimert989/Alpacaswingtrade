import pandas as pd
import numpy as np


class PrimaryModel:
    def __init__(self, config):
        self.adx_threshold = config["primary_adx_threshold"]
        self.cs_min_rank   = config["cs_momentum_min_rank"]

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        """입력: features_df. 출력: {+1, -1, 0}"""
        f   = features_df
        sig = pd.Series(0, index=f.index)

        trending    = f["adx_14"]    > self.adx_threshold
        above_ma200 = f["close"]     > f["sma_200"]
        bull_market = f["spy_price"] > f["spy_ma200"]   # relaxed: MA200 vs MA60
        no_earnings = ~f["earnings_proximity"].astype(bool)
        cs_eligible = f["cs_momentum_rank"] >= self.cs_min_rank

        long_cond = (
            trending & above_ma200 & bull_market &
            no_earnings & cs_eligible &
            (f["ma_cross_20_60"] > 0)
        )
        short_cond = (
            trending & ~above_ma200 & ~bull_market &
            no_earnings &
            (f["cs_momentum_rank"] <= (1 - self.cs_min_rank)) &
            (f["ma_cross_20_60"] < 0)
        )
        sig[long_cond]  = 1
        sig[short_cond] = -1
        return sig
