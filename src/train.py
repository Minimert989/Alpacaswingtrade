"""
First-time training script. Run once before starting live trading.

Usage (from project root):
    python src/train.py

What it does:
  1. Downloads 1,500 days of daily OHLCV + market context
  2. Downloads up to 729 days of 1h OHLCV (yfinance hard limit)
  3. Engineers features for all tickers
  4. Attaches triple-barrier labels PER TICKER (prevents cross-ticker leakage)
  5. Trains MetaModel with PurgedKFold OOF calibration
  6. Optimises decision threshold on calib set
  7. Saves model.pkl, calibrator.pkl, manifest.json
  8. Updates threshold in config.yaml
"""
import sys, logging, yaml
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from features import engineer
from primary_model import PrimaryModel
from labeling import attach_barrier_to_features
from train_production import monthly_retrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    loader  = DataLoader()
    primary = PrimaryModel(config)

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=1500)).strftime("%Y-%m-%d")

    log.info(f"Fetching daily data {start} → {end} ...")
    ohlcv   = loader.fetch(config["tickers"], start, end, "1d")
    context = loader.fetch_market_context(start, end)

    log.info("Fetching 1h data (last 729 days) ...")
    ohlcv_1h = loader.fetch_1h(config["tickers"], start, end)

    log.info("Engineering features ...")
    feat_dict = engineer(ohlcv, context, ohlcv_1h, config)

    # ── Per-ticker barrier labeling ──────────────────────────────────────────
    # Must run per-ticker: triple_barrier_label iterates prices sequentially,
    # so concatenating tickers first would compare prices across companies.
    log.info("Attaching triple-barrier labels per ticker ...")
    labeled_frames = []
    for ticker, feat_df in feat_dict.items():
        psigs = primary.predict(feat_df)
        feat_df = feat_df.copy()
        feat_df["primary_signal"] = psigs
        try:
            feat_df = attach_barrier_to_features(feat_df, config)
            labeled_frames.append(feat_df)
            log.info(f"  {ticker}: {len(feat_df)} labeled rows")
        except Exception as e:
            log.warning(f"  {ticker}: barrier labeling failed — {e}")

    if not labeled_frames:
        raise RuntimeError("No labeled data produced. Check data quality.")

    # Sort by date across all tickers before integer-indexing.
    # This gives a time-based train/test split instead of ticker-sequential.
    df_all = pd.concat(labeled_frames).sort_index().reset_index(drop=True)
    signals_all = df_all["primary_signal"]

    log.info(
        f"Total rows: {len(df_all)}  |  "
        f"Active signals: {(signals_all != 0).sum()}"
    )
    log.info("Starting monthly_retrain (this takes ~10-20 min) ...")

    model, calibrator, threshold = monthly_retrain(
        df_all, signals_all, config,
        barriers_attached=True,   # barriers already applied per-ticker above
    )

    log.info(f"Training complete. Threshold={threshold:.3f}")
    log.info("Artifacts saved to models/production/")


if __name__ == "__main__":
    main()
