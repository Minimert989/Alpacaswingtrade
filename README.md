# QuantAlpaca — ML Swing Trading System

A machine learning–driven swing trading strategy running on Alpaca's paper trading API. Trades 30 large-cap US equities across Tech, Financials, Healthcare, Energy, and Consumer sectors.

---

## Backtest Results (Jul 2025 – Feb 2026, out-of-sample)

| Metric | Value |
|---|---|
| Trades | 202 over 153 trading days |
| Win rate | 59.4% gross / 58.4% net |
| Avg return | +1.55% gross / +1.25% net per trade |
| Total P&L | **+$13,260 (+13.26%)** on $100,000 |
| Annualized return | **+21.84%** |
| Sharpe ratio | **1.96** |
| Max drawdown | **-2.29%** (on equity curve) |
| Profit factor | 2.01 |

> Costs include 0.1% commission + 0.05% slippage per leg. Position sizing uses Kelly criterion capped at 5% per trade.

---

## Architecture

```
yfinance / Alpaca Data API
        │
        ▼
   DataLoader  ──────────────────────────────────────────────┐
   (1d OHLCV + 1h OHLCV + market context: SPY, VIX, sectors) │
        │                                                      │
        ▼                                                      │
  features.py                                                  │
  30+ technical indicators per ticker:                         │
  • Trend: SMA200, EMA slope, MACD, MA cross (20/60)          │
  • Momentum: RSI(14), CCI, Aroon, ADX                        │
  • Volatility: Bollinger Bands, HV20d                         │
  • Cross-sectional: CS momentum rank vs. sector ETF          │
  • Macro: SPY MA200 regime, VIX regime                       │
        │                                                      │
        ▼                                                      │
  PrimaryModel  (rule-based filter)                            │
  • ADX ≥ 15                                                   │
  • close > SMA200 (trend)                                     │
  • SPY > MA200 (bull market regime)                           │
  • CS momentum rank ≥ 0.3                                     │
  • MA20 / MA60 cross (entry timing)                           │
        │ signal ∈ {+1, 0, -1}                                │
        ▼                                                      │
  Triple Barrier Labeling  (Lopez de Prado)                   │
  • TP = +5%,  SL = -3%,  max_hold = 15 days                 │
        │                                                      │
        ▼                                                      │
  MetaModel  (soft-voting ensemble)                            │
  • LightGBM  (n=500, lr=0.01)                                │
  • XGBoost   (n=300, lr=0.05)                                │
  • LogisticRegression  (balanced)                             │
  • PurgedKFold CV (5 folds, 1% embargo)                      │
        │ P(win | features)                                    │
        ▼                                                      │
  PlattCalibrator  (LogisticRegression on logit(raw_prob))    │
        │ calibrated probability                               │
        ▼                                                      │
  Signal Filters                                               │
  • Volume ≥ 0.5× 20d avg                                     │
  • Earnings blackout ±5 days                                  │
  • VIX regime: +0.05 threshold in high-VIX environment       │
        │ prob ≥ threshold (0.62)                              │
        ▼                                                      │
  Executor  (Alpaca bracket orders)                            │
  • Kelly position sizing (max 5% per trade, max 60% exposure)│
  • Take-profit: +5%,  Stop-loss: -3%                         │
  • Runs daily at 10:00 AM ET                                  │
```

---

## Universe (30 Tickers)

| Sector | Tickers |
|---|---|
| Tech / Software | AAPL, MSFT, NVDA, GOOGL, META, AMD, AVGO, ORCL, CRM, ADBE, MU, QCOM, NOW, PANW |
| Consumer | AMZN, TSLA, NFLX, HD, WMT, COST, MCD, PG |
| Financials | JPM, V, MA, BAC, GS |
| Healthcare | UNH, LLY |
| Energy | XOM |

---

## Project Structure

```
QuantAlpaca/
├── config.yaml              # Parameters + Alpaca credentials (fill in your own)
├── requirements.txt
└── src/
    ├── train.py             # Full retrain entry point
    ├── train_production.py  # monthly_retrain(), weekly_threshold_refresh()
    ├── main.py              # Live daily trading loop
    ├── backtest.py          # Held-out test set backtest (dollar P&L + equity curve)
    ├── pipeline.py          # Quarterly audit
    ├── data_loader.py       # yfinance fetch + parquet cache
    ├── features.py          # Feature engineering (30+ indicators)
    ├── primary_model.py     # Rule-based pre-filter
    ├── meta_model.py        # LightGBM + XGBoost + LR ensemble + PlattCalibrator
    ├── labeling.py          # Triple barrier labels + sample weights
    ├── signals.py           # Live signal generation
    ├── executor.py          # Alpaca order execution
    ├── cost_model.py        # Commission + slippage model
    └── report.py            # Threshold optimization + metrics
```

---

## Setup

**1. Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure credentials**

Copy `config.yaml` and fill in your Alpaca paper trading credentials:
```yaml
alpaca_key: YOUR_ALPACA_PAPER_KEY
alpaca_secret: YOUR_ALPACA_PAPER_SECRET
paper: true
```

Get free paper trading credentials at [alpaca.markets](https://alpaca.markets).

**3. Train the model**
```bash
cd /path/to/QuantAlpaca
python src/train.py
```
Downloads ~4 years of daily + 1h data, trains the ensemble, saves artifacts to `models/production/`.

**4. Run the backtest**
```bash
python src/backtest.py
```
Evaluates on the held-out test set (last 15% by date). Shows dollar P&L, equity curve, per-ticker breakdown.

**5. Start live paper trading**
```bash
python src/main.py
```
Or set up a scheduled task to run daily at market open (10 AM ET).

---

## Retraining Schedule

| Task | Frequency | Command |
|---|---|---|
| Full retrain | Monthly (1st Sunday) | `python src/train.py` |
| Threshold refresh | Weekly (Sunday) | `train_production.weekly_threshold_refresh()` |
| Test set audit | Quarterly | `pipeline.evaluate()` |

---

## Key Design Decisions

- **Time-based train/test split**: `pd.concat(frames).sort_index()` ensures all tickers appear in the test set, preventing look-ahead bias from ticker-sequential splits.
- **PurgedKFold**: Prevents leakage between overlapping barrier windows during cross-validation.
- **Raw ensemble probs**: IsotonicRegression and naive Platt scaling both compress probability spread. PlattCalibrator uses `logit(raw_prob)` as input for proper scaling.
- **Volume filter at 0.5× avg**: Avoids illiquid days without blocking too many signals.
- **SPY MA200 regime**: Less restrictive than MA60; avoids missing early bull market signals.

---

## Disclaimer

This project is for **educational and research purposes only**. It runs on a paper trading account with simulated money. Past backtest performance does not guarantee future results. Do not use this system with real capital without thorough independent validation.
