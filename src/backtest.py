"""
Backtest on held-out test set (last 15% of labeled data).
Uses the production model + calibrator already saved in models/production/.
Applies the same signal filters as live trading (volume, earnings, VIX).

Equity simulation uses realistic position sizing (Kelly fraction) and
round-trip transaction costs (commission + slippage).

Usage (from project root):
    python src/backtest.py
"""
import sys, logging, yaml, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from features import engineer, split_inputs
from primary_model import PrimaryModel
from meta_model import MetaModel, calibrate
from labeling import attach_barrier_to_features, make_meta_labels
from signals import generate_signal, is_bear_market

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtest")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SLIPPAGE_PER_LEG = 0.0005   # 0.05% per leg (conservative intraday estimate)


def _kelly_frac(prob, tp, sl, kelly_fraction, max_position_pct):
    b     = tp / sl
    kelly = max((prob * b - (1 - prob)) / b, 0.0)
    return min(kelly * kelly_fraction, max_position_pct)


def run_backtest(test_start_override=None, test_end_override=None, bear_demo=False):
    """
    test_start_override / test_end_override : str "YYYY-MM-DD"
        When set, df_test is filtered to this date range instead of the
        automatic last-15% split.  Useful for bear market validation on 2022.

    bear_demo : bool
        If True, prints a disclaimer that results are in-sample (the
        production model was trained on this period).
    """
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    log.info("Loading production model ...")
    meta_model = MetaModel()
    meta_model.load("models/production/model.pkl")
    calibrator = joblib.load("models/production/calibrator.pkl")

    loader  = DataLoader()
    primary = PrimaryModel(config)

    end   = datetime.today().strftime("%Y-%m-%d")
    # Extend window to cover 2022 bear market when running bear demo
    lookback_days = 1500 if not bear_demo else 1600
    start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    log.info(f"Fetching data {start} → {end} (uses cache if fresh) ...")
    ohlcv    = loader.fetch(config["tickers"], start, end, "1d")
    context  = loader.fetch_market_context(start, end)
    ohlcv_1h = loader.fetch_1h(config["tickers"], start, end)

    log.info("Engineering features ...")
    feat_dict = engineer(ohlcv, context, ohlcv_1h, config)

    log.info("Attaching barriers per ticker ...")
    labeled_frames = []
    for ticker, feat_df in feat_dict.items():
        psigs = primary.predict(feat_df)
        feat_df = feat_df.copy()
        feat_df["primary_signal"] = psigs
        try:
            feat_df = attach_barrier_to_features(feat_df, config)
            labeled_frames.append(feat_df)
            log.info(f"  {ticker}: {len(feat_df)} rows")
        except Exception as e:
            log.warning(f"  {ticker}: failed — {e}")

    if not labeled_frames:
        raise RuntimeError("No labeled data produced.")

    # Sort by date (time-based split); preserve date before dropping index
    df_all = pd.concat(labeled_frames).sort_index()
    df_all.insert(0, "date", df_all.index.normalize())
    df_all = df_all.reset_index(drop=True)

    y_meta = make_meta_labels(df_all["primary_signal"], df_all)
    idx    = y_meta.index
    df_all = df_all.loc[idx].reset_index(drop=True)

    n     = len(df_all)
    t_end = int(n * config["train_ratio"])
    c_end = int(n * (config["train_ratio"] + config["calib_ratio"]))

    # ── Test set selection ───────────────────────────────────────────────────
    if test_start_override or test_end_override:
        # Custom date range (e.g. 2022 bear market demo)
        df_test = df_all.copy()
        if test_start_override:
            df_test = df_test[df_test["date"] >= pd.Timestamp(test_start_override)]
        if test_end_override:
            df_test = df_test[df_test["date"] <= pd.Timestamp(test_end_override)]
        df_test = df_test.reset_index(drop=True)
    else:
        df_test = df_all.iloc[c_end:].copy().reset_index(drop=True)

    test_start = df_test["date"].min().date()
    test_end   = df_test["date"].max().date()
    n_cal_days = max((df_test["date"].max() - df_test["date"].min()).days, 1)
    n_trading_days = int(n_cal_days * 252 / 365)

    log.info(f"Test set: {len(df_test)} rows  "
             f"({test_start} → {test_end}, ~{n_trading_days} trading days)")

    _, X_test = split_inputs(df_test)
    threshold       = config["threshold"]
    threshold_short = config.get("threshold_short", 0.45)
    log.info(f"Threshold (long): {threshold}  |  Threshold (short): {threshold_short}")

    # ── Pre-compute calibrated probs for all test rows ─────────────────────
    all_probs, all_signals, all_rows = [], [], []
    for j in range(len(df_test)):
        feat_row = df_test.iloc[j].to_dict()
        X_row    = X_test.iloc[j].values
        psig     = int(feat_row.get("primary_signal", 0))
        if psig == 0:
            continue
        if feat_row.get("earnings_proximity", False):
            continue
        vol  = feat_row.get("volume", 0)
        avg  = feat_row.get("avg_vol_20d", 1)
        if vol < avg * 0.5:
            continue
        raw = meta_model.predict_proba_positive(X_row.reshape(1, -1))
        all_probs.append(float(raw[0]))  # raw ensemble prob
        all_signals.append(psig)
        all_rows.append(feat_row)

    if not all_probs:
        log.warning("No signals survived pre-filters.")
        return

    all_probs   = np.array(all_probs)
    all_signals = np.array(all_signals)
    print(f"\n  Signals after pre-filters: {len(all_probs)}")
    print(f"  Prob distribution: "
          f"p25={np.percentile(all_probs,25):.3f}  "
          f"p50={np.percentile(all_probs,50):.3f}  "
          f"p75={np.percentile(all_probs,75):.3f}  "
          f"p90={np.percentile(all_probs,90):.3f}")

    # ── Threshold grid (gross, for tuning reference) ────────────────────────
    print("\n  Threshold grid (gross, no costs):")
    print(f"  {'Thresh':>7}  {'N':>5}  {'WinRate':>8}  {'AvgRet':>8}  "
          f"{'Sharpe*':>8}")
    for t in np.arange(0.48, 0.66, 0.02):
        mask = all_probs >= t
        if mask.sum() < 5:
            break
        tr = []
        for i in np.where(mask)[0]:
            e = all_rows[i].get("close", np.nan)
            x = all_rows[i].get("barrier_exit_price", np.nan)
            if pd.isna(e) or pd.isna(x):
                continue
            tr.append((x - e) / (e + 1e-9) * all_signals[i])
        if not tr:
            continue
        tr = np.array(tr)
        sh = tr.mean() / (tr.std() + 1e-9) * np.sqrt(252)
        print(f"  {t:>7.2f}  {len(tr):>5}  {(tr>0).mean():>8.1%}  "
              f"{tr.mean():>8.4f}  {sh:>8.2f}")

    # ── CS Selection: group by date, pick top-N per direction ──────────────
    # This is the cross-sectional L/S layer:
    # On each day, rank all candidates by MetaModel prob, then:
    #   - Long:  top cs_long_n  by prob (highest conviction longs)
    #   - Short: top cs_short_n by prob (highest conviction shorts)
    # Combined with VIX stop: no new positions when VIX > vix_stop_threshold.
    from collections import defaultdict
    vix_stop     = config.get("vix_stop_threshold", 30)
    vix_caution  = config.get("vix_caution_threshold", 20)
    cs_long_n    = config.get("cs_long_n", 3)
    cs_short_n   = config.get("cs_short_n", 3)

    daily_candidates = defaultdict(list)
    n_vix_stopped = 0
    for prob, psig, feat_row in zip(all_probs, all_signals, all_rows):
        # VIX hard stop
        vix_val = feat_row.get("vix", 0)
        if vix_val > vix_stop:
            n_vix_stopped += 1
            continue
        # Threshold with VIX caution bump
        vix_bump = 0.03 if vix_val > vix_caution else 0.0
        if psig == -1:
            adj_thresh = threshold_short + vix_bump
        else:
            adj_thresh = threshold + vix_bump
        if prob < adj_thresh:
            continue
        date = feat_row.get("date")
        daily_candidates[date].append((prob, psig, feat_row))

    # CS selection: top-N per direction per day
    selected_trades = []
    for date in sorted(daily_candidates.keys()):
        day_cands = daily_candidates[date]
        longs  = sorted([c for c in day_cands if c[1] == +1], key=lambda x: -x[0])[:cs_long_n]
        shorts = sorted([c for c in day_cands if c[1] == -1], key=lambda x: -x[0])[:cs_short_n]
        selected_trades.extend(longs + shorts)

    n_after_cs = len(selected_trades)
    print(f"  VIX-stopped signals: {n_vix_stopped}  (VIX > {vix_stop})")
    print(f"  After CS selection:  {n_after_cs}  (top-{cs_long_n}L / top-{cs_short_n}S per day)")

    # ── Portfolio simulation on CS-selected trades ──────────────────────────
    capital     = float(config["capital"])
    commission  = config["commission_rate"]
    equity      = capital
    equity_ts   = [capital]   # equity after each closed trade
    trade_dates = [None]      # dates aligned with equity_ts

    trades = []
    for prob, psig, feat_row in selected_trades:

        entry = feat_row.get("close", np.nan)
        exit_ = feat_row.get("barrier_exit_price", np.nan)
        if pd.isna(entry) or pd.isna(exit_) or entry <= 0:
            continue

        signal = int(psig)

        # Kelly position sizing — use short-specific TP/SL for shorts
        if signal == -1:
            tp_k = config.get("short_tp", 0.08)
            sl_k = config.get("short_sl", 0.04)
        else:
            tp_k = config["tp"]
            sl_k = config["sl"]
        frac = _kelly_frac(prob, tp_k, sl_k,
                           config["kelly_fraction"], config["max_position_pct"])
        if frac <= 0:
            continue
        dollar_pos = equity * frac
        shares     = int(dollar_pos / entry)
        if shares == 0:
            continue

        # P&L
        gross_pnl   = shares * (exit_ - entry) * signal
        cost_entry  = shares * entry * (commission + SLIPPAGE_PER_LEG)
        cost_exit   = shares * exit_ * (commission + SLIPPAGE_PER_LEG)
        net_pnl     = gross_pnl - cost_entry - cost_exit

        equity += net_pnl
        equity_ts.append(equity)
        trade_dates.append(feat_row.get("date"))

        raw_ret = (exit_ - entry) / (entry + 1e-9) * signal
        net_ret = net_pnl / (shares * entry)   # net % on position size

        trades.append({
            "date":          str(feat_row.get("date", "?"))[:10],
            "ticker":        feat_row.get("ticker", "?"),
            "signal":        signal,
            "prob":          round(prob, 4),
            "entry":         round(entry, 2),
            "exit":          round(exit_, 2),
            "shares":        shares,
            "dollar_pos":    round(dollar_pos, 0),
            "gross_pnl":     round(gross_pnl, 2),
            "costs":         round(cost_entry + cost_exit, 2),
            "net_pnl":       round(net_pnl, 2),
            "raw_ret":       round(raw_ret, 4),
            "net_ret":       round(net_ret, 4),
            "barrier_label": feat_row.get("barrier_label", "?"),
            "vix":           round(feat_row.get("vix", np.nan), 2),
        })

    if not trades:
        log.warning(f"No trades at threshold={threshold}.")
        return

    df_trades = pd.DataFrame(trades)
    df_trades.to_csv(OUTPUT_DIR / "backtest_trades.csv", index=False)

    # ── SH Bear Market Overlay Simulation ───────────────────────────────────
    # Simulate holding SH (inverse S&P500) during bear market periods.
    # Bear = SPY < MA200 AND breadth_pct < breadth_min_long.
    # Capital allocated: bear_etf_alloc × portfolio equity at entry.
    bear_etf_alloc      = config.get("bear_etf_alloc", 0.25)
    bear_etf_exit_buf   = config.get("bear_etf_exit_buffer", 0.01)
    sh_pnl_total        = 0.0
    bear_periods        = []

    try:
        spy_series  = context["SPY"]["Close"]
        sh_series   = context["SH"]["Close"]
        spy_ma200_s = spy_series.rolling(200).mean()
        sh_daily_ret = sh_series.pct_change().fillna(0)

        # Build daily breadth_pct for test period
        breadth_by_date = (
            df_test.groupby("date")["breadth_pct"].mean()
            .rename(lambda d: pd.Timestamp(d))
        )

        in_bear        = False
        sh_entry_equity = 0.0
        sh_current     = 0.0   # dollar value of SH holding
        bear_start_date = None

        test_biz_days = pd.bdate_range(str(test_start), str(test_end))
        for day in test_biz_days:
            spy_p = spy_series.get(day)
            spy_m = spy_ma200_s.get(day)
            if spy_p is None or spy_m is None or pd.isna(spy_p) or pd.isna(spy_m):
                continue
            breadth = float(breadth_by_date.get(day, 0.5))

            bear_now = is_bear_market(float(spy_p), float(spy_m), breadth, config)

            if bear_now and not in_bear:
                # Enter SH position
                in_bear = True
                bear_start_date = day
                sh_current = equity * bear_etf_alloc   # use current equity at entry
                log.info(f"[SH] Bear mode ENTER on {day.date()}  "
                         f"  SPY={spy_p:.1f} < MA200={spy_m:.1f}")

            if in_bear:
                # Compound SH daily
                day_ret = float(sh_daily_ret.get(day, 0.0))
                if pd.isna(day_ret):
                    day_ret = 0.0
                sh_current *= (1 + day_ret)

            # Exit SH: SPY must recross MA200 with exit buffer
            if in_bear and not bear_now:
                exit_confirmed = float(spy_p) > float(spy_m) * (1 + bear_etf_exit_buf)
                if exit_confirmed:
                    period_pnl = sh_current - equity * bear_etf_alloc
                    sh_pnl_total += period_pnl
                    equity += period_pnl
                    bear_periods.append((bear_start_date, day, period_pnl))
                    log.info(f"[SH] Bear mode EXIT on {day.date()}  "
                             f"  Period P&L: ${period_pnl:+,.0f}")
                    in_bear = False
                    sh_current = 0.0

        # Close open SH position at end of test period if still in bear mode
        if in_bear and sh_current > 0:
            period_pnl = sh_current - equity * bear_etf_alloc
            sh_pnl_total += period_pnl
            equity += period_pnl
            bear_periods.append((bear_start_date, test_biz_days[-1], period_pnl))

    except Exception as e:
        log.warning(f"SH simulation failed: {e}")

    # ── Metrics ─────────────────────────────────────────────────────────────
    equity_arr  = np.array(equity_ts)
    total_pnl   = equity - capital
    total_ret   = total_pnl / capital
    annual_ret  = total_ret * (252 / max(n_trading_days, 1))

    # Proper max drawdown on equity curve
    eq_s  = pd.Series(equity_arr)
    peak  = eq_s.cummax()
    dd    = (eq_s - peak) / (peak + 1e-9)
    max_dd = float(dd.min())

    # Sharpe: annualize per-trade net returns by average hold period
    avg_hold = config.get("max_hold", 15) / 2   # rough mid-point
    net_rets = df_trades["net_ret"].values
    sharpe   = (net_rets.mean() / (net_rets.std() + 1e-9)
                * np.sqrt(252 / max(avg_hold, 1)))

    gross_rets = df_trades["raw_ret"].values
    wins  = net_rets > 0
    gross_wins = gross_rets > 0

    print("\n" + "=" * 60)
    if bear_demo:
        print("  ⚠️  BEAR MARKET DEMO  (IN-SAMPLE — 모델이 이 기간 데이터로 학습됨)")
        print("  성능 수치는 과적합 가능성 있음. 로직 검증 목적으로만 사용.")
    else:
        print("  BACKTEST RESULTS  (held-out test set, last 15%)")
    print(f"  Period: {test_start} → {test_end}  (~{n_trading_days} trading days)")
    print(f"  Strategy: CS L/S (top-{cs_long_n}L / top-{cs_short_n}S per day) + VIX stop @ {vix_stop}")
    print("=" * 60)
    print(f"  Trades:              {len(trades)}")
    print(f"  Win rate (gross):    {gross_wins.mean():.1%}")
    print(f"  Win rate (net):      {wins.mean():.1%}")
    print(f"  Avg gross ret:       {gross_rets.mean():.3%}  per trade")
    print(f"  Avg net ret:         {net_rets.mean():.3%}  per trade (after costs)")
    print(f"  Avg costs/trade:     {df_trades['costs'].mean():.2f} USD")
    print()
    print(f"  Starting capital:    ${capital:>10,.0f}")
    print(f"  Ending equity:       ${equity:>10,.0f}")
    print(f"  Total P&L:           ${total_pnl:>+10,.0f}")
    print(f"  Total return:        {total_ret:>+.2%}")
    print(f"  Annualized return:   {annual_ret:>+.2%}")
    print(f"  Sharpe (ann.)*:      {sharpe:.2f}  (* avg hold = {avg_hold:.0f}d)")
    print(f"  Max drawdown:        {max_dd:.2%}  (on equity curve)")
    pf = (net_rets[wins].sum() / (abs(net_rets[~wins].sum()) + 1e-9))
    print(f"  Profit factor:       {pf:.2f}")
    print()

    # ── Bear market overlay summary ──────────────────────────────────────────
    n_short_trades = (df_trades["signal"] == -1).sum()
    print(f"── Bear Market Mode ─────────────────────────────────────────")
    print(f"  Short trades (individual stocks): {n_short_trades}")
    if bear_periods:
        for bs, be, bpnl in bear_periods:
            print(f"  SH period: {bs.date()} → {be.date()}  P&L: ${bpnl:+,.0f}")
        print(f"  SH total P&L:  ${sh_pnl_total:+,.0f}")
    else:
        print(f"  SH overlay: no bear market in test period → $0 (code ready)")
    print()

    print("  By ticker:")
    ticker_stats = (
        df_trades.groupby("ticker")
        .agg(
            n=("net_pnl", "count"),
            net_pnl=("net_pnl", "sum"),
            avg_net_ret=("net_ret", "mean"),
            win_rate=("net_ret", lambda x: (x > 0).mean()),
        )
        .sort_values("net_pnl", ascending=False)
        .round({"net_pnl": 0, "avg_net_ret": 4, "win_rate": 3})
    )
    print(ticker_stats.to_string())

    print("\n  By barrier outcome:")
    barrier_stats = (
        df_trades.groupby("barrier_label")
        .agg(n=("net_pnl", "count"), avg_net_ret=("net_ret", "mean"))
        .round(4)
    )
    print(barrier_stats.to_string())

    print("=" * 60)
    print(f"\n  Trades saved  → {OUTPUT_DIR}/backtest_trades.csv")

    # ── Equity curve plot ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(13, 10))

        ax1, ax2, ax3 = axes

        # Equity curve in dollars
        ax1.plot(equity_arr, lw=1.5, color="steelblue")
        ax1.axhline(capital, color="gray", lw=0.8, ls="--")
        ax1.fill_between(range(len(equity_arr)),
                         capital, equity_arr,
                         where=equity_arr >= capital, alpha=0.2, color="green")
        ax1.fill_between(range(len(equity_arr)),
                         capital, equity_arr,
                         where=equity_arr < capital, alpha=0.2, color="red")
        ax1.set_title(f"Equity Curve — ${capital:,.0f} start  |  "
                      f"{len(trades)} trades  |  "
                      f"Total P&L: ${total_pnl:+,.0f} ({total_ret:+.1%})")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2.fill_between(range(len(dd)), dd.values, 0,
                         color="red", alpha=0.5)
        ax2.set_title(f"Drawdown  (max: {max_dd:.1%})")
        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.grid(True, alpha=0.3)

        # Per-trade net P&L
        colors = ["green" if p > 0 else "red" for p in df_trades["net_pnl"]]
        ax3.bar(range(len(trades)), df_trades["net_pnl"].values,
                color=colors, width=0.8)
        ax3.axhline(0, color="gray", lw=0.8, ls="--")
        ax3.set_title("Per-Trade Net P&L ($)")
        ax3.set_ylabel("Net P&L ($)")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = OUTPUT_DIR / "backtest_equity.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Equity curve  → {out_path}")
    except Exception as e:
        log.warning(f"Plot failed: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QuantAlpaca Backtest")
    parser.add_argument(
        "--bear-test", action="store_true",
        help="Run on 2022 bear market period (in-sample demo, logic validation only)"
    )
    args = parser.parse_args()

    if args.bear_test:
        print("\n[BEAR TEST] 2022년 bear market 구간으로 테스트합니다.")
        print("[BEAR TEST] 주의: 이 기간은 학습 데이터에 포함됨 → 성능 수치는 참고용")
        run_backtest(
            test_start_override="2022-03-01",
            test_end_override="2022-12-31",
            bear_demo=True,
        )
    else:
        run_backtest()
