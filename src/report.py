import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


def max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.cummax()
    dd   = (cum_returns - peak) / (peak.abs() + 1e-9)
    return float(dd.min())


def _realized_returns(primary_signals, entry_prices, exit_prices,
                       adv, hv, tickers, exit_types, cost_model,
                       mask=None, capital=None, config=None):
    """
    mask:    boolean array (None = 전체)
    capital, config: position_size() 기반 order_size 계산용.
                     None이면 entry_price * 100 근사.
    """
    if mask is not None:
        psig = primary_signals.values[mask]
        ep   = entry_prices.values[mask]
        xp   = exit_prices.values[mask]
        a    = adv[mask]; v = hv[mask]; t = tickers[mask]
        et   = [exit_types[i] for i in np.where(mask)[0]]
    else:
        psig = primary_signals.values
        ep   = entry_prices.values
        xp   = exit_prices.values
        a    = adv; v = hv; t = tickers; et = exit_types

    raw = (xp - ep) / (ep + 1e-9) * psig

    # position size 기반 order_size (config 있을 때)
    if capital is not None and config is not None:
        from signals import position_size as _ps
        sizes = np.array([
            _ps(int(ps), 0.65, capital,
                vi * pi, pi,
                {"daily_pnl": 0, "total_exposure": 0,
                 "positions": {}},
                config) * pi
            for ps, pi, vi in zip(psig, ep, v)
        ], dtype=float)
        sizes = np.where(sizes == 0, ep * 100, sizes)  # fallback
    else:
        sizes = ep * 100   # 근사

    net = cost_model.adjust_returns(raw, sizes, a, v, t, et)
    return raw, net


def optimize_threshold_economic(cal_probs, primary_signals,
                                  entry_prices, exit_prices,
                                  barrier_outcomes, adv, hv,
                                  tickers, exit_types,
                                  cost_model,
                                  capital=None, config=None,
                                  min_trades=20,
                                  max_drawdown_limit=0.15):
    grid, results = np.arange(0.40, 0.76, 0.05), []
    for t in grid:
        mask = (cal_probs >= t) & (primary_signals.values != 0)
        if mask.sum() < min_trades:
            continue
        _, net = _realized_returns(
            primary_signals, entry_prices, exit_prices,
            adv, hv, tickers, exit_types, cost_model,
            mask, capital, config,
        )
        if len(net) < min_trades:
            continue
        sharpe = net.mean() / (net.std() + 1e-9) * np.sqrt(252)
        mdd    = max_drawdown(pd.Series(net).cumsum())
        if mdd < -max_drawdown_limit:
            continue
        results.append({
            "threshold": t, "n_trades": len(net),
            "net_sharpe": sharpe, "net_er": net.mean(),
            "max_dd": mdd, "win_rate": (net > 0).mean(),
        })
    if not results:
        log.warning("[THRESHOLD] No valid threshold. Default 0.60.")
        return 0.60
    df   = pd.DataFrame(results)
    best = df.loc[df["net_sharpe"].idxmax()]
    log.info(f"\nThreshold grid:\n{df.to_string(index=False)}")
    log.info(f"Selected: {best['threshold']:.2f} "
             f"Sharpe={best['net_sharpe']:.2f} n={int(best['n_trades'])}")
    return float(best["threshold"])


def probability_bucket_report(cal_probs, primary_signals,
                               entry_prices, exit_prices,
                               barrier_outcomes, adv, hv,
                               tickers, exit_types, cost_model,
                               min_bucket_trades=15):
    bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"]
    active     = primary_signals.values != 0
    probs_a    = cal_probs[active]
    bucket_col = pd.cut(probs_a, bins=bins, labels=labels, right=False)
    rows, mono_check = [], []
    for label in labels:
        mask_b = active.copy(); mask_b[active] = (bucket_col == label)
        n      = mask_b.sum()
        if n == 0:
            rows.append({"bucket": label, "n_trades": 0, "status": "empty"})
            continue
        if n < min_bucket_trades:
            rows.append({"bucket": label, "n_trades": n,
                         "status": f"WARN n<{min_bucket_trades}",
                         "net_avg_ret": np.nan, "net_sharpe": np.nan})
            continue
        raw, net = _realized_returns(
            primary_signals, entry_prices, exit_prices,
            adv, hv, tickers, exit_types, cost_model, mask_b,
        )
        se = net.std() / np.sqrt(n)
        rows.append({
            "bucket":      label, "n_trades": n, "status": "ok",
            "win_rate":    (raw > 0).mean().round(3),
            "gross_avg":   raw.mean().round(4),
            "net_avg_ret": net.mean().round(4),
            "stderr":      se.round(4),
            "ci95_lo":     (net.mean() - 1.96*se).round(4),
            "ci95_hi":     (net.mean() + 1.96*se).round(4),
            "net_sharpe":  (net.mean()/(net.std()+1e-9)*np.sqrt(252)).round(2),
            "cost_drag":   (raw.mean()-net.mean()).round(4),
        })
        mono_check.append((label, net.mean()))
    df = pd.DataFrame(rows)
    df.to_csv("outputs/prob_bucket_report.csv", index=False)
    print("\n=== Probability Bucket Report ===")
    print(df.to_string(index=False))
    if len(mono_check) >= 2:
        vals = [v for _, v in mono_check]
        if not np.all(np.diff(vals) >= -0.001):
            log.warning(
                "[MONOTONICITY BROKEN] "
                f"Buckets checked: {[l for l,_ in mono_check]}. "
                "Re-check calibration."
            )
    else:
        log.warning(
            f"[MONOTONICITY SKIP] Only {len(mono_check)} valid bucket(s)."
        )
    return df


def regime_performance_report(cal_probs, primary_signals,
                               features_df, threshold,
                               cost_model, exit_types, min_trades=5):
    active  = (primary_signals.values != 0) & (cal_probs >= threshold)
    feat_a  = features_df[active].copy()
    psig_a  = primary_signals.values[active]
    ep_a    = features_df["close"].values[active]
    xp_a    = features_df["barrier_exit_price"].values[active]
    adv_a   = features_df["adv_20d"].values[active]
    hv_a    = features_df["hv_20d"].values[active]
    tick_a  = features_df["ticker"].values[active]
    et_a    = [exit_types[i] for i in np.where(active)[0]]

    raw, net = _realized_returns(
        pd.Series(psig_a), pd.Series(ep_a), pd.Series(xp_a),
        adv_a, hv_a, tick_a, et_a, cost_model,
    )
    feat_a["_net_ret"] = net
    feat_a["vix_regime_label"] = pd.cut(
        feat_a["vix"], bins=[0, 15, 25, 999], labels=["low", "mid", "high"]
    )
    feat_a["market_regime"] = np.where(
        feat_a["spy_price"] > feat_a["spy_ma60"], "bull", "bear"
    )
    feat_a["blackout_label"] = feat_a["earnings_proximity"].map(
        {True: "blackout", False: "normal"}
    )

    segments = [
        ("vix_regime_label", ["low", "mid", "high"]),
        ("market_regime",    ["bull", "bear"]),
        ("blackout_label",   ["normal", "blackout"]),
    ]
    report = {}
    for col, groups in segments:
        for g in groups:
            sub = feat_a[feat_a[col] == g]["_net_ret"]
            if len(sub) < min_trades:
                continue
            report[f"{col}={g}"] = {
                "n_trades":      len(sub),
                "win_rate":      (sub > 0).mean().round(3),
                "avg_ret":       sub.mean().round(4),
                "sharpe":        (sub.mean()/(sub.std()+1e-9)*np.sqrt(252)).round(2),
                "max_dd":        round(max_drawdown(sub.cumsum()), 4),
                "profit_factor": (sub[sub>0].sum()
                                  / (abs(sub[sub<0].sum())+1e-9)).round(2),
            }

    all_r = feat_a["_net_ret"]
    report["ALL"] = {
        "n_trades":      len(all_r),
        "win_rate":      (all_r > 0).mean().round(3),
        "avg_ret":       all_r.mean().round(4),
        "sharpe":        (all_r.mean()/(all_r.std()+1e-9)*np.sqrt(252)).round(2),
        "max_dd":        round(max_drawdown(all_r.cumsum()), 4),
        "profit_factor": (all_r[all_r>0].sum()
                          / (abs(all_r[all_r<0].sum())+1e-9)).round(2),
    }

    rdf = pd.DataFrame(report).T
    rdf.to_csv("outputs/regime_performance.csv")
    print("\n=== Regime Performance ===")
    print(rdf.to_string())
    return rdf
