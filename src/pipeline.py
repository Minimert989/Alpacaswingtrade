"""
test set은 여기서만 사용.
재학습에 절대 포함 금지.
"""
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression
from features import split_inputs
from labeling import (compute_sample_weights, barrier_to_exit_type,
                      attach_barrier_to_features)
from meta_model import MetaModel, calibrate, build_oof_probs
from cost_model import TransactionCostModel
from report import (optimize_threshold_economic,
                    regime_performance_report,
                    probability_bucket_report)

log = logging.getLogger(__name__)


def evaluate(features_df_all, primary_signals_all, config):
    """
    features_df_all: barrier_* 컬럼 포함
                     (attach_barrier_to_features 호출 후 전달)
    primary_signals_all: PrimaryModel.predict() 결과
    """
    from labeling import make_meta_labels
    y_meta_all = make_meta_labels(primary_signals_all, features_df_all)

    # active & NaN 제거 행에 맞게 정렬
    idx = y_meta_all.index
    features_df_all     = features_df_all.loc[idx]
    primary_signals_all = primary_signals_all.loc[idx]

    _, X_all = split_inputs(features_df_all)
    X_all    = X_all.values

    n     = len(X_all)
    t_end = int(n * config["train_ratio"])
    c_end = int(n * (config["train_ratio"] + config["calib_ratio"]))

    X_train, y_train = X_all[:t_end],  y_meta_all.values[:t_end]
    X_calib, y_calib = X_all[t_end:c_end], y_meta_all.values[t_end:c_end]
    X_test,  y_test  = X_all[c_end:],  y_meta_all.values[c_end:]

    feat_c = features_df_all.iloc[t_end:c_end]
    feat_t = features_df_all.iloc[c_end:]
    psig_c = primary_signals_all.iloc[t_end:c_end]
    psig_t = primary_signals_all.iloc[c_end:]

    cost_model = TransactionCostModel(
        commission=config["commission_rate"],
        slippage_model=config["slippage_model"],
        conservative=config["slippage_conservative"],
    )

    sw_train   = compute_sample_weights(y_meta_all.iloc[:t_end])
    meta_model = MetaModel()
    meta_model.train(X_train, y_train, sample_weight=sw_train)

    oof_probs  = build_oof_probs(meta_model, X_train, y_train, sw_train)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probs, y_train)

    raw_c  = meta_model.predict_proba_positive(X_calib)
    cal_c  = calibrate(calibrator, raw_c)
    et_c   = barrier_to_exit_type(feat_c["barrier_label"].values, psig_c.values)

    best_thresh = optimize_threshold_economic(
        cal_c, psig_c,
        feat_c["close"], feat_c["barrier_exit_price"],
        feat_c["barrier_label"],
        feat_c["adv_20d"].values, feat_c["hv_20d"].values,
        feat_c["ticker"].values, et_c,
        cost_model,
        capital=config["capital"], config=config,
        min_trades=config["threshold_weekly_min_trades"],
    )

    raw_t  = meta_model.predict_proba_positive(X_test)
    cal_t  = calibrate(calibrator, raw_t)
    et_t   = barrier_to_exit_type(feat_t["barrier_label"].values, psig_t.values)

    probability_bucket_report(
        cal_t, psig_t,
        feat_t["close"], feat_t["barrier_exit_price"],
        feat_t["barrier_label"],
        feat_t["adv_20d"].values, feat_t["hv_20d"].values,
        feat_t["ticker"].values, et_t, cost_model,
        min_bucket_trades=config["min_bucket_trades"],
    )

    regime_performance_report(
        cal_t, psig_t, feat_t, best_thresh, cost_model, et_t,
        min_trades=config["regime_min_trades"],
    )

    log.info(f"[EVAL] Best threshold: {best_thresh:.3f}")
    return best_thresh
