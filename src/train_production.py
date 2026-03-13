import numpy as np
import joblib
import yaml
import logging
from datetime import datetime
from pathlib import Path
from features import split_inputs, MODEL_FEATURE_COLS
from labeling import (compute_sample_weights, barrier_to_exit_type,
                      attach_barrier_to_features, make_meta_labels)
from meta_model import (MetaModel, PlattCalibrator, calibrate,
                        build_oof_probs, save_manifest)
from cost_model import TransactionCostModel
from report import optimize_threshold_economic

log         = logging.getLogger(__name__)
MODEL_DIR   = Path("models/production")
CONFIG_PATH = Path("config.yaml")


def _cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _save_cfg(c):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(c, f, default_flow_style=False)


def update_config(key, value):
    c = _cfg(); c[key] = value; _save_cfg(c)


def monthly_retrain(features_df_all, primary_signals_all, config,
                    barriers_attached=False):
    """
    features_df_all: barrier 미포함이면 여기서 per-ticker attach.
                     barriers_attached=True이면 이미 부착된 상태.
    test set(마지막 15%)은 절대 건드리지 않음.
    """
    if barriers_attached:
        # train.py에서 이미 per-ticker 처리 + reset_index 완료
        feat = features_df_all.copy()
    else:
        feat = attach_barrier_to_features(features_df_all, config)
        feat["primary_signal"] = primary_signals_all.reindex(feat.index).fillna(0)

    y_meta = make_meta_labels(
        feat["primary_signal"], feat
    )
    idx  = y_meta.index
    feat = feat.loc[idx]
    _, X = split_inputs(feat)
    X    = X.values
    y    = y_meta.values

    n     = len(X)
    t_end = int(n * config["train_ratio"])
    c_end = int(n * (config["train_ratio"] + config["calib_ratio"]))

    X_train, y_train = X[:t_end], y[:t_end]
    X_calib          = X[t_end:c_end]
    feat_c           = feat.iloc[t_end:c_end]
    psig_c           = primary_signals_all.reindex(feat_c.index)

    cost_model = TransactionCostModel(
        commission=config["commission_rate"],
        slippage_model=config["slippage_model"],
        conservative=config["slippage_conservative"],
    )

    # Step 1: train set 학습
    sw1    = compute_sample_weights(y_meta.iloc[:t_end])
    model1 = MetaModel()
    model1.train(X_train, y_train, sample_weight=sw1)

    # Step 2: OOF calibrator
    oof1   = build_oof_probs(model1, X_train, y_train, sw1)
    calib1 = PlattCalibrator()
    calib1.fit(oof1, y_train)

    # Step 3: calib set으로 threshold 최적화
    raw_c  = model1.predict_proba_positive(X_calib)
    cal_c  = calibrate(calib1, raw_c)
    et_c   = barrier_to_exit_type(
        feat_c["barrier_label"].values, psig_c.values
    )
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

    # Step 4: train+calib 전체로 최종 재학습
    X_prod, y_prod = X[:c_end], y[:c_end]
    sw_prod        = compute_sample_weights(y_meta.iloc[:c_end])
    model_final    = MetaModel()
    model_final.train(X_prod, y_prod, sample_weight=sw_prod)

    # Step 5: calibrator도 train+calib OOF로 재fit
    oof_prod    = build_oof_probs(model_final, X_prod, y_prod, sw_prod)
    calib_final = PlattCalibrator()
    calib_final.fit(oof_prod, y_prod)

    # Step 6: 저장
    model_final.save(MODEL_DIR / "model.pkl")
    joblib.dump(calib_final,          MODEL_DIR / "calibrator.pkl")
    save_manifest(
        threshold       = best_thresh,
        prod_end_idx    = c_end,
        feature_cols    = MODEL_FEATURE_COLS,
        config_snapshot = config,
        note            = f"monthly_retrain {datetime.utcnow().date()}",
    )
    update_config("threshold",            best_thresh)
    update_config("threshold_base",       best_thresh)
    update_config("threshold_updated_at", datetime.utcnow().isoformat())
    log.info(f"[MONTHLY] Done. threshold={best_thresh:.3f} "
             f"prod_size={c_end}")
    return model_final, calib_final, best_thresh


def weekly_threshold_refresh(model, calibrator,
                              features_df_recent, primary_signals_recent,
                              config):
    """
    최근 60거래일 슬라이딩 윈도우.
    barrier 컬럼 포함된 features_df_recent 전달 필요.
    """
    cost_model = TransactionCostModel(
        commission=config["commission_rate"],
        slippage_model=config["slippage_model"],
        conservative=config["slippage_conservative"],
    )
    base_thresh = config["threshold_base"]
    active_mask = primary_signals_recent.values != 0
    n_active    = active_mask.sum()
    if n_active < config["threshold_weekly_min_trades"]:
        log.warning(
            f"[FREEZE] n_active={n_active} < "
            f"min={config['threshold_weekly_min_trades']}. "
            f"Keeping {base_thresh:.3f}"
        )
        return base_thresh

    feat_a = features_df_recent[active_mask]
    psig_a = primary_signals_recent[active_mask]
    _, X_a = split_inputs(features_df_recent)
    raw    = model.predict_proba_positive(X_a.values[active_mask])
    cal    = calibrate(calibrator, raw)
    et_a   = barrier_to_exit_type(
        feat_a["barrier_label"].values, psig_a.values
    )
    candidate = optimize_threshold_economic(
        cal, psig_a,
        feat_a["close"], feat_a["barrier_exit_price"],
        feat_a["barrier_label"],
        feat_a["adv_20d"].values, feat_a["hv_20d"].values,
        feat_a["ticker"].values, et_a,
        cost_model,
        capital=config["capital"], config=config,
        min_trades=config["threshold_weekly_min_trades"],
    )
    drift = candidate - base_thresh
    max_d = config["threshold_weekly_max_drift"]
    if abs(drift) > max_d:
        candidate = base_thresh + np.sign(drift) * max_d
        log.warning(
            f"[CLIP] candidate clipped: drift={drift:+.3f} "
            f"→ {candidate:.3f}"
        )
    update_config("threshold",            candidate)
    update_config("threshold_updated_at", datetime.utcnow().isoformat())
    log.info(f"[WEEKLY] {base_thresh:.3f} → {candidate:.3f}")
    return candidate
