"""
Triple Barrier Labeling.
- log-return 기반 배리어 (단순 % 아님)
- atr_series: 배리어 크기 scaling에 실제 사용
- Meta label 생성: barrier=0 구간 변동성 정규화
- features_df 병합 헬퍼 포함
"""
import numpy as np
import pandas as pd


def triple_barrier_label(prices: pd.Series,
                          atr_series: pd.Series,
                          tp: float = 0.03,
                          sl: float = 0.02,
                          max_hold: int = 10):
    """
    log-return 기반 배리어.
      log_tp = log(1 + tp)
      log_sl = log(1 + sl)  (하락 기준: -log_sl)
    atr_series: 현재 미사용이 아닌 실제 반영.
      배리어 크기를 고정 tp/sl 대신
      max(tp, 0.5 * ATR/price) 방식으로 종목별 동적 조정.
      → 저변동 구간에서 배리어가 너무 좁아지는 것 방지.
    반환:
      barrier_labels:   pd.Series {+1, -1, 0}
      barrier_exit_px:  pd.Series (실제 청산가 근사)
    """
    labels, exit_pxs = [], []
    log_prices = np.log(prices.values)
    for i in range(len(prices) - max_hold):
        entry   = prices.iloc[i]
        atr_val = atr_series.iloc[i]
        # 동적 배리어 (고정값과 ATR 기반 중 큰 값)
        dyn_tp = max(tp, 0.5 * atr_val / entry)
        dyn_sl = max(sl, 0.4 * atr_val / entry)
        log_tp = np.log(1 + dyn_tp)
        log_sl = np.log(1 + dyn_sl)  # 하락은 -log_sl

        future_log = log_prices[i+1 : i+max_hold+1] - log_prices[i]
        tp_mask = future_log >=  log_tp
        sl_mask = future_log <= -log_sl

        tp_hit = int(tp_mask.argmax()) if tp_mask.any() else None
        sl_hit = int(sl_mask.argmax()) if sl_mask.any() else None

        if tp_hit is not None and (sl_hit is None or tp_hit <= sl_hit):
            labels.append(1)
            exit_pxs.append(entry * (1 + dyn_tp))
        elif sl_hit is not None and (tp_hit is None or sl_hit < tp_hit):
            labels.append(-1)
            exit_pxs.append(entry * (1 - dyn_sl))
        else:
            labels.append(0)
            exit_pxs.append(prices.iloc[i + max_hold])

    idx = prices.index[:len(prices) - max_hold]
    return (
        pd.Series(labels,   index=idx, name="barrier_label"),
        pd.Series(exit_pxs, index=idx, name="barrier_exit_price"),
    )


def attach_barrier_to_features(features_df: pd.DataFrame,
                                 config: dict) -> pd.DataFrame:
    """
    features_df에 barrier_label, barrier_exit_price 컬럼 병합.
    학습/평가 파이프라인 진입 전 반드시 호출.
    운영(main.py)에서는 호출하지 않음.
    사용 예:
        features_df = attach_barrier_to_features(features_df, config)
    """
    prices = features_df["close"]
    atr    = features_df["atr_14"]
    barrier_labels, barrier_exit_px = triple_barrier_label(
        prices, atr,
        tp=config["tp"],
        sl=config["sl"],
        max_hold=config["max_hold"],
    )
    features_df = features_df.join(barrier_labels, how="left")
    features_df = features_df.join(barrier_exit_px, how="left")
    # max_hold 이후 행은 배리어 계산 불가 → 학습에서 제외
    features_df = features_df.dropna(subset=["barrier_label", "barrier_exit_price"])
    features_df["barrier_label"] = features_df["barrier_label"].astype(int)
    return features_df


def make_meta_labels(primary_signals: pd.Series,
                      features_df: pd.DataFrame) -> pd.Series:
    """
    반환: pd.Series {0.0, 1.0} — active & NaN 제거 행만.
    barrier=0 처리: 변동성 정규화 기준
      threshold = max(0.25 * ATR/close, 0.002)
    """
    barrier_labels = features_df["barrier_label"]
    prices_df      = features_df
    atr_series     = features_df["atr_14"]
    active = primary_signals != 0
    meta   = pd.Series(np.nan, index=primary_signals.index)
    for idx in primary_signals[active].index:
        if idx not in barrier_labels.index:
            continue
        primary = primary_signals[idx]
        barrier = barrier_labels[idx]
        if barrier == primary:
            meta[idx] = 1.0
        elif barrier == -primary:
            meta[idx] = 0.0
        else:
            i       = prices_df.index.get_loc(idx)
            end_i   = min(i + 10, len(prices_df) - 1)
            entry_p = prices_df["close"].iloc[i]
            exit_p  = prices_df["close"].iloc[end_i]
            dir_ret = (exit_p - entry_p) / (entry_p + 1e-9) * primary
            atr_val   = atr_series.iloc[i]
            threshold = max(0.25 * atr_val / (entry_p + 1e-9), 0.002)
            if   dir_ret >  threshold: meta[idx] = 1.0
            elif dir_ret < -threshold: meta[idx] = 0.0
            # else: NaN 유지 (학습 제외)
    return meta[active].dropna()


def compute_sample_weights(meta_labels: pd.Series) -> np.ndarray:
    return np.where(meta_labels.values == 1.0, 1.5, 1.0)


def barrier_to_exit_type(barrier_labels, primary_signals):
    """tp | sl | time"""
    result = []
    for b, p in zip(barrier_labels, primary_signals):
        if   b ==  p: result.append("tp")
        elif b == -p: result.append("sl")
        else:         result.append("time")
    return result
