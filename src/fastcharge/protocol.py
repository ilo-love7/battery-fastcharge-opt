"""协议特征工程：把 63 种充电协议变成可回归的结构化特征。

设计要点
--------
1. **只用协议可达的信息**：真实场景下"给定协议"就能算出这些量；不偷看寿命结局。
2. **峰倍率 + SOC 位置交互**：文献共识是"高倍率落在高 SOC 最伤"，单看峰值倍率抓不到。
3. **时间代价显式建模**：chargetime 是优化目标之一，必须能从协议参数直接算出。

特征分三族:
    R 族 rate    : 峰值/均值/中位倍率、高倍率段占比、倍率变化幅度
    S 族 soc     : SOC 窗口、窗口起点、峰值倍率所在 SOC、高 SOC 高倍率暴露
    T 族 thermal : 倍率×时间暴露（产热代理）、平均充电倍率对应的理论时长
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 高 SOC 阈值（多数 NMC 文献把 4.0 V 以上视作高 SOC，对应 ~70% SOC）
HIGH_SOC_THRESHOLD = 0.70

#: 判定"高倍率"的阈（C）
HIGH_RATE_THRESHOLD = 1.5


def protocol_features(row: pd.Series | dict) -> dict[str, float]:
    """单协议 → 特征字典。输入可以是 cells 表的一行。"""
    r = dict(row) if not isinstance(row, dict) else row
    rate_peak = float(r.get("charge_rate_peak", np.nan))
    rate_min = float(r.get("charge_rate_min", np.nan))
    rate_mean = float(r.get("charge_rate_mean", np.nan))
    rate_med = float(r.get("charge_rate_median", np.nan))
    soc_lo = float(r.get("soc_lo", 0.0))
    soc_hi = float(r.get("soc_hi", 1.0))
    span = float(r.get("soc_span", max(soc_hi - soc_lo, 1e-6)))
    peak_soc = float(r.get("peak_rate_soc_start", np.nan))
    dis_rate = float(r.get("discharge_rate_mean", np.nan))

    feats: dict[str, float] = {}

    # ---- R 族 ----
    feats["f_rate_peak"] = rate_peak
    feats["f_rate_mean"] = rate_mean
    feats["f_rate_median"] = rate_med
    feats["f_rate_range"] = rate_peak - rate_min if np.isfinite(rate_min) else 0.0
    feats["f_rate_is_cc"] = float(bool(r.get("charge_rate_is_constant", True)))
    feats["f_high_rate_flag"] = float(rate_peak >= HIGH_RATE_THRESHOLD)

    # ---- S 族 ----
    feats["f_soc_lo"] = soc_lo
    feats["f_soc_span"] = span
    feats["f_soc_start_deficit"] = 1.0 - soc_lo  # 需要补多少电
    # 峰值倍率起点：若协议只有一段，就是窗口起点
    feats["f_peak_rate_soc"] = peak_soc if np.isfinite(peak_soc) else soc_lo
    # 高 SOC 段（>0.7）是否被高倍率覆盖 —— 关键交互项
    feats["f_highrate_highsoc"] = float(
        np.isfinite(rate_peak) and rate_peak >= HIGH_RATE_THRESHOLD and feats["f_peak_rate_soc"] >= HIGH_SOC_THRESHOLD
    )
    feats["f_highsoc_exposure"] = max(soc_hi - max(soc_lo, HIGH_SOC_THRESHOLD), 0.0) * (
        rate_peak if np.isfinite(rate_peak) else 0.0
    )

    # ---- T 族 ----
    # 名义充电时长（小时）: 各段 = SOC 跨度 / 倍率，全部满窗口近似
    if np.isfinite(rate_peak) and rate_peak > 0:
        feats["f_nominal_charge_time_h"] = span / rate_mean if np.isfinite(rate_mean) and rate_mean > 0 else span / rate_peak
    else:
        feats["f_nominal_charge_time_h"] = np.nan
    feats["f_c_rate_squared"] = rate_peak ** 2 if np.isfinite(rate_peak) else np.nan  # 产热 ~ I²R
    feats["f_discharge_rate"] = dis_rate
    # 充放电倍率失配（充电快于放电 → 充电侧可能更热）
    feats["f_charge_minus_discharge_rate"] = (rate_mean - dis_rate) if (np.isfinite(rate_mean) and np.isfinite(dis_rate)) else np.nan

    return feats


def build_feature_matrix(cells: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """cells 表 → (特征矩阵, 特征名列表)。索引与 cells 对齐。"""
    feats = [protocol_features(row) for _, row in cells.iterrows()]
    X = pd.DataFrame(feats, index=cells.index)
    names = list(X.columns)
    return X, names


# --------------------------------------------------------------------------
# 目标变量
# --------------------------------------------------------------------------
def log_lifetime(cells: pd.DataFrame, *, censored_value: float | None = None) -> pd.Series:
    """寿命结局 → log(EOL 循环数)。

    删失电芯（跑完测试仍未跌破 80% SOH）只有下界，没有真实结局。
    基础版处理：给一个保守下界（用其实际循环数），并单列 censored 掩码供分层报告。
    """
    eol = cells["eol_cycle"].astype(float).copy()
    mask = cells["censored"].astype(bool)
    fill = censored_value if censored_value is not None else cells["n_cycles"].astype(float)
    eol[mask] = fill[mask]
    return np.log(eol.clip(lower=1.0))
