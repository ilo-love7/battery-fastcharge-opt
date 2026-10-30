"""充电时长模型：从 240 只电芯的实测充电段反推"协议 → 充电时长"。

为什么必须单独建这个模型
------------------------
优化目标之一就是充电时长。若直接用"窗口宽度 / 倍率"算 CC 时长，会漏掉 CV 段
（ISU-ILCC 全部协议都是 CC-CV 到 4.2 V、截止 ~0.05 C）。忽略它会得出"倍率越高
越快、可以无限堆倍率"的错误结论。

从数据反推的形式（三参数线性）
------------------------------
    t_charge = a·span/rate  +  c·span  +  d
               └── CC 主体 ──┘   └ CV 电荷 ┘  └固定尾部┘

拟合结果（240 电芯，前 20 圈中位充电时长）:
    a≈0.87, c≈0.43 h/FEC, d≈0.17 h
    R² = 0.978, MAPE = 5.7%（纯 CC 假设只有 R²=0.665 / MAPE 27.5%）

物理含义 —— 这是本项目的重要工程结论:
    CV 阶段的时长几乎**与倍率无关**（c·span 项不含量率），相当于一个时长下限。
    倍率从 2C 推到 3C，CC 段从 0.5→0.33 h/span，但 CV 段仍要 0.43 h/span，
    总时长只从 0.93→0.76 h。**充电时长随倍率递减收益，存在硬地板**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def charge_time_h(span, rate, a: float, c: float, d: float):
    """协议 (SOC 窗口, 峰值倍率) → 预计充电时长（小时）。"""
    span = np.asarray(span, dtype=float)
    rate = np.asarray(rate, dtype=float)
    return a * span / rate + c * span + d


def fit_charge_time(cycles: pd.DataFrame, n_head: int = 20) -> dict:
    """用前 n_head 圈的实测充电时长拟合 (a, c, d)。"""
    head = cycles[cycles["cycle_index"] < n_head]
    d_ = (
        head.groupby(["cell_id", "group"])
        .agg(t_charge_s=("t_charge_s", "median"), span=("soc_span", "first"),
             rate=("charge_rate_peak", "first"),
             t_discharge_s=("t_discharge_s", "median"))
        .reset_index()
        .dropna()
    )
    d_ = d_[(d_["rate"] > 0) & (d_["span"] > 0) & (d_["t_charge_s"] > 0)].copy()

    span = d_["span"].to_numpy(float)
    rate = d_["rate"].to_numpy(float)
    y = d_["t_charge_s"].to_numpy(float) / 3600.0

    A = np.column_stack([span / rate, span, np.ones_like(span)])
    (a, c, d), *_ = np.linalg.lstsq(A, y, rcond=None)

    pred = charge_time_h(span, rate, a, c, d)
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mape = float(np.mean(np.abs(resid / y)) * 100)

    cc_only = span / rate
    r2_cc = 1 - float(np.sum((y - cc_only) ** 2)) / ss_tot
    mape_cc = float(np.mean(np.abs((y - cc_only) / y)) * 100)

    d_["measured_h"] = y
    d_["predicted_h"] = pred
    d_["cc_only_h"] = cc_only
    d_["cv_time_h"] = c * span + d
    d_["cv_share"] = d_["cv_time_h"] / pred

    return {
        "a_cc": float(a),
        "c_cv": float(c),
        "d_tail": float(d),
        "r2": float(r2),
        "rmse_h": rmse,
        "mape_pct": mape,
        "r2_cc_only": float(r2_cc),
        "mape_cc_only_pct": mape_cc,
        "n_points": int(len(y)),
        "median_ratio_to_cc": float(np.median(y / cc_only)),
        "cv_share_median": float(np.nanmedian(d_["cv_share"])),
        "cv_share_at_3C": float(np.nanmedian(d_["cv_share"][d_["rate"] >= 2.8])),
        "fit_table": d_,
    }


def time_savings_law(params: dict, span: float = 0.6, rates=(0.5, 1, 2, 3, 4, 6, 8)) -> pd.DataFrame:
    """扫描倍率 → 充电时长，量化"收益递减地板"。"""
    t = charge_time_h(span, np.array(rates, float), params["a_cc"], params["c_cv"], params["d_tail"])
    cc = params["a_cc"] * span / np.array(rates, float)
    cv = params["c_cv"] * span + params["d_tail"]
    df = pd.DataFrame(
        {
            "rate_C": list(rates),
            "charge_time_h": t,
            "cc_part_h": cc,
            "cv_part_h": cv,
            "cv_share": cv / t,
        }
    )
    df["time_vs_1C"] = t / t[0]
    df["marginal_gain_h"] = -np.diff(t, prepend=t[0])
    df["efc_per_hour"] = span / t
    return df


# --------------------------------------------------------------------------
# 多段阶梯协议：逐段时长与总时长
# --------------------------------------------------------------------------
def stage_time_h(rate: float, soc_from: float, soc_to: float, params: dict) -> float:
    """单段充电时长。

    段内电荷 = (soc_to - soc_from)，CC 时间按该段倍率算；CV 尾巴只在末尾出现，
    因此每段只有"段内 CC"，最后一段额外承担 CV。
    """
    dq = max(soc_to - soc_from, 0.0)
    if dq <= 0:
        return 0.0
    t_cc = params["a_cc"] * dq / rate
    return t_cc


def multistage_time_h(stages: list[tuple[float, float, float]], params: dict,
                      v_switch: float = 4.2) -> float:
    """多段阶梯协议总时长。stages = [(rate, soc_from, soc_to), ...] 按 SOC 升序。

    CV 尾巴只算一次，挂在最后一段（这是 CC-CV 协议的物理事实）。
    """
    if not stages:
        return 0.0
    total = sum(stage_time_h(r, a, b, params) for r, a, b in stages[:-1])
    r_last, a_last, b_last = stages[-1]
    total += stage_time_h(r_last, a_last, b_last, params)
    span = stages[-1][2] - stages[0][1]
    total += params["c_cv"] * span + params["d_tail"]
    return total


def cascade_time_h(rates_by_soc: list[tuple[float, float]], soc_lo: float, soc_hi: float,
                   params: dict) -> float:
    """阶梯协议：rates_by_soc = [(soc_start, rate), ...] 升序，最后一段到 soc_hi。"""
    stages = []
    for i, (s0, r) in enumerate(rates_by_soc):
        s1 = rates_by_soc[i + 1][0] if i + 1 < len(rates_by_soc) else soc_hi
        stages.append((r, s0, s1))
    return multistage_time_h(stages, params)


def biggest_gain_rate(params: dict, span: float = 0.6) -> dict:
    """收益递减律的定量表述：每多 1C 买回多少分钟，何时不值得。

    返回:
        - 边际收益曲线（每档倍率的分钟收益）
        - "值得的上限"：边际收益跌破 2 min / 档（经验门槛）时的倍率
        - 边际收益减半点
    """
    rates = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10], float)
    t = charge_time_h(span, rates, params["a_cc"], params["c_cv"], params["d_tail"])
    gain_min = -np.diff(t) * 60
    df = pd.DataFrame({
        "rate_C": rates[:-1],
        "time_h_at_rate": t[:-1],
        "gain_min_to_next_step": gain_min,
        "marginal_min_per_C": gain_min / np.diff(rates),
    })
    worth = df[df["marginal_min_per_C"] >= 2.0]["rate_C"].max()
    first_gain = gain_min[0]
    half_point = df[df["marginal_min_per_C"] <= first_gain / np.diff(rates)[0] / 2]["rate_C"].min()
    return {
        "table": df,
        "worthwhile_max_rate": float(worth) if np.isfinite(worth) else float(rates[-2]),
        "marginal_halving_rate": float(half_point) if np.isfinite(half_point) else np.nan,
        "span": span,
        "floor_h": float(params["c_cv"] * span + params["d_tail"]),
    }
