"""阶梯快充协议搜索：把"恒定倍率"升级为"SOC 分段阶梯"（深化 D2 的实质）。

为什么阶梯能赢
--------------
损伤密度 δ(s,r) = α·r^p·exp(β·s) 对 SOC 是指数敏感的（本数据集拟合 β ≈ 2.0）。
这意味着同样 1 个 C 的倍率，在 SOC 0.2 处造成的损伤只有 SOC 0.9 处的 e^{-1.4} ≈ 25%。

→ 最优结构必然是 **低 SOC 段猛充、高 SOC 段收油门**。
恒定倍率协议被"最苛刻的那一段"绑死，阶梯协议可以把倍率分配出去。

搜索空间
--------
把 [soc_lo, soc_hi] 切成 K 段（K=1..5），每段倍率 ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3}，
枚举所有组合（K=5 时 8^5=32768，可全枚举）。约束：
    · 每段倍率 ≤ 该 SOC 区间的析锂安全上限 rate_max(s, T)
    · 段间倍率单调不增（工程实现简单，且最优解必然如此——高 SOC 更该降速）

目标：在给定 SOC 窗口下，最小化「充满 1 FEC 的时长」同时最大化 EFC 寿命。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .chargetime import charge_time_h, stage_time_h
from .damage import damage_stages, efc_stages

#: 候选倍率档位
RATE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def make_stages(soc_lo: float, soc_hi: float, n_stages: int,
                rates: tuple[float, ...]) -> list[tuple[float, float, float]]:
    """按等 SOC 间隔切段，返回 [(rate, s0, s1), ...]。"""
    edges = np.linspace(soc_lo, soc_hi, n_stages + 1)
    return [(float(rates[i]), float(edges[i]), float(edges[i + 1])) for i in range(n_stages)]


def cascade_time(stages, params: dict) -> float:
    """阶梯协议总时长 = 各段 CC + 一次 CV 尾巴。"""
    if not stages:
        return 0.0
    span = stages[-1][2] - stages[0][1]
    t = sum(stage_time_h(r, a, b, params) for r, a, b in stages)
    return t + params["c_cv"] * span + params["d_tail"]


def _nonincreasing(rates) -> bool:
    return all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))


def enumerate_pareto(soc_lo: float, soc_hi: float, damage_pars: dict, ct_params: dict,
                     max_stages: int = 5, rate_grid=RATE_GRID,
                     rate_limit_fn=None, min_stage_span: float = 0.05) -> pd.DataFrame:
    """枚举全阶梯协议空间，返回 (时长, EFC) 与损伤分解。

    rate_limit_fn(soc_mid) -> 该处最大安全倍率（None 表示不设约束）
    """
    span = soc_hi - soc_lo
    rows = []
    for k in range(1, max_stages + 1):
        if span / k < min_stage_span:
            break
        edges = np.linspace(soc_lo, soc_hi, k + 1)
        for rates in itertools.combinations_with_replacement(rate_grid, k):
            rt = tuple(sorted(rates, reverse=True))   # 单调不增
            if not _nonincreasing(rt):
                continue
            ok = True
            if rate_limit_fn is not None:
                for i, r in enumerate(rt):
                    s_mid = (edges[i] + edges[i + 1]) / 2
                    if r > rate_limit_fn(s_mid) + 1e-9:
                        ok = False
                        break
            if not ok:
                continue
            stages = make_stages(soc_lo, soc_hi, k, rt)
            t = cascade_time(stages, ct_params)
            efc = efc_stages(stages, damage_pars)
            rows.append({
                "n_stages": k,
                "rates": "|".join(f"{r:g}" for r in rt),
                "soc_lo": soc_lo,
                "soc_hi": soc_hi,
                "time_h": t,
                "time_per_fec_h": t / span,
                "pred_efc": efc,
                "efc_per_hour": span / t,
                "peak_rate": max(rt),
                "mean_rate": float(np.mean(rt)),
                "rate_drop": float(max(rt) - min(rt)),
            })
    return pd.DataFrame(rows)


def pareto_non_dominated(df: pd.DataFrame, time_col: str = "time_per_fec_h",
                         life_col: str = "pred_efc") -> pd.DataFrame:
    """两目标（时长最小、寿命最大）的非支配筛选。"""
    pts = df[[time_col, life_col]].to_numpy(float)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        dominated = (pts[:, 0] <= pts[i, 0]) & (pts[:, 1] >= pts[i, 1]) & (
            (pts[:, 0] < pts[i, 0]) | (pts[:, 1] > pts[i, 1]))
        if dominated.any():
            keep[i] = False
    return df[keep].sort_values(time_col).reset_index(drop=True)


def best_multistage_vs_constant(df: pd.DataFrame, cst: pd.DataFrame,
                                time_col: str = "time_per_fec_h",
                                life_col: str = "pred_efc") -> pd.DataFrame:
    """在同一 SOC 窗口下，比较"最优阶梯" vs "最优恒定倍率"，量化阶梯的净收益。"""
    best_ms = df.sort_values(time_col).iloc[0]
    # 恒定倍率基线：在 df 里筛 n_stages==1 的最快
    c1 = cst[cst["n_stages"] == 1].sort_values(time_col)
    best_c1 = c1.iloc[0]
    # 同寿命约束下比速度：阶梯在"不低于恒定最优寿命"的前提下能快多少
    lc = best_c1[life_col]
    ms_fast = df[df[life_col] >= lc].sort_values(time_col).iloc[0] if (df[life_col] >= lc).any() else best_ms
    return pd.DataFrame([
        {"scheme": "最优恒定倍率", "rates": best_c1["rates"], "time_per_fec_h": best_c1[time_col],
         "pred_efc": best_c1[life_col], "n_stages": 1},
        {"scheme": "最优阶梯（同寿命更快）", "rates": ms_fast["rates"],
         "time_per_fec_h": ms_fast[time_col], "pred_efc": ms_fast[life_col],
         "n_stages": int(ms_fast["n_stages"])},
        {"scheme": "最优阶梯（绝对最快）", "rates": best_ms["rates"],
         "time_per_fec_h": best_ms[time_col], "pred_efc": best_ms[life_col],
         "n_stages": int(best_ms["n_stages"])},
    ]).assign(
        speedup_pct_vs_constant=lambda d: (1 - d["time_per_fec_h"] / d["time_per_fec_h"].iloc[0]) * 100,
        life_delta_pct_vs_constant=lambda d: (d["pred_efc"] / d["pred_efc"].iloc[0] - 1) * 100,
    )


def decompose(rates: str, soc_lo: float, soc_hi: float, damage_pars: dict) -> pd.DataFrame:
    """把一条阶梯协议的损伤逐段拆开（回答"哪段最伤"）。"""
    rt = [float(x) for x in rates.split("|")]
    stages = make_stages(soc_lo, soc_hi, len(rt), tuple(rt))
    from .damage import stage_damage_breakdown
    return stage_damage_breakdown(stages, damage_pars)
