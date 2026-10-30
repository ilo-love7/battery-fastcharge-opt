"""损伤密度模型：把"恒定倍率 → EFC 寿命"拆成可按 SOC 积分的损伤密度。

为什么需要它
------------
代理模型（GP/树）只能回答"**恒定** span+rate 下寿命多少"。但真实快充协议是
**阶梯**的：低 SOC 段用大倍率、高 SOC 段降倍率。要评估这种协议，必须有
"局部损伤"的概念，才能沿 SOC 积分出来。

模型形式（三参数，可解析积分）
-----------------------------
    δ(s, r) = α · r^p · exp(β·s)          [每单位 FEC 的寿命损耗]
    D(协议) = ∫_{s_lo}^{s_hi} δ(s, r(s)) ds
    EFC_life = 1 / D

其中:
    α   整体尺度
    p   倍率损伤指数（文献里通常 1–3，对应欧姆热/极化/析锂的加速）
    β   高 SOC 敏感指数（exp 形式，抓住"高 SOC 快充最伤"）

**恒倍率协议下的闭合解**（用于拟合）:
    D = α·r^p·(e^{β·1} − e^{β·s_lo}) / β
    EFC_life = 1 / D

这个形式的价值:
    1. 可解析积分，阶梯协议只需逐段相加 → 秒级评估任意协议结构
    2. 参数有物理含义（p = 倍率加速，β = SOC 敏感）
    3. 与 GP 代理交叉验证：两者在同一数据上的误差可对比
    4. 让"哪一段最伤"变成可分解的（每段 D_i 直接给出）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

EPS = 1e-12


def _int_exp(beta: float, s_lo, s_hi):
    """∫_{s_lo}^{s_hi} exp(β·s) ds，在 β→0 时退化为 (s_hi - s_lo)。支持数组。"""
    s_lo = np.asarray(s_lo, dtype=float)
    s_hi = np.asarray(s_hi, dtype=float)
    if abs(beta) < 1e-8:
        return s_hi - s_lo
    return (np.exp(beta * s_hi) - np.exp(beta * s_lo)) / beta


def damage_density_params(alpha: float, p: float, beta: float) -> dict:
    return {"alpha": alpha, "p_rate": p, "beta_soc": beta}


def damage_constant_rate(span: float, rate: float, s_lo: float, pars: dict) -> float:
    """恒倍率协议的损伤总量（= 1 / EFC_life）。"""
    return float(pars["alpha"] * rate ** pars["p_rate"]
                 * _int_exp(pars["beta_soc"], s_lo, s_lo + span))


def efc_constant_rate(span: float, rate: float, s_lo: float, pars: dict) -> float:
    d = damage_constant_rate(span, rate, s_lo, pars)
    return float(1.0 / max(d, EPS))


def damage_stages(stages: list[tuple[float, float, float]], pars: dict) -> float:
    """阶梯协议的总损伤。stages = [(rate, soc_from, soc_to), ...]。"""
    total = 0.0
    for rate, s0, s1 in stages:
        if s1 <= s0:
            continue
        total += pars["alpha"] * rate ** pars["p_rate"] * _int_exp(pars["beta_soc"], s0, s1)
    return float(total)


def stage_damage_breakdown(stages: list[tuple[float, float, float]], pars: dict,
                           total_time_h: float | None = None) -> pd.DataFrame:
    """逐段损伤分解：哪一段吃掉最多寿命。"""
    rows = []
    D_total = damage_stages(stages, pars)
    for i, (rate, s0, s1) in enumerate(stages):
        d = pars["alpha"] * rate ** pars["p_rate"] * _int_exp(pars["beta_soc"], s0, s1) \
            if s1 > s0 else 0.0
        rows.append({
            "stage": i + 1,
            "rate_C": rate,
            "soc_from": s0,
            "soc_to": s1,
            "dQ_span": s1 - s0,
            "damage": d,
            "damage_share": d / D_total if D_total > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def efc_stages(stages, pars: dict) -> float:
    return float(1.0 / max(damage_stages(stages, pars), EPS))


def _pred_efc(alpha, p, beta, span, rate, s_lo):
    d = alpha * rate ** p * _int_exp(beta, s_lo, s_lo + span)
    return 1.0 / np.maximum(d, EPS)


def fit_damage_model(cells: pd.DataFrame, *, loss: str = "log") -> dict:
    """用全部电芯的 (span, rate, s_lo) → 观测 EFC 拟合三参数。

    loss="log" 时在 log(EFC) 残差上做最小二乘，对浅充的小数值更公平。
    """
    c = cells.dropna(subset=["eol_cycle"]).copy()
    span = c["soc_span"].to_numpy(float)
    rate = c["charge_rate_peak"].to_numpy(float)
    s_lo = c["soc_lo"].to_numpy(float)
    efc_obs = (c["eol_cycle"].to_numpy(float) * span)

    def resid(x):
        alpha, p, beta = x[0], x[1], x[2]
        if alpha <= 0 or p <= 0:
            return np.full_like(efc_obs, 1e3)
        pred = _pred_efc(alpha, p, beta, span, rate, s_lo)
        if loss == "log":
            return np.log(np.maximum(pred, EPS)) - np.log(np.maximum(efc_obs, EPS))
        return (pred - efc_obs) / np.maximum(efc_obs, 1.0)

    best = None
    for p0 in ([1.0, 1.0, 1.0], [0.01, 2.0, 2.0], [0.1, 0.5, 0.5], [0.001, 3.0, 3.0]):
        sol = least_squares(resid, p0, bounds=([1e-6, 1e-3, -5], [10, 6, 10]),
                            loss="soft_l1", f_scale=0.5, max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol

    alpha, p, beta = best.x
    pred = _pred_efc(alpha, p, beta, span, rate, s_lo)
    err_log = np.log(np.maximum(pred, EPS)) - np.log(np.maximum(efc_obs, EPS))
    ss_res = float(np.sum(err_log ** 2))
    ss_tot = float(np.sum((np.log(np.maximum(efc_obs, EPS)) - np.log(np.maximum(efc_obs, EPS)).mean()) ** 2))

    return {
        "alpha": float(alpha),
        "p_rate": float(p),
        "beta_soc": float(beta),
        "r2_log": float(1 - ss_res / ss_tot),
        "mape_pct": float(np.mean(np.abs(pred / efc_obs - 1)) * 100),
        "rmse_log": float(np.sqrt(np.mean(err_log ** 2))),
        "n": int(len(efc_obs)),
        "fit_table": c.assign(pred_efc=pred, err_pct=(pred / efc_obs - 1) * 100),
        "solver_success": bool(best.success),
    }


def soc_sensitivity_curve(pars: dict, rate: float = 1.0) -> pd.DataFrame:
    """把 δ(s,r) 沿 SOC 展开：哪个 SOC 区间最伤。"""
    s = np.linspace(0.0, 1.0, 101)
    dens = pars["alpha"] * rate ** pars["p_rate"] * np.exp(pars["beta_soc"] * s)
    return pd.DataFrame({"soc": s, "damage_density": dens,
                         "relative": dens / dens.max()})


def rate_sensitivity_report(pars: dict, span: float = 0.6, s_lo: float = 0.4,
                            rates=(0.5, 1, 1.5, 2, 3, 4)) -> pd.DataFrame:
    """每个倍率下的 EFC 与"每 0.1C 增益"——倍率损伤的边际成本。"""
    rows = []
    for r in rates:
        efc = efc_constant_rate(span, r, s_lo, pars)
        rows.append({"rate_C": r, "pred_efc": efc,
                     "damage": damage_constant_rate(span, r, s_lo, pars)})
    df = pd.DataFrame(rows)
    df["efc_delta_pct_per_C"] = df["pred_efc"].pct_change() / df["rate_C"].diff()
    return df
