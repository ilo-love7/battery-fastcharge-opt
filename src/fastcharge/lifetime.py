"""寿命结局与统计分析。

对应 v1.1 阶段 1：63 协议寿命分布对比 + 按协议要素分组做统计检验与相关性分析。

核心问题:
    Q1 峰值倍率与寿命是什么关系？是单调还是存在拐点？
    Q2 同样峰值倍率下，SOC 窗口的作用有多大？（"高倍率落在高 SOC" 假说的直接检验）
    Q3 放电倍率是否独立贡献？（ISU-ILCC 的独特之处：充放电倍率解耦）

**关键口径问题（本项目最重要的方法论发现）**
    表观 EOL 循环数 eol_cycle 不是"材料耐久度"的无偏度量:
        浅充（SOC 窗口 0.3）每循环只搬运 0.3 FEC，跑 3000 循环 = 900 EFC
        全充（SOC 窗口 1.0）跑 400 循环         = 400 EFC
    直接把 eol_cycle 当目标，会让"浅充"看起来优异、把倍率效应完全掩盖
    （辛普森悖论）。因此本模块同时提供两种口径:
        eol_cycle  —— 表观循环数（用户视角："我能用多少次"）
        efc        —— 等效满循环 = eol_cycle × soc_span（材料视角："材料被用掉多少"）
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# 分组统计
# --------------------------------------------------------------------------
def group_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """按 group（= 一个协议）汇总，含组内离散度与删失比例。"""
    g = cells.groupby("group")
    out = pd.DataFrame(
        {
            "charge_rate_peak": g["charge_rate_peak"].first(),
            "charge_rate_mean": g["charge_rate_mean"].first(),
            "discharge_rate_mean": g["discharge_rate_mean"].first(),
            "soc_lo": g["soc_lo"].first(),
            "soc_span": g["soc_span"].first(),
            "n_cells": g.size(),
            "n_censored": g["censored"].sum(),
            "eol_median": g["eol_cycle"].median(),
            "eol_min": g["eol_cycle"].min(),
            "eol_max": g["eol_cycle"].max(),
            "eol_mean": g["eol_cycle"].mean(),
            "eol_std": g["eol_cycle"].std(),
        }
    )
    out["eol_cv"] = out["eol_std"] / out["eol_mean"]
    out["censored_frac"] = out["n_censored"] / out["n_cells"]
    return out.sort_values("eol_median").reset_index()


def rate_dose_response(cells: pd.DataFrame) -> pd.DataFrame:
    """峰值充电倍率 → 寿命的剂量-反应表（固定放电 0.5C 分组的纯效应）。"""
    d = cells[cells["discharge_rate_mean"].round(2) == 0.5].copy()
    out = (
        d.groupby("charge_rate_peak")
        .agg(
            n_cells=("cell_id", "size"),
            eol_median=("eol_cycle", "median"),
            eol_q25=("eol_cycle", lambda s: s.quantile(0.25)),
            eol_q75=("eol_cycle", lambda s: s.quantile(0.75)),
            soc_lo_median=("soc_lo", "median"),
        )
        .reset_index()
        .sort_values("charge_rate_peak")
    )
    return out


# --------------------------------------------------------------------------
# 假设检验
# --------------------------------------------------------------------------
def spearman_report(cells: pd.DataFrame, features: list[str], target: str = "eol_cycle") -> pd.DataFrame:
    """协议特征 vs 寿命的 Spearman 秩相关（对非线性和离群稳健）。"""
    rows = []
    y = cells[target].astype(float)
    for f in features:
        x = cells[f].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() < 8 or x[ok].nunique() < 2:
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append({"feature": f, "spearman_rho": rho, "p_value": p, "n": int(ok.sum())})
    df = pd.DataFrame(rows).sort_values("spearman_rho", key=np.abs, ascending=False)
    return df.reset_index(drop=True)


def kruskal_by_bins(cells: pd.DataFrame, feature: str, bins: list[float], label: str = "") -> dict[str, Any]:
    """按分箱切组做 Kruskal-Wallis（非参数方差分析），检验该要素是否显著影响寿命。"""
    x = pd.to_numeric(cells[feature], errors="coerce")
    y = cells["eol_cycle"].astype(float)
    ok = x.notna() & y.notna()
    cut = pd.cut(x[ok], bins=bins, include_lowest=True)
    groups = [y[ok][cut == c].to_numpy() for c in cut.cat.categories if (cut == c).sum() >= 2]
    if len(groups) < 2:
        return {"feature": feature, "label": label, "H": np.nan, "p": np.nan, "n_groups": len(groups)}
    H, p = stats.kruskal(*groups)
    n = int(ok.sum())
    k = len(groups)
    # epsilon-squared 效应量
    eps = (H - k + 1) / (n - k) if n > k else np.nan
    medians = {str(c): float(np.median(y[ok][cut == c])) for c in cut.cat.categories}
    return {
        "feature": feature,
        "label": label,
        "H": float(H),
        "p": float(p),
        "n_groups": k,
        "n_total": n,
        "epsilon_sq": float(eps),
        "group_medians": medians,
    }


def factorial_interaction(cells: pd.DataFrame, a: str, b: str, bins_a: list[float], bins_b: list[float]) -> pd.DataFrame:
    """双因素交互表：a×b 各格的中位寿命（检验"高倍率×高SOC"叠加效应）。"""
    xa = pd.cut(pd.to_numeric(cells[a], errors="coerce"), bins=bins_a, include_lowest=True)
    xb = pd.cut(pd.to_numeric(cells[b], errors="coerce"), bins=bins_b, include_lowest=True)
    d = pd.DataFrame({"a": xa, "b": xb, "eol": cells["eol_cycle"].astype(float)}).dropna()
    piv = d.pivot_table(index="a", columns="b", values="eol", aggfunc="median", observed=True)
    return piv


# ------------------------------------------------------------------
# 协议要素归因：逐元素"留一"对比
# ------------------------------------------------------------------
def pairwise_same_peak(cells: pd.DataFrame) -> pd.DataFrame:
    """同一峰值倍率下，不同 SOC 起点的寿命对比（隔离 SOC 窗口的独立贡献）。"""
    d = cells[cells["discharge_rate_mean"].round(2) == 0.5].copy()
    rows = []
    for rate, sub in d.groupby("charge_rate_peak"):
        if len(sub) < 8:
            continue
        # SOC 起点高低两组（以中位数切）
        med = sub["soc_lo"].median()
        hi = sub[sub["soc_lo"] >= med]["eol_cycle"]
        lo = sub[sub["soc_lo"] < med]["eol_cycle"]
        if len(hi) < 3 or len(lo) < 3:
            continue
        u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        rows.append(
            {
                "charge_rate_peak": rate,
                "n_high_soc_start": len(hi),
                "n_low_soc_start": len(lo),
                "eol_soc_start_high": float(hi.median()),
                "eol_soc_start_low": float(lo.median()),
                "delta_pct": float((hi.median() - lo.median()) / lo.median() * 100),
                "mannwhitney_p": float(p),
            }
        )
    return pd.DataFrame(rows)


def protocol_ranking(group_df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """63 协议按寿命结局排序，输出"最伤/最温和"榜单 + 单位充电时间收益。"""
    g = group_df.copy()
    g["eol_per_charge_hour"] = g["eol_median"] / g["soc_span"].clip(lower=1e-6)  # 每小时充电换来的循环数
    g = g.sort_values("eol_median", ascending=False)
    cols = ["group", "charge_rate_peak", "charge_rate_mean", "discharge_rate_mean", "soc_lo",
            "soc_span", "n_cells", "eol_median", "eol_min", "eol_max", "censored_frac"]
    head = g[cols].head(top)
    tail = g[cols].tail(top).iloc[::-1]
    return pd.concat([head.assign(rank="best"), tail.assign(rank="worst")], ignore_index=True)


# --------------------------------------------------------------------------
# 口径工具：表观循环 vs 等效满循环
# --------------------------------------------------------------------------
def add_throughput_columns(cells: pd.DataFrame) -> pd.DataFrame:
    """派生吞吐口径列。

    efc        : 等效满循环数 = EOL 循环 × SOC 窗口宽度
    q_total_Ah : 累计吞吐电荷量 (Ah) = efc × 标称容量
    """
    c = cells.copy()
    eol = c["eol_cycle"].astype(float)
    c["efc"] = eol * c["soc_span"].astype(float)
    c["q_total_Ah"] = c["efc"] * c["q_ref_Ah"].astype(float)
    return c


def dual_objective_report(cells: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """同一批特征分别对 eol_cycle 与 efc 做 Spearman，暴露口径差异。

    返回的 DataFrame 以 feature 为索引、列为 eol_cycle / efc / delta。
    特征缺失时该行跳过（不写 NaN 行）。
    """
    c = add_throughput_columns(cells)
    rows = []
    for f in features:
        if f not in c.columns:
            continue
        x = pd.to_numeric(c[f], errors="coerce")
        for tgt in ("eol_cycle", "efc"):
            y = pd.to_numeric(c[tgt], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() < 8 or x[ok].nunique() < 2:
                continue
            rho, p = stats.spearmanr(x[ok], y[ok])
            rows.append({"feature": f, "target": tgt, "rho": rho, "p_value": p, "n": int(ok.sum())})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["eol_cycle", "efc", "delta"])
    piv = df.pivot_table(index="feature", columns="target", values="rho", observed=True)
    piv["delta"] = piv.get("efc", np.nan) - piv.get("eol_cycle", np.nan)
    return piv.sort_values("delta")


def rate_effect_within_window(cells: pd.DataFrame, min_span: float = 0.95) -> pd.DataFrame:
    """**控制 SOC 窗口**后的倍率效应：只在近满窗口子集内比较倍率。

    这是唯一能把"倍率伤害"从"窗口宽度混淆"里分离出来的干净对照。
    子集为空时返回空表（附 attrs 里的说明），不抛异常。
    """
    c = add_throughput_columns(cells)
    sub = c[(c["soc_span"] >= min_span) & c["eol_cycle"].notna()].copy()
    if sub.empty or len(sub) < 4:
        out = pd.DataFrame(columns=[
            "charge_rate_peak", "n_cells", "span_median",
            "eol_median", "efc_median", "discharge_rate_median",
        ])
        out.attrs["spearman_vs_cycles"] = (np.nan, np.nan, len(sub))
        out.attrs["spearman_vs_efc"] = (np.nan, np.nan, len(sub))
        out.attrs["note"] = f"窗口 ≥{min_span} 的子集只有 {len(sub)} 只电芯，样本不足"
        return out

    rows = []
    for rate, s in sub.groupby(sub["charge_rate_peak"].round(2), observed=True):
        rows.append(
            {
                "charge_rate_peak": float(rate),
                "n_cells": len(s),
                "span_median": float(s["soc_span"].median()),
                "eol_median": float(s["eol_cycle"].median()),
                "efc_median": float(s["efc"].median()),
                "discharge_rate_median": float(s["discharge_rate_mean"].median()),
            }
        )
    out = pd.DataFrame(rows).sort_values("charge_rate_peak").reset_index(drop=True)
    rho_c, p_c = stats.spearmanr(sub["charge_rate_peak"], sub["eol_cycle"])
    rho_e, p_e = stats.spearmanr(sub["charge_rate_peak"], sub["efc"])
    out.attrs["spearman_vs_cycles"] = (float(rho_c), float(p_c), len(sub))
    out.attrs["spearman_vs_efc"] = (float(rho_e), float(p_e), len(sub))
    return out

