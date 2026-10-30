"""早期寿命预测（early-life prediction）：用前 N 圈预测终寿命，加速优化闭环。

为什么需要
----------
贝叶斯优化/强化学习每评估一个候选协议，理论上要跑到 EOL（数百到上万个循环）。
若能用前 N 圈（N=20–200）的容量轨迹预测最终寿命，闭环迭代成本能降 1–2 个数量级。
这是 v1.1 深化清单里 A3-D5「早期寿命代理指标」的实质内容。

本模块做四件事
--------------
1. **特征**：前 N 圈的
      · 容量衰减斜率（线性拟合的 slope，归一化到每循环容量损失）
      · 非线性度（二次项系数 / 相对残差）
      · 首圈容量 vs 标称（初始落差，"形成损失"）
      · 充电时长漂移（内阻增长的代理）
      · 库仑效率均值（SEI 生长速率的代理）
2. **预测目标**：log(EOL 循环数)
3. **评估**：按协议 LOGO 交叉验证（与主报告同口径），报告"用 N 圈能达到多少误差"
4. **扫描 N**：绘制"观测圈数-预测精度"曲线，回答"最少要看多少圈"

产出:
    results/metrics/early_features.csv        逐电芯的早期特征（N=20/50/100/200）
    results/metrics/early_prediction.csv      N × 模型 的 LOGO-CV 精度
    results/metrics/early_curve.csv           精度随 N 的曲线（主图数据）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARLY_NS = (10, 20, 50, 100, 200, 400)


def _fit_poly(y: np.ndarray, deg: int = 2) -> tuple[float, float, float]:
    """y 对循环序号做多项式拟合，返回 (线性斜率, 二次项, 相对残差)。"""
    if len(y) < 5:
        return np.nan, np.nan, np.nan
    x = np.arange(len(y), dtype=float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return np.nan, np.nan, np.nan
    x, y = x[ok], y[ok]
    lin = np.polyfit(x, y, 1)
    qua = np.polyfit(x, y, min(deg, 3))
    pred = np.polyval(qua, x)
    resid = float(np.sqrt(np.nanmean((y - pred) ** 2)) / max(np.nanmean(np.abs(y)), 1e-9))
    return float(lin[0]), float(qua[0]), resid


def early_features(cycles: pd.DataFrame, cells: pd.DataFrame, n_head: int) -> pd.DataFrame:
    """前 n_head 圈的早期特征（一电芯一行）。"""
    head = cycles[cycles["cycle_index"] < n_head]
    rows = []
    for cid, g in head.groupby("cell_id", observed=True):
        g = g.sort_values("cycle_index")
        q = g["q_norm_Ah"].to_numpy(float)
        q0 = float(np.nanmedian(q[:3])) if len(q) >= 3 else np.nan
        qn = q / q0 if np.isfinite(q0) and q0 > 0 else q
        slope, quad, resid = _fit_poly(qn, 2)
        t_c = g["t_charge_s"].to_numpy(float)
        ce = g["ce"].to_numpy(float)
        rows.append({
            "cell_id": cid,
            f"n{n_head}_slope": slope,
            f"n{n_head}_quad": quad,
            f"n{n_head}_resid": resid,
            f"n{n_head}_q0": q0,
            f"n{n_head}_ce_mean": float(np.nanmean(ce)) if np.isfinite(ce).any() else np.nan,
            f"n{n_head}_ce_std": float(np.nanstd(ce)) if np.isfinite(ce).any() else np.nan,
            f"n{n_head}_t_drift": (
                float(np.polyfit(np.arange(len(t_c)), t_c, 1)[0]) if len(t_c) >= 5
                and np.isfinite(t_c).all() else np.nan
            ),
            f"n{n_head}_t_mean": float(np.nanmean(t_c)) if np.isfinite(t_c).any() else np.nan,
        })
    df = pd.DataFrame(rows)
    return cells[["cell_id", "group", "eol_cycle", "censored", "soc_span",
                  "charge_rate_peak", "discharge_rate_mean"]].merge(df, on="cell_id", how="left")


def feature_cols(n_head: int) -> list[str]:
    return [f"n{n_head}_slope", f"n{n_head}_quad", f"n{n_head}_resid",
            f"n{n_head}_ce_mean", f"n{n_head}_t_drift", f"n{n_head}_q0"]


def evaluate_early(feat_df: pd.DataFrame, n_head: int, *, model: str = "ridge") -> dict:
    """按协议 LOGO 评估"用前 n_head 圈预测 log(EOL)"的精度。"""
    from fastcharge.surrogate import FACTORIES, logo_cv

    d = feat_df.dropna(subset=["eol_cycle"]).copy()
    cols = [c for c in feature_cols(n_head) if c in d.columns]
    X = d[cols].to_numpy(float)
    y = np.log(d["eol_cycle"].to_numpy(float).clip(min=1.0))
    groups = d["group"].to_numpy()

    # 缺失值用列中位数填补（早期特征偶有 NaN）
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)

    met = logo_cv(X, y, groups, FACTORIES[model])
    met.update({"n_head": n_head, "model": model, "n_features": len(cols)})
    return met


def vs_full_protocol_baseline(feat_df: pd.DataFrame, n_head: int) -> dict:
    """对照：只用协议参数（不看任何早期曲线）能预测到什么程度。"""
    from fastcharge.protocol import build_feature_matrix
    from fastcharge.surrogate import CORE_FEATURES, FACTORIES, logo_cv

    d = feat_df.dropna(subset=["eol_cycle"]).copy()
    feats, _ = build_feature_matrix(d)
    X = feats[CORE_FEATURES].to_numpy(float)
    y = np.log(d["eol_cycle"].to_numpy(float).clip(min=1.0))
    met = logo_cv(X, y, d["group"].to_numpy(), FACTORIES["ridge"])
    met.update({"n_head": "协议参数（无曲线）", "model": "ridge",
                "n_features": len(CORE_FEATURES)})
    return met


def build_all(cycles: pd.DataFrame, cells: pd.DataFrame, ns=EARLY_NS) -> dict[int, pd.DataFrame]:
    return {n: early_features(cycles, cells, n) for n in ns}


def compute_generation_hours(cycles: pd.DataFrame, cells: pd.DataFrame,
                             holdout_frac: float = 0.2) -> pd.DataFrame:
    """用实测数据算"早期预测能省多少实验时间"。

    思路：对每个协议，比较"跑满 EOL 需要的循环数"与"只看前 N 圈就能出结论"。
    折算出电池测试台的实际时长（用该协议每圈实测周期）。
    """
    eol = cells.set_index("cell_id")["eol_cycle"]
    rows = []
    for n in EARLY_NS:
        head = cycles[cycles["cycle_index"] < n]
        t_per_cycle = head.groupby("cell_id", observed=True)["t_total_s"].median()
        full_h = (eol * t_per_cycle).dropna() / 3600
        early_h = (t_per_cycle * n).dropna() / 3600
        rows.append({
            "n_head": n,
            "mean_full_test_h": float(full_h.mean()),
            "mean_early_test_h": float(early_h.mean()),
            "saving_pct": float((1 - early_h.mean() / full_h.mean()) * 100),
            "median_full_test_h": float(full_h.median()),
        })
    return pd.DataFrame(rows)
