"""16 — 深化⑦：充电时长随老化的漂移（"新电池标定的策略，用到老电池还安全吗？"）。

问题
----
CC+CV 时长律 `t = 0.869·span/rate + 0.432·span + 0.173` 是在**全部循环**上拟合的均值。
但电池会老化：内阻上升 → CV 段变长 → 同样的协议在老化后期会明显更慢。
更重要的是：**若协议按新电池的时间基准设计，老化后期可能超出用户容忍**。

本脚本回答三件事
----------------
1. CV 段系数 `c` 与 `d` 在循环早期 vs 末期是否显著不同？（用同一批电芯的前后两半对比）
2. 老化后期充电时长慢多少？速率与 SOC 窗口的依赖是什么？
3. 若按新电池时间设计协议，老化到 80% SOH 时时长会漂移多少？给出修正系数。

产出:
    results/metrics/charge_time_drift.csv        前后半段的时长对比
    results/metrics/charge_time_drift_by_group.csv  逐协议漂移
    results/metrics/charge_time_drift_model.csv   漂移的比例模型
    results/figures/17_time_drift.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fastcharge.chargetime import fit_charge_time  # noqa: E402
from fastcharge.config import FIGURES, METRICS, ensure_dirs  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
warnings.filterwarnings("ignore")


def early_late_split(cycles: pd.DataFrame, cells: pd.DataFrame, frac: float = 0.5) -> pd.DataFrame:
    """把每个电芯的循环分成前/后两段，对比充电时长。

    **必须归一化**：老化的直接后果是容量下降，同样充到 4.2 V 需要搬的电荷变少，
    绝对时长反而缩短（本数据实测 t_late/t_early 中位 0.51）。如果不归一，
    会得出"老化让充电变快"的错误结论。

    正确做法看**单位电荷所需时间** τ = t_charge / q_charge（秒每 Ah）。
    τ 上升 = 内阻上升（CV 段拉长）；τ 不变 = 时间缩短纯粹因为充得少。
    """
    eol = cells.set_index("cell_id")["eol_cycle"]
    qref = cells.set_index("cell_id")["q_ref_Ah"]
    rows = []
    for cid, g in cycles.groupby("cell_id", observed=True):
        e = eol.get(cid, np.nan)
        if not np.isfinite(e) or e < 40:
            continue
        g = g.sort_values("cycle_index")

        # 前段：最开始的 50 圈（新电池状态）
        head = g.iloc[:50]
        # 后段：**EOL 之前**的 50 圈。不能取文件末尾——EOL 之后容量已趋近于零，
        #       t/q 会发散（实测最高 5.6 倍，纯属数值假象）。
        eol_pos = int(np.argmin(np.abs(g["cycle_number"].to_numpy() - e)))
        lo = max(eol_pos - 50, 50)
        tail = g.iloc[lo:eol_pos]
        if len(head) < 5 or len(tail) < 5:
            continue

        t_e = float(head["t_charge_s"].median())
        t_l = float(tail["t_charge_s"].median())
        q_e = float(head["q_charge_Ah"].median())
        q_l = float(tail["q_charge_Ah"].median())
        if q_e <= 0 or q_l <= 0 or t_e <= 0:
            continue

        tau_e = t_e / q_e        # 秒每 Ah（初期）
        tau_l = t_l / q_l        # 秒每 Ah（后期，EOL 前）

        rows.append({
            "cell_id": cid,
            "group": int(g["group"].iloc[0]),
            "soc_span": float(g["soc_span"].iloc[0]),
            "rate": float(g["charge_rate_peak"].iloc[0]),
            "eol_cycle": float(e),
            "t_early_h": t_e / 3600,
            "t_late_h": t_l / 3600,
            "t_ratio": t_l / max(t_e, 1e-9),
            "tau_ratio": tau_l / max(tau_e, 1e-9),
            "q_ratio": q_l / max(q_e, 1e-9),
            "tau_early_s_per_ah": tau_e,
            "tau_late_s_per_ah": tau_l,
            "extra_s_per_ah": tau_l - tau_e,
            "soh_at_tail": float(q_l / qref.get(cid, np.nan) * g["soc_span"].iloc[0])
            if np.isfinite(qref.get(cid, np.nan)) else np.nan,
        })
    return pd.DataFrame(rows)


def drift_model(dr: pd.DataFrame) -> dict:
    """归一化漂移 tau_ratio 对 (span, rate, EOL) 的线性模型。"""
    from sklearn.linear_model import LinearRegression

    d = dr.dropna(subset=["tau_ratio", "soc_span", "rate", "eol_cycle"]).copy()
    d = d[(d["tau_ratio"] > 0.5) & (d["tau_ratio"] < 3.0)]
    X = np.column_stack([d["soc_span"], d["rate"], np.log(d["eol_cycle"])])
    y = d["tau_ratio"]
    m = LinearRegression().fit(X, y)
    return {
        "coef_span": float(m.coef_[0]),
        "coef_rate": float(m.coef_[1]),
        "coef_logeol": float(m.coef_[2]),
        "intercept": float(m.intercept_),
        "r2": float(m.score(X, y)),
        "n": int(len(d)),
        "median_tau_ratio": float(d["tau_ratio"].median()),
        "mean_tau_ratio": float(d["tau_ratio"].mean()),
        "median_t_ratio": float(d["t_ratio"].median()) if "t_ratio" in d else np.nan,
        "data": d,
    }


def project_drift(dr: pd.DataFrame, span: float, rate: float, ct: dict) -> pd.DataFrame:
    """给定协议，把**单位电荷时间** τ 按 SOH 外推，再折算回绝对时长。

    τ 上升（内阻恶化） × 容量下降（搬的电荷变少） → 净效应可能是净增长、持平或净缩短。
    这正是工程上最反直觉的地方：老化后期充电**可能更快**，但充进去的电更少。
    """
    from fastcharge.chargetime import charge_time_h

    d = dr.dropna(subset=["tau_ratio"])
    median_tau = float(d["tau_ratio"].median())
    base_q = 0.25 * span                      # 标称容量 × 窗口
    rows = []
    for soh in (1.00, 0.95, 0.90, 0.85, 0.80):
        # τ 从 1.0 线性升到 median_tau（在 80% SOH 处）
        tau_ratio = 1.0 + (1.0 - soh) / 0.20 * (median_tau - 1.0)
        q = base_q * soh                      # 可用容量按 SOH 缩
        t_h = float(charge_time_h(span, rate, ct["a_cc"], ct["c_cv"], ct["d_tail"]))
        t_new = t_h * tau_ratio * soh         # 时间 ∝ τ × 电荷量
        rows.append({
            "soh": soh,
            "tau_ratio": tau_ratio,
            "usable_q_Ah": q,
            "charge_time_h": t_new,
            "time_vs_new_h": t_new - t_h,
            "time_vs_new_min": (t_new - t_h) * 60,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")
    ct = fit_charge_time(cycles)

    print("=== 1) 前段 vs 后段 充电时长 ===")
    dr = early_late_split(cycles, cells)
    dr.to_csv(METRICS / "charge_time_drift.csv", index=False, encoding="utf-8-sig")
    print(f"  {len(dr)} 电芯")

    print("\n  **绝对时长比 t_late/t_early**（受容量衰减主导，会掩盖内阻效应）:")
    q_abs = dr["t_ratio"].quantile([0.10, 0.50, 0.90])
    print(f"    P10={q_abs.loc[0.10]:.3f}  中位={q_abs.loc[0.50]:.3f}  P90={q_abs.loc[0.90]:.3f}")
    print(f"    → 中位 {(dr['t_ratio'].median()-1)*100:+.0f}%（充得少，所以时间缩短）")

    print("\n  **归一化后 τ = 每 Ah 所需时间比**（真实的内阻恶化）:")
    q_tau = dr["tau_ratio"].quantile([0.10, 0.50, 0.90])
    print(f"    P10={q_tau.loc[0.10]:.3f}  中位={q_tau.loc[0.50]:.3f}  P90={q_tau.loc[0.90]:.3f}")
    print(f"    → 中位 {(dr['tau_ratio'].median()-1)*100:+.0f}%：同样的电荷量，后期要多花这么多时间")
    print(f"    → 中位每 Ah 多 {dr['extra_s_per_ah'].median():.0f} 秒")

    print(f"\n  电荷量比 q_late/q_early 中位: {dr['q_ratio'].median():.3f}")
    print(f"    → 绝对时长比 ≈ q_ratio × tau_ratio = "
          f"{dr['q_ratio'].median() * dr['tau_ratio'].median():.3f}"
          f"（内阻 +{(dr['tau_ratio'].median()-1)*100:.0f}% 被 电荷 −{(1-dr['q_ratio'].median())*100:.0f}% 抵消）")

    # ---- 按 SOC 窗口分层 ----
    print("\n=== 2) 漂移与 SOC 窗口 / 倍率的关系 ===")
    dr["span_bin"] = pd.cut(dr["soc_span"], [0, 0.3, 0.5, 0.7, 0.85, 1.01])
    by_span = dr.groupby("span_bin", observed=True).agg(
        n=("cell_id", "size"), tau_ratio_median=("tau_ratio", "median"),
        t_ratio_median=("t_ratio", "median"),
        extra_s_per_ah=("extra_s_per_ah", "median"),
    ).reset_index()
    by_span["span_bin"] = by_span["span_bin"].astype(str)
    print(by_span.round(3).to_string(index=False))

    dr["rate_bin"] = pd.cut(dr["rate"], [0.4, 1.0, 1.5, 2.0, 2.5, 3.1])
    by_rate = dr.groupby("rate_bin", observed=True).agg(
        n=("cell_id", "size"), tau_ratio_median=("tau_ratio", "median"),
        extra_s_per_ah=("extra_s_per_ah", "median"),
    ).reset_index()
    by_rate["rate_bin"] = by_rate["rate_bin"].astype(str)
    print("\n" + by_rate.round(3).to_string(index=False))

    # ---- 逐协议 ----
    print("\n=== 3) 逐协议 τ 漂移（最严重的前 10）===")
    byg = dr.groupby("group").agg(
        n=("cell_id", "size"), span=("soc_span", "first"), rate=("rate", "first"),
        tau_ratio_median=("tau_ratio", "median"), tau_ratio_max=("tau_ratio", "max"),
        t_ratio_median=("t_ratio", "median"),
        extra_s_per_ah=("extra_s_per_ah", "median"),
    ).reset_index().sort_values("tau_ratio_median", ascending=False)
    byg.to_csv(METRICS / "charge_time_drift_by_group.csv", index=False, encoding="utf-8-sig")
    print(byg.head(10).round(3).to_string(index=False))

    # ---- 漂移模型 ----
    print("\n=== 4) 归一化漂移模型 ===")
    dm = drift_model(dr)
    pd.DataFrame([{k: v for k, v in dm.items() if not callable(v) and k != "data"}]).to_csv(
        METRICS / "charge_time_drift_model.csv", index=False, encoding="utf-8-sig")
    print(f"  τ_late/τ_early = {dm['intercept']:.3f} "
          f"+ ({dm['coef_span']:.3f})·span + ({dm['coef_rate']:.3f})·rate "
          f"+ ({dm['coef_logeol']:.3f})·log(EOL)")
    print(f"  R² = {dm['r2']:.3f}, n = {dm['n']}")
    print(f"  中位 τ 比 = {dm['median_tau_ratio']:.3f}")

    # ---- 外推到 80% SOH ----
    print("\n=== 5) 按新电池标定，老化到各 SOH 时的净充电时长 ===")
    for span, rate, label in [(0.50, 0.99, "A 0.99C/50%"), (0.45, 1.50, "B 1.50C/45%"),
                              (1.00, 0.50, "D 基线 0.5C/100%")]:
        p = project_drift(dr, span, rate, ct)
        print(f"\n  [{label}] 新电池 {p['charge_time_h'].iloc[0]:.3f} h")
        print(p.round(3).to_string(index=False))

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))

    ax = axes[0]
    ax.hist(dr["tau_ratio"].clip(0.8, 2.0), bins=40, color="#bee3f8", edgecolor="#2b6cb0",
            label="归一化 τ 比")
    ax.axvline(1.0, color="gray", ls=":", lw=1, label="无内阻恶化")
    ax.axvline(dr["tau_ratio"].median(), color="#e53e3e", ls="--", lw=1.6,
               label=f"中位 {dr['tau_ratio'].median():.2f}")
    ax.set_xlabel("τ_late / τ_early（每 Ah 所需时间比）")
    ax.set_ylabel("电芯数")
    ax.set_title(f"内阻恶化：同样电荷后期多花 {(dr['tau_ratio'].median()-1)*100:+.0f}%")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    from fastcharge.chargetime import charge_time_h
    soh_grid = np.linspace(1.0, 0.80, 40)
    tau_med = dr["tau_ratio"].median()
    for span_val in (0.3, 0.5, 0.7, 1.0):
        t_h = float(charge_time_h(span_val, 1.0, ct["a_cc"], ct["c_cv"], ct["d_tail"]))
        # 净时长 = τ 上升 × SOH 容量减少
        net = [t_h * (1 + (1 - s) / 0.20 * (tau_med - 1)) * s for s in soh_grid]
        ax2.plot(soh_grid * 100, net, lw=1.8, label=f"窗口 {span_val:.0%}")
    ax2.invert_xaxis()
    ax2.set_xlabel("SOH (%)")
    ax2.set_ylabel("预计绝对充电时长 (h)")
    ax2.set_title("净效应：内阻↑ 与 容量↓ 相互抵消")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "17_time_drift.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
