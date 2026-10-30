"""06 — 受机理约束的重优化（阶段 3 × 阶段 4 联动）。

做了什么
--------
无约束 BO 的 Pareto 前沿有 10/12 条协议超出析锂安全边界——纯靠统计建议是不安全的。
本脚本把 PyBaMM 的析锂判据编译成一张**倍率上限表 rate_max(起始SOC, 温度)**，
再把它作为硬约束塞回优化器，得到"物理可行域内的 Pareto 前沿"，并量化：
    约束吃掉多少速度收益？ 约束后还能不能比 0.5C 基线更快更耐用？

产出:
    results/metrics/rate_limit_table.csv        起始 SOC × 温度 → 安全倍率上限
    results/metrics/opt_pareto_constrained.csv  受约束 Pareto 协议
    results/metrics/opt_constraint_cost.csv     约束的代价（时长/寿命损失）
    results/figures/10_constrained_vs_free.png
"""
from __future__ import annotations

import pickle
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

from fastcharge.chargetime import charge_time_h  # noqa: E402
from fastcharge.config import FIGURES, METRICS, MODELS, ensure_dirs  # noqa: E402
from fastcharge.optimize import (  # noqa: E402
    ApplicabilityDomain,
    BOUNDS,
    VAR_ORDER,
    pareto_front,
    run_bo,
)
from fastcharge.physics import (  # noqa: E402
    SAFETY_MARGIN_V,
    SimulationSetup,
    min_plating_overpotential,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# 1. 析锂上限表 rate_max(soc_start, T)
# --------------------------------------------------------------------------
def build_rate_limit_table(soc_starts=(0.0, 0.2, 0.4, 0.6, 0.8),
                           temps=(15.0, 30.0, 45.0),
                           rates=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0),
                           model: str = "SPM") -> pd.DataFrame:
    rows = []
    for T in temps:
        for s0 in soc_starts:
            etas, cs = [], []
            for c in rates:
                try:
                    r = min_plating_overpotential(
                        SimulationSetup(model=model, c_rate=c, temperature_C=T, soc_start=s0))
                    etas.append(r["min_plating_overpotential_V"])
                    cs.append(c)
                except Exception:
                    continue
            etas = np.array(etas, float)
            cs = np.array(cs, float)
            ok = np.isfinite(etas)
            etas, cs = etas[ok], cs[ok]
            if len(etas) == 0:
                limit = np.nan
            else:
                # **保守单调口径**：取"从最小倍率起连续安全"的最大倍率。
                # 为什么不直接找 0 V 交点：SPM 下 η_pl 对倍率并非单调（高速段在
                # 极片表面饱和前就结束，反而"更安全"），单一交点会给出 4C 以上
                # 仍判安全的荒谬结论。连续安全段的上界才是工程上可承诺的上限。
                xs = np.sort(cs)
                e = etas[np.argsort(cs)]
                safe_flag = e >= SAFETY_MARGIN_V
                if not safe_flag.any():
                    limit = 0.0
                else:
                    n_ok = 0
                    for f in safe_flag:
                        if f:
                            n_ok += 1
                        else:
                            break
                    limit = float(xs[n_ok - 1])
            rows.append({"temperature_C": T, "soc_start": s0, "rate_limit_C": float(limit)})
            print(f"  T={T:.0f}°C soc_start={s0:.1f} → rate_max={limit:.2f}C", flush=True)
    return pd.DataFrame(rows)


def interpolate_limit(table: pd.DataFrame, soc_start: float, temperature_C: float = 30.0) -> float:
    """在 (soc_start, T) 平面上做双线性插值，得到该条件下的安全倍率上限。"""
    sub = table.copy()
    # 温度方向: 取最近的两档
    Ts = np.sort(sub["temperature_C"].unique())
    if temperature_C <= Ts[0]:
        rows = sub[sub["temperature_C"] == Ts[0]]
    elif temperature_C >= Ts[-1]:
        rows = sub[sub["temperature_C"] == Ts[-1]]
    else:
        t_lo = Ts[Ts <= temperature_C].max()
        t_hi = Ts[Ts > temperature_C].min()
        w = (temperature_C - t_lo) / (t_hi - t_lo)
        lo = sub[sub["temperature_C"] == t_lo].set_index("soc_start")["rate_limit_C"]
        hi = sub[sub["temperature_C"] == t_hi].set_index("soc_start")["rate_limit_C"]
        rows = ((lo * (1 - w) + hi * w)).reset_index().rename(
            columns={"rate_limit_C": "rate_limit_C"})
    s = rows.set_index("soc_start")["rate_limit_C"].sort_index()
    return float(np.interp(soc_start, s.index.to_numpy(float), s.to_numpy(float)))


# --------------------------------------------------------------------------
# 2. 受约束重优化
# --------------------------------------------------------------------------
def main() -> int:
    ensure_dirs()

    print("=== 1) 建立析锂倍率上限表（PyBaMM）===")
    table = build_rate_limit_table()
    table.to_csv(METRICS / "rate_limit_table.csv", index=False, encoding="utf-8-sig")

    print("\n上限表（30°C 行）:")
    print(table[table["temperature_C"] == 30.0].round(2).to_string(index=False))

    # 载入 GP 代理 + 充电时长参数
    with open(MODELS / "gp_cycles.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    gp, ct = bundle["model"], bundle["ct_params"]

    cells = pd.read_parquet(METRICS / "cells.parquet")
    from fastcharge.optimize import decision_space
    Xdec = decision_space(cells)

    # ---- 受约束的目标包装：域内 + 析锂安全 ----
    T_REF = 30.0

    def constraint(X: np.ndarray) -> np.ndarray:
        spans, rates = X[:, 0], X[:, 1]
        soc_start = 1.0 - spans
        limits = np.array([interpolate_limit(table, s, T_REF) for s in soc_start])
        return rates <= limits + 1e-9

    print("\n=== 2) 受约束 BO（析锂安全 + 适用域）===")
    res_free = run_bo(gp, ct, X_decision=Xdec, n_init=48, n_iter=16, seed=0)
    print(f"  无约束: Pareto {len(res_free['pareto_bo'])} 点, HV={res_free['hv_bo']:.3f}")

    # 复用 run_bo，但用约束过滤候选：先粗略生成，再筛，再补采样
    from fastcharge.optimize import ApplicabilityDomain, objectives
    from scipy.stats import qmc

    AD = ApplicabilityDomain(Xdec, quantile=0.95)
    lo = np.array([BOUNDS[v][0] for v in VAR_ORDER])
    hi = np.array([BOUNDS[v][1] for v in VAR_ORDER])
    sampler = qmc.Sobol(d=3, scramble=True, seed=1)
    pool = []
    for _ in range(60):
        b = qmc.scale(sampler.random(400), lo, hi)
        ok = AD(b) & constraint(b)
        pool.append(b[ok])
        if sum(len(p) for p in pool) >= 120:
            break
    Xc = np.vstack(pool)[:96]
    Yc = objectives(Xc, gp, ct)
    i_c = pareto_front(Yc)
    print(f"  受约束: 可行采样 {len(Xc)} 点, Pareto {len(i_c)} 点")

    def decode(X: np.ndarray) -> pd.DataFrame:
        span, rate = X[:, 0], X[:, 1]
        t_win = charge_time_h(span, rate, ct["a_cc"], ct["c_cv"], ct["d_tail"])
        t_per_fec = t_win / span
        F = objectives(X, gp, ct)
        return pd.DataFrame({
            "soc_window": span,
            "peak_rate_C": rate,
            "discharge_rate_C": X[:, 2],
            "time_to_charge_window_h": t_win,
            "time_per_fec_h": t_per_fec,
            "pred_cycles": np.exp(-F[:, 1]),
        })

    pc = decode(Xc[i_c]).sort_values("time_per_fec_h").reset_index(drop=True)
    pc.to_csv(METRICS / "opt_pareto_constrained.csv", index=False, encoding="utf-8-sig")
    print("\n=== 受约束 Pareto 前沿 ===")
    print(pc.round(3).to_string(index=False))

    # ---- 3. 约束代价 ----
    free = pd.read_csv(METRICS / "opt_pareto_bo.csv")
    rows = [
        {
            "metric": "最快端 time_per_fec_h",
            "unconstrained": free["time_per_fec_h"].min(),
            "constrained": pc["time_per_fec_h"].min(),
        },
        {
            "metric": "最快端 pred_cycles",
            "unconstrained": float(free.loc[free["time_per_fec_h"].idxmin(), "pred_cycles"]),
            "constrained": float(pc.loc[pc["time_per_fec_h"].idxmin(), "pred_cycles"]),
        },
        {
            "metric": "最长寿端 time_per_fec_h",
            "unconstrained": free["time_per_fec_h"].max(),
            "constrained": pc["time_per_fec_h"].max(),
        },
        {
            "metric": "最长寿端 pred_cycles",
            "unconstrained": float(free.loc[free["time_per_fec_h"].idxmax(), "pred_cycles"]),
            "constrained": float(pc.loc[pc["time_per_fec_h"].idxmax(), "pred_cycles"]),
        },
    ]
    cost = pd.DataFrame(rows)
    cost["delta_pct"] = (cost["constrained"] / cost["unconstrained"] - 1) * 100
    cost.to_csv(METRICS / "opt_constraint_cost.csv", index=False, encoding="utf-8-sig")
    print("\n=== 约束的代价 ===")
    print(cost.round(3).to_string(index=False))

    # ---- 4. 图 ----
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.scatter(free["pred_cycles"], free["time_per_fec_h"], s=64, marker="o",
               c="#e53e3e", label="无约束 Pareto（含析锂风险）", zorder=3)
    ax.scatter(pc["pred_cycles"], pc["time_per_fec_h"], s=64, marker="s",
               c="#2b6cb0", label="受约束 Pareto（无析锂）", zorder=4)
    base_t = charge_time_h(1.0, 0.5, ct["a_cc"], ct["c_cv"], ct["d_tail"])
    ax.scatter([467.5], [base_t], s=110, marker="*", c="black", label="0.5C 满窗口基线", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("预测可用循环数")
    ax.set_ylabel("等效充满 1 FEC 的时长 (h)")
    ax.set_title("机理约束的代价：删掉「不可能」的那半个前沿")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "10_constrained_vs_free.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
