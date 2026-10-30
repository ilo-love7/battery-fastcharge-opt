"""04 — 多目标贝叶斯优化（阶段 3）：在（等效充满时长，可用循环数）平面求 Pareto 前沿。

产出:
    results/metrics/chargetime_params.csv   充电时长模型参数
    results/metrics/opt_pareto_bo.csv       BO 采样得到的 Pareto 协议
    results/metrics/opt_pareto_grid.csv     稠密网格参照 Pareto
    results/metrics/opt_report.csv          关键场景对照
    results/metrics/opt_baseline_protocols.csv  数据集中真实协议在目标平面上的位置
    results/models/gp_cycles.pkl            优化用 GP 代理
    results/figures/08_pareto_and_law.png
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

from fastcharge.chargetime import charge_time_h, fit_charge_time, time_savings_law  # noqa: E402
from fastcharge.config import FIGURES, METRICS, MODELS, ensure_dirs  # noqa: E402
from fastcharge.lifetime import add_throughput_columns  # noqa: E402
from fastcharge.optimize import (  # noqa: E402
    decision_space, design_space_coverage, objectives, run_bo,
)
from fastcharge.surrogate import fit_final  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

warnings.filterwarnings("ignore")


def decode(X: np.ndarray, gp_model, ct_params: dict) -> pd.DataFrame:
    """决策变量 → 可读协议表。"""
    span, rate, d_rate = X[:, 0], X[:, 1], X[:, 2]
    t_win = charge_time_h(span, rate, ct_params["a_cc"], ct_params["c_cv"], ct_params["d_tail"])
    F = objectives(X, gp_model, ct_params)
    cycles = np.exp(-F[:, 1])
    t_full = charge_time_h(np.ones_like(span), rate, ct_params["a_cc"], ct_params["c_cv"], ct_params["d_tail"])
    return pd.DataFrame(
        {
            "soc_window": span,
            "peak_rate_C": rate,
            "discharge_rate_C": d_rate,
            "time_to_charge_window_h": t_win,
            "time_per_fec_h": F[:, 0],
            "time_to_charge_full_h": t_full,
            "pred_cycles": cycles,
            "cycles_per_fec_at_full": cycles,
            "efc_per_hour": span / t_win,
        }
    )


def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")

    # ---- 1. 充电时长模型 ----
    print("拟合充电时长模型…")
    ct = fit_charge_time(cycles)
    print(f"  a={ct['a_cc']:.3f} c={ct['c_cv']:.3f} d={ct['d_tail']:.3f} "
          f"R²={ct['r2']:.3f} MAPE={ct['mape_pct']:.1f}%  (纯CC假设 R²={ct['r2_cc_only']:.3f})")
    ct["fit_table"].to_csv(METRICS / "chargetime_fit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in ct.items() if k != "fit_table"}]).to_csv(
        METRICS / "chargetime_params.csv", index=False, encoding="utf-8-sig")
    sav = time_savings_law(ct)
    sav.to_csv(METRICS / "chargetime_savings_law.csv", index=False, encoding="utf-8-sig")
    print("\n=== 充电时长收益递减（60% 窗口）===")
    print(sav.round(3).to_string(index=False))

    # ---- 2. GP 代理（目标 = 可用循环数）----
    print("\n拟合 GP 代理（全数据，目标 log(可用循环数)）…")
    gp, Xtr, ytr, names = fit_final(cells, target="cycles", model_name="gp")
    with open(MODELS / "gp_cycles.pkl", "wb") as fh:
        pickle.dump({"model": gp, "features": names, "ct_params": ct}, fh)
    print(f"  训练集 n={len(ytr)}, 目标范围 log-cycles = [{ytr.min():.2f}, {ytr.max():.2f}]")

    # ---- 3. BO ----
    print("\n多目标 BO：Sobol 初始化 + 双轴 EI 采集 + 适用域约束 + 网格参照…")
    cov = design_space_coverage(cells)
    cov.to_csv(METRICS / "design_space_coverage.csv", encoding="utf-8-sig")
    print("  设计空间覆盖（格内电芯数）:")
    print(cov.to_string())
    res = run_bo(gp, ct, X_decision=decision_space(cells), n_init=48, n_iter=16, seed=0)
    print(f"  BO 评估 {res['n_evals_bo']} 点 / 网格参照 {res['n_evals_grid']} 点")
    print(f"  超体积 BO={res['hv_bo']:.3f}  grid={res['hv_grid']:.3f}  "
          f"效率比={res['efficiency_ratio']:.3f}")

    pareto_bo = decode(res["pareto_bo"], gp, ct).sort_values("time_per_fec_h").reset_index(drop=True)
    pareto_bo.to_csv(METRICS / "opt_pareto_bo.csv", index=False, encoding="utf-8-sig")
    pareto_grid = decode(res["pareto_grid"], gp, ct).sort_values("time_per_fec_h").reset_index(drop=True)
    pareto_grid.to_csv(METRICS / "opt_pareto_grid.csv", index=False, encoding="utf-8-sig")

    show = ["soc_window", "peak_rate_C", "discharge_rate_C",
            "time_to_charge_window_h", "time_per_fec_h", "pred_cycles"]
    print("\n=== BO Pareto 前沿（按等效充满时长排序）===")
    print(pareto_bo[show].round(3).to_string(index=False))

    # ---- 4. 真实协议在目标平面上的位置（用来检验优化是否可信）----
    c = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    gs = c.groupby("group").agg(
        span=("soc_span", "median"), rate=("charge_rate_peak", "median"),
        drate=("discharge_rate_mean", "median"), cycles=("eol_cycle", "median")).reset_index()
    real = decode(gs[["span", "rate", "drate"]].to_numpy(float), gp, ct)
    real["group"] = gs["group"].to_numpy()
    real["true_cycles_median"] = gs["cycles"].to_numpy()
    real["pred_vs_true_pct"] = (real["pred_cycles"] / real["true_cycles_median"] - 1) * 100
    real.to_csv(METRICS / "opt_baseline_protocols.csv", index=False, encoding="utf-8-sig")
    print("\n=== 真实协议中已落在前沿附近的（按等效时长）===")
    print(real.sort_values("time_per_fec_h").head(8)[
        ["group", "soc_window", "peak_rate_C", "time_per_fec_h", "pred_cycles", "true_cycles_median"]
    ].round(3).to_string(index=False))

    # ---- 5. 关键场景对照 ----
    rows = []
    ref = decode(np.array([[1.0, 0.5, 0.5]]), gp, ct).iloc[0]
    picks = {
        "baseline_0.5C_full": ref,
        "pareto_fastest": pareto_bo.iloc[0],
        "pareto_balanced": pareto_bo.iloc[len(pareto_bo) // 2],
        "pareto_longest_life": pareto_bo.iloc[-1],
        # 数据集中实测最耐久的协议 G57（1.78C 只充 0.21 窗口）
        "measured_best_G57": real[real["group"] == 57].iloc[0] if (real["group"] == 57).any() else None,
    }
    for name, r in picks.items():
        if r is None:
            continue
        rows.append(
            {
                "scenario": name,
                "soc_window": float(r["soc_window"]),
                "peak_rate_C": float(r["peak_rate_C"]),
                "charge_time_window_h": float(r["time_to_charge_window_h"]),
                "time_per_fec_h": float(r["time_per_fec_h"]),
                "pred_cycles": float(r["pred_cycles"]),
                "time_vs_baseline_pct": (1 - float(r["time_per_fec_h"]) / float(ref["time_per_fec_h"])) * 100,
                "cycles_vs_baseline_pct": (float(r["pred_cycles"]) / float(ref["pred_cycles"]) - 1) * 100,
            }
        )
    rep = pd.DataFrame(rows)
    rep.to_csv(METRICS / "opt_report.csv", index=False, encoding="utf-8-sig")
    print("\n=== 关键场景对照（相对 0.5C 满窗口基线）===")
    print(rep.round(2).to_string(index=False))

    # ---- 6. 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax = axes[0]
    ax.scatter(-res["Y_grid"][:, 1], res["Y_grid"][:, 0], s=7, c="#e2e8f0", label="网格候选", zorder=1)
    ax.scatter(-res["pareto_grid_obj"][:, 1], res["pareto_grid_obj"][:, 0], s=46,
               c="#a0aec0", marker="s", label="网格 Pareto", zorder=2)
    ax.scatter(-res["pareto_bo_obj"][:, 1], res["pareto_bo_obj"][:, 0], s=62,
               c="#e53e3e", marker="o", label="BO Pareto", zorder=3)
    ax.scatter(real["pred_cycles"], real["time_per_fec_h"], s=34, c="#2b6cb0",
               marker="^", alpha=0.8, label="数据集真实协议 (60)", zorder=2)
    ax.set_xlabel("预测可用循环数")
    ax.set_ylabel("等效充满 1 FEC 的时长 (h)")
    ax.set_title("Pareto 前沿：等效充电时长 vs 可用寿命")
    ax.set_xscale("log")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(sav["rate_C"], sav["charge_time_h"], "o-", color="#2c5282", lw=1.8, label="充满 60% 总时长")
    ax2.plot(sav["rate_C"], sav["cc_part_h"], "s--", color="#38a169", label="CC 段")
    ax2.plot(sav["rate_C"], sav["cv_part_h"], "^--", color="#dd6b20", label="CV 段（与倍率无关）")
    ax2.axhline(sav["cv_part_h"].iloc[0], ls=":", color="#dd6b20", lw=1)
    ax2.set_xlabel("峰值充电倍率 (C)")
    ax2.set_ylabel("充满 60% 窗口的时长 (h)")
    ax2.set_title(f"充电时长收益递减：CV 硬地板 {ct['c_cv']:.2f}·span + {ct['d_tail']:.2f} h")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "08_pareto_and_law.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
