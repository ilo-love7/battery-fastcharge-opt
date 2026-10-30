"""08 — README 主图与报告素材生成（一键出图，供 README 引用）。

产出:
    docs/figures/main_pareto.png         主图：受约束 Pareto + 真实协议散点
    docs/figures/main_chargetime_law.png 副图：充电时长收益递减
    docs/figures/main_plating.png        机理图：析锂安全 map
    docs/figures/main_dual_metric.png    口径图：循环数 vs EFC 的相关性差
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fastcharge.chargetime import charge_time_h, fit_charge_time  # noqa: E402
from fastcharge.config import METRICS, MODELS  # noqa: E402
from fastcharge.lifetime import add_throughput_columns  # noqa: E402
from fastcharge.optimize import decision_space, objectives  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig_main_pareto(ct, gp, cells) -> None:
    free = pd.read_csv(METRICS / "opt_pareto_bo.csv")
    cons = pd.read_csv(METRICS / "opt_pareto_constrained.csv")
    card = pd.read_csv(METRICS / "recommendation_card.csv")

    c = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    gs = c.groupby("group").agg(span=("soc_span", "median"), rate=("charge_rate_peak", "median"),
                                drate=("discharge_rate_mean", "median"))
    X = gs.to_numpy(float)
    t_win = charge_time_h(X[:, 0], X[:, 1], ct["a_cc"], ct["c_cv"], ct["d_tail"])
    F = objectives(X, gp, ct)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.scatter(np.exp(-F[:, 1]), t_win / X[:, 0], s=30, c="#cbd5e0", alpha=0.85,
               label=f"数据集 60 组真实协议", zorder=1)
    ax.scatter(free["pred_cycles"], free["time_per_fec_h"], s=62, marker="o", c="#e53e3e",
               alpha=0.9, label="无约束 Pareto（含析锂风险）", zorder=3)
    ax.scatter(cons["pred_cycles"], cons["time_per_fec_h"], s=68, marker="s", c="#2b6cb0",
               label="受机理约束 Pareto（推荐）", zorder=4)

    marks = {"A 机理保守版（1C 上限）": ("A", "#38a169"),
             "B 数据实证版（1.5C）": ("B", "#805ad5"),
             "C 折中版（1.25C）": ("C", "#dd6b20"),
             "D 基线对照 0.5C 满窗口": ("D", "black")}
    for _, r in card.iterrows():
        m, col = marks.get(r["protocol"], ("?", "gray"))
        ax.scatter([r["pred_cycles"]], [r["time_per_fec_h"]], s=190, marker="*",
                   c=col, edgecolors="white", linewidths=1.2, zorder=6)
        ax.annotate(m, (r["pred_cycles"], r["time_per_fec_h"]),
                    textcoords="offset points", xytext=(9, 7), fontsize=11,
                    fontweight="bold", color=col)

    ax.set_xscale("log")
    ax.set_xlabel("预测可用循环数（等效满循环口径）")
    ax.set_ylabel("等效充满 1 FEC 的时长 (h)")
    ax.set_title("快充协议权衡：速度 vs 寿命（含析锂机理约束）")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "main_pareto.png", bbox_inches="tight")
    plt.close(fig)


def fig_law(ct) -> None:
    law = pd.read_csv(METRICS / "chargetime_savings_law.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(law["rate_C"], law["charge_time_h"], "o-", color="#2c5282", lw=1.8, label="总时长")
    ax.fill_between(law["rate_C"], 0, law["cc_part_h"], color="#c6f6d5", alpha=0.75, label="CC 段")
    ax.plot(law["rate_C"], law["cc_part_h"], "s--", color="#38a169", lw=1.2)
    ax.plot(law["rate_C"], law["cv_part_h"], "^--", color="#dd6b20", lw=1.4,
            label="CV 段（与倍率无关）")
    ax.axhline(law["cv_part_h"].iloc[0], ls=":", color="#dd6b20", lw=1)
    ax.annotate("CV 硬地板 ≈ 0.43·span + 0.17 h\n→ 倍率再高也快不过它",
                xy=(4, law["cv_part_h"].iloc[0]), xytext=(3.2, law["cv_part_h"].iloc[0] + 0.32),
                fontsize=8.5, color="#dd6b20",
                arrowprops=dict(arrowstyle="->", color="#dd6b20", lw=1))
    ax.set_xlabel("峰值充电倍率 (C)")
    ax.set_ylabel("充满 60% 窗口的时长 (h)")
    ax.set_title(f"充电时长收益递减定律（实测拟合 R²={ct['r2']:.3f}）")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "main_chargetime_law.png", bbox_inches="tight")
    plt.close(fig)


def fig_plating() -> None:
    tmap = pd.read_csv(METRICS / "plating_temp_map.csv")
    piv = tmap.pivot_table(index="temperature_C", columns="c_rate", values="eta_min_V", observed=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="RdYlGn", origin="lower",
                   vmin=-0.07, vmax=0.07)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c:.1f}" for c in piv.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{t:.0f}" for t in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        fontweight="bold" if v < 0 else "normal")
    ax.set_xlabel("充电倍率 (C)")
    ax.set_ylabel("温度 (°C)")
    ax.set_title("PyBaMM 析锂判据：min η_pl（负值=析锂）")
    fig.colorbar(im, ax=ax, label="min η_pl (V)", fraction=0.045)
    fig.tight_layout()
    fig.savefig(OUT / "main_plating.png", bbox_inches="tight")
    plt.close(fig)


def fig_dual_metric(cells) -> None:
    c = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    ax = axes[0]
    sc = ax.scatter(c["soc_span"], c["eol_cycle"], c=c["charge_rate_peak"], cmap="viridis", s=26)
    ax.set_xlabel("SOC 窗口宽度")
    ax.set_ylabel("表观 EOL 循环数")
    ax.set_yscale("log")
    ax.set_title("口径① 循环数：ρ(窗口, 寿命) = −0.875")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="峰值倍率 (C)")

    ax2 = axes[1]
    sc2 = ax2.scatter(c["soc_span"], c["efc"], c=c["charge_rate_peak"], cmap="viridis", s=26)
    ax2.set_xlabel("SOC 窗口宽度")
    ax2.set_ylabel("等效满循环 EFC")
    ax2.set_title("口径② EFC：ρ 降到 −0.633，伪优势被削掉 28%")
    ax2.grid(alpha=0.3)
    fig.colorbar(sc2, ax=ax2, label="峰值倍率 (C)")
    fig.tight_layout()
    fig.savefig(OUT / "main_dual_metric.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")
    ct = fit_charge_time(cycles)
    with open(MODELS / "gp_cycles.pkl", "rb") as fh:
        gp = pickle.load(fh)["model"]

    fig_main_pareto(ct, gp, cells)
    fig_law(ct)
    fig_plating()
    fig_dual_metric(cells)
    print(f"主图已写入 {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}  {f.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
