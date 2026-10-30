"""02 — 协议 → 寿命 统计分析（v1.1 阶段 1 的全部内容）。

产出:
    results/metrics/group_summary.csv          63 协议逐组汇总
    results/metrics/dose_response.csv          倍率 → 寿命 剂量-反应
    results/metrics/spearman.csv               协议特征 vs 寿命 秩相关
    results/metrics/kruskal.csv                单要素 Kruskal-Wallis 检验
    results/metrics/interaction_rate_soc.csv   倍率 × SOC 窗口 交互表
    results/metrics/protocol_ranking.csv       最优/最差协议榜单
    results/figures/*.png                      EDA 图

用法:
    python scripts/02_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fastcharge.config import FIGURES, METRICS, ensure_dirs  # noqa: E402
from fastcharge.lifetime import (  # noqa: E402
    factorial_interaction,
    group_summary,
    kruskal_by_bins,
    pairwise_same_peak,
    protocol_ranking,
    rate_dose_response,
    spearman_report,
)
from fastcharge.protocol import build_feature_matrix  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def fig_lifetime_distribution(gs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = gs.sort_values("eol_median")
    ax.scatter(order["charge_rate_peak"], order["eol_median"], s=28, c="#2b6cb0", zorder=3)
    for _, r in order.iterrows():
        ax.plot([r["charge_rate_peak"] * 0.99, r["charge_rate_peak"] * 1.01],
                [r["eol_min"], r["eol_max"]], color="#cbd5e0", lw=0.8, zorder=1)
    ax.set_xlabel("峰值充电倍率 (C)")
    ax.set_ylabel("EOL 循环数（中位数，竖线=组内 min–max）")
    ax.set_title("63 协议：峰值充电倍率 vs 寿命结局")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_rate_vs_lifetime.png")
    plt.close(fig)


def fig_soc_window(cells: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sc = axes[0].scatter(cells["soc_lo"], cells["eol_cycle"], c=cells["charge_rate_peak"],
                         cmap="viridis", s=26, alpha=0.85)
    axes[0].set_xlabel("充电起始 SOC")
    axes[0].set_ylabel("EOL 循环数")
    axes[0].set_yscale("log")
    axes[0].set_title("起始 SOC：越浅充越长寿")
    fig.colorbar(sc, ax=axes[0], label="峰值倍率 (C)")
    axes[0].grid(alpha=0.3)

    sc2 = axes[1].scatter(cells["soc_span"], cells["eol_cycle"], c=cells["charge_rate_peak"],
                          cmap="viridis", s=26, alpha=0.85)
    axes[1].set_xlabel("SOC 窗口宽度")
    axes[1].set_ylabel("EOL 循环数")
    axes[1].set_yscale("log")
    axes[1].set_title("SOC 窗口宽度 vs 寿命")
    fig.colorbar(sc2, ax=axes[1], label="峰值倍率 (C)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_soc_window.png")
    plt.close(fig)


def fig_interaction(piv: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    data = piv.to_numpy(dtype=float)
    im = ax.imshow(np.log10(data), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c}" for c in piv.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{c}" for c in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("SOC 窗口宽度分箱")
    ax.set_ylabel("峰值倍率分箱")
    ax.set_title("倍率 × SOC 窗口 交互（中位 EOL，色=log10）")
    fig.colorbar(im, ax=ax, label="log10(EOL 循环)")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_rate_soc_interaction.png")
    plt.close(fig)


def fig_dose(resp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(resp["charge_rate_peak"], resp["eol_q25"], resp["eol_q75"],
                    color="#bee3f8", alpha=0.7, label="25–75 分位")
    ax.plot(resp["charge_rate_peak"], resp["eol_median"], "o-", color="#2c5282", label="中位数")
    ax.set_xlabel("峰值充电倍率 (C)")
    ax.set_ylabel("EOL 循环数")
    ax.set_yscale("log")
    ax.set_title("放电固定 0.5C 分组的倍率-寿命剂量反应")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_dose_response.png")
    plt.close(fig)


def fig_capacity_curves(cells: pd.DataFrame, cycles: pd.DataFrame) -> None:
    """挑 6 个典型协议画容量衰减轨迹（归一化到 q_ref）。"""
    gs = group_summary(cells)
    picks = []
    for target in (1, 3, 6, 16, 24, 49):
        if target in gs["group"].values:
            picks.append(target)
    # 用 EOL 分位数补充
    qs = gs["eol_median"].quantile([0.15, 0.5, 0.85]).to_numpy()
    for qv in qs:
        g = gs.iloc[(gs["eol_median"] - qv).abs().argsort()[:1]]["group"].iloc[0]
        if g not in picks:
            picks.append(int(g))
    picks = picks[:7]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(picks):
        sub = cycles[cycles["group"] == g]
        cid = cells[cells["group"] == g].sort_values("n_cycles").iloc[len(cells[cells['group']==g])//2]["cell_id"]
        s = sub[sub["cell_id"] == cid]
        if s.empty:
            continue
        qref = float(cells.loc[cells["cell_id"] == cid, "q_ref_Ah"].iloc[0])
        ax.plot(s["cycle_number"], s["q_norm_Ah"] / qref, lw=1.2, color=cmap(i),
                label=f"G{g} ({s['charge_rate_peak'].iloc[0]:.2f}C/{s['discharge_rate_mean'].iloc[0]:.2f}C, "
                      f"span {s['soc_span'].iloc[0]:.2f})")
    ax.axhline(0.8, ls=":", color="gray", label="EOL 80%")
    ax.set_xlabel("循环数")
    ax.set_ylabel("归一化容量 SOH")
    ax.set_title("典型协议的容量衰减轨迹")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_capacity_curves.png")
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")
    print(f"载入: {len(cells)} 电芯 / {len(cycles):,} 循环")

    gs = group_summary(cells)
    gs.to_csv(METRICS / "group_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n协议组: {len(gs)}")
    print(gs[["group", "charge_rate_peak", "discharge_rate_mean", "soc_lo", "soc_span",
              "n_cells", "eol_median"]].head(8).to_string(index=False))

    resp = rate_dose_response(cells)
    resp.to_csv(METRICS / "dose_response.csv", index=False, encoding="utf-8-sig")

    X, names = build_feature_matrix(cells)
    feats_df = pd.concat([cells[["cell_id", "group", "eol_cycle", "censored"]], X], axis=1)
    sp = spearman_report(feats_df, names)
    sp.to_csv(METRICS / "spearman.csv", index=False, encoding="utf-8-sig")
    print("\n=== 协议特征 vs 寿命 Spearman 前 8 ===")
    print(sp.head(8).to_string(index=False))

    kr = []
    kr.append(kruskal_by_bins(cells, "charge_rate_peak", [0, 0.75, 1.25, 1.85, 2.5, 3.1], "峰值充电倍率"))
    kr.append(kruskal_by_bins(cells, "soc_lo", [-0.01, 0.15, 0.35, 0.55, 0.75, 0.85], "充电起始 SOC"))
    kr.append(kruskal_by_bins(cells, "soc_span", [0.0, 0.3, 0.5, 0.7, 0.8, 1.01], "SOC 窗口宽度"))
    kr.append(kruskal_by_bins(cells, "discharge_rate_mean", [0, 0.7, 1.2, 1.8, 2.5], "放电倍率"))
    kr_df = pd.DataFrame([{k: v for k, v in r.items() if k != "group_medians"} for r in kr])
    kr_df.to_csv(METRICS / "kruskal.csv", index=False, encoding="utf-8-sig")
    print("\n=== 单要素 Kruskal-Wallis ===")
    print(kr_df.to_string(index=False))

    piv = factorial_interaction(
        cells, "charge_rate_peak", "soc_span",
        [0, 0.5, 1.0, 1.6, 2.2, 3.1], [0.0, 0.3, 0.5, 0.7, 0.8, 1.01],
    )
    piv.to_csv(METRICS / "interaction_rate_soc.csv", encoding="utf-8-sig")

    pw = pairwise_same_peak(cells)
    pw.to_csv(METRICS / "pairwise_soc_within_rate.csv", index=False, encoding="utf-8-sig")
    print("\n=== 同倍率下 SOC 起点高低对比 ===")
    if len(pw):
        print(pw.round(2).to_string(index=False))

    rk = protocol_ranking(gs, top=8)
    rk.to_csv(METRICS / "protocol_ranking.csv", index=False, encoding="utf-8-sig")
    print("\n=== 协议寿命榜单 ===")
    print(rk.round(2).to_string(index=False))

    fig_lifetime_distribution(gs)
    fig_soc_window(cells)
    fig_interaction(piv)
    fig_dose(resp)
    fig_capacity_curves(cells, cycles)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
