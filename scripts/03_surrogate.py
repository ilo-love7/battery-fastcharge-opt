"""03 — 代理模型：协议 → 寿命结局（阶段 2）。

产出:
    results/metrics/model_leaderboard.csv   LOGO-CV 四模型 × 双目标 榜单
    results/metrics/shap_attribution.csv    SHAP 归因
    results/metrics/predictions_logo.csv    逐电芯 LOGO 预测（误差分解用）
    results/figures/06_model_leaderboard.png
    results/figures/07_parity.png
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
from fastcharge.surrogate import (  # noqa: E402
    CORE_FEATURES,
    evaluate_all,
    make_targets,
    shap_attribution,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")
    print(f"载入 {len(cells)} 电芯，特征 {len(CORE_FEATURES)} 维，按协议 LOGO 交叉验证…")

    board, preds = evaluate_all(cells, seed=0)
    board.to_csv(METRICS / "model_leaderboard.csv", index=False, encoding="utf-8-sig")
    print("\n=== LOGO-CV 榜单 ===")
    print(board.round(3).to_string(index=False))

    # 逐电芯预测表（用 GP 的 efc 预测）
    from fastcharge.surrogate import modeling_rows

    tgt = make_targets(cells)
    keep = modeling_rows(tgt)
    pred_df = pd.DataFrame(
        {
            "cell_id": tgt.loc[keep, "cell_id"].to_numpy(),
            "group": tgt.loc[keep, "group"].to_numpy(),
            "true_efc": tgt.loc[keep, "efc"].to_numpy(),
            "true_cycles": tgt.loc[keep, "eol_cycle"].to_numpy(),
            "soc_span": tgt.loc[keep, "soc_span"].to_numpy(),
            "pred_efc_log": preds["gp_efc"],
            "pred_cycles_log": preds["gp_cycles"],
        }
    )
    pred_df["pred_efc"] = np.exp(pred_df["pred_efc_log"])
    pred_df["pred_cycles"] = np.exp(pred_df["pred_cycles_log"])
    pred_df["resid_log"] = pred_df["pred_efc_log"] - np.log(pred_df["true_efc"])
    pred_df["err_pct"] = (pred_df["pred_efc"] / pred_df["true_efc"] - 1) * 100
    pred_df.to_csv(METRICS / "predictions_logo.csv", index=False, encoding="utf-8-sig")

    # SHAP
    print("\n计算 SHAP 归因…")
    shap_df = shap_attribution(cells, target="efc", model_name="rf")
    shap_df.to_csv(METRICS / "shap_attribution_efc.csv", index=False, encoding="utf-8-sig")
    print(shap_df.round(3).to_string(index=False))

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, t in zip(axes, ("efc", "cycles")):
        sub = board[board["target"] == t].sort_values("r2")
        labels = sub["model"] + " → R²=" + sub["r2"].round(3).astype(str)
        colors = ["#cbd5e0"] * (len(sub) - 1) + ["#2b6cb0"]
        ax.barh(range(len(sub)), sub["mape_pct"], color=colors)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("LOGO-CV MAPE (%)")
        ax.set_title(f"目标: log({t})")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("协议 → 寿命 代理模型：按协议留一交叉验证")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_model_leaderboard.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (xc, yc, lab) in zip(
        axes,
        [("true_efc", "pred_efc", "等效满循环 EFC"), ("true_cycles", "pred_cycles_log", "EOL 循环数 (log)")],
    ):
        if yc == "pred_cycles_log":
            yv = np.exp(pred_df[yc])
            xv = pred_df[xc]
            ax.set_yscale("log")
        else:
            yv = pred_df[yc]
            xv = pred_df[xc]
        ax.scatter(xv, yv, s=22, c=pred_df["group"] % 10, cmap="tab10", alpha=0.85)
        lim = [min(xv.min(), yv.min()) * 0.9, max(xv.max(), yv.max()) * 1.1]
        ax.plot(lim, lim, "k--", lw=1, label="理想")
        ax.set_xlabel(f"实测 {lab}")
        ax.set_ylabel(f"LOGO 预测 {lab}")
        ax.set_title(f"Parity 图（GP，R²={board[(board.target==('efc' if yc=='pred_efc' else 'cycles'))&(board.model=='gp')].r2.iloc[0]:.3f}）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_parity.png")
    plt.close(fig)

    # ---- 按 SOC 窗口分层的误差分析 ----
    merged = pred_df.copy()
    merged["span_bin"] = pd.cut(merged["soc_span"], [0.0, 0.35, 0.55, 0.75, 0.95, 1.01])
    strat = merged.groupby("span_bin", observed=True).agg(
        n=("cell_id", "size"),
        mape=("err_pct", lambda s: float(np.mean(np.abs(s)))),
        bias=("err_pct", "mean"),
        resid_std=("resid_log", "std"),
    ).reset_index()
    strat["span_bin"] = strat["span_bin"].astype(str)
    strat.to_csv(METRICS / "error_by_span_bin.csv", index=False, encoding="utf-8-sig")
    print("\n=== 按 SOC 窗口分层的误差 ===")
    print(strat.round(2).to_string(index=False))
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
