"""17 — 深化⑧：温度自适应协议表（工程交付物）。

为什么这是最有价值的一步
------------------------
前七步都是"分析"，这一步是**能直接贴到 BMS 规格书里的东西**：
一张按温度分档的协议表，给出每档可用的最大倍率及其析锂裕度。

数据来源
--------
PyBaMM 的 rate_limit_table（起始 SOC × 温度 → 安全倍率上限），
已在 scripts/06 生成。本脚本把它编译成：
    1. 温度分档协议表（含裕度、预计时长、寿命预测）
    2. 若超出上限，给出"降到多少才安全"的具体建议
    3. 冬季/夏季两套预设

产出:
    results/metrics/protocol_table_by_temp.csv   温度分档协议表
    results/metrics/compliance_check.csv         一组实测协议的合规性检查
    results/metrics/winter_summer_presets.csv    冬/夏两套预设
    docs/figures/main_temp_table.png             交付图
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
from fastcharge.config import FIGURES, METRICS, MODELS  # noqa: E402
from fastcharge.optimize import objectives, feature_rows  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
warnings.filterwarnings("ignore")

OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

#: 温度分档（对应实际气候带）
TEMP_BINS = [(0, 10, "严寒 0–10°C"), (10, 20, "低温 10–20°C"),
             (20, 30, "常温 20–30°C"), (30, 40, "偏热 30–40°C"),
             (40, 50, "高温 40–50°C")]

#: 候选窗口
WINDOWS = (0.40, 0.50, 0.60, 0.75)


def load_limit_table() -> pd.DataFrame:
    return pd.read_csv(METRICS / "rate_limit_table.csv")


def limit_at(table: pd.DataFrame, soc_start: float, temp: float) -> float:
    """二维插值：给定起始 SOC 与温度。"""
    temps = np.sort(table["temperature_C"].unique().astype(float))
    socs = np.sort(table["soc_start"].unique().astype(float))
    vals = np.array([np.interp(soc_start, socs,
                               table[table["temperature_C"] == T]
                               .sort_values("soc_start")["rate_limit_C"].to_numpy(float))
                     for T in temps])
    return float(np.interp(temp, temps, vals))


def build_table(table: pd.DataFrame, gp, ct) -> pd.DataFrame:
    """编译温度分档协议表。"""
    rows = []
    for lo_t, hi_t, label in TEMP_BINS:
        t_mid = (lo_t + hi_t) / 2
        for span in WINDOWS:
            soc_lo = 1.0 - span
            lim = limit_at(table, soc_lo, t_mid)
            # 保守取整到 0.05C
            rate = np.floor(lim * 20) / 20 if np.isfinite(lim) else np.nan
            if not np.isfinite(rate) or rate <= 0:
                continue
            t_win = float(charge_time_h(span, rate, ct["a_cc"], ct["c_cv"], ct["d_tail"]))
            X = np.array([[span, rate, 1.0]])
            F = objectives(X, gp, ct)
            rows.append({
                "temp_bin": label, "temp_C": t_mid,
                "soc_window": span, "soc_lo": soc_lo,
                "safe_rate_ceiling_C": lim,
                "recommended_rate_C": rate,
                "margin_C": lim - rate,
                "charge_time_window_h": t_win,
                "time_per_fec_h": t_win / span,
                "pred_cycles": float(np.exp(-F[0, 1])),
            })
    return pd.DataFrame(rows)


def compliance(cells: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """检查数据集里的实测协议在各温度下是否合规。"""
    rows = []
    for _, c in cells.iterrows():
        for lo_t, hi_t, label in TEMP_BINS:
            t_mid = (lo_t + hi_t) / 2
            lim = limit_at(table, c["soc_lo"], t_mid)
            rate = c["charge_rate_peak"]
            rows.append({
                "group": int(c["group"]), "cell_id": c["cell_id"],
                "temp_bin": label, "rate_C": rate,
                "safe_limit_C": lim,
                "compliant": bool(rate <= lim + 1e-9),
                "overshoot_C": max(rate - lim, 0.0),
            })
    agg = (rows and pd.DataFrame(rows).groupby(["temp_bin", "rate_C"]).agg(
        n=("cell_id", "size"), compliant_n=("compliant", "sum"),
        compliant_frac=("compliant", "mean"), mean_overshoot=("overshoot_C", "mean"),
    ).reset_index())
    return agg


def presets(tbl: pd.DataFrame) -> pd.DataFrame:
    """冬/夏两套预设：给 BMS 标定一份即用的参数。"""
    rows = []
    for name, t_lo, t_hi, window in [
        ("冬季预设 (0–15°C)", 5, 12, 0.60),
        ("夏季预设 (30–45°C)", 30, 42, 0.50),
    ]:
        sub = tbl[(tbl["temp_C"] >= t_lo) & (tbl["temp_C"] <= t_hi)
                  & ((tbl["soc_window"] - window).abs() < 0.06)]
        if sub.empty:
            continue
        best = sub.sort_values("charge_time_window_h").iloc[0]
        rows.append({
            "preset": name,
            "soc_window": best["soc_window"],
            "max_rate_C": best["recommended_rate_C"],
            "ceiling_C": best["safe_rate_ceiling_C"],
            "charge_time_window_h": best["charge_time_window_h"],
            "time_per_fec_h": best["time_per_fec_h"],
            "pred_cycles": best["pred_cycles"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")
    table = load_limit_table()
    from fastcharge.chargetime import fit_charge_time
    ct = fit_charge_time(cycles)

    with open(MODELS / "gp_cycles.pkl", "rb") as fh:
        gp = pickle.load(fh)["model"]

    print("=== 1) 温度分档协议表 ===")
    tbl = build_table(table, gp, ct)
    tbl.to_csv(METRICS / "protocol_table_by_temp.csv", index=False, encoding="utf-8-sig")
    print(tbl.round(3).to_string(index=False))

    print("\n=== 2) 实测协议的温度合规性 ===")
    comp = compliance(cells, table)
    comp.to_csv(METRICS / "compliance_check.csv", index=False, encoding="utf-8-sig")
    piv = comp.pivot_table(index="temp_bin", values="compliant_frac", aggfunc="mean")
    print(piv.round(3).to_string())
    for _, r in comp.groupby("temp_bin", as_index=False).agg(
            frac=("compliant_frac", "mean"), n=("n", "sum")).iterrows():
        print(f"  {r['temp_bin']:16s} 合规率 {r['frac']*100:5.1f}%  (n={int(r['n'])})")

    print("\n=== 3) 冬/夏预设 ===")
    pre = presets(tbl)
    pre.to_csv(METRICS / "winter_summer_presets.csv", index=False, encoding="utf-8-sig")
    print(pre.round(3).to_string(index=False))

    # ---- 图（交付级） ----
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    pivot = tbl.pivot(index="temp_bin", columns="soc_window", values="recommended_rate_C")
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}C", ha="center", va="center",
                        fontsize=11, fontweight="bold")
    ax.set_xlabel("SOC 窗口")
    ax.set_ylabel("温度分档")
    ax.set_title("温区自适应协议表：每格推荐的最大安全倍率")
    fig.colorbar(im, ax=ax, label="推荐倍率 (C)", fraction=0.046)

    ax2 = axes[1]
    for span in pivot.columns:
        ax2.plot(range(len(pivot)), pivot[span].to_numpy(float), "o-", lw=1.8,
                 label=f"窗口 {span:.0%}")
    ax2.set_xticks(range(len(pivot)))
    ax2.set_xticklabels([b.split(" ")[0] for b in pivot.index], rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("推荐最大倍率 (C)")
    ax2.set_xlabel("温度分档")
    ax2.set_title("温区越高 → 可用倍率越高（但高温本身伤寿命）")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "main_temp_table.png", bbox_inches="tight")
    fig.savefig(FIGURES / "18_temp_table.png")
    plt.close(fig)
    print(f"\n图已写入 {OUT} 与 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
