"""11 — 深化③：早期寿命预测（用前 N 圈预测终寿命，加速优化闭环）。

产出:
    results/metrics/early_prediction.csv        N × 模型 的 LOGO-CV 精度
    results/metrics/early_features.csv          逐电芯早期特征（N=50/200）
    results/metrics/early_test_time.csv         早期预测能省多少电池台时
    results/figures/13_early_prediction.png
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

from fastcharge.config import FIGURES, METRICS, ensure_dirs  # noqa: E402
from fastcharge.early import (  # noqa: E402
    EARLY_NS,
    build_all,
    compute_generation_hours,
    evaluate_early,
    vs_full_protocol_baseline,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
warnings.filterwarnings("ignore")


def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")
    print(f"载入 {len(cells)} 电芯 / {len(cycles):,} 循环")

    print("\n=== 1) 构建早期特征 ===")
    feats = build_all(cycles, cells)
    for n, df in feats.items():
        n_ok = int(df["eol_cycle"].notna().sum())
        print(f"  N={n:>3}: {df.shape[1]-7} 个特征, {n_ok} 电芯")

    print("\n=== 2) 早期预测精度（按协议 LOGO-CV）===")
    rows = [vs_full_protocol_baseline(feats[EARLY_NS[0]], 0)]
    print(f"  基线（只用协议参数）: R²={rows[0]['r2']:.3f} MAPE={rows[0]['mape_pct']:.1f}%")
    for n in EARLY_NS:
        for model in ("ridge", "rf", "xgb"):
            met = evaluate_early(feats[n], n, model=model)
            rows.append(met)
            print(f"  N={n:>3} {model:>5}: R²={met['r2']:.3f}  MAPE={met['mape_pct']:5.1f}%  "
                  f"RMSE(log)={met['rmse']:.3f}")
    pred = pd.DataFrame(rows)
    pred.to_csv(METRICS / "early_prediction.csv", index=False, encoding="utf-8-sig")

    # ---- 最优曲线（每个 N 取最佳模型）----
    curve = (pred[pred["n_head"].apply(lambda x: isinstance(x, (int, np.integer)))]
             .sort_values("mape_pct").groupby("n_head", as_index=False).first())
    curve.to_csv(METRICS / "early_curve.csv", index=False, encoding="utf-8-sig")
    print("\n=== 3) 精度随观测圈数 ===")
    print(curve[["n_head", "model", "r2", "mape_pct"]].round(3).to_string(index=False))

    base = pred.iloc[0]
    gain = pd.DataFrame({
        "n_head": curve["n_head"],
        "mape_pct": curve["mape_pct"],
        "vs_protocol_only_pct": (base["mape_pct"] - curve["mape_pct"]) / base["mape_pct"] * 100,
        "promise_ok": curve["mape_pct"] < 15,
    })

    # ---- 4. 台时节省 ----
    print("\n=== 4) 早期预测能省多少电池台时 ===")
    hours = compute_generation_hours(cycles, cells)
    hours.to_csv(METRICS / "early_test_time.csv", index=False, encoding="utf-8-sig")
    print(hours.round(1).to_string(index=False))

    # ---- 5. 单只电芯的早期轨迹展示 ----
    pick = cells.dropna(subset=["eol_cycle"]).sort_values("eol_cycle")
    short = pick.iloc[len(pick) // 10]
    long_ = pick.iloc[int(len(pick) * 0.9)]

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))

    ax = axes[0]
    for r in (short, long_):
        s = cycles[cycles["cell_id"] == r["cell_id"]].sort_values("cycle_index")
        q = s["q_norm_Ah"].to_numpy(float) / float(r["q_ref_Ah"])
        ax.plot(s["cycle_number"], q, lw=1.2,
                label=f"{r['cell_id']} (EOL={int(r['eol_cycle'])})")
    ax.axhline(0.8, ls=":", color="gray", label="EOL 80%")
    for n in (20, 50, 100, 200):
        ax.axvline(n, color="#e2e8f0", lw=0.8)
    ax.set_xlabel("循环数")
    ax.set_ylabel("归一化容量")
    ax.set_title("早期观测窗口 vs 终寿命")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(600, int(long_["eol_cycle"] * 1.05)))

    ax2 = axes[1]
    ints = curve["n_head"].astype(int)
    ax2.plot(ints, curve["mape_pct"], "o-", color="#2c5282", lw=1.8, label="早期曲线特征")
    ax2.axhline(base["mape_pct"], ls="--", color="#dd6b20", lw=1.2, label="只用协议参数")
    best_n = int(curve.sort_values("mape_pct").iloc[0]["n_head"])
    ax2.axvline(best_n, ls=":", color="#38a169")
    ax2.set_xlabel("观测圈数 N")
    ax2.set_ylabel("LOGO-CV MAPE (%)")
    ax2.set_title("精度随观测圈数（按协议留一）")
    ax2.set_xscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    ax3 = axes[2]
    ax3.plot(hours["n_head"], hours["saving_pct"], "s-", color="#805ad5", lw=1.8)
    ax3.set_xlabel("观测圈数 N")
    ax3.set_ylabel("相对跑满 EOL 的台时节省 (%)")
    ax3.set_title("预测提速：省下的电池测试时间")
    ax3.set_xscale("log")
    ax3.grid(alpha=0.3, which="both")
    for _, r in hours.iterrows():
        ax3.annotate(f"{r['saving_pct']:.0f}%", (r["n_head"], r["saving_pct"]),
                     textcoords="offset points", xytext=(3, 5), fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGURES / "13_early_prediction.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
