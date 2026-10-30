"""05 — PyBaMM 机理闭环验证（阶段 4）：析锂安全边界 + Pareto 协议一致性核查。

产出:
    results/metrics/plating_envelope_SPM.csv     倍率 × 温度 SPM 包络
    results/metrics/plating_envelope_DFN.csv     DFN 关键点复核
    results/metrics/plating_temp_map.csv         温度 × 倍率 安全 map
    results/metrics/safe_rate_limits.csv          各温度下的安全倍率上限
    results/metrics/pareto_plating_check.csv     Pareto 协议逐条析锂核查
    results/figures/09_plating_map.png
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
from fastcharge.physics import (  # noqa: E402
    SAFETY_MARGIN_V,
    check_protocols,
    safe_rate_limit,
    scan_plating_envelope,
    temperature_rate_map,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def main() -> int:
    ensure_dirs()

    print("=== 1) SPM 倍率 × 温度 析锂包络 ===")
    env_spm = scan_plating_envelope(
        rates=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0),
        temps=(0.0, 15.0, 30.0, 45.0), model="SPM")
    env_spm.to_csv(METRICS / "plating_envelope_SPM.csv", index=False, encoding="utf-8-sig")

    print("\n=== 2) DFN 关键点复核 ===")
    env_dfn = scan_plating_envelope(
        rates=(1.0, 2.0, 3.0, 4.0), temps=(0.0, 30.0), model="DFN")
    env_dfn.to_csv(METRICS / "plating_envelope_DFN.csv", index=False, encoding="utf-8-sig")

    print("\n=== 3) 温度 × 倍率 安全 map ===")
    tmap = temperature_rate_map(
        rates=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0),
        temps=(0.0, 10.0, 20.0, 30.0, 45.0, 60.0))
    tmap.to_csv(METRICS / "plating_temp_map.csv", index=False, encoding="utf-8-sig")

    print("\n=== 4) 各温度安全倍率上限 ===")
    limits = []
    for T in sorted(env_spm["temperature_C"].unique()):
        r = safe_rate_limit(env_spm, temperature_C=T)
        limits.append({"temperature_C": T, "safe_rate_limit_C": r})
        print(f"  {T:.0f}°C → {r:.2f}C")
    pd.DataFrame(limits).to_csv(METRICS / "safe_rate_limits.csv", index=False, encoding="utf-8-sig")

    # ---------- 5) Pareto 协议一致性核查 ----------
    pf_path = METRICS / "opt_pareto_bo.csv"
    if pf_path.exists():
        pareto = pd.read_csv(pf_path)
        print(f"\n=== 5) Pareto 协议析锂核查（{len(pareto)} 条，30°C）===")
        chk = check_protocols(pareto, model="SPM", temperature_C=30.0)
        chk["within_safe_limit"] = chk["min_plating_overpotential_V"] >= SAFETY_MARGIN_V
        chk.to_csv(METRICS / "pareto_plating_check.csv", index=False, encoding="utf-8-sig")
        cols = ["peak_rate_C", "soc_window", "min_plating_overpotential_V",
                "plated", "within_safe_limit", "pred_cycles"]
        print(chk[[c for c in cols if c in chk.columns]].round(4).to_string(index=False))
        n_viol = int((~chk["within_safe_limit"]).sum())
        print(f"\n  越界协议数: {n_viol} / {len(chk)}")

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax = axes[0]
    piv = tmap.pivot_table(index="temperature_C", columns="c_rate",
                           values="eta_min_V", observed=True)
    im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{c:.1f}" for c in piv.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{t:.0f}" for t in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_xlabel("充电倍率 (C)")
    ax.set_ylabel("温度 (°C)")
    ax.set_title("析锂安全 map：min η_pl (V)，绿色=安全")
    fig.colorbar(im, ax=ax, label="min η_pl (V)")

    ax2 = axes[1]
    for T, sub in env_spm.groupby("temperature_C"):
        s = sub.sort_values("c_rate")
        ax2.plot(s["c_rate"], s["min_plating_overpotential_V"], "o-", label=f"{T:.0f}°C")
    ax2.axhline(0, color="red", ls="--", lw=1, label="析锂阈值 0 V")
    ax2.axhline(SAFETY_MARGIN_V, color="orange", ls=":", lw=1, label="工程裕度 5 mV")
    ax2.set_xlabel("充电倍率 (C)")
    ax2.set_ylabel("min η_pl (V)")
    ax2.set_title("倍率 ↑ / 温度 ↓ → 析锂风险 ↑")
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "09_plating_map.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
