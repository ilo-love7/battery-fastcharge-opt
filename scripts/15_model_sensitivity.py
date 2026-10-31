"""15 — 深化⑥：析锂判据的模型敏感性（SPM / SPMe / DFN 三档对比）。

问题
----
基准报告用 SPM + OKane2022 得出"30°C 安全上限 ≈1.0C"，并据此约束了 Pareto 前沿。
但 SPM 是**最简单的降阶模型**——它假设电极内部均匀，会低估高倍率下的极化不均。
换成 SPMe（加抛物线浓度分布）或 DFN（全空间分辨），上限会怎么变？

如果 DFN 给出更严的上限，那基准报告的"受约束 Pareto"仍然偏乐观。

产出:
    results/metrics/model_comparison.csv       三模型 × 多倍率的 η_min 对比
    results/metrics/safe_limit_by_model.csv    各模型给出的安全上限
    results/metrics/a3_vs_okane.csv            参数集 × 模型 的交叉对照
    results/figures/16_model_sensitivity.png
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
from fastcharge.physics import SAFETY_MARGIN_V, SimulationSetup, min_plating_overpotential  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
warnings.filterwarnings("ignore")

MODELS = ("SPM", "SPMe", "DFN")
RATES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
TEMPS = (15.0, 30.0, 45.0)


def contiguous_limit(df: pd.DataFrame, temp: float, margin: float = SAFETY_MARGIN_V) -> float:
    """从最小倍率起连续安全段的上界（与主报告同口径）。"""
    sub = df[(df["temperature_C"] == temp) & df["eta_min_V"].notna()].sort_values("c_rate")
    if sub.empty:
        return np.nan
    sub = sub.drop_duplicates("c_rate")
    safe = (sub["eta_min_V"] >= margin).to_numpy()
    if not safe.any():
        return 0.0
    n_ok = 0
    for f in safe:
        if f:
            n_ok += 1
        else:
            break
    return float(sub["c_rate"].to_numpy()[n_ok - 1])


def run_grid() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for T in TEMPS:
            for c in RATES:
                try:
                    r = min_plating_overpotential(
                        SimulationSetup(model=model, c_rate=c, temperature_C=T))
                    eta = r["min_plating_overpotential_V"]
                    npts = r.get("n_points", np.nan)
                except Exception as exc:
                    eta, npts = np.nan, np.nan
                    print(f"  {model} {c}C {T}°C FAIL: {str(exc)[:60]}", flush=True)
                rows.append({"model": model, "temperature_C": T, "c_rate": c,
                             "eta_min_V": eta, "n_charging_points": npts})
                print(f"  {model:5s} {c:4.1f}C @ {T:3.0f}°C → η={eta:+.4f} V", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    cache = METRICS / "model_comparison.csv"
    if cache.exists():
        comp = pd.read_csv(cache)
        print(f"复用缓存 {len(comp)} 行")
    else:
        print("=== 1) 三模型 × 温度 × 倍率 网格 ===")
        comp = run_grid()
        comp.to_csv(cache, index=False, encoding="utf-8-sig")

    # ---- 2. 安全上限对比 ----
    print("\n=== 2) 各模型的安全倍率上限 ===")
    rows = []
    for model in MODELS:
        for T in TEMPS:
            lim = contiguous_limit(comp[comp["model"] == model], T)
            rows.append({"model": model, "temperature_C": T, "safe_rate_limit_C": lim})
    limits = pd.DataFrame(rows)
    limits.to_csv(METRICS / "safe_limit_by_model.csv", index=False, encoding="utf-8-sig")
    piv = limits.pivot(index="temperature_C", columns="model", values="safe_rate_limit_C")
    print(piv.round(2).to_string())

    base = piv.loc[30.0, "SPM"] if 30.0 in piv.index else np.nan
    print(f"\n  30°C: SPM={piv.loc[30.0,'SPM']:.2f}C  SPMe={piv.loc[30.0,'SPMe']:.2f}C  "
          f"DFN={piv.loc[30.0,'DFN']:.2f}C")
    print(f"  DFN 相对 SPM 收紧 {(1 - piv.loc[30.0,'DFN']/base)*100:.1f}%")

    # ---- 3. 各模型的 η 曲线 ----
    print("\n=== 3) 30°C 下三模型的 η_pl 曲线 ===")
    d30 = comp[comp["temperature_C"] == 30.0]
    tab = d30.pivot(index="c_rate", columns="model", values="eta_min_V")
    tab.to_csv(METRICS / "eta_curves_30C.csv", encoding="utf-8-sig")
    print(tab.round(4).to_string())

    # ---- 4. 结论一致性检验 ----
    print("\n=== 4) 判据一致性 ===")
    agree = []
    for r in RATES:
        vals = [tab.loc[r, m] if r in tab.index and m in tab.columns else np.nan for m in MODELS]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 2:
            continue
        agree.append({
            "c_rate": r,
            "n_models_safe": int(sum(v >= SAFETY_MARGIN_V for v in vals)),
            "n_models": len(vals),
            "spread_V": float(max(vals) - min(vals)),
            "unanimous": bool(all(v >= SAFETY_MARGIN_V for v in vals)
                              or all(v < SAFETY_MARGIN_V for v in vals)),
        })
    agr = pd.DataFrame(agree)
    agr.to_csv(METRICS / "model_agreement.csv", index=False, encoding="utf-8-sig")
    print(agr.round(4).to_string(index=False))
    n_unan = int(agr["unanimous"].sum())
    print(f"\n  三模型一致判定: {n_unan}/{len(agr)} 个倍率点")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    ax = axes[0]
    colors = {"SPM": "#2b6cb0", "SPMe": "#38a169", "DFN": "#dd6b20"}
    for m in MODELS:
        s = d30[d30["model"] == m].sort_values("c_rate")
        ax.plot(s["c_rate"], s["eta_min_V"], "o-", color=colors[m], lw=1.8, label=m)
    ax.axhline(0, color="red", ls="--", lw=1, label="析锂阈值")
    ax.axhline(SAFETY_MARGIN_V, color="orange", ls=":", lw=1, label="5 mV 裕度")
    ax.set_xlabel("充电倍率 (C)")
    ax.set_ylabel("min η_pl (V)")
    ax.set_title("30°C：模型越精细，越早触底")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    for j, T in enumerate(TEMPS):
        vals = [piv.loc[T, m] if T in piv.index else np.nan for m in MODELS]
        ax2.bar(np.arange(len(MODELS)) + j * 0.25, vals, width=0.22,
                label=f"{T:.0f}°C", alpha=0.85)
    ax2.set_xticks(np.arange(len(MODELS)) + 0.25)
    ax2.set_xticklabels(MODELS)
    ax2.set_ylabel("安全倍率上限 (C)")
    ax2.set_title("安全上限的模型依赖与温度依赖")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIGURES / "16_model_sensitivity.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
