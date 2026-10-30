"""09 — 深化①：阶梯（多段/CC-CV 递减）快充协议，PyBaMM 仿真直采。

为什么不用拟合的损伤模型
------------------------
损伤密度 δ(s,r)=α·r^p·exp(β·s) 在 ISU-ILCC 上拟合出 p≈0.12（倍率几乎不伤），
log-R² 只有 0.23——**数据本身没有足够的倍率对比度**来约束 p。若用它做阶梯搜索，
结论必然是"恒 3C 最优、阶梯无收益"，那是模型缺陷而不是物理事实。

阶梯的真实机制来自**析锂约束**：高 SOC 段必须降速，否则负极触底。
所以本脚本直接用 PyBaMM 模拟"电压触发的阶梯协议"（`Charge at X C until Y V` 序列），
逐条测 η_pl 与时长——这是唯一如实的做法。

产出:
    results/metrics/cascade_sweep.csv        阶梯协议扫描（η_min / 时长 / 容量）
    results/metrics/cascade_pareto.csv       同等时长下的最优阶梯结构
    results/metrics/cascade_baseline.csv     恒定倍率基线对照
    results/figures/11_cascade.png
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
from fastcharge.physics import SAFETY_MARGIN_V, sweep_cascades  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# 协议设计
# --------------------------------------------------------------------------
def constant_baselines() -> list[list[tuple[float, float]]]:
    return [[(r, 4.2)] for r in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)]


def cascade_designs() -> list[list[tuple[float, float]]]:
    """阶梯结构：首段猛、后续逐级降速，切换点用电压（因为电压≈SOC）。"""
    return [
        # 2 段
        [(3.0, 4.00), (2.0, 4.20)],
        [(3.0, 4.05), (1.5, 4.20)],
        [(3.0, 4.10), (1.0, 4.20)],
        [(2.5, 4.00), (1.5, 4.20)],
        [(2.5, 4.10), (1.0, 4.20)],
        [(4.0, 3.90), (1.5, 4.20)],
        [(4.0, 3.95), (1.0, 4.20)],
        # 3 段
        [(3.0, 3.95), (2.0, 4.10), (1.0, 4.20)],
        [(3.0, 4.00), (2.0, 4.10), (1.0, 4.20)],
        [(3.0, 4.05), (1.5, 4.15), (0.75, 4.20)],
        [(4.0, 3.90), (2.0, 4.05), (1.0, 4.20)],
        [(4.0, 3.95), (2.0, 4.10), (0.75, 4.20)],
        [(2.5, 3.95), (1.5, 4.10), (0.75, 4.20)],
        # 4 段
        [(3.0, 3.90), (2.0, 4.00), (1.5, 4.10), (1.0, 4.20)],
        [(4.0, 3.88), (2.5, 3.98), (1.5, 4.10), (0.75, 4.20)],
        [(4.0, 3.90), (2.0, 4.00), (1.0, 4.10), (0.5, 4.20)],
        # 5 段（最细）
        [(4.0, 3.85), (3.0, 3.92), (2.0, 4.02), (1.2, 4.12), (0.6, 4.20)],
        [(3.5, 3.88), (2.5, 3.96), (1.8, 4.04), (1.0, 4.14), (0.5, 4.20)],
    ]


def ramped_consecutive() -> list[list[tuple[float, float]]]:
    """"高倍率贯穿到高 SOC"的对照：验证高 SOC 降速是否真的必要。"""
    return [
        [(3.0, 4.18), (1.0, 4.20)],   # 几乎全程 3C
        [(2.0, 4.18), (1.0, 4.20)],
        [(1.5, 4.18), (1.0, 4.20)],
    ]


def main() -> int:
    ensure_dirs()

    print("=== 1) 恒定倍率基线 ===")
    base = sweep_cascades(constant_baselines())
    base["scheme"] = "constant"
    base.to_csv(METRICS / "cascade_baseline.csv", index=False, encoding="utf-8-sig")

    print("\n=== 2) 阶梯协议扫描 ===")
    casc = sweep_cascades(cascade_designs())
    casc["scheme"] = "cascade"

    print("\n=== 3) 高倍率贯穿对照 ===")
    ramp = sweep_cascades(ramped_consecutive())
    ramp["scheme"] = "ramp"

    all_df = pd.concat([base, casc, ramp], ignore_index=True)
    all_df["n_stages"] = all_df["steps_str"].str.count("→") + 1
    all_df["peak_rate"] = all_df["steps_str"].str.extract(r"(\d+(?:\.\d+)?)C@")[0].astype(float)
    all_df["safe"] = all_df["min_plating_overpotential_V"] >= SAFETY_MARGIN_V
    all_df["plated"] = all_df["min_plating_overpotential_V"] < 0.0
    all_df.to_csv(METRICS / "cascade_sweep.csv", index=False, encoding="utf-8-sig")

    # ---- 分析：同等时长下，阶梯能把 η_min 抬多高 ----
    print("\n=== 4) 阶梯 vs 恒定：析锂裕度对比 ===")
    casc = casc.assign(
        safe=casc["min_plating_overpotential_V"] >= SAFETY_MARGIN_V,
        plated=casc["min_plating_overpotential_V"] < 0.0)
    base = base.assign(
        safe=base["min_plating_overpotential_V"] >= SAFETY_MARGIN_V,
        plated=base["min_plating_overpotential_V"] < 0.0)
    rows = []
    for _, c in casc.iterrows():
        t_c = c["charge_time_h"]
        # 找时长最接近的恒定协议
        ref = base.iloc[(base["charge_time_h"] - t_c).abs().argsort()[:1]].iloc[0]
        rows.append({
            "cascade": c["steps_str"],
            "cascade_time_h": t_c,
            "cascade_eta_min_V": c["min_plating_overpotential_V"],
            "cascade_safe": bool(c["safe"]),
            "matched_constant": ref["steps_str"],
            "matched_constant_time_h": ref["charge_time_h"],
            "matched_constant_eta_min_V": ref["min_plating_overpotential_V"],
            "matched_constant_safe": bool(ref["min_plating_overpotential_V"] >= SAFETY_MARGIN_V),
            "eta_gain_V": c["min_plating_overpotential_V"] - ref["min_plating_overpotential_V"],
            "time_cost_pct": (t_c / ref["charge_time_h"] - 1) * 100,
        })
    cmp = pd.DataFrame(rows).sort_values("eta_gain_V", ascending=False)
    cmp.to_csv(METRICS / "cascade_vs_constant.csv", index=False, encoding="utf-8-sig")
    print(cmp.head(10).round(4).to_string(index=False))

    # ---- 前沿：时长 vs η_min（两目标，都最小化 → η 取负）----
    safe_df = all_df[np.isfinite(all_df["min_plating_overpotential_V"])].copy()
    pts = safe_df[["charge_time_h", "min_plating_overpotential_V"]].to_numpy(float)
    keep = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        dom = (pts[:, 0] <= pts[i, 0]) & (pts[:, 1] >= pts[i, 1]) & (
            (pts[:, 0] < pts[i, 0]) | (pts[:, 1] > pts[i, 1]))
        if dom.any():
            keep[i] = False
    pf = safe_df[keep].sort_values("charge_time_h")
    pf.to_csv(METRICS / "cascade_pareto.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 5) 时长 × 析锂裕度 Pareto 前沿（{len(pf)} 条）===")
    print(pf[["scheme", "steps_str", "charge_time_h",
              "min_plating_overpotential_V", "safe"]].round(4).to_string(index=False))

    # ---- 关键结论数字 ----
    fastest_safe = safe_df[safe_df["safe"]].sort_values("charge_time_h")
    fastest_unsafe = safe_df[~safe_df["safe"]].sort_values("charge_time_h")
    concl = pd.DataFrame([
        {"metric": "最快恒定安全协议",
         "value": base[base["min_plating_overpotential_V"] >= SAFETY_MARGIN_V]["steps_str"]
         .iloc[base[base["min_plating_overpotential_V"] >= SAFETY_MARGIN_V]["charge_time_h"].values.argmin()]
         if (base["min_plating_overpotential_V"] >= SAFETY_MARGIN_V).any() else "无",
         "time_h": float(base[base["min_plating_overpotential_V"] >= SAFETY_MARGIN_V]["charge_time_h"].min())
         if (base["min_plating_overpotential_V"] >= SAFETY_MARGIN_V).any() else np.nan},
        {"metric": "最快阶梯安全协议",
         "value": fastest_safe["steps_str"].iloc[0] if len(fastest_safe) else "无",
         "time_h": float(fastest_safe["charge_time_h"].iloc[0]) if len(fastest_safe) else np.nan},
        {"metric": "最快不安全协议（证明约束必要）",
         "value": fastest_unsafe["steps_str"].iloc[0] if len(fastest_unsafe) else "无",
         "time_h": float(fastest_unsafe["charge_time_h"].iloc[0]) if len(fastest_unsafe) else np.nan},
    ])
    concl.to_csv(METRICS / "cascade_conclusion.csv", index=False, encoding="utf-8-sig")
    print("\n=== 6) 关键结论 ===")
    print(concl.to_string(index=False))

    # ---- 6b. 双优解：阶梯同时在"更快"与"更安全"上打赢恒定 ----
    print("\n=== 6b) 双优（dominance）：同时更快且更安全 ===")
    dom_rows = []
    casc_df = all_df[all_df["scheme"] == "cascade"]
    for _, c in casc_df.iterrows():
        for _, b in base.iterrows():
            if (c["charge_time_h"] < b["charge_time_h"]
                    and c["min_plating_overpotential_V"] > b["min_plating_overpotential_V"]):
                dom_rows.append({
                    "cascade": c["steps_str"],
                    "cascade_time_h": c["charge_time_h"],
                    "cascade_eta_V": c["min_plating_overpotential_V"],
                    "dominated_constant": b["steps_str"],
                    "constant_time_h": b["charge_time_h"],
                    "constant_eta_V": b["min_plating_overpotential_V"],
                    "faster_pct": (1 - c["charge_time_h"] / b["charge_time_h"]) * 100,
                    "safety_multiple": c["min_plating_overpotential_V"]
                    / max(b["min_plating_overpotential_V"], 1e-9),
                })
    dom = pd.DataFrame(dom_rows)
    if len(dom):
        # 只保留最优的一条（对每个被支配对象，挑最快那个）
        dom = dom.sort_values(["dominated_constant", "cascade_time_h"])
        dom.to_csv(METRICS / "cascade_dominance.csv", index=False, encoding="utf-8-sig")
        best_dom = dom.sort_values("constant_time_h", ascending=False).iloc[0]
        print(f"  共 {len(dom)} 组支配关系")
        print(f"  最优示例: {best_dom['cascade']}")
        print(f"    vs 恒定 {best_dom['dominated_constant']}: "
              f"快 {best_dom['faster_pct']:.2f}%，析锂裕度 {best_dom['safety_multiple']:.1f} 倍")
        print(dom.head(8).round(4).to_string(index=False))
    else:
        dom = pd.DataFrame()
        print("  无支配关系")

    # ---- 6c. 显式纳指标：每多 1 mV 裕度要付多少时间 ----
    safe_sorted = all_df[all_df["safe"]].sort_values("min_plating_overpotential_V")
    if len(safe_sorted) >= 2:
        safe_sorted = safe_sorted.assign(
            dt_per_mV=lambda d: d["charge_time_h"].diff() / (d["min_plating_overpotential_V"].diff() * 1000)
        )
        safe_sorted[["scheme", "steps_str", "charge_time_h", "min_plating_overpotential_V",
                     "dt_per_mV"]].to_csv(METRICS / "cascade_safety_price.csv",
                                          index=False, encoding="utf-8-sig")
        price = safe_sorted.dropna(subset=["dt_per_mV"])["dt_per_mV"].median()
        print(f"\n=== 6c) 安全裕度的价格 ===")
        print(f"  中位: 每多 1 mV 析锂裕度 ≈ {price * 3600:.1f} 秒充电时间"
              f"（{price:.4f} h/V，即 {price * 1000:.4f} h 每 1000 mV）")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    ax = axes[0]
    for scheme, marker, color, lab in [("constant", "o", "#a0aec0", "恒定倍率"),
                                        ("cascade", "s", "#2b6cb0", "阶梯（逐级降速）"),
                                        ("ramp", "^", "#dd6b20", "高倍率贯穿对照")]:
        s = all_df[all_df["scheme"] == scheme]
        ax.scatter(s["charge_time_h"], s["min_plating_overpotential_V"], s=70, marker=marker,
                   c=color, label=lab, zorder=3)
    ax.axhline(0, color="red", ls="--", lw=1, label="析锂阈值 0 V")
    ax.axhline(SAFETY_MARGIN_V, color="orange", ls=":", lw=1, label="工程裕度 5 mV")
    ax.set_xlabel("充电时长 (h)")
    ax.set_ylabel("min η_pl (V)")
    ax.set_title("阶梯协议用少量时间换安全裕度")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(pf["charge_time_h"], pf["min_plating_overpotential_V"], s=80, c="#2b6cb0",
                zorder=3, label="Pareto 前沿")
    for _, r in pf.iterrows():
        if r["n_stages"] >= 3:
            ax2.annotate(r["steps_str"].split(" → ")[0], (r["charge_time_h"], r["min_plating_overpotential_V"]),
                         fontsize=6.5, rotation=35, color="#4a5568")
    ax2.axhline(0, color="red", ls="--", lw=1)
    ax2.axhline(SAFETY_MARGIN_V, color="orange", ls=":", lw=1)
    ax2.set_xlabel("充电时长 (h)")
    ax2.set_ylabel("min η_pl (V)")
    ax2.set_title("时长 × 析锂裕度 的 Pareto 前沿")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "11_cascade.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
