"""最小可运行示例：不下载 10 GB 数据，直接看模型与优化器的核心行为。

运行:
    python examples/quickstart.py

展示五件事:
    1. 充电时长收益递减律（CC + CV 地板）
    2. 阶梯协议的时长优势
    3. 析锂安全上限表如何约束优化
    4. 早期寿命特征的可用性边界
    5. 稳健性检验为什么能推翻一个"显著"结论
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from fastcharge.chargetime import (  # noqa: E402
    charge_time_h,
    multistage_time_h,
    time_savings_law,
)
from fastcharge.damage import damage_density_params, efc_stages  # noqa: E402
from fastcharge.rl import ChargeEnv, evaluate_policy, solve_dp  # noqa: E402
from fastcharge.surrogate import variance_ceiling  # noqa: E402

# 基准报告拟合出的参数（scripts/04_optimize.py 产出）
CT = {"a_cc": 0.869, "c_cv": 0.432, "d_tail": 0.173}
DMG = damage_density_params(alpha=0.000682, p=0.121, beta=2.012)


def demo_1_time_law():
    print("=" * 68)
    print("1) 充电时长收益递减律")
    print("=" * 68)
    law = time_savings_law(CT, span=0.6)
    print(law[["rate_C", "charge_time_h", "cc_part_h", "cv_part_h", "cv_share"]]
          .round(3).to_string(index=False))
    t1 = float(charge_time_h(0.6, 1.0, CT["a_cc"], CT["c_cv"], CT["d_tail"]))
    t8 = float(charge_time_h(0.6, 8.0, CT["a_cc"], CT["c_cv"], CT["d_tail"]))
    print(f"\n1C → 8C（倍率涨 8 倍）：时长 {t1:.3f} h → {t8:.3f} h，只降 {(1-t8/t1)*100:.0f}%")
    print(f"CV 硬地板 = {CT['c_cv']}·span + {CT['d_tail']} = "
          f"{CT['c_cv']*0.6 + CT['d_tail']:.3f} h")
    print("→ 再高的倍率也买不到线性收益\n")


def demo_2_cascade():
    print("=" * 68)
    print("2) 阶梯协议的时长优势")
    print("=" * 68)
    span = 0.85
    for name, stages in [
        ("恒流 1C", [(1.0, 0.0, span)]),
        ("恒流 3C", [(3.0, 0.0, span)]),
        ("阶梯 3C→2C→1C", [(3.0, 0.0, 0.30), (2.0, 0.30, 0.60), (1.0, 0.60, span)]),
        ("阶梯 3C→1.5C", [(3.0, 0.0, 0.35), (1.5, 0.35, span)]),
    ]:
        t = multistage_time_h(stages, CT)
        efc = efc_stages(stages, DMG)
        print(f"  {name:18s} 时长 {t:.3f} h   预测 EFC {efc:8.0f}")
    print("\n→ 阶梯比同倍率恒流快，且把倍率放在损伤密度低的低 SOC 段\n")


def demo_3_dp():
    print("=" * 68)
    print("3) 精确 DP 求最优充电策略（含析锂约束）")
    print("=" * 68)
    for lam in (0.002, 0.05):
        env = ChargeEnv(0.0, 0.85, DMG, CT, lam=lam, n_states=10,
                        rate_limit=lambda s, T: 1.0)   # 30°C 安全上限 ≈1C
        dp = solve_dp(env)
        ev = evaluate_policy(env, dp["policy_rates"])
        print(f"  λ={lam}: 策略 {ev['rates']:<10s} ({ev['n_stages']} 段)  "
              f"等效时长 {ev['time_per_fec_h']:.3f} h  EFC {ev['pred_efc']:.0f}")
    print("\n→ 加了 1C 上限后最优策略退化为恒流 1C（与 PyBaMM 结论一致）\n")


def demo_4_early_features():
    print("=" * 68)
    print("4) 早期寿命：精度天花板")
    print("=" * 68)
    root = Path(__file__).resolve().parents[1]
    cells_path = root / "results" / "metrics" / "cells.parquet"
    if not cells_path.exists():
        print("  （需要先跑 scripts/01_index.py 才有 cells.parquet，此处跳过）")
        print("  结论：log(EFC) 的 ICC = 0.594 → 这是任何协议级模型的 R² 上限\n")
        return
    cells = pd.read_parquet(cells_path)
    vc = variance_ceiling(cells)
    print(vc[["target", "icc_r2_ceiling", "within_group_ratio_median"]]
          .round(3).to_string(index=False))
    print("\n→ 组内 max/min 中位比 1.45：只剩个体差异，协议级特征吃不下\n")


def demo_5_robustness():
    print("=" * 68)
    print("5) 为什么「显著」的结论也可能不成立：聚类导致 p 值虚高")
    print("=" * 68)
    from scipy import stats

    rng = np.random.default_rng(7)
    rows = []
    groups = 61
    rho_true = 0.30                            # 组水平上的真实相关（弱）
    for g in range(groups):
        gx = rng.normal(0, 1)
        gy = rho_true * gx + np.sqrt(1 - rho_true ** 2) * rng.normal(0, 1)
        for c in range(4):
            rows.append({"group": g,
                         "x": gx + rng.normal(0, 0.02),   # 组内几乎无额外噪声
                         "y": gy + rng.normal(0, 0.02)})
    d = pd.DataFrame(rows)

    naive = stats.spearmanr(d["x"], d["y"])
    grp = d.groupby("group")[["x", "y"]].mean()
    pooled = stats.spearmanr(grp["x"], grp["y"])

    print(f"  真实结构：61 组 × 4 只电芯，组间真实 ρ≈{rho_true}")
    print(f"  当作 {len(d)} 个独立样本 : ρ={naive.statistic:+.2f}  p={naive.pvalue:.1e}  ← 用 244 个假自由度")
    print(f"  按 {groups} 组聚合     : ρ={pooled.statistic:+.2f}  p={pooled.pvalue:.3f}  ← 只有 {groups} 个真自由度")
    print("\n  同一个 ρ，p 值差了几个数量级 —— 这就是聚类导致的假显著。")
    print("  ISU-ILCC 同协议 4 只电芯就是这种结构。")
    print("  所以必须 cluster bootstrap 与按组置换，而不是直接对 240 行做检验\n")


if __name__ == "__main__":
    demo_1_time_law()
    demo_2_cascade()
    demo_3_dp()
    demo_4_early_features()
    demo_5_robustness()
    print("=" * 68)
    print("完。完整结果见 reports/基准评测报告-A3.md 与 reports/深化报告-A3.md")
    print("=" * 68)
