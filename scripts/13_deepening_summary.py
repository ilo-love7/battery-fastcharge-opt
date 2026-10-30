"""13 — 深化汇总：把四个深化的结论压缩成一张表 + 一张图。

产出:
    results/metrics/deepening_summary.csv   四个深化的核心结论（含置信区间）
    docs/figures/main_deepening.png         汇总主图
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

from fastcharge.config import FIGURES, METRICS  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
warnings.filterwarnings("ignore")

OUT = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    rows = []

    # ① 阶梯协议
    try:
        sw = pd.read_csv(METRICS / "cascade_sweep.csv")
        dom = pd.read_csv(METRICS / "cascade_dominance.csv")
        best = sw[sw["safe"]].sort_values("charge_time_h").iloc[0]
        top = dom.sort_values("safety_multiple", ascending=False).iloc[0]
        rows.append({
            "深化": "① 阶梯协议",
            "结论": f"最优阶梯 {top['cascade']} 同时快 {top['faster_pct']:.1f}% 且析锂裕度 "
                    f"{top['safety_multiple']:.1f}×（vs 同族恒流）",
            "关键数字": f"最快安全 {best['charge_time_h']:.3f} h / η={best['min_plating_overpotential_V']:+.4f} V",
            "置信": "仿真（PyBaMM SPM，30°C）",
        })
    except Exception as exc:
        print("阶梯汇总跳过:", exc)

    # ② RL / DP
    try:
        ag = pd.read_csv(METRICS / "rl_dp_agreement.csv")
        rand = pd.read_csv(METRICS / "rl_random_env.csv")
        n_ok = int(ag["policy_identical"].sum())
        g = rand[rand["temp_sigma_C"] > 0]
        rows.append({
            "深化": "② RL / DP",
            "结论": f"Q-learning 在 {n_ok}/{len(ag)} 个 λ 下与 DP 完全一致（交叉验证）；"
                    f"随机环境下 RL 联合策略收益仅 {g['rl_gain_pct'].mean():+.1f}%",
            "关键数字": f"朴素贪心反而 {g['greedy_gain_pct'].mean():+.1f}%",
            "置信": "确定性 MDP 精确；随机环境为负结果",
        })
    except Exception as exc:
        print("RL 汇总跳过:", exc)

    # ③ 早期预测
    try:
        ec = pd.read_csv(METRICS / "early_curve.csv")
        et = pd.read_csv(METRICS / "early_test_time.csv")
        base = pd.read_csv(METRICS / "early_prediction.csv").iloc[0]
        best = ec.sort_values("mape_pct").iloc[0]
        row_n = et[et["n_head"] == int(best["n_head"])].iloc[0]
        rows.append({
            "深化": "③ 早期寿命预测",
            "结论": f"N={int(best['n_head'])} 圈时 MAPE 降到 {best['mape_pct']:.1f}%"
                    f"（协议参数基线 {base['mape_pct']:.1f}%），省 {row_n['saving_pct']:.0f}% 台时",
            "关键数字": f"N≤200 时被基线反超（34–43%），曲线先降后平再降",
            "置信": f"LOGO-CV，n={int(best['n'])}",
        })
    except Exception as exc:
        print("早期汇总跳过:", exc)

    # ④ 稳健性
    try:
        cs = pd.read_csv(METRICS / "claim_summary.csv")
        strong = cs[cs["strength"].str.startswith("强")]
        weak = cs[~cs["strength"].str.startswith("强")]
        rows.append({
            "深化": "④ 结论稳健性",
            "结论": f"{len(strong)}/{len(cs)} 条结论通过置换检验且区间窄；"
                    f"倍率效应（C2）perm p=0.17 → 不稳健",
            "关键数字": "SOC 窗口 ρ=0.875 [0.796,0.928]；Pareto 越界率 83% [56%,96%]",
            "置信": "cluster bootstrap B=2000 + 置换 2000",
        })
    except Exception as exc:
        print("稳健性汇总跳过:", exc)

    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "deepening_summary.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    # ---- 汇总图 ----
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))

    # (a) 阶梯
    ax = axes[0, 0]
    sw = pd.read_csv(METRICS / "cascade_sweep.csv")
    for scheme, marker, color, lab in [
        ("constant", "o", "#a0aec0", "恒定倍率"),
        ("cascade", "s", "#2b6cb0", "阶梯（逐级降速）"),
        ("ramp", "^", "#dd6b20", "高倍率贯穿"),
    ]:
        s = sw[sw["scheme"] == scheme]
        ax.scatter(s["charge_time_h"], s["min_plating_overpotential_V"], s=64,
                   marker=marker, c=color, label=lab, zorder=3)
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.axhline(0.005, color="orange", ls=":", lw=1, label="5 mV 裕度")
    ax.set_xlabel("充电时长 (h)")
    ax.set_ylabel("min η_pl (V)")
    ax.set_title("① 阶梯协议：用少量时间换安全裕度")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # (b) RL
    ax2 = axes[0, 1]
    rand = pd.read_csv(METRICS / "rl_random_env.csv")
    sig = rand[rand["temp_sigma_C"] > 0]
    base_cost = float(rand[rand["temp_sigma_C"] == 0]["fixed_cost"].iloc[0])
    ax2.plot(sig["temp_sigma_C"], (sig["fixed_cost"] / base_cost - 1) * 100, "o-",
             color="#a0aec0", label="固定策略")
    ax2.plot(sig["temp_sigma_C"], (sig["greedy_cost"] / base_cost - 1) * 100, "^-",
             color="#dd6b20", label="朴素贪心")
    ax2.plot(sig["temp_sigma_C"], (sig["rl_soc_temp_cost"] / base_cost - 1) * 100, "s-",
             color="#2b6cb0", label="RL (SOC×T)")
    ax2.axhline(0, color="k", ls=":", lw=1)
    ax2.set_xlabel("温度波动 σ (°C)")
    ax2.set_ylabel("相对无波动时的成本变化 (%)")
    ax2.set_title("② RL 的诚实边界：联合策略收益微弱，贪心还倒亏")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # (c) 早期预测
    ax3 = axes[1, 0]
    ec = pd.read_csv(METRICS / "early_curve.csv")
    base = pd.read_csv(METRICS / "early_prediction.csv").iloc[0]
    ax3.plot(ec["n_head"], ec["mape_pct"], "o-", color="#2c5282", lw=1.8, label="早期曲线特征")
    ax3.axhline(base["mape_pct"], ls="--", color="#dd6b20", lw=1.2,
                label=f"只用协议参数 ({base['mape_pct']:.1f}%)")
    ax3.set_xscale("log")
    ax3.set_xlabel("观测圈数 N")
    ax3.set_ylabel("LOGO-CV MAPE (%)")
    ax3.set_title("③ 早期预测：N≤200 打不过协议基线")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, which="both")

    # (d) 稳健性
    ax4 = axes[1, 1]
    cs = pd.read_csv(METRICS / "claim_summary.csv").dropna(subset=["ci_low"])
    for i, r in cs.reset_index(drop=True).iterrows():
        ax4.plot([r["ci_low"], r["ci_high"]], [i, i], color="#a0aec0", lw=7,
                 solid_capstyle="butt", zorder=2)
        color = "#38a169" if r["perm_p"] < 0.05 else "#dd6b20"
        ax4.scatter([r["observed"]], [i], s=90, color=color, zorder=3)
    ax4.axvline(0, color="k", ls=":", lw=1)
    ax4.set_yticks(range(len(cs)))
    ax4.set_yticklabels([c.split(" ", 1)[0] for c in cs["claim"]], fontsize=9)
    ax4.set_xlabel("效应量（95% CI）")
    ax4.set_title("④ 结论强度：绿=稳健，橙=不成立")
    ax4.grid(alpha=0.3, axis="x")

    fig.suptitle("四个深化的核心结论", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "main_deepening.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
