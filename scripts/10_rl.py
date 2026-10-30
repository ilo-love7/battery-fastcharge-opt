"""10 — 深化②：强化学习 / 动态规划求解充电策略，与 BO 同台对比。

三件事
------
1. **验证**：确定性 MDP 上，表格 Q-learning 应收敛到 DP 的最优解。完全一致 = DP 实现正确。
2. **约束下的结构**：加入析锂安全上限后，DP 的最优策略必然呈"逐级降速"阶梯——
                     这从最优控制层面解释了第 9 步的阶梯现象。
3. **随机环境的真实收益**：温度波动时，固定策略只能保守，观测式策略可以"热就冲、冷就收"。
                     这是 RL 相对 DP 的唯一实质优势，本脚本量化它。

产出:
    results/metrics/rl_dp_agreement.csv     DP vs Q-learning 一致性
    results/metrics/rl_lambda_sweep.csv     权衡系数 λ 扫描下的最优策略
    results/metrics/rl_constrained.csv      析锂约束下的最优策略
    results/metrics/rl_random_env.csv       随机环境：固定 vs 观测式
    results/figures/12_rl_policy.png
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

from fastcharge.chargetime import fit_charge_time  # noqa: E402
from fastcharge.config import FIGURES, METRICS, ensure_dirs  # noqa: E402
from fastcharge.damage import damage_density_params, fit_damage_model  # noqa: E402
from fastcharge.rl import (  # noqa: E402
    ChargeEnv,
    build_rate_limit_2d,
    evaluate_policy,
    q_learning,
    solve_dp,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
warnings.filterwarnings("ignore")

SOC_LO, SOC_HI = 0.0, 0.85
N_STATES = 20


def build_rate_limit_from_table(table: pd.DataFrame, temperature_C: float = 30.0):
    """单温度下的 rate_limit(soc_mid) —— 确定性环境用。"""
    sub = table[table["temperature_C"] == temperature_C]
    s = sub.set_index("soc_start")["rate_limit_C"].astype(float).sort_index()
    socs, vals = s.index.to_numpy(float), s.to_numpy(float)

    def f(soc_start: float, T: float = temperature_C) -> float:
        soc_start = float(np.clip(soc_start, socs.min(), socs.max()))
        return float(np.interp(soc_start, socs, vals))

    return f


def main() -> int:
    ensure_dirs()
    cells = pd.read_parquet(METRICS / "cells.parquet")
    cycles = pd.read_parquet(METRICS / "cycles.parquet")

    fit = fit_damage_model(cells)
    pars = damage_density_params(fit["alpha"], fit["p_rate"], fit["beta_soc"])
    ct = fit_charge_time(cycles)
    pct = {"a_cc": ct["a_cc"], "c_cv": ct["c_cv"], "d_tail": ct["d_tail"]}

    # ========== 1. DP vs Q-learning 一致性 ==========
    print("=== 1) DP vs 表格 Q-learning（确定性环境）===")
    rows = []
    for lam in (0.0, 0.001, 0.002, 0.005, 0.02, 0.2, 1.0):
        env = ChargeEnv(SOC_LO, SOC_HI, pars, pct, lam=lam, n_states=N_STATES)
        dp = solve_dp(env)
        rl = q_learning(env, episodes=6000, seed=0)
        ev_dp = evaluate_policy(env, dp["policy_rates"])
        ev_rl = evaluate_policy(env, rl["policy_rates"])
        match = bool(np.allclose(dp["policy_rates"], rl["policy_rates"]))
        rows.append({
            "lambda": lam,
            "dp_rates": ev_dp["rates"],
            "rl_rates": ev_rl["rates"],
            "dp_time_per_fec_h": ev_dp["time_per_fec_h"],
            "dp_pred_efc": ev_dp["pred_efc"],
            "rl_time_per_fec_h": ev_rl["time_per_fec_h"],
            "rl_pred_efc": ev_rl["pred_efc"],
            "policy_identical": match,
            "dp_value": dp["value"],
        })
        print(f"  λ={lam:<6} DP={ev_dp['rates']:<12} RL={ev_rl['rates']:<12} "
              f"一致={match}  t={ev_dp['time_per_fec_h']:.3f}h EFC={ev_dp['pred_efc']:.0f}")
    agree = pd.DataFrame(rows)
    agree.to_csv(METRICS / "rl_dp_agreement.csv", index=False, encoding="utf-8-sig")
    n_agree = int(agree["policy_identical"].sum())
    print(f"  → {n_agree}/{len(agree)} 个 λ 下策略完全一致"
          f"（{n_agree/len(agree)*100:.0f}%），DP 实现可信")

    # ========== 2. 析锂约束下的最优策略 ==========
    print("\n=== 2) 析锂安全约束下的最优策略 ===")
    limit_tbl = pd.read_csv(METRICS / "rate_limit_table.csv")
    rl_fn = build_rate_limit_from_table(limit_tbl, 30.0)
    rl_fn_2d = build_rate_limit_2d(limit_tbl)
    print("  上限表在 30°C:")
    print(limit_tbl[limit_tbl["temperature_C"] == 30.0].round(2).to_string(index=False))
    constrained_rows = []
    for lam in (0.001, 0.002, 0.01):
        env_c = ChargeEnv(SOC_LO, SOC_HI, pars, pct, lam=lam, n_states=N_STATES,
                          rate_limit=lambda s, T: rl_fn(s, T))
        # 注意: 训练表的 soc_start 与这里的 soc 同义（充电起点），直接传 s
        dp_c = solve_dp(env_c)
        ev = evaluate_policy(env_c, dp_c["policy_rates"])
        env_free = ChargeEnv(SOC_LO, SOC_HI, pars, pct, lam=lam, n_states=N_STATES)
        dp_f = solve_dp(env_free)
        evf = evaluate_policy(env_free, dp_f["policy_rates"])
        constrained_rows.append({
            "lambda": lam,
            "constrained_rates": ev["rates"],
            "constrained_n_stages": ev["n_stages"],
            "constrained_time_per_fec_h": ev["time_per_fec_h"],
            "constrained_pred_efc": ev["pred_efc"],
            "free_rates": evf["rates"],
            "free_time_per_fec_h": evf["time_per_fec_h"],
            "time_penalty_pct": (ev["time_per_fec_h"] / evf["time_per_fec_h"] - 1) * 100,
        })
        print(f"  λ={lam}: 受约束={ev['rates']} ({ev['n_stages']} 段) t={ev['time_per_fec_h']:.3f}h"
              f" | 无约束={evf['rates']} t={evf['time_per_fec_h']:.3f}h"
              f" | 时间代价 {(ev['time_per_fec_h']/evf['time_per_fec_h']-1)*100:+.1f}%")
    cons = pd.DataFrame(constrained_rows)
    cons.to_csv(METRICS / "rl_constrained.csv", index=False, encoding="utf-8-sig")

    # ========== 3. 随机环境 ==========
    print("\n=== 3) 随机温度环境：种策略对比 ===")
    rand_rows = []
    for sigma in (0.0, 2.0, 5.0, 8.0):
        env_det = ChargeEnv(SOC_LO, SOC_HI, pars, pct, lam=0.002, n_states=N_STATES)
        dp_det = solve_dp(env_det)
        # ① 固定策略（DP 的产物，与温度无关）
        env_rand = ChargeEnv(SOC_LO, SOC_HI, pars, pct, lam=0.002, n_states=N_STATES,
                             temp_sigma=sigma, rate_limit=rl_fn_2d, seed=1)
        from fastcharge.rl import simulate_random, q_learning_temp, simulate_temp_policy
        fixed = simulate_random(env_rand, dp_det["policy_rates"], observe=False, seed=2)
        # ② 朴素观测式：每步取该温度允许的最高倍率（贪心，不看 SOC）
        greedy = simulate_random(env_rand, dp_det["policy_rates"], observe=True, seed=2)
        # ③ RL（SOC × 温度）联合策略
        rl_t = q_learning_temp(env_rand, episodes=20000, seed=3)
        rl_sim = simulate_temp_policy(env_rand, rl_t["policy_grid"], rl_t["temp_bins"],
                                       n_rollouts=300, seed=2)
        gain_rl = (fixed["cost"].mean() / rl_sim["cost"].mean() - 1) * 100
        gain_greedy = (fixed["cost"].mean() / greedy["cost"].mean() - 1) * 100
        rand_rows.append({
            "temp_sigma_C": sigma,
            "fixed_cost": float(fixed["cost"].mean()),
            "greedy_cost": float(greedy["cost"].mean()),
            "rl_soc_temp_cost": float(rl_sim["cost"].mean()),
            "greedy_gain_pct": float(gain_greedy),
            "rl_gain_pct": float(gain_rl),
        })
        print(f"  σ={sigma:>4.1f}°C: 固定 {fixed['cost'].mean():.5f} | 朴素贪心 {greedy['cost'].mean():.5f}"
              f" ({gain_greedy:+.1f}%) | RL(SOC×T) {rl_sim['cost'].mean():.5f} ({gain_rl:+.1f}%)")
    rand_df = pd.DataFrame(rand_rows)
    rand_df.to_csv(METRICS / "rl_random_env.csv", index=False, encoding="utf-8-sig")

    # ========== 4. 与 BO 对比 ==========
    print("\n=== 4) RL/DP vs BO（同一目标平面）===")
    bo = pd.read_csv(METRICS / "opt_pareto_constrained.csv")
    cmp_rows = []
    # DP 在 λ 网格上给出多条策略（等价于沿 Pareto 前沿取点）
    for lam in np.logspace(-4, 1, 12):
        env_c = ChargeEnv(SOC_LO, SOC_HI, pars, pct, lam=float(lam), n_states=N_STATES,
                          rate_limit=rl_fn)
        dp_c = solve_dp(env_c)
        ev = evaluate_policy(env_c, dp_c["policy_rates"])
        cmp_rows.append({
            "lambda": float(lam),
            "solver": "DP(带析锂约束)",
            "time_per_fec_h": ev["time_per_fec_h"],
            "pred_efc": ev["pred_efc"],
            "n_stages": ev["n_stages"],
            "rates": ev["rates"],
        })
    for _, r in bo.iterrows():
        cmp_rows.append({
            "lambda": np.nan, "solver": "BO",
            "time_per_fec_h": float(r["time_per_fec_h"]),
            "pred_efc": float(r["pred_cycles"]),
            "n_stages": 1, "rates": f"{r['peak_rate_C']:.2f}",
        })
    cmp_df = pd.DataFrame(cmp_rows).sort_values("time_per_fec_h")
    cmp_df = cmp_df.drop_duplicates(subset=["time_per_fec_h", "pred_efc"])
    cmp_df.to_csv(METRICS / "rl_vs_bo.csv", index=False, encoding="utf-8-sig")
    print(cmp_df.round(3).to_string(index=False))

    # ========== 图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    ax = axes[0]
    dp_pts = cmp_df[cmp_df["solver"].str.startswith("DP")]
    ax.scatter(dp_pts["time_per_fec_h"], dp_pts["pred_efc"], s=80, c="#2b6cb0",
               marker="s", label="DP（带析锂约束，λ 扫描）", zorder=3)
    bo_pts = cmp_df[cmp_df["solver"] == "BO"]
    ax.scatter(bo_pts["time_per_fec_h"], bo_pts["pred_efc"], s=70, c="#e53e3e",
               marker="o", label="BO（GP 代理 + 机理约束）", zorder=3)
    ax.set_xlabel("等效充满 1 FEC 的时长 (h)")
    ax.set_ylabel("预测 EFC 寿命")
    ax.set_yscale("log")
    ax.set_title("两条求解路线给出同一组权衡点")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax2 = axes[1]
    sig = rand_df[rand_df["temp_sigma_C"] > 0]
    ax2.plot(sig["temp_sigma_C"], sig["fixed_cost"], "o-", color="#a0aec0",
             label="固定策略（DP 产物，与温度无关）")
    ax2.plot(sig["temp_sigma_C"], sig["greedy_cost"], "^-", color="#dd6b20",
             label="朴素贪心（只看温度不看 SOC）")
    ax2.plot(sig["temp_sigma_C"], sig["rl_soc_temp_cost"], "s-", color="#2b6cb0",
             label="RL（SOC×温度 联合策略）")
    ax2.set_xlabel("温度波动 σ (°C)")
    ax2.set_ylabel("总成本（λ·时间 + 损伤）")
    ax2.set_title("环境不确定时：RL 的联合状态策略 vs 固定/贪心")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    for _, r in sig.iterrows():
        ax2.annotate(f"RL {r['rl_gain_pct']:+.1f}%\n贪心 {r['greedy_gain_pct']:+.1f}%",
                     (r["temp_sigma_C"], r["rl_soc_temp_cost"]),
                     textcoords="offset points", xytext=(6, -22), fontsize=7, color="#2b6cb0")

    fig.tight_layout()
    fig.savefig(FIGURES / "12_rl_policy.png")
    plt.close(fig)
    print(f"\n图已写入 {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
