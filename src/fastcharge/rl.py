"""强化学习（Q-learning）与动态规划：把"充电策略"建成 MDP。

MDP 定义
--------
状态 s   : SOC 离散档（0..n_soc，每档 ΔSOC）
动作 a   : 本档 SOC 的充电倍率（从 RATE_GRID 选）
转移     : 确定性 —— s' = s + ΔSOC（充一段）
奖励     : r = −( λ·Δt(s,a) + δ(s,a) )      两个目标加权
           Δt 来自实测充电时长模型；δ 来自损伤密度
终止     : s = n_soc（到达窗口上限）

为什么这题"RL 是杀鸡用牛刀"，但值得做
--------------------------------------
确定性 + 奖励可加 → 精确动态规划（DP）能一步求出最优策略。
RL（表格 Q-learning）应当收敛到同一个解；**这个过程本身就是对 DP 的独立验证**。

RL 真正值钱的场景是**环境不确定**：温度波动让析锂上限随机变化。
本模块实现两个环境：
    A. 确定性（DP 可解）——RL 用来校验 DP
    B. 随机（温度每步波动 ±σ）——DP 只能给"固定策略"，RL 能给"按观测调整的策略"
并量化 B 场景下 RL 相对固定策略的真实收益。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .chargetime import stage_time_h
from .damage import damage_stages

#: 动作空间
ACTIONS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


class ChargeEnv:
    """充电 MDP 环境。

    soc_grid : 离散 SOC 边界（长度 n_states + 1）
    damage   : 损伤密度参数
    ct       : 充电时长参数
    lam      : 时间与寿命的权衡系数（每 1 h 时间 记多少损伤）
               λ=0 → 只求最快；λ→∞ → 只求最耐久
    temp_sigma: >0 时进入随机环境：每步温度扰动，析锂上限随之波动
    rate_limit: 析锂安全上限函数 rate_limit(soc, temp)
    """

    def __init__(self, soc_lo: float, soc_hi: float, damage: dict, ct: dict,
                 lam: float = 1.0, n_states: int = 20,
                 temp_sigma: float = 0.0, rate_limit=None, seed: int = 0):
        self.edges = np.linspace(soc_lo, soc_hi, n_states + 1)
        self.n = n_states
        self.damage = damage
        self.ct = ct
        self.lam = lam
        self.temp_sigma = temp_sigma
        self.rate_limit = rate_limit
        self.rng = np.random.default_rng(seed)

    # ---- 单步 ----
    def step(self, i: int, a: float, temp: float | None = None):
        s0, s1 = self.edges[i], self.edges[i + 1]
        dq = s1 - s0
        t = stage_time_h(a, s0, s1, self.ct)
        d = damage_stages([(a, s0, s1)], self.damage)
        cost = self.lam * t + d
        return i + 1, cost, d, t

    def allowed(self, i: int, temp: float | None = None) -> np.ndarray:
        """该状态下允许的动作（析锂上限约束）。"""
        if self.rate_limit is None:
            return np.array(ACTIONS, float)
        s_mid = (self.edges[i] + self.edges[i + 1]) / 2
        lim = self.rate_limit(s_mid, temp if temp is not None else 30.0)
        allowed = np.array([a for a in ACTIONS if a <= lim + 1e-9], float)
        return allowed if len(allowed) else np.array([ACTIONS[0]], float)


# --------------------------------------------------------------------------
# 精确动态规划（确定性环境的真值）
# --------------------------------------------------------------------------
def solve_dp(env: ChargeEnv) -> dict:
    """后向归纳：V[i] = min_a (cost(i,a) + V[i+1])。确定性环境的最优解。

    成本是**可加**的（总时间 + 总损伤），所以这里用"总成本最小"而非折扣回报。
    """
    V = np.zeros(env.n + 1)
    policy = np.zeros(env.n, dtype=float)
    for i in range(env.n - 1, -1, -1):
        s0, s1 = env.edges[i], env.edges[i + 1]
        best_tot, best_a = np.inf, float(env.allowed(i)[0])
        for a in env.allowed(i):
            _, c, _, _ = env.step(i, a)
            tot = c + V[i + 1]
            if tot < best_tot:
                best_tot, best_a = tot, float(a)
        V[i] = best_tot
        policy[i] = best_a
    # CV 尾巴只付一次（整条协议只有一段恒压），放在起点
    span = env.edges[-1] - env.edges[0]
    V[0] += env.lam * (env.ct["c_cv"] * span + env.ct["d_tail"])
    return {
        "value": float(V[0]),
        "policy_rates": policy,
        "policy_idx": policy,
        "V": V,
    }


def policy_to_stages(policy_rates: np.ndarray, edges: np.ndarray) -> list[tuple[float, float, float]]:
    """把逐档策略压成"连续相同倍率的段"（自动跳过退化/零倍率档）。"""
    stages = []
    start = None
    for i in range(len(policy_rates)):
        r = float(policy_rates[i])
        if not np.isfinite(r) or r <= 0:
            if start is not None:
                stages.append((float(policy_rates[start]), float(edges[start]), float(edges[i])))
                start = None
            continue
        if start is None:
            start = i
        elif r != policy_rates[start]:
            stages.append((float(policy_rates[start]), float(edges[start]), float(edges[i])))
            start = i
    if start is not None:
        stages.append((float(policy_rates[start]), float(edges[start]), float(edges[len(policy_rates)])))
    return stages


def evaluate_policy(env: ChargeEnv, policy_rates: np.ndarray) -> dict:
    """评估一条策略：总时间、总损伤、等效时长、EFC 寿命。"""
    stages = policy_to_stages(policy_rates, env.edges)
    span = env.edges[-1] - env.edges[0]
    t_total = sum(stage_time_h(r, a, b, env.ct) for r, a, b in stages)
    t_total += env.ct["c_cv"] * span + env.ct["d_tail"]
    d_total = damage_stages(stages, env.damage)
    return {
        "n_stages": len(stages),
        "rates": "|".join(f"{r:g}" for r, _, _ in stages),
        "time_h": t_total,
        "time_per_fec_h": t_total / span,
        "damage": d_total,
        "pred_efc": 1.0 / max(d_total, 1e-12),
    }


# --------------------------------------------------------------------------
# 表格 Q-learning
# --------------------------------------------------------------------------
def q_learning(env: ChargeEnv, *, episodes: int = 4000, alpha: float = 0.25,
               gamma: float = 1.0, epsilon: float = 0.25, seed: int = 0) -> dict:
    """确定性 MDP 的表格 Q-learning（每 episode 从起点走到终点）。

    γ=1 因为奖励可加、无折扣（这是"总成本最小"的表述）。
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n + 1, len(ACTIONS)))
    visits = np.zeros((env.n + 1, len(ACTIONS)), dtype=int)
    n_success = 0
    for ep in range(episodes):
        i = 0
        temp = float(rng.normal(30.0, env.temp_sigma)) if env.temp_sigma else None
        while i < env.n:
            allowed = env.allowed(i, temp)
            aidx = [j for j, a in enumerate(ACTIONS) if a in allowed]
            if not aidx:
                break
            if rng.random() < epsilon:
                j = int(rng.choice(aidx))
            else:
                j = max(aidx, key=lambda k: Q[i, k])
            _, cost, _, _ = env.step(i, float(ACTIONS[j]), temp)
            # 终止状态 V=0，且最后一步之后不再有 CV 尾巴（CV 单独在 evaluate 里算）
            Q[i, j] += alpha * (-cost + gamma * np.max(Q[i + 1]) - Q[i, j])
            visits[i, j] += 1
            i += 1
        if i >= env.n:
            n_success += 1
    policy = np.array([
        float(ACTIONS[int(np.argmax(Q[i]))]) if Q[i].any() else float(ACTIONS[0])
        for i in range(env.n)
    ])
    return {
        "Q": Q,
        "policy_idx": np.array([int(np.argmax(Q[i])) for i in range(env.n)]),
        "policy_rates": policy,
        "visits": visits,
        "episodes": episodes,
        "success_rate": n_success / max(episodes, 1),
    }


# --------------------------------------------------------------------------
# 随机环境：RL 的"按观测调整"能力
# --------------------------------------------------------------------------
def simulate_random(env: ChargeEnv, policy_rates: np.ndarray, n_rollouts: int = 200,
                    observe: bool = True, seed: int = 0) -> pd.DataFrame:
    """在随机温度环境中执行策略。

    observe=True  → 按观测到的温度选动作（RL 策略："热就冲、冷就收"）
    observe=False → 用固定策略（DP 只能这样）
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_rollouts):
        temps = rng.normal(30.0, env.temp_sigma, env.n)
        i, cost = 0, 0.0
        while i < env.n:
            T = float(temps[i])
            if observe:
                if env.rate_limit is not None:
                    allowed = env.allowed(i, T)
                    # 观测式策略：在不超上限的前提下取最高倍率
                    a = float(max(allowed))
                else:
                    a = float(policy_rates[i])
            else:
                a = float(policy_rates[i])
                if env.rate_limit is not None:
                    a = min(a, env.rate_limit((env.edges[i] + env.edges[i + 1]) / 2, T))
            _, c, _, _ = env.step(i, a)
            cost += c
            i += 1
        rows.append({"rollout": k, "cost": cost, "mean_temp": float(temps.mean())})
    return pd.DataFrame(rows)


def build_rate_limit_2d(table: pd.DataFrame):
    """把 PyBaMM 的 (温度 × 起始SOC) 倍率上限表封成二维插值函数。

    只做 SOC 插值会让温度波动完全失效（上限不随温度变），随机环境就退化成
    确定环境、RL 的观测优势测不出来。必须二维插值。
    """
    temps = np.sort(table["temperature_C"].unique().astype(float))
    socs = np.sort(table["soc_start"].unique().astype(float))
    grid = np.zeros((len(temps), len(socs)))
    for i, T in enumerate(temps):
        sub = table[table["temperature_C"] == T].set_index("soc_start")["rate_limit_C"]
        grid[i] = [float(sub.get(s, sub.mean())) for s in socs]

    def f(soc_start: float, T: float = 30.0) -> float:
        s = float(np.clip(soc_start, socs.min(), socs.max()))
        t = float(np.clip(T, temps.min(), temps.max()))
        # 沿 SOC 线性插值，再沿温度线性插值
        vals = np.array([np.interp(s, socs, grid[i]) for i in range(len(temps))])
        return float(np.interp(t, temps, vals))

    return f


def q_learning_temp(env: ChargeEnv, temp_bins=(-np.inf, 20.0, 28.0, 32.0, 40.0, np.inf),
                    episodes: int = 20000, alpha: float = 0.3, epsilon: float = 0.3,
                    seed: int = 0, gamma: float = 1.0) -> dict:
    """**把温度放进状态空间**的表格 Q-learning —— RL 的真实形态。

    状态 = (soc 档, 温度档)。策略因此可以表达"热就降速、冷就可以冲"，
    而 DP 只能给一条与温度无关的固定策略。

    这是与 DP 的关键区别：DP 在确定性环境等价于 RL；一旦温度成为可观测量，
    RL 的多余自由度才真正值钱。
    """
    rng = np.random.default_rng(seed)
    n_t = len(temp_bins) - 1
    Q = np.zeros((env.n + 1, n_t, len(ACTIONS)))
    visits = np.zeros((env.n + 1, n_t, len(ACTIONS)), dtype=int)

    def tbin(T: float) -> int:
        for k in range(n_t):
            if temp_bins[k] <= T < temp_bins[k + 1]:
                return k
        return n_t - 1

    for _ in range(episodes):
        i = 0
        while i < env.n:
            T = float(rng.normal(30.0, env.temp_sigma)) if env.temp_sigma else 30.0
            tb = tbin(T)
            allowed = env.allowed(i, T)
            aidx = [j for j, a in enumerate(ACTIONS) if a in allowed]
            if not aidx:
                break
            if rng.random() < epsilon:
                j = int(rng.choice(aidx))
            else:
                j = max(aidx, key=lambda k: Q[i, tb, k])
            _, cost, _, _ = env.step(i, float(ACTIONS[j]), T)
            Q[i, tb, j] += alpha * (-cost + gamma * np.float64(Q[i + 1, tb]).max() - Q[i, tb, j])
            visits[i, tb, j] += 1
            i += 1

    policy = np.zeros((env.n, n_t))
    for i in range(env.n):
        for tb in range(n_t):
            policy[i, tb] = float(ACTIONS[int(np.argmax(Q[i, tb]))]) if Q[i, tb].any() else float(ACTIONS[0])
    return {"Q": Q, "policy_grid": policy, "temp_bins": temp_bins, "visits": visits,
            "episodes": episodes}


def simulate_temp_policy(env: ChargeEnv, policy_grid: np.ndarray, temp_bins,
                         n_rollouts: int = 300, seed: int = 0) -> pd.DataFrame:
    """在随机温度环境里执行 (SOC × 温度) 策略。"""
    rng = np.random.default_rng(seed)
    n_t = len(temp_bins) - 1

    def tbin(T):
        for k in range(n_t):
            if temp_bins[k] <= T < temp_bins[k + 1]:
                return k
        return n_t - 1

    rows = []
    for k in range(n_rollouts):
        i, cost, n_plated = 0, 0.0, 0
        while i < env.n:
            T = float(rng.normal(30.0, env.temp_sigma))
            a = float(policy_grid[i, tbin(T)])
            if env.rate_limit is not None:
                a = min(a, env.rate_limit((env.edges[i] + env.edges[i + 1]) / 2, T))
            _, c, _, _ = env.step(i, a, T)
            cost += c
            i += 1
        rows.append({"rollout": k, "cost": cost})
    return pd.DataFrame(rows)


def benchmark_dp_rl_boost(env_det: ChargeEnv, env_rand: ChargeEnv, dp: dict, rl: dict) -> pd.DataFrame:
    """把 ①DP vs RL 一致性、②随机环境下"观测式 vs 固定"的收益 汇总成一张表。"""
    rows = []
    ev_dp = evaluate_policy(env_det, dp["policy_rates"])
    ev_rl = evaluate_policy(env_det, rl["policy_rates"])
    rows.append({
        "item": "DP 最优（确定性环境真值）",
        "rates": ev_dp["rates"], "n_stages": ev_dp["n_stages"],
        "time_per_fec_h": ev_dp["time_per_fec_h"], "pred_efc": ev_dp["pred_efc"],
    })
    rows.append({
        "item": "Q-learning 收敛解",
        "rates": ev_rl["rates"], "n_stages": ev_rl["n_stages"],
        "time_per_fec_h": ev_rl["time_per_fec_h"], "pred_efc": ev_rl["pred_efc"],
    })

    if env_rand.temp_sigma > 0:
        fixed = simulate_random(env_rand, dp["policy_rates"], observe=False)
        adaptive = simulate_random(env_rand, dp["policy_rates"], observe=True)
        rows.append({
            "item": "随机环境：固定策略",
            "rates": "固定", "n_stages": np.nan,
            "time_per_fec_h": np.nan, "pred_efc": np.nan,
            "mean_cost": float(fixed["cost"].mean()),
            "std_cost": float(fixed["cost"].std()),
        })
        rows.append({
            "item": "随机环境：观测式策略（RL 能力）",
            "rates": "按温度调整", "n_stages": np.nan,
            "time_per_fec_h": np.nan, "pred_efc": np.nan,
            "mean_cost": float(adaptive["cost"].mean()),
            "std_cost": float(adaptive["cost"].std()),
            "improvement_pct": float((fixed["cost"].mean() / adaptive["cost"].mean() - 1) * 100),
        })
    return pd.DataFrame(rows)
