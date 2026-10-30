"""多目标贝叶斯优化：在（等效充满时间，可用循环寿命）平面上找 Pareto 前沿（阶段 3）。

问题定义
--------
决策变量（协议参数，3 维）:
    span   ∈ [0.30, 0.85]   SOC 窗口宽度
    rate   ∈ [0.50, 4.00]   峰值充电倍率（CC 段）
    d_rate ∈ [0.50, 2.50]   放电倍率（使用强度）

目标 —— **都按"用户真实体验"归一化**（这点最容易做错）:
    f1 = 充满 1 FEC 的等效时长 = t_charge(span, rate) / span
         （浅充不能只看单次时长：充 30% 当然快，但要充 3 次才算 1 FEC）
    f2 = -log(可用循环数)  ← 用户在固定 SOC 窗口下能跑多少次

为什么要这样归一化
------------------
若 f1 用"单次充电时长"、f2 用"EFC 等效寿命"，优化器会同时把 span 压到最小、
rate 拉到最大，得到"3.99C 只充 30%"这种看起来又快又长寿的协议——但那不是
用户要的东西（相当于每隔几小时充一次）。按 FEC 归一后，前沿才落到真实权衡上。

代理模型
--------
寿命目标用 GP（Matern + White），既有均值也有方差，EI 采集能用。
代理的诚实精度在 reports 里给（LOGO-CV），优化结论以"相对排序"解读，
不承诺绝对寿命值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .chargetime import charge_time_h

#: 决策空间边界（严格取自数据实测覆盖范围，不外推）
BOUNDS = {
    "span": (0.30, 0.85),
    "rate": (0.50, 3.00),   # 数据实测峰值倍率上限 3.0C
    "d_rate": (0.50, 2.50),  # 数据实测放电倍率上限 ≈2.48C
}
VAR_ORDER = ("span", "rate", "d_rate")

#: 归一化目标时用的参照窗口（1 FEC 的定义与窗口无关，这里只做量纲统一）
FEC_REF_H = 1.0


# --------------------------------------------------------------------------
# 适用域（applicability domain）
# --------------------------------------------------------------------------
class ApplicabilityDomain:
    """GP 的"可信区间"：到最近训练点的标准化距离不得超过阈值。

    为什么必须有
    ------------
    GP 在远离训练数据处会**回落到先验均值**。若不加约束，贝叶斯优化会主动
    跑到"4C + 0.85 窗口"这种数据里根本不存在的角落，拿到一个"寿命 ≈ 均值"
    的漂亮结果——纯属外推幻觉。本类用训练点两两距离的分位数做阈值，
    把搜索限制在数据真正覆盖的区域。

    注意: 决策变量 3 维，但代理特征 5 维。适用域在**决策空间**里算
    （span, rate, d_rate），因为那才是优化器实际搜索、且人能解释的空间。

    阈值口径: 用训练集**两两距离**的分位数（不是最近邻分位数）。
    最近邻分位数会退化成"只允许训练点自己"，把整个搜索空间封死；
    两两距离分位数能表达"多大间隔仍算被数据支撑"。
    """

    def __init__(self, X_decision: np.ndarray, quantile: float = 0.95):
        self.mean = X_decision.mean(axis=0)
        self.std = X_decision.std(axis=0) + 1e-9
        Z = (X_decision - self.mean) / self.std
        self._Z_train = Z
        n = len(Z)
        D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
        # 两两距离（只取上三角）
        iu = np.triu_indices(n, k=1)
        self.threshold = float(np.quantile(D[iu], quantile))
        self.quantile = quantile
        self.n_train = n
        np.fill_diagonal(D, np.inf)
        self.nn_distance_p90 = float(np.quantile(D.min(axis=1), 0.90))

    def distance(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.sqrt(((Z[:, None, :] - self._Z_train[None, :, :]) ** 2).sum(-1)).min(axis=1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """返回布尔掩码：True = 在适用域内。"""
        return self.distance(X) <= self.threshold


def objectives(X: np.ndarray, gp_model, ct_params: dict) -> np.ndarray:
    """X (n,3) → 目标矩阵 (n,2)，两目标都是**最小化**。"""
    span, rate, d_rate = X[:, 0], X[:, 1], X[:, 2]
    t_window = charge_time_h(span, rate, ct_params["a_cc"], ct_params["c_cv"], ct_params["d_tail"])
    t_per_fec = t_window / np.clip(span, 1e-6, None)   # 充满 1 FEC 的等效时长（小时）
    feats = feature_rows(span, rate, d_rate)
    mu = gp_model.predict(feats)                       # log(可用循环数)
    return np.column_stack([t_per_fec / FEC_REF_H, -mu])


def feature_rows(span: np.ndarray, rate: np.ndarray, d_rate: np.ndarray) -> np.ndarray:
    """决策变量 → 代理模型特征（顺序必须与 CORE_FEATURES 一致）。"""
    soc_lo = 1.0 - span
    highsoc_exposure = np.maximum(1.0 - np.maximum(soc_lo, 0.70), 0.0) * rate
    return np.column_stack([span, rate, d_rate, highsoc_exposure, soc_lo])


def decision_space(cells: pd.DataFrame) -> np.ndarray:
    """cells 表 → 决策变量矩阵 (n,3)，即适用域的训练点。"""
    return np.column_stack([
        cells["soc_span"].to_numpy(float),
        cells["charge_rate_peak"].to_numpy(float),
        cells["discharge_rate_mean"].to_numpy(float),
    ])


def design_space_coverage(cells: pd.DataFrame) -> pd.DataFrame:
    """报告数据在 (span, rate) 平面上的实际覆盖格，作为优化"可信区"的说明。"""
    c = cells.dropna(subset=["soc_span", "charge_rate_peak"]).copy()
    c["span_bin"] = pd.cut(c["soc_span"], [0.25, 0.35, 0.50, 0.70, 0.85, 1.01])
    c["rate_bin"] = pd.cut(c["charge_rate_peak"], [0.4, 0.8, 1.3, 1.9, 2.5, 3.05])
    piv = c.pivot_table(index="rate_bin", columns="span_bin", values="cell_id",
                        aggfunc="count", observed=True).fillna(0).astype(int)
    return piv


# --------------------------------------------------------------------------
# Pareto 工具
# --------------------------------------------------------------------------
def pareto_front(F: np.ndarray) -> np.ndarray:
    """返回非支配点的索引（两目标最小化）。"""
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominated.any():
            keep[i] = False
    return np.flatnonzero(keep)


def hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    """二目标超体积（最小化口径；ref 应比所有点都差）。"""
    if len(F) == 0:
        return 0.0
    pts = np.array([p for p in F if np.all(p <= ref)])
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


# --------------------------------------------------------------------------
# EI 贝叶斯优化（纯 sklearn/scipy）
# --------------------------------------------------------------------------
def _ei(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    from scipy.stats import norm

    sigma = np.maximum(sigma, 1e-9)
    z = (best - mu - xi) / sigma
    return (best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def _make_gp(n_dim: int, seed: int):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    k = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
        length_scale=np.ones(n_dim), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(1e-3, (1e-7, 1e-1))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=k, normalize_y=True, n_restarts_optimizer=1, random_state=seed, alpha=1e-6))


def bayes_opt_multistart(X, Y, bounds, n_iter, n_candidates=3000, seed=0, xi=0.01):
    """对每个目标轴各跑一次 EI 采集（把前沿两端拉开），返回新增点。"""
    import warnings

    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    new_points = []
    for tgt_col in range(Y.shape[1]):
        y = Y[:, tgt_col].copy()
        for _ in range(n_iter):
            m = _make_gp(X.shape[1], seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X, y)
            cand = rng.uniform(lo, hi, size=(n_candidates, X.shape[1]))
            mu, sd = m.predict(cand, return_std=True)
            ei = _ei(mu, sd, best=float(y.min()), xi=xi)
            x_new = cand[int(np.argmax(ei))]
            new_points.append(x_new)
    return np.array(new_points)


def run_bo(gp_model, ct_params: dict, *, X_decision: np.ndarray, n_init: int = 48,
           n_iter: int = 16, seed: int = 0, domain_quantile: float = 0.90,
           n_grid=(18, 18, 12)) -> dict:
    """两阶段优化（Sobol 铺底 + EI 主动采样）+ 稠密网格参照。

    全部候选点都先过 **适用域** 筛（到最近训练点的距离 ≤ 训练点距离的 P90），
    避免 GP 外推幻觉。
    """
    rng = np.random.default_rng(seed)
    lo = np.array([BOUNDS[v][0] for v in VAR_ORDER])
    hi = np.array([BOUNDS[v][1] for v in VAR_ORDER])
    AD = ApplicabilityDomain(X_decision, quantile=domain_quantile)
    print(f"  适用域: 训练点 {AD.n_train} 个，P{int(domain_quantile*100)} 最近邻距离阈值 = {AD.threshold:.2f}")

    sampler = qmc.Sobol(d=len(VAR_ORDER), scramble=True, seed=seed)
    # 反复采样直到凑够域内的初始化点
    X0_pool = []
    for _ in range(40):
        batch = qmc.scale(sampler.random(200), lo, hi)
        ok = AD(batch)
        X0_pool.append(batch[ok])
        if sum(len(b) for b in X0_pool) >= n_init * 2:
            break
    X0 = np.vstack(X0_pool)[:n_init]
    Y0 = objectives(X0, gp_model, ct_params)
    print(f"  初始化: {len(X0)} 个域内点（原始采样命中率 "
          f"{len(X0)/max(1,200*1):.0%} 量级）")

    Xbo_raw = bayes_opt_multistart(X0, Y0, np.column_stack([lo, hi]), n_iter=n_iter, seed=seed)
    ok_bo = AD(Xbo_raw)
    Xbo = Xbo_raw[ok_bo]
    Ybo = objectives(Xbo, gp_model, ct_params) if len(Xbo) else np.zeros((0, 2))
    print(f"  BO 采集 {len(Xbo_raw)} 点 → 域内 {len(Xbo)} 点")

    X_all = np.vstack([X0, Xbo]) if len(Xbo) else X0
    Y_all = np.vstack([Y0, Ybo]) if len(Ybo) else Y0
    _, uniq = np.unique(np.round(X_all, 3), axis=0, return_index=True)
    X_all, Y_all = X_all[uniq], Y_all[uniq]

    # 稠密网格参照（同样过滤适用域）
    axes = [np.linspace(BOUNDS[v][0], BOUNDS[v][1], n) for v, n in zip(VAR_ORDER, n_grid)]
    G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(len(VAR_ORDER), -1).T
    G = G[AD(G)]
    YG = objectives(G, gp_model, ct_params)

    i_all = pareto_front(Y_all)
    i_grid = pareto_front(YG)

    # 参照点必须**劣于所有点**（两目标都最小化 → 取两轴的最大值再加余量）
    ref = np.array([max(Y_all[:, 0].max(), YG[:, 0].max()) * 1.05,
                    max(Y_all[:, 1].max(), YG[:, 1].max()) * 1.05])
    hv_bo = hypervolume_2d(Y_all[i_all], ref=ref)
    hv_grid = hypervolume_2d(YG[i_grid], ref=ref)
    return {
        "X_samples": X_all, "Y_samples": Y_all,
        "X_grid": G, "Y_grid": YG,
        "pareto_bo": X_all[i_all], "pareto_bo_obj": Y_all[i_all],
        "pareto_grid": G[i_grid], "pareto_grid_obj": YG[i_grid],
        "n_evals_bo": len(X_all), "n_evals_grid": len(G),
        "hv_bo": hv_bo, "hv_grid": hv_grid,
        "hv_ref": ref,
        "efficiency_ratio": hv_bo / max(hv_grid, 1e-9),
        "domain_threshold": AD.threshold,
    }
