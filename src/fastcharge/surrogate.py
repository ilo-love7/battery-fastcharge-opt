"""代理模型：协议 → 寿命结局（v1.1 阶段 2）。

关键设计决策
------------
1. **交叉验证按协议分组（Leave-One-Group-Out）**
   同一协议的 4 只电芯是重复实验，随机 KFold 会让同组样本同时进训练和测试，
   精度虚高。60 个协议 → 60 折 LOGO，这是唯一诚实的泛化口径。

2. **双目标建模**
   log(efc)       材料耐久度：每搬运 1 FEC 的容量损失（口径干净，倍率效应可见）
   log(eol_cycle) 用户可用循环数（受 SOC 窗口强烈影响，表观值）
   两个都报，让"口径差"变成结论而不是误差。

3. **小样本 → GP 优先**
   n=239、特征 5 维，高斯过程比树模型更不容易过拟合，且自带不确定性，
   可直接喂给贝叶斯优化（阶段 3）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .lifetime import add_throughput_columns

#: 建模用的物理特征（不含任何泄漏变量）
CORE_FEATURES = [
    "f_soc_span",
    "f_rate_peak",
    "f_discharge_rate",
    "f_highsoc_exposure",
    "f_soc_lo",
]

TARGET_COL = {"efc": "y_efc", "cycles": "y_cycles"}


def make_targets(cells: pd.DataFrame) -> pd.DataFrame:
    """构造双目标（log 空间）与配套列。

    删失电芯（跑完测试仍未跌破 80% SOH）没有确定的 eol_cycle → 从建模样本里剔除，
    但保留在表里（censored=True），供报告中单列说明。
    """
    c = add_throughput_columns(cells)
    out = pd.DataFrame(
        {
            "cell_id": c["cell_id"],
            "group": c["group"],
            "y_efc": np.log(c["efc"].clip(lower=1.0)),
            "y_cycles": np.log(c["eol_cycle"].clip(lower=1.0)),
            "censored": c["censored"].astype(bool),
            "efc": c["efc"],
            "eol_cycle": c["eol_cycle"],
            "soc_span": c["soc_span"],
        }
    ).reset_index(drop=True)
    return out


def modeling_rows(tgt: pd.DataFrame) -> pd.Series:
    """可用于建模的行掩码：双目标都有限。"""
    return tgt["y_efc"].notna() & tgt["y_cycles"].notna()


# --------------------------------------------------------------------------
# 模型工厂：每个返回 **新的未拟合** estimator
# --------------------------------------------------------------------------
def _make_ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))


def _make_gp(seed: int = 0):
    n_feat = len(CORE_FEATURES)
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
        length_scale=np.ones(n_feat), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-4, 1.0))
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed, alpha=1e-6
        ),
    )


def _make_rf(seed: int = 0):
    return RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1
    )


def _make_xgb(seed: int = 0):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=0,
    )


FACTORIES = {
    "ridge": _make_ridge,
    "gp": _make_gp,
    "rf": _make_rf,
    "xgb": _make_xgb,
}


# --------------------------------------------------------------------------
# 分组交叉验证
# --------------------------------------------------------------------------
def logo_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, factory, *, return_preds: bool = False):
    """Leave-One-Group-Out 交叉验证。"""
    preds = np.full(len(y), np.nan)
    uniq = np.unique(groups)
    for g in uniq:
        te = groups == g
        tr = ~te
        if tr.sum() < 6 or te.sum() == 0:
            continue
        m = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ok = np.isfinite(preds) & np.isfinite(y)
    metrics = {
        "n": int(ok.sum()),
        "n_groups": int(len(uniq)),
        "r2": float(r2_score(y[ok], preds[ok])) if ok.sum() > 2 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y[ok], preds[ok]))) if ok.sum() else np.nan,
        "mae": float(mean_absolute_error(y[ok], preds[ok])) if ok.sum() else np.nan,
        "mape_pct": float(np.mean(np.abs(np.expm1(preds[ok]) - np.expm1(y[ok])) / np.expm1(y[ok])) * 100)
        if ok.sum() else np.nan,
        "rmse_physical": float(np.sqrt(np.mean((np.exp(preds[ok]) - np.exp(y[ok])) ** 2))) if ok.sum() else np.nan,
    }
    if return_preds:
        return metrics, preds
    return metrics


def evaluate_all(cells: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    from .protocol import build_feature_matrix

    feats, _ = build_feature_matrix(cells)
    tgt = make_targets(cells)
    keep = modeling_rows(tgt)
    X = feats.loc[keep, CORE_FEATURES].to_numpy(float)
    groups = tgt.loc[keep, "group"].to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"  剔除 {n_dropped} 只删失电芯（无确定寿命结局）")

    rows = []
    preds_store: dict[str, np.ndarray] = {}
    for name, factory in FACTORIES.items():
        for tgt_name in ("efc", "cycles"):
            y = tgt.loc[keep, TARGET_COL[tgt_name]].to_numpy(float)
            met, p = logo_cv(X, y, groups, factory, return_preds=True)
            met.update({"model": name, "target": tgt_name})
            rows.append(met)
            preds_store[f"{name}_{tgt_name}"] = p
    df = pd.DataFrame(rows)[
        ["model", "target", "n", "n_groups", "r2", "rmse", "mae", "mape_pct", "rmse_physical"]
    ]
    df = df.sort_values(["target", "mape_pct"], ascending=[True, True]).reset_index(drop=True)
    return df, preds_store


def variance_ceiling(cells: pd.DataFrame) -> pd.DataFrame:
    """方差分解：协议能解释多少？剩下的个体差异是**任何协议级模型的误差天花板**。

    做法: 对 log 目标做单因素方差分解，
        ICC = between-group SS / total SS
    若模型完美预测每个协议的组均值，R² 上限就是 ICC。超出部分只能靠
    "单电芯级特征"（本数据集没有）来解释。
    """
    c = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    rows = []
    for col in ("efc", "eol_cycle"):
        y = np.log(c[col].astype(float))
        mu = y.mean()
        ss_total = float(((y - mu) ** 2).sum())
        ss_between = float(
            c.assign(_y=y).groupby("group")["_y"].apply(lambda s: len(s) * (s.mean() - mu) ** 2).sum()
        )
        g = c.groupby("group")
        rows.append(
            {
                "target": col,
                "ss_total": ss_total,
                "ss_between": ss_between,
                "ss_within": ss_total - ss_between,
                "icc_r2_ceiling": ss_between / ss_total,
                "within_group_ratio_median": float(
                    (g[col].max() / g[col].min()).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def group_oracle_baseline(cells: pd.DataFrame) -> pd.DataFrame:
    """协议级"最优可能"基线: 直接用本协议组内中位数当预测（LOGO 下这是作弊上限）。"""
    c = add_throughput_columns(cells).dropna(subset=["eol_cycle"])
    rows = []
    for col in ("efc", "eol_cycle"):
        y = c[col].astype(float)
        gm = c.groupby("group")[col].transform("median")
        rows.append(
            {
                "target": col,
                "r2_group_median": float(r2_score(y, gm)),
                "mape_group_median_pct": float(np.mean(np.abs(gm - y) / y) * 100),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 最终模型 + 归因
# --------------------------------------------------------------------------
def fit_final(cells: pd.DataFrame, target: str = "efc", model_name: str = "gp"):
    from .protocol import build_feature_matrix

    feats, _ = build_feature_matrix(cells)
    tgt = make_targets(cells)
    keep = modeling_rows(tgt)
    X = feats.loc[keep, CORE_FEATURES].to_numpy(float)
    y = tgt.loc[keep, TARGET_COL[target]].to_numpy(float)
    model = FACTORIES[model_name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model, X, y, CORE_FEATURES


def shap_attribution(cells: pd.DataFrame, target: str = "efc", model_name: str = "rf") -> pd.DataFrame:
    """树模型 SHAP 归因（GP 无 SHAP，树模型是标准做法）。"""
    import shap

    model, X, _, names = fit_final(cells, target=target, model_name=model_name)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
