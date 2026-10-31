"""fastcharge — 快充策略优化（A3）核心库.

模块划分:
    config      路径与常量
    protocol    协议结构化参数与特征
    index       ISU-ILCC zip 直读索引器（cycle 级摘要，不落盘解压）
    lifetime    寿命结局判定（EOL / 删失）+ 双口径（cycles / EFC）
    chargetime  充电时长模型（CC + CV + 尾部）+ 阶梯协议时长
    surrogate   代理模型（GP / RF / XGB / Ridge）
    optimize    多目标贝叶斯优化（时长 × 寿命 Pareto）
    physics     PyBaMM 析锂判据与机理闭环验证 + 阶梯仿真

深化 v1.1:
    damage      损伤密度模型 δ(s,r)，含闭合积分
    multistage  阶梯协议搜索与逐段损伤分解
    rl          MDP + 精确 DP + 表格 Q-learning + 随机环境
    early       早期寿命特征与预测
"""

__version__ = "1.2.0"
