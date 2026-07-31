"""合成面板数据生成器。

⚠️  全部是合成数据, 与任何真实行情/策略无关 —— 只用于演示与自测。

生成一个 (date × asset) 面板, 内置四种因子, 每种的"真相"已知:

  factor_good   — 真实有预测力: 由横截面 alpha 驱动未来收益(合规构造)
  factor_bad    — 纯随机噪声: 无预测力
  factor_cheat  — 作弊因子: 直接偷看"下一期收益"(典型的未来函数/数据泄露)
  factor_decay  — 前半段有预测力, 后半段退化为噪声: 用于演示衰减诊断

构造纪律(和真实研究一致):
  alpha 逐日独立抽样(横截面信号), ret_t 只依赖 alpha_{t-1}(前一期信息);
  factor_cheat 故意违反这一纪律, 让检测工具能当场抓住它。
"""

import numpy as np
import pandas as pd
from typing import Optional

FWD_PERIODS = [1, 3, 5, 10, 20]


def generate_panel(n_dates: int = 300, n_assets: int = 50, seed: int = 42,
                   decay_cutoff: Optional[float] = None,
                   alpha_beta: float = 0.15, alpha_sd: float = 0.0015,
                   noise_sd: float = 0.5) -> pd.DataFrame:
    """生成 (date × asset) 面板。

    返回列: date, asset, ret, fwd_ret_1/3/5/10/20,
            factor_good, factor_bad, factor_cheat, factor_decay, alpha
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2025-01-01', periods=n_dates, freq='B')
    assets = ['A%02d' % i for i in range(n_assets)]

    # 横截面 alpha: 每日独立抽样(不同资产不同, 与时间无关)
    alpha = pd.DataFrame(rng.normal(0, alpha_sd, (n_dates, n_assets)),
                         index=dates, columns=assets)

    # 收益: ret_t = alpha_beta * alpha_{t-1} + 噪声(严格只用前一期信息)
    noise = rng.normal(0, noise_sd, (n_dates, n_assets)) * 0.002  # 收益噪声 ~0.1%
    alpha_lag = alpha.shift(1).fillna(0.0)
    ret = (alpha_beta * alpha_lag + noise).values

    panel = pd.DataFrame(ret, index=dates, columns=assets).stack().reset_index()
    panel.columns = ['date', 'asset', 'ret']

    # 前瞻收益: fwd_ret_k = 未来 k 期收益之和(按 asset 分组计算)
    g = panel.groupby('asset')['ret']
    for k in FWD_PERIODS:
        shifted = [g.shift(-i).rename('r%d' % i) for i in range(1, k + 1)]
        panel['fwd_ret_%d' % k] = pd.concat(shifted, axis=1).sum(axis=1)

    # alpha 对齐到面板
    panel['alpha'] = alpha.stack().reset_index(level=0, drop=True).values
    panel = panel.sort_values(['date', 'asset']).reset_index(drop=True)

    # 因子构造
    n = len(panel)
    f_noise = 0.001  # 因子观测噪声: 远小于 alpha 幅度, 不淹没信号
    panel['factor_good'] = panel['alpha'] + rng.normal(0, f_noise, n)   # alpha + 截面噪声
    panel['factor_bad'] = rng.normal(0, 1.0, n)                         # 纯噪声
    # 作弊因子: 直接用了下一期收益(未来函数)
    panel['factor_cheat'] = panel['fwd_ret_1'] * 0.9 + rng.normal(0, 1e-4, n)

    # 衰减因子: 前 decay_cutoff 比例有预测力, 之后退化为噪声
    cutoff = decay_cutoff if decay_cutoff is not None else 0.5
    cut = int(n * cutoff)
    clean = panel['alpha'].iloc[:cut] + rng.normal(0, f_noise, cut)
    dirty = rng.normal(0, 1.0, n - cut)
    panel['factor_decay'] = pd.concat([clean, pd.Series(dirty, index=panel.index[cut:])],
                                      axis=0).sort_index()

    return panel
