"""数据预处理流水线 + 未来函数防护。

量化的工程铁律(本模块把它们变成可复用代码):

  1. 去极值用 MAD(中位数绝对偏差) —— 对离群值比 3σ 稳健得多
  2. 标准化用**截面** z-score(按日期分组) —— 整体 z-score 会把跨期风格混进因子
  3. 中性化: 因子对风格暴露(市值/行业/波动率等)回归取残差, 剔除风格干扰
  4. 未来函数防护:
     - 因子计算必须只用截至当期收盘的信息, 评估前 shift(1) 是最后一道保险
     - detect_lookahead() 检查因子对**下一期收益**的 IC(1) 是否高到不合理 ——
       合规因子的 IC(1) 有限(0.02~0.3), 偷看未来的因子 IC(1)≈1.0,
       "完美得不像真的"就是数据泄露的指纹
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def mad_winsorize(series: pd.Series, n: float = 5.0) -> pd.Series:
    """MAD 去极值: 把 |x - median| > n×MAD 的点压缩到边界。

    MAD(Median Absolute Deviation)比标准差稳健, 不受离群值自身影响。
    """
    s = series.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad <= 0:
        return s
    scale = 1.4826 * mad  # 使 MAD 与 σ 同尺度(正态假设下)
    lo, hi = med - n * scale, med + n * scale
    return s.clip(lo, hi)


def cross_sectional_zscore(df: pd.DataFrame, factor: str, date_col: str = 'date',
                           min_obs: int = 10) -> pd.Series:
    """截面 z-score: 每个日期**内部**做标准化。

    整体 z-score 会让"某天全体因子偏高"这种风格波动留在因子里;
    按日期分组标准化后, 每个截面均值≈0、标准差≈1, 跨期才可比。
    日期内样本过少置 NaN(截面样本不足时 z-score 没有统计意义)。
    """
    def _z(g: pd.Series) -> pd.Series:
        if len(g) < min_obs:
            return pd.Series(np.nan, index=g.index)
        return (g - g.mean()) / (g.std() + 1e-12)

    return df.groupby(date_col)[factor].transform(_z)


def neutralize(df: pd.DataFrame, factor: str, expose_cols: List[str],
               add_const: bool = True) -> pd.Series:
    """中性化: 因子对风格暴露列做 OLS 回归, 取残差。

    返回的残差与风格暴露正交 —— 剩下的才是"剔除风格干扰"后的因子贡献。
    典型用途: 剔除市值/行业/波动率等风格, 避免因子收益其实是风格收益。
    """
    X = df[expose_cols].astype(float).copy()
    if add_const:
        X['_const'] = 1.0
    y = df[factor].astype(float)
    mask = y.notna() & X.notna().all(axis=1)
    resid = pd.Series(np.nan, index=df.index)
    if mask.sum() < len(expose_cols) + 2:
        return resid
    beta, *_ = np.linalg.lstsq(X[mask].values, y[mask].values, rcond=None)
    resid.loc[mask] = y[mask].values - X[mask].values @ beta
    return resid


def shift_factor(df: pd.DataFrame, factor: str, periods: int = 1) -> pd.Series:
    """评估前强制把因子整体 shift(periods) —— 防未来函数的最后一道保险。

    因子在 t 时刻计算出来后, 最早只能影响 t+1 的持仓/收益。
    """
    return df[factor].shift(periods)


def detect_lookahead(df: pd.DataFrame, factor: str, fwd_col: str = 'fwd_ret_1',
                     threshold: float = 0.7, min_obs: int = 30) -> Dict:
    """未来函数检测: 因子对**下一期收益**的 IC 是否高到不合理。

    原理: 合规因子的预测力是有限的(典型 |IC(1)| 在 0.02~0.3 之间, 且随周期
    增长而衰减); 若 IC(1) 接近 1.0, 几乎可以确定因子值本身就是未来收益 ——
    "完美得不像真的"就是数据泄露的指纹。真实因子永远做不到 IC(1)≈1。

    返回: {'factor', 'ic_1', 'status', 'n', 'threshold'}
      status: 'OK' | 'LIKELY_LEAK' | 'INSUFFICIENT'
    """
    f = df[factor].astype(float)
    fr = df[fwd_col].astype(float)
    mask = f.notna() & fr.notna()
    out = {'factor': factor, 'threshold': threshold, 'n': int(mask.sum())}
    if mask.sum() < min_obs:
        out['status'] = 'INSUFFICIENT'
        out['ic_1'] = None
        return out
    ic = f[mask].corr(fr[mask], method='spearman')
    if pd.isna(ic):
        out['status'] = 'INSUFFICIENT'
        out['ic_1'] = None
        return out
    out['ic_1'] = round(float(ic), 4)
    out['status'] = 'LIKELY_LEAK' if abs(ic) > threshold else 'OK'
    return out


def detect_lookahead_all(df: pd.DataFrame, factors: List[str],
                         fwd_col: str = 'fwd_ret_1', **kw) -> Dict[str, Dict]:
    """批量未来函数检测。"""
    return {f: detect_lookahead(df, f, fwd_col, **kw) for f in factors}
