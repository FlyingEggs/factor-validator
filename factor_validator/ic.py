"""IC/IR 系列指标: 因子预测力的经典度量。

  Rank IC  — Spearman 秩相关系数(factor_t vs fwd_ret_{t→t+N}), 对离群值稳健
  IR       — mean(滚动 IC) / std(滚动 IC), 衡量预测力的稳定性
  IC 衰减  — 因子在不同前瞻周期(N=1/3/5/10/20)的 IC, 看预测力随周期怎么衰减
  分位数收益 — 按因子值分组, 看各组未来收益的单调性(top vs bottom)
  滚动 IC  — 预测力在时间上的稳定性(窗口正收益占比 / IR)

一切函数都只接受 Series 或 DataFrame, 不依赖任何列名约定 ——
你用自己的列名, 传进来即可。
"""

import warnings

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr


def rank_ic(factor: pd.Series, fwd_ret: pd.Series, min_obs: int = 10) -> Tuple[float, float]:
    """Rank IC: 返回 (ic, p_value)。样本不足返回 (0.0, 1.0)。"""
    mask = factor.notna() & fwd_ret.notna()
    fv, fr = factor[mask], fwd_ret[mask]
    if len(fv) < min_obs:
        return 0.0, 1.0
    with warnings.catch_warnings():
        # 常量输入时 spearmanr 会告警并返回 nan —— 这里统一按无效处理
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ic, pval = spearmanr(fv, fr)
    if np.isnan(ic):
        return 0.0, 1.0
    return float(ic), float(pval)


def ic_series(factor: pd.Series, fwd_ret: pd.Series, window: int = 60,
              step: int = 10, min_obs: int = 10) -> pd.Series:
    """滚动窗口 IC 序列(索引 = 各窗口的结束位置)。"""
    mask = factor.notna() & fwd_ret.notna()
    fv, fr = factor[mask].values, fwd_ret[mask].values
    idx = factor[mask].index
    ics, dates = [], []
    for start in range(0, len(fv) - window + 1, step):
        end = start + window
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            ic, _ = spearmanr(fv[start:end], fr[start:end])
        if not np.isnan(ic):
            ics.append(float(ic))
            dates.append(idx[end - 1])
    return pd.Series(ics, index=dates)


def ir(ic_series: pd.Series) -> float:
    """Information Ratio = mean(IC) / std(IC)。窗口太少或 IC 无波动 → 0。"""
    if len(ic_series) < 3:
        return 0.0
    std = ic_series.std()
    if std < 1e-10:
        return 0.0
    return float(ic_series.mean() / std)


def ic_decay(factor: pd.Series, forward_returns: Dict[int, pd.Series],
             min_obs: int = 10) -> Dict[int, Dict]:
    """不同前瞻周期的 IC: {period: {'ic': float, 'pval': float}}。

    健康的因子: 短周期 IC 高, 随周期增长单调衰减;
    异常因子: 所有周期 IC 都异常高且不衰减 —— 配合 lookahead 检测一起看。
    """
    out: Dict[int, Dict] = {}
    for period, fr in sorted(forward_returns.items()):
        ic, pval = rank_ic(factor, fr, min_obs)
        out[period] = {'ic': ic, 'pval': pval}
    return out


def quantile_returns(factor: pd.Series, fwd_ret: pd.Series,
                     n_quantiles: int = 5, min_obs: int = 15) -> Dict:
    """按因子值分位分组, 看各组未来收益。

    有效因子: top 组收益 > bottom 组收益(单调), long_short_spread > 0;
    无效因子: 各组收益无差异。
    """
    mask = factor.notna() & fwd_ret.notna()
    sub = pd.DataFrame({'factor': factor[mask], 'fwd_ret': fwd_ret[mask]})
    if len(sub) < n_quantiles * min_obs:
        return {}
    sub['quantile'] = pd.qcut(sub['factor'], n_quantiles, labels=False, duplicates='drop')
    out: Dict = {}
    for q in sorted(sub['quantile'].unique()):
        g = sub[sub['quantile'] == q]
        out['Q%d' % (q + 1)] = {
            'mean_ret': float(g['fwd_ret'].mean()),
            'hit_rate': float((g['fwd_ret'] > 0).mean()),
            'count': int(len(g)),
        }
    qs = sub['quantile']
    top = sub[qs == qs.max()]['fwd_ret'].mean()
    bottom = sub[qs == qs.min()]['fwd_ret'].mean()
    out['long_short_spread'] = float(top - bottom)
    return out


def rolling_analysis(factor: pd.Series, fwd_ret: pd.Series,
                     window: int = 100, step: int = 20) -> Dict:
    """滚动 IC 稳定性分析。"""
    s = ic_series(factor, fwd_ret, window, step)
    if len(s) < 2:
        return {'n_windows': int(len(s)), 'ic_series': s}
    return {
        'n_windows': int(len(s)),
        'ic_series': s,
        'mean_ic': float(s.mean()),
        'std_ic': float(s.std()),
        'ir': ir(s),
        'min_ic': float(s.min()),
        'max_ic': float(s.max()),
        'positive_ratio': float((s > 0).mean()),
    }


def factor_correlation(df: pd.DataFrame, factors: List[str],
                       min_obs: int = 10) -> pd.DataFrame:
    """因子间 Spearman 相关矩阵 —— 发现冗余因子(高相关=在重复表达同一信息)。"""
    valid = [f for f in factors if f in df.columns]
    if len(valid) < 2:
        return pd.DataFrame()
    sub = df[valid].dropna()
    if len(sub) < min_obs:
        return pd.DataFrame()
    return sub.corr(method='spearman')


def panel_ic_series(df: pd.DataFrame, factor: str, fwd_col: str,
                    date_col: str = 'date', min_obs: int = 10) -> pd.Series:
    """面板数据: 逐日截面 Rank IC 序列(索引 = 日期)。

    面板(多资产 × 多日期)的稳定性分析必须以**日期**为窗口:
    每个日期对全部资产的横截面算一次 IC, 得到的 IC 序列再算 IR 才有意义
    (扁平序列滑窗会把不同日期的截面混在一起)。
    """
    rows = {}
    for d, g in df.groupby(date_col, sort=True):
        ic, _ = rank_ic(g[factor], g[fwd_col], min_obs)
        rows[d] = ic
    return pd.Series(rows)
