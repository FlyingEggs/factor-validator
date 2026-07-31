"""因子衰减诊断: 预测力是否在时间上消失。

"因子会衰减"是量化圈最贵的学费 —— 上个月还能用的因子, 这个月开始亏钱。
两个可量化的信号:

  1. 分段 IC —— 把样本按时间切 N 段, 看 IC 逐段变化;
     末期均值显著弱于早期, 或 IC 线性斜率显著为负 → 衰减
  2. 滚动 IC —— 结合 ic.rolling_analysis 的 positive_ratio / 末期窗口 IC

判定(三态):
  STABLE     — 后期 IC 没有显著弱于早期(含基本持平)
  DECAYING   — 后期 IC 显著弱于早期(或斜率显著为负)
  INSUFFICIENT — 样本不够
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .ic import rank_ic


def segment_ic(factor: pd.Series, fwd_ret: pd.Series, segments: int = 4,
               min_obs: int = 30) -> List[Dict]:
    """按时间顺序把样本切 N 段, 逐段算 Rank IC。

    返回: [{'segment': 1..N, 'ic': float, 'pval': float, 'n': int}, ...]
    """
    mask = factor.notna() & fwd_ret.notna()
    sub = pd.DataFrame({'factor': factor[mask], 'fwd_ret': fwd_ret[mask]})
    if len(sub) < segments * min_obs:
        return []
    n = len(sub)
    size = n // segments
    out: List[Dict] = []
    for s in range(segments):
        start = s * size
        end = start + size if s < segments - 1 else n
        seg = sub.iloc[start:end]
        ic, pval = rank_ic(seg['factor'], seg['fwd_ret'])
        out.append({'segment': s + 1, 'ic': ic, 'pval': pval, 'n': int(len(seg))})
    return out


def decay_assessment(segments: List[Dict], first_drop: float = -0.02,
                     slope_threshold: float = -0.01) -> Dict:
    """根据分段 IC 判断因子是否衰减。"""
    if not segments or len(segments) < 2:
        return {'status': 'INSUFFICIENT', 'segments': segments}
    ics = np.array([s['ic'] for s in segments], dtype=float)
    half = len(ics) // 2
    early = float(ics[:half].mean()) if half else float(ics[0])
    late = float(ics[half:].mean()) if len(ics) - half else float(ics[-1])
    slope = float(np.polyfit(np.arange(len(ics)), ics, 1)[0])

    if late - early < first_drop or slope < slope_threshold:
        status = 'DECAYING'
    else:
        status = 'STABLE'

    return {
        'status': status,
        'early_mean_ic': round(early, 4),
        'late_mean_ic': round(late, 4),
        'ic_slope': round(slope, 6),
        'segments': segments,
    }


def panel_segment_ic(df: pd.DataFrame, factor: str, fwd_col: str,
                     date_col: str = 'date', segments: int = 4,
                     min_obs: int = 30) -> List[Dict]:
    """面板数据: 按日期区间切段, 每段算一次截面 Rank IC。

    面板的"时间分段"必须以日期为界(扁平切段会把同一日期的截面劈开)。
    返回: [{'segment', 'ic', 'pval', 'n', 'range'}, ...]
    """
    dates = sorted(df[date_col].unique())
    if len(dates) < segments * 2:
        return []
    out: List[Dict] = []
    for i, idx in enumerate(np.array_split(np.arange(len(dates)), segments)):
        d0, d1 = dates[idx[0]], dates[idx[-1]]
        sub = df[(df[date_col] >= d0) & (df[date_col] <= d1)]
        ic, pval = rank_ic(sub[factor], sub[fwd_col], min_obs)
        out.append({'segment': i + 1, 'ic': ic, 'pval': pval, 'n': int(len(sub)),
                    'range': '%s~%s' % (str(d0)[:10], str(d1)[:10])})
    return out
