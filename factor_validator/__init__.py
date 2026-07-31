"""factor-validator — 因子验证工具链。

回答一个问题: "这个因子, 信还是不信?"
  - 预处理    MAD 去极值 / 截面 z-score / 中性化
  - 未来函数  detect_lookahead(): 因子是否偷看了当期收益(数据泄露)
  - 预测力    Rank IC / IC 衰减曲线 / IR / 分位数收益
  - 衰减诊断  分段 IC + 斜率, 判定 STABLE / MIXED / DECAYING
  - 评级      A/B/C/D, 可入 CI

本工具**不挖因子**, 只验证因子 —— "因子有没有用"是一个独立的工程问题,
应该在回测之前回答。
"""

from .decay import decay_assessment, panel_segment_ic, segment_ic
from .ic import (
    factor_correlation,
    ic_decay,
    ic_series,
    ir,
    panel_ic_series,
    quantile_returns,
    rank_ic,
    rolling_analysis,
)
from .preprocess import (
    cross_sectional_zscore,
    detect_lookahead,
    detect_lookahead_all,
    mad_winsorize,
    neutralize,
    shift_factor,
)
from .report import (
    DEFAULT_PERIODS,
    evaluate_all,
    evaluate_factor,
    evaluate_panel_factor,
    grade_factor,
    render_report,
)

__all__ = [
    # 预处理
    'mad_winsorize',
    'cross_sectional_zscore',
    'neutralize',
    'shift_factor',
    # 未来函数防护
    'detect_lookahead',
    'detect_lookahead_all',
    # IC/IR
    'rank_ic',
    'ic_series',
    'ir',
    'ic_decay',
    'quantile_returns',
    'rolling_analysis',
    'factor_correlation',
    'panel_ic_series',
    # 衰减
    'segment_ic',
    'panel_segment_ic',
    'decay_assessment',
    # 报告
    'evaluate_factor',
    'evaluate_all',
    'evaluate_panel_factor',
    'grade_factor',
    'render_report',
    'DEFAULT_PERIODS',
]

__version__ = '0.1.0'
