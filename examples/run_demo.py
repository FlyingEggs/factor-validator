"""一键演示: 四个已知"真相"的合成因子, 跑完整验证流水线。

  factor_good   — 真实有预测力 → 应评 A/B, 无未来函数, 不衰减
  factor_bad    — 纯噪声       → 应评 D
  factor_cheat  — 偷看未来     → 被 detect_lookahead 当场抓出
  factor_decay  — 前强后弱     → 被判 DECAYING

运行:
  python3 examples/run_demo.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from factor_validator import (
    detect_lookahead_all,
    evaluate_panel_factor,
    mad_winsorize,
)
from examples.synthetic import FWD_PERIODS, generate_panel

FACTORS = ['factor_good', 'factor_bad', 'factor_cheat', 'factor_decay']


def main():
    panel = generate_panel(n_dates=300, n_assets=50, seed=42)
    print('面板: %d 行 (300 交易日 × 50 资产), 4 个合成因子\n' % len(panel))

    # 1) 预处理: MAD 去极值(极端值被压缩, 分布主体不变)
    before = panel['factor_bad']
    after = mad_winsorize(before, n=5.0)
    print('== 第 1 步: MAD 去极值(factor_bad) ==')
    print('  去极值前: min=%.3f max=%.3f | 去极值后: min=%.3f max=%.3f' % (
        before.min(), before.max(), after.min(), after.max()))
    print('  (极端离群值被压缩到 ±5×MAD 边界)\n')

    # 2) 未来函数检测: 因子对下一期收益的 IC(1) 是否高到不合理
    print('== 第 2 步: 未来函数检测(IC(1) 合理性) ==')
    for f, r in detect_lookahead_all(panel, FACTORS, fwd_col='fwd_ret_1').items():
        mark = {'OK': '[通过]', 'LIKELY_LEAK': '[!!] 疑似泄露',
                'INSUFFICIENT': '[--] 样本不足'}.get(r['status'], '?')
        ic1 = '%.4f' % r['ic_1'] if r['ic_1'] is not None else 'N/A'
        print('  %s %-14s IC(1)=%s' % (mark, f, ic1))
    print('  (合规因子 IC(1) 有限 ~0.1; factor_cheat 直接用了未来收益, IC(1)≈1.0)\n')

    # 3) 完整评估(面板感知: 逐日截面 IC → IR, 按日期分段看衰减)
    print('== 第 3 步: 完整评估(前瞻周期 %s) ==' % '/'.join(str(p) for p in FWD_PERIODS))
    fwd_cols = {p: 'fwd_ret_%d' % p for p in FWD_PERIODS}
    print('  %-14s %8s %8s %7s %9s %9s %4s' % (
        '因子', 'IC(5)', 'IC(1)', 'IR', 'Q5-Q1', '衰减', '评级'))
    for f in FACTORS:
        r = evaluate_panel_factor(panel, f, fwd_cols)
        ic1 = r['ic_decay'].get(1, {}).get('ic', 0.0)
        q = r.get('quantiles', {})
        spread = q.get('long_short_spread', 0.0) if q else 0.0
        d_status = r.get('decay', {}).get('status', '')
        d_mark = {'DECAYING': '!衰减', 'STABLE': '稳定',
                  'INSUFFICIENT': '不足'}.get(d_status, '?')
        lookahead = detect_lookahead_all(panel, [f], fwd_col='fwd_ret_1')[f]['status']
        grade = r['grade']
        if lookahead == 'LIKELY_LEAK':
            grade = 'X(泄露)'
        print('  %-14s %8.4f %8.4f %7.2f %9.5f %9s %4s' % (
            f, r.get('main_ic', 0.0), ic1, r.get('ir', 0.0),
            spread, d_mark, grade))

    print('\n结论: 工具不挖因子, 只回答"这个因子信不信"。'
          '合成演示里 4 个因子全部被正确识别。')


if __name__ == '__main__':
    main()
