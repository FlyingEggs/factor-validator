"""单因子/全因子评估报告 + A/B/C/D 评级。

把 IC/IR/衰减/分位数/lookahead 检查拼成一个完整的因子体检报告,
并给一个可入 CI 的结论: A/B/C/D。
"""

from typing import Dict, List, Optional

from .decay import decay_assessment, panel_segment_ic, segment_ic
from .ic import (
    factor_correlation,
    ic_decay,
    ir,
    panel_ic_series,
    quantile_returns,
    rank_ic,
    rolling_analysis,
)
from .preprocess import detect_lookahead

DEFAULT_PERIODS = [1, 3, 5, 10, 20]


def grade_factor(main_ic: float, ir_val: float) -> str:
    """评级: |IC| 和 IR 双维度。

      A — |IC5| >= 0.05 且 IR >= 0.5    强且稳定
      B — |IC5| >= 0.03 且 IR >= 0.3    有效且较稳定
      C — |IC5| >= 0.02                 有微弱信号
      D — 其余                          无效
    """
    a = abs(main_ic)
    if a >= 0.05 and ir_val >= 0.5:
        return 'A'
    if a >= 0.03 and ir_val >= 0.3:
        return 'B'
    if a >= 0.02:
        return 'C'
    return 'D'


def evaluate_factor(df, factor: str, forward_returns: Dict[int, object],
                    main_period: int = 5, ic_window: int = 100,
                    ic_step: int = 20, n_segments: int = 4,
                    ret_col: Optional[str] = None) -> Dict:
    """单因子完整评估。

    参数:
      df               — 面板数据(每行一个观测)
      factor           — 因子列名
      forward_returns  — {period: pd.Series(与 df 对齐的未来收益)}
      ret_col          — 若提供, 附做 lookahead(未来函数)检查

    返回: 完整评估 dict, 含 grade。
    """
    if factor not in df.columns:
        return {'factor': factor, 'error': '因子列不存在'}
    fv = df[factor]
    n_obs = int(fv.notna().sum())
    result: Dict = {'factor': factor, 'n_obs': n_obs}

    decay = ic_decay(fv, forward_returns)
    result['ic_decay'] = decay

    main_ic = decay.get(main_period, {}).get('ic', 0.0)
    main_pval = decay.get(main_period, {}).get('pval', 1.0)
    result['main_ic'] = round(main_ic, 4)
    result['main_pval'] = main_pval

    if main_period in forward_returns:
        fr = forward_returns[main_period]
        rolling = rolling_analysis(fv, fr, ic_window, ic_step)
        result['rolling'] = {k: v for k, v in rolling.items() if k != 'ic_series'}
        result['ic_series'] = rolling.get('ic_series')
        result['quantiles'] = quantile_returns(fv, fr)
        result['segments'] = segment_ic(fv, fr, n_segments)
        result['decay'] = decay_assessment(result['segments'])
        result['ir'] = rolling.get('ir', 0.0)
    else:
        result['ir'] = 0.0
        result['decay'] = {'status': 'INSUFFICIENT', 'segments': []}

    result['grade'] = grade_factor(main_ic, result.get('ir', 0.0))

    if ret_col:
        result['lookahead'] = detect_lookahead(df, factor, ret_col)

    return result


def evaluate_all(df, factors: List[str], forward_returns: Dict[int, object],
                 main_period: int = 5, ret_col: Optional[str] = None,
                 **kw) -> List[Dict]:
    """批量评估, 按 |main_ic| 降序。"""
    results = []
    for f in factors:
        r = evaluate_factor(df, f, forward_returns, main_period=main_period,
                            ret_col=ret_col, **kw)
        if 'error' not in r:
            results.append(r)
    results.sort(key=lambda x: abs(x.get('main_ic', 0.0)), reverse=True)
    return results


def evaluate_panel_factor(df, factor: str, forward_cols: Dict[int, str],
                          date_col: str = 'date', main_period: int = 5,
                          min_obs: int = 10, n_segments: int = 4) -> Dict:
    """面板(多资产×多日期)因子的完整评估。

    与 evaluate_factor 的区别: 时间稳定性指标(IC 序列/IR/衰减)全部
    以**日期**为窗口计算截面 IC, 而不是把扁平序列滑窗 —— 这才是面板
    数据的正确姿势。

    参数:
      df             — 面板 DataFrame(每行一个 date×asset 观测)
      factor         — 因子列名
      forward_cols   — {period: 前瞻收益列名}
    """
    if factor not in df.columns:
        return {'factor': factor, 'error': '因子列不存在'}

    result: Dict = {'factor': factor, 'n_obs': int(df[factor].notna().sum())}

    # 混合截面 IC 衰减(全面板 pooled)
    decay = {}
    for period, col in sorted(forward_cols.items()):
        ic, pval = rank_ic(df[factor], df[col], min_obs)
        decay[period] = {'ic': ic, 'pval': pval}
    result['ic_decay'] = decay

    main_ic = decay.get(main_period, {}).get('ic', 0.0)
    result['main_ic'] = round(main_ic, 4)

    # 逐日截面 IC 序列 → IR(时间稳定性)
    if main_period in forward_cols:
        ic_by_date = panel_ic_series(df, factor, forward_cols[main_period],
                                     date_col, min_obs)
        result['ic_by_date'] = ic_by_date
        result['ir'] = ir(ic_by_date)
        result['positive_ratio'] = float((ic_by_date > 0).mean()) if len(ic_by_date) else 0.0
        result['quantiles'] = quantile_returns(df[factor], df[forward_cols[main_period]])
        result['segments'] = panel_segment_ic(df, factor, forward_cols[main_period],
                                              date_col, n_segments, min_obs)
        result['decay'] = decay_assessment(result['segments'])
    else:
        result['ir'] = 0.0
        result['decay'] = {'status': 'INSUFFICIENT', 'segments': []}

    result['grade'] = grade_factor(main_ic, result.get('ir', 0.0))
    return result


def main(argv=None) -> None:
    """CLI: factor-validator --data panel.csv --factor momentum_20d --fwd 1:fwd_ret_1 --fwd 5:fwd_ret_5

    退出码: 任何因子评级 A/B 且未泄露 → 0, 否则 1, 可接 CI。
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog='factor-validator',
        description='因子验证: 评级 A/B/C/D + 未来函数检测 + 衰减诊断。')
    parser.add_argument('--data', required=True, help='CSV 面板数据(date/asset + 因子列 + 前瞻收益列)')
    parser.add_argument('--factor', action='append', required=True, help='因子列名(可多次)')
    parser.add_argument('--fwd', action='append', required=True,
                        help='前瞻收益: "周期:列名", 如 1:fwd_ret_1(可多次)')
    parser.add_argument('--json', default=None, help='输出结构化结果到 JSON 文件')
    args = parser.parse_args(argv)

    import pandas as pd
    df = pd.read_csv(args.data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    forward_cols = {}
    for spec in args.fwd:
        period, _, col = spec.partition(':')
        forward_cols[int(period)] = col
    fwd_cols = {p: c for p, c in forward_cols.items() if c in df.columns}

    rows = []
    for f in args.factor:
        r = evaluate_panel_factor(df, f, fwd_cols)
        leak = detect_lookahead(df, f, fwd_cols.get(1)) if 1 in fwd_cols else None
        rows.append({'factor': f, 'grade': r.get('grade', '?'),
                     'main_ic': r.get('main_ic', 0.0), 'ir': r.get('ir', 0.0),
                     'decay': r.get('decay', {}).get('status', '?'),
                     'lookahead': leak['status'] if leak else None})

    for row in rows:
        flag = '[!!]' if row['lookahead'] == 'LIKELY_LEAK' else '[OK]'
        print('%s %-20s grade=%-2s IC5=%.4f IR=%.2f decay=%s%s' % (
            flag, row['factor'], row['grade'], row['main_ic'], row['ir'],
            row['decay'], ' 疑似泄露!' if row['lookahead'] == 'LIKELY_LEAK' else ''))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)

    usable = all(r['lookahead'] != 'LIKELY_LEAK' and r['grade'] in ('A', 'B')
                 for r in rows)
    sys.exit(0 if usable else 1)


def render_report(results: List[Dict], title: str = '因子评估报告',
                  top_n: int = 20) -> str:
    """终端文本渲染(与存储的 dict 分离, 展示层可自由替换)。"""
    lines = []
    lines.append('=' * 96)
    lines.append('  %s' % title)
    lines.append('=' * 96)
    header = ('  %-20s %8s %8s %7s %8s %8s %8s %8s %6s %4s %12s' % (
        '因子', '|IC5|', 'IC5', 'IR', 'IC1', 'IC3', 'IC10', 'IC20',
        'Obs', '评级', '衰减'))
    lines.append(header)
    lines.append('  ' + '-' * 94)

    for r in results[:top_n]:
        decay = r.get('ic_decay', {})
        g = r.get('grade', '?')
        d_status = r.get('decay', {}).get('status', '')
        d_mark = {'DECAYING': '!衰减', 'STABLE': '稳定',
                  'INSUFFICIENT': '不足'}.get(d_status, '?')
        lines.append('  %-20s %8.4f %8.4f %7.2f %8.4f %8.4f %8.4f %8.4f %6d %4s %12s' % (
            r.get('factor', '?')[:20],
            abs(r.get('main_ic', 0.0)),
            r.get('main_ic', 0.0),
            r.get('ir', 0.0),
            decay.get(1, {}).get('ic', 0.0),
            decay.get(3, {}).get('ic', 0.0),
            decay.get(10, {}).get('ic', 0.0),
            decay.get(20, {}).get('ic', 0.0),
            r.get('n_obs', 0),
            g,
            d_mark,
        ))

    valid = [r for r in results if abs(r.get('main_ic', 0.0)) >= 0.03]
    a_grade = [r for r in results if r.get('grade') == 'A']
    b_grade = [r for r in results if r.get('grade') == 'B']
    leaks = [r for r in results if r.get('lookahead', {}).get('status') == 'LIKELY_LEAK']
    lines.append('  ' + '-' * 94)
    lines.append('  汇总: 因子 %d | |IC|>=0.03: %d | A级: %d | B级: %d | 疑似未来函数: %d' % (
        len(results), len(valid), len(a_grade), len(b_grade), len(leaks)))
    lines.append('=' * 96)
    return '\n'.join(lines)
