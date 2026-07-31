"""框架自测: 预处理/未来函数/IC 指标/衰减诊断/面板评估。

运行:
  python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from factor_validator import (
    cross_sectional_zscore,
    decay_assessment,
    detect_lookahead,
    detect_lookahead_all,
    evaluate_panel_factor,
    grade_factor,
    ic_decay,
    ir,
    mad_winsorize,
    neutralize,
    panel_ic_series,
    panel_segment_ic,
    quantile_returns,
    rank_ic,
    segment_ic,
    shift_factor,
)
from examples.synthetic import FWD_PERIODS, generate_panel


def _panel(**kw):
    return generate_panel(n_dates=100, n_assets=20, seed=7, **kw)


class PreprocessTest(unittest.TestCase):

    def test_mad_winsorize_clips_outliers(self):
        s = pd.Series(np.r_[np.random.randn(500), 100.0, -80.0])
        out = mad_winsorize(s, n=5.0)
        self.assertLess(out.max(), 10.0)
        self.assertGreater(out.min(), -10.0)
        # 主体分布几乎不变
        self.assertAlmostEqual(s[s.abs() < 3].median(), out[s.abs() < 3].median(), places=6)

    def test_cross_sectional_zscore(self):
        panel = _panel()
        z = cross_sectional_zscore(panel, 'factor_bad', 'date')
        means = panel.groupby('date')['factor_bad'].transform('mean')  # noqa
        per_date = pd.DataFrame({'z': z, 'date': panel['date']}).groupby('date')['z']
        self.assertTrue((per_date.mean().abs() < 1e-8).all(), '每日期内均值应≈0')
        self.assertTrue(((per_date.std() - 1.0).abs() < 1e-6).all(), '每日期内标准差应≈1')

    def test_neutralize_removes_style(self):
        n = 300
        x = pd.Series(np.random.randn(n))
        factor = 2.0 * x + np.random.randn(n) * 0.01  # 因子 = 风格 + 少量噪声
        df = pd.DataFrame({'x': x, 'f': factor})
        resid = neutralize(df, 'f', ['x'])
        # 残差应与风格列正交
        self.assertTrue(resid.dropna().corr(x) < 0.05)
        # 原始因子与风格高度相关, 中性化后相关性被剔除
        self.assertGreater(abs(df['f'].corr(x)), 0.9)

    def test_shift_factor(self):
        panel = _panel()
        shifted = shift_factor(panel, 'factor_good', periods=1)
        self.assertTrue(shifted.iloc[1] == panel['factor_good'].iloc[0])
        self.assertTrue(pd.isna(shifted.iloc[0]))


class RankIcTest(unittest.TestCase):

    def test_perfect_positive(self):
        x = pd.Series(np.arange(100, dtype=float))
        ic, p = rank_ic(x, x)
        self.assertAlmostEqual(ic, 1.0, places=9)

    def test_perfect_negative(self):
        x = pd.Series(np.arange(100, dtype=float))
        ic, _ = rank_ic(x, -x)
        self.assertAlmostEqual(ic, -1.0, places=9)

    def test_min_obs_guard(self):
        x = pd.Series([1.0, 2.0])
        ic, p = rank_ic(x, x, min_obs=10)
        self.assertEqual(ic, 0.0)

    def test_ic_decay_monotonic(self):
        panel = generate_panel()  # 15000 行
        good = panel['factor_good']
        decay = ic_decay(good, {p: panel['fwd_ret_%d' % p] for p in FWD_PERIODS})
        # 合规因子: IC 随周期衰减(1期 > 5期 > 20期)
        self.assertGreater(decay[1]['ic'], decay[20]['ic'])

    def test_quantile_monotonic(self):
        panel = generate_panel()
        q = quantile_returns(panel['factor_good'], panel['fwd_ret_5'])
        self.assertGreater(q['Q5']['mean_ret'], q['Q1']['mean_ret'])
        self.assertGreater(q['long_short_spread'], 0)


class LookaheadTest(unittest.TestCase):

    def test_cheat_factor_caught(self):
        panel = generate_panel()
        r = detect_lookahead(panel, 'factor_cheat', 'fwd_ret_1')
        self.assertEqual(r['status'], 'LIKELY_LEAK')
        self.assertGreater(r['ic_1'], 0.9)

    def test_good_and_bad_pass(self):
        panel = generate_panel()
        for f in ('factor_good', 'factor_bad'):
            self.assertEqual(detect_lookahead(panel, f, 'fwd_ret_1')['status'], 'OK')

    def test_lookahead_all(self):
        panel = generate_panel()
        r = detect_lookahead_all(panel, ['factor_cheat', 'factor_good'], 'fwd_ret_1')
        self.assertEqual(r['factor_cheat']['status'], 'LIKELY_LEAK')
        self.assertEqual(r['factor_good']['status'], 'OK')


class DecayTest(unittest.TestCase):

    def test_declining_segments_decaying(self):
        segs = [{'segment': i + 1, 'ic': ic, 'n': 100}
                for i, ic in enumerate([0.10, 0.08, 0.03, -0.02])]
        r = decay_assessment(segs)
        self.assertEqual(r['status'], 'DECAYING')

    def test_flat_segments_stable(self):
        segs = [{'segment': i + 1, 'ic': 0.07 + (0.005 if i % 2 else -0.005), 'n': 100}
                for i in range(4)]
        r = decay_assessment(segs)
        self.assertEqual(r['status'], 'STABLE')

    def test_insufficient(self):
        self.assertEqual(decay_assessment([])['status'], 'INSUFFICIENT')

    def test_panel_segment_and_assessment(self):
        panel = generate_panel()
        segs = panel_segment_ic(panel, 'factor_decay', 'fwd_ret_5', 'date', segments=4)
        self.assertEqual(len(segs), 4)
        self.assertEqual(decay_assessment(segs)['status'], 'DECAYING')
        segs_good = panel_segment_ic(panel, 'factor_good', 'fwd_ret_5', 'date', segments=4)
        self.assertEqual(decay_assessment(segs_good)['status'], 'STABLE')


class PanelEvaluationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.panel = generate_panel()
        cls.fwd = {p: 'fwd_ret_%d' % p for p in FWD_PERIODS}

    def test_good_factor_grades_a(self):
        r = evaluate_panel_factor(self.panel, 'factor_good', self.fwd)
        self.assertEqual(r['grade'], 'A')
        self.assertGreater(r['main_ic'], 0.05)
        self.assertGreater(r['ir'], 0.5)

    def test_bad_factor_grades_d(self):
        r = evaluate_panel_factor(self.panel, 'factor_bad', self.fwd)
        self.assertEqual(r['grade'], 'D')

    def test_cheat_factor_flagged_by_lookahead(self):
        r = detect_lookahead(self.panel, 'factor_cheat', 'fwd_ret_1')
        self.assertEqual(r['status'], 'LIKELY_LEAK')

    def test_decay_factor_detected(self):
        r = evaluate_panel_factor(self.panel, 'factor_decay', self.fwd)
        self.assertEqual(r['decay']['status'], 'DECAYING')

    def test_panel_ic_series_length(self):
        s = panel_ic_series(self.panel, 'factor_good', 'fwd_ret_5', 'date')
        self.assertEqual(len(s), len(self.panel['date'].unique()))
        self.assertGreater(ir(s), 0.3)


class GradeTest(unittest.TestCase):

    def test_boundaries(self):
        self.assertEqual(grade_factor(0.05, 0.5), 'A')
        self.assertEqual(grade_factor(0.0499, 0.5), 'B')
        self.assertEqual(grade_factor(0.03, 0.29), 'C')
        self.assertEqual(grade_factor(0.019, 0.0), 'D')


if __name__ == '__main__':
    unittest.main()
