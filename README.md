# factor-validator — Can This Factor Be Trusted?

> A factor validation toolchain: preprocessing, look-ahead detection, IC/IR, decay
> diagnostics, and A/B/C/D grading.
> The tool **does not mine factors — it validates them**. Whether a factor works is an
> engineering problem independent of how it was discovered.
>
> 因子验证工具链:预处理、未来函数检测、IC/IR、衰减诊断、A/B/C/D 评级。
> 工具**不挖因子**,只验证因子 —— "因子有没有用"是独立于挖掘的工程问题。

---

## Why? / 为什么需要它

The biggest risk of mining factors is not "finding nothing" — it is **finding something
you cannot use**:

- The factor peeked at the future during computation (look-ahead / data leakage) →
  beautiful backtest, blown-up live account;
- Last month IC was 0.08, this month it turns negative (factor decay) — and you are still
  trading the stale signal;
- The factor's returns were actually market-cap or sector style returns (forgot to
  neutralize) → fails the moment the environment shifts;
- It was discovered in-sample and collapses out-of-sample (overfitting) → your "alpha"
  was just noise combinatorics.

挖因子最大的风险不是"挖不出来",而是**挖出来的不能用**:

- 因子计算时偷看了未来(未来函数/数据泄露) → 回测漂亮,实盘打穿;
- 上个月 IC 还有 0.08,这个月开始为负(因子衰减) → 你还在按旧信号交易;
- 因子收益其实是市值/行业风格的收益(忘记中性化) → 换市场环境就失效;
- 样本内挖出来的,样本外一测就塌(过拟合) → 你的"alpha"只是噪声的排列组合。

This tool standardizes the validation workflow, so every factor passes a health check
*before* it ever enters a backtest.

本工具把验证流程标准化,让每个因子在进回测之前先过一遍体检。

## Features / 功能一览

| Module 模块 | What it does 功能 | Problem it solves 解决的问题 |
|---|---|---|
| `preprocess` | MAD winsorization / cross-sectional z-score / neutralization / shift(1) | Dirty data, style interference, look-ahead 数据不干净、风格干扰、未来函数 |
| `detect_lookahead` | Is IC(1) of the factor vs *next-period* returns implausibly high? | **The fingerprint of data leakage**: a look-ahead factor has IC(1) ≈ 1.0 **数据泄露的指纹** |
| `ic` | Rank IC / IC decay curve / IR / quantile returns / rolling analysis / factor correlation | How much predictive power, and how stable 预测力到底有多少、稳不稳 |
| `decay` | Segmented IC + slope → STABLE / DECAYING | Is the factor decaying right now 因子是不是正在衰减 |
| `report` | A/B/C/D grading; single-factor / full-factor / panel evaluation | CI-ready verdicts 结论可入 CI |

## Quick Start / 快速开始

### Panel data (assets × dates) — the typical case / 面板数据 —— 最典型场景

```python
import pandas as pd
from factor_validator import evaluate_panel_factor, detect_lookahead, render_report

panel = pd.read_parquet('panel.parquet')   # date, asset, factor_xxx, fwd_ret_1/5/10
fwd = {p: f'fwd_ret_{p}' for p in (1, 5, 10)}

for f in ['momentum_20d', 'vol_factor', 'your_factor']:
    r = evaluate_panel_factor(panel, f, fwd)          # daily cross-sectional IC → IR → grade
                                                      # 逐日截面 IC → IR → 评级
    leak = detect_lookahead(panel, f, 'fwd_ret_1')    # look-ahead check 未来函数检查
    if leak['status'] == 'LIKELY_LEAK':
        print(f'{f}: suspected data leakage! IC(1)={leak["ic_1"]:.3f} — discard')
        # 疑似数据泄露! —— 弃用
    elif r['grade'] in ('A', 'B') and r['decay']['status'] == 'STABLE':
        print(f'{f}: usable 可用 ({r["grade"]}, IC(5)={r["main_ic"]:.3f}, IR={r["ir"]:.2f})')
```

### Preprocessing pipeline / 预处理流水线

```python
from factor_validator import mad_winsorize, cross_sectional_zscore, neutralize

panel['f_mad']     = mad_winsorize(panel['raw_factor'], n=5.0)          # de-extreme 去极值
panel['f_z']       = cross_sectional_zscore(panel, 'f_mad', 'date')     # cross-sectional z-score 截面标准化
panel['f_neutral'] = neutralize(panel, 'f_z', ['market_cap', 'sector_dummy'])  # neutralize 中性化
```

### Full demo / 完整演示

```bash
python3 examples/run_demo.py
```

Four synthetic factors with **known ground truth** (good / random / cheat / decaying),
all correctly identified by the framework in one pass:

四个**真相已知**的合成因子(好/随机/作弊/衰减),框架一次全部正确识别:

```
factor_good     IC(1)=0.178  IC(5)=0.081  IR=0.61  STABLE  A   ← real predictive power 真实有预测力
factor_bad      IC(1)=-0.025 IC(5)=-0.001 IR=-0.01 STABLE  D   ← pure noise 纯噪声
factor_cheat    IC(1)=0.994  →  [!!] suspected leak (peeked at future returns)
                                                              疑似泄露(偷看了未来收益)
factor_decay    IC(5)=0.034  DECAYING  B  ← signal on the surface, decaying — don't use
                                            表面有信号, 正在衰减, 别用
```

### Run the test suite / 运行自测

```bash
python3 -m unittest discover -s tests -v   # 22 test cases 22 个用例
```

## Design Decisions / 设计决策

| Decision 决策 | Rationale 理由 |
|---|---|
| Look-ahead detection via implausibly high **IC(1)** (> 0.7) 未来函数检测看 IC(1) 是否高到不合理 | A compliant factor has limited IC(1) (0.02–0.3); a look-ahead factor has IC(1) ≈ 1.0 — more reliable than correlating the factor with *current* returns (momentum factors would be falsely flagged) 合规因子 IC(1) 有限;偷看未来的因子 IC(1)≈1.0 —— 比"因子 vs 当期收益相关性"可靠(动量因子会误伤) |
| Time-series metrics always computed **per-date cross-section** (panel-aware) 时间序列指标一律按日期切截面 | Flat-window rolling mixes cross-sections from different dates; IR / decay become meaningless 扁平序列滑窗会把不同日期的截面混在一起 |
| MAD winsorization instead of 3σ MAD 去极值而非 3σ | Outliers poison mean/std; MAD is median-based and robust 离群值会污染均值/标准差,MAD 基于中位数,稳健 |
| Cross-sectional z-score instead of global z-score 截面 z-score 而非整体 z-score | Global standardization keeps style waves ("every factor high today") inside the factor 整体标准化会把"某天全体因子偏高"的风格波动留在因子里 |
| Neutralization via OLS residuals 中性化用 OLS 残差 | Residuals are orthogonal to style; factor returns no longer contain style returns 残差与风格正交,因子收益里不再混着风格收益 |
| Grade and decay are separate dimensions 评级与衰减分离 | An A-grade factor may be decaying right now — never let one dimension mask the other 两个维度独立看,不能互相掩盖 |

## Project Layout / 目录结构

```
factor-validator/
├── factor_validator/          # core library 核心库
│   ├── preprocess.py          #   MAD / z-score / neutralization / shift(1) / look-ahead detection
│   │                          #   MAD/z-score/中性化/shift(1)/未来函数检测
│   ├── ic.py                  #   Rank IC / IC decay / IR / quantiles / rolling / panel IC series
│   │                          #   Rank IC/IC衰减/IR/分位数/滚动/面板IC序列
│   ├── decay.py               #   segmented IC + STABLE/DECAYING verdict 分段IC + 衰减判定
│   └── report.py              #   A/B/C/D grading + single/full-factor/panel evaluation
│                              #   A/B/C/D评级 + 单因子/全因子/面板评估
├── examples/                  # synthetic demos, unrelated to any real market data
│                              # 合成演示(与任何真实行情无关)
│   ├── synthetic.py           #   (date × asset) panel generator with four "ground-truth" factors
│   │                          #   面板生成器, 四个"真相已知"的因子
│   └── run_demo.py            #   one-shot demo 一键演示
└── tests/                     # 22 self-tests 22 个自测
```

## Background / 背景

This tool was distilled from multi-factor research on a futures live-trading system,
built on three non-negotiable rules: **compute returns on adjusted prices first,
always shift(1) to rule out look-ahead, and preprocess via MAD → cross-sectional z-score
→ neutralization.** All examples and demos use synthetic data — no real factors,
parameters, or credentials.

本工具从一套期货多因子研究实践中提炼,贯穿的三条铁律:**收益计算优先后复权、
运算务必 shift(1) 杜绝未来函数、预处理走 MAD→截面 z-score→中性化 标准流程**。
示例与演示全部使用合成数据,不包含任何真实因子、参数或凭证。

## Author & Freelance / 作者与接单

> Open to freelance collaboration on quantitative data engineering, backtest validation,
> and Python/Java development. Every deliverable ships with tests and is verifiable.
>
> 承接量化数据工程、回测验证与 Python/Java 开发私活。每个交付物都带测试、可验证。

**Services 接单范围:**

- Backtest framework construction & audit — look-ahead, overfitting, data-leakage review
  回测框架搭建与审计(未来函数/过拟合/数据泄露排查)
- Factor validation & quant data pipeline — cleaning, price adjustment, neutralization
  因子验证与量化数据管线(清洗、复权、中性化)
- Python / Java development — data tools, automation, APIs
  Python/Java 开发(数据处理工具、自动化、接口)
- AI-agent-assisted programming
  AI Agent 编程

**Acceptance criteria 验收标准:**

- Data accuracy first: adjusted-price returns, `shift(1)`, zero look-ahead or leakage
  数据准确优先:后复权收益、shift(1)、零未来函数与数据泄露
- Logic verifiable: every deliverable ships with tests and a reproducible demo
  逻辑可验证:交付物带测试与可复现示例
- Clear handover: documented code + runnable examples + report
  交付明确:文档化代码 + 可运行示例 + 报告

**Contact 联系方式:**

- GitHub: [FlyingEggs](https://github.com/FlyingEggs)
- Email: `357378143@qq.com` (freelance 接单)

## License

MIT
