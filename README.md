# factor-validator — 这个因子, 信还是不信?

> 因子验证工具链: 预处理、未来函数检测、IC/IR、衰减诊断、A/B/C/D 评级。
> 工具**不挖因子**, 只验证因子 —— "因子有没有用"是独立于挖掘的工程问题。

## 为什么需要它

挖因子最大的风险不是"挖不出来", 而是**挖出来的不能用**:

- 因子计算时偷看了未来(未来函数/数据泄露) → 回测漂亮, 实盘打穿;
- 上个月 IC 还有 0.08, 这个月开始为负(因子衰减) → 你还在按旧信号交易;
- 因子收益其实是市值/行业风格的收益(忘记中性化) → 换市场环境就失效;
- 样本内挖出来的, 样本外一测就塌(过拟合) → 你的"alpha"只是噪声的排列组合。

本工具把验证流程标准化, 让每个因子在进回测之前先过一遍体检。

## 功能一览

| 模块 | 功能 | 解决的问题 |
|---|---|---|
| `preprocess` | MAD 去极值 / 截面 z-score / 中性化 / shift(1) | 数据不干净、风格干扰、未来函数 |
| `detect_lookahead` | 因子对下期收益的 IC(1) 是否高到不合理 | **数据泄露的指纹**: 偷看未来的因子 IC(1)≈1.0 |
| `ic` | Rank IC / IC 衰减曲线 / IR / 分位数收益 / 滚动分析 / 因子相关性 | 预测力到底有多少、稳不稳 |
| `decay` | 分段 IC + 斜率 → STABLE / DECAYING | 因子是不是正在衰减 |
| `report` | A/B/C/D 评级, 单因子/全因子/面板评估 | 结论可入 CI |

## 快速开始

### 面板数据(多资产 × 多日期)—— 最典型场景

```python
import pandas as pd
from factor_validator import evaluate_panel_factor, detect_lookahead, render_report

panel = pd.read_parquet('panel.parquet')   # date, asset, factor_xxx, fwd_ret_1/5/10
fwd = {p: f'fwd_ret_{p}' for p in (1, 5, 10)}

for f in ['momentum_20d', 'vol_factor', 'your_factor']:
    r = evaluate_panel_factor(panel, f, fwd)          # 逐日截面 IC → IR → 评级
    leak = detect_lookahead(panel, f, 'fwd_ret_1')    # 未来函数检查
    if leak['status'] == 'LIKELY_LEAK':
        print(f'{f}: 疑似数据泄露! IC(1)={leak["ic_1"]:.3f} —— 弃用')
    elif r['grade'] in ('A', 'B') and r['decay']['status'] == 'STABLE':
        print(f'{f}: 可用 ({r["grade"]}, IC(5)={r["main_ic"]:.3f}, IR={r["ir"]:.2f})')
```

### 预处理流水线

```python
from factor_validator import mad_winsorize, cross_sectional_zscore, neutralize

panel['f_mad']     = mad_winsorize(panel['raw_factor'], n=5.0)          # 去极值
panel['f_z']       = cross_sectional_zscore(panel, 'f_mad', 'date')      # 截面标准化
panel['f_neutral'] = neutralize(panel, 'f_z', ['market_cap', 'sector_dummy'])  # 中性化
```

### 完整演示

```bash
python3 examples/run_demo.py
```

四个**真相已知**的合成因子(好/随机/作弊/衰减), 框架一次全部正确识别:

```
factor_good     IC(1)=0.178  IC(5)=0.081  IR=0.61  稳定  A   ← 真实有预测力
factor_bad      IC(1)=-0.025 IC(5)=-0.001 IR=-0.01 稳定  D   ← 纯噪声
factor_cheat    IC(1)=0.994  →  [!!] 疑似泄露(偷看了未来收益)
factor_decay    IC(5)=0.034  衰减  B  ← 表面有信号, 正在衰减, 别用
```

### 运行自测

```bash
python3 -m unittest discover -s tests -v   # 22 个用例
```

## 设计决策

| 决策 | 理由 |
|---|---|
| 未来函数检测看 **IC(1) 是否高到不合理**(>0.7) | 合规因子 IC(1) 有限(0.02~0.3); 偷看未来的因子 IC(1)≈1.0 —— 比"因子 vs 当期收益相关性"可靠(动量因子会误伤) |
| 时间序列指标一律**按日期切截面**(面板感知) | 扁平序列滑窗会把不同日期的截面混在一起, IR/衰减没有意义 |
| MAD 去极值而非 3σ | 离群值会污染均值/标准差, MAD 基于中位数, 稳健 |
| 截面 z-score 而非整体 z-score | 整体标准化会把"某天全体因子偏高"的风格波动留在因子里 |
| 中性化用 OLS 残差 | 残差与风格正交, 因子收益里不再混着风格收益 |
| 评级与衰减分离 | A 级因子也可能正在衰减 —— 两个维度独立看, 不能互相掩盖 |

## 目录结构

```
factor-validator/
├── factor_validator/          # 核心库
│   ├── preprocess.py          #   MAD/z-score/中性化/shift(1)/未来函数检测
│   ├── ic.py                  #   Rank IC/IC衰减/IR/分位数/滚动/面板IC序列
│   ├── decay.py               #   分段IC + STABLE/DECAYING判定
│   └── report.py              #   A/B/C/D评级 + 单因子/全因子/面板评估
├── examples/                  # 合成演示(与任何真实行情无关)
│   ├── synthetic.py           #   (date×asset)面板生成器, 四个"真相已知"的因子
│   └── run_demo.py            #   一键演示
└── tests/                     # 22 个自测
```

## 背景

本工具从一套期货多因子研究实践中提炼, 贯穿的三条铁律: **收益计算优先后复权、
运算务必 shift(1) 杜绝未来函数、预处理走 MAD→截面 z-score→中性化 标准流程**。
示例与演示全部使用合成数据, 不包含任何真实因子、参数或凭证。

## License

MIT
