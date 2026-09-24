# RateMA_T10Timing

基于 **20 日 / 120 日简单移动平均线** 的利率择时策略回测。

- **标的**：中债-10年期国债净价(总值)指数（`CBA04502.CS`）
- **信号源**：5 个底层资金利率指标（`DR001` / `R001` / `DR007` / `R007` / `M0017139`(SHIBOR:1周)）
- **样本**：2010-12-31 ~ 2026-09-16，共 3927 个交易日
- **运行时**：Python 3.11 + [uv](https://docs.astral.sh/uv/) 管理

---

## 1. 快速开始

```bash
# 1) 安装依赖（自动创建 .venv，无需手动装 Python）
uv sync

# 2) 一键跑通：Excel -> CSV -> 回测 -> 阈值扫描 -> 图表
uv run python run.py

# 3) 看结果（产出落在 <outdir>/<run_name>/，默认 output/default/）
#    output/default/report.md    完整中文报告
#    output/default/charts/      图表
```

就这两条命令。`uv` 会在项目内建好隔离环境，依赖版本由 `uv.lock` 锁定，换台机器结果一致。

### 改参数只改一个文件

**所有可调参数都集中在 `run.py` 顶部的 `CONFIG` 字典里**，改完直接运行即可，
不需要记命令行参数：

```python
CONFIG = {
    "run_name": "5bp_多空",  # ← 输出文件夹名，产出写到 output/5bp_多空/
    "tol_mode": "bp",  # 重合阈值单位：基点（推荐，不易搞错量级）
    "tol": 5.0,  # 5bp 以内视为两线重合
    "direction": "long_short",  # 多空双向
    "steps": {
        "convert": True,
        "backtest": True,
        "composite": True,
        "journal": True,
        "sweep": True,
    },  # 关掉不需要的步骤会快很多
}
```

也可以临时覆盖输出文件夹名，不用改文件：

```bash
uv run python run.py 我的方案A     # 产出写到 output/我的方案A/
```

Windows 下直接**双击 `run.bat`** 即可（它调用 `run.py`，参数以 `run.bat 方案名` 形式透传）。
首次运行不确定参数怎么填时，建议先只开 `backtest`、关掉 `sweep` 和 `charts` 试跑。

### 命令行用法（可选）

上面的 `run.py` 底层调用的是同一套 CLI 子命令，参数完全一致；
习惯命令行的可以直接用：

| 我想…… | 命令 |
| --- | --- |
| 一键跑通（等价于 `run.py` 全开） | `uv run ratema all` |
| 跑默认回测（阈值 0） | `uv run ratema backtest` |
| 用 5bp 作为重合阈值 | `uv run ratema backtest --tol-mode bp --tol 5` |
| 换均线（如 10/60 日） | `uv run ratema backtest --short 10 --long 60` |
| 只看某段时间 | `uv run ratema backtest --start 2020-01-01 --end 2024-12-31` |
| 改交易成本（如单边 5bp） | `uv run ratema backtest --cost-bps 5` |
| 改初始资金（仅影响净值绝对水平） | `uv run ratema backtest --initial-capital 1000000` |
| 成本按"往返合计"理解 | `uv run ratema backtest --cost-bps 2 --cost-mode round_trip` |
| **只做空** | `uv run ratema backtest --direction short_only` |
| **多空双向** | `uv run ratema backtest --direction long_short` |
| 只跑某一个利率指标 | `uv run ratema backtest --rate-cols DR007` |
| **每利率单独测 + 五利率等权综合** | `uv run ratema composite --tol-mode bp --tol 5` |
| 换综合方式（全票 / 任一） | `uv run ratema composite --mode unanimous`（或 `any`） |
| **输出每日交易日志** | `uv run ratema journal --tol-mode bp --tol 5` |
| 日志只看动作日 | `uv run ratema journal --events-only` |
| 日志追加到台账 | `uv run ratema journal --append output/ledger.csv` |
| 看最新一天的信号 | `uv run ratema signal --tol-mode bp --tol 5` |
| 扫描阈值找最优 | `uv run ratema sweep --tol-mode bp` |
| 扫描综合信号 | `uv run ratema sweep --target composite --tol-mode bp` |
| 自定义阈值网格 | `uv run ratema sweep --tol-mode bp --tol-grid 0,1,2,3,5,8,10` |
| 不生成图表（更快） | `uv run ratema backtest --no-charts` |
| 多组参数各存一份结果 | `uv run ratema backtest --tol-mode bp --tol 5 --tag bp5` |
| 换一份 Excel 数据 | `uv run ratema convert 新数据.xlsx --outdir data/csv2` 再 `uv run ratema backtest data/csv2/panel.csv` |
| 跑测试 | `uv run pytest` |
| 交叉验证引擎 | `uv run python scripts/cross_check.py` |

**关于 `--tag`**：加 `--tag bp5` 后结果写到 `output/bp5/`，不会覆盖默认的 `output/`，
方便把多组参数并排比较。

**重复运行是安全的**：`convert` 会覆盖 `data/csv/`，`backtest` 会覆盖 `output/<tag>/`，
都是幂等的，不会累积垃圾。

### 方向对比图（独立入口）

`compare_directions.py` 把**同一套信号**在三种交易方向下的净值，和买入持有基准
画到**同一张图**上。默认出 2 + N 张（N = 利率指标个数）：

- `compare_directions.png` —— 五利率**综合信号**（五票合成一个仓位）
- `compare_portfolio.png` —— 五利率**等权组合**（五份资金各跟一个信号）
- `compare_directions_DR001.png`、`_R001.png`、`_DR007.png`、`_R007.png`、
  `_M0017139.png` —— **每个利率单独**一张，用的是该利率自己的信号
- 每个主图配一张 `*_drawdowns.png` 回撤附图（可用 `--no-drawdown` 关掉）
- `compare_annual_returns.png` / `compare_annual_returns_portfolio.png` —— **年度收益热力图**
  （仿 `charts/05_annual_returns.png` 的样式：绿=赚 红=亏，行=方向，列=年份）
- `compare_annual_indicator_<方向>.png` —— 同款热力图，**行=等权综合/各利率/基准**（每个方向一张）
- `compare_annual_dd_<方向>.png` —— **年度最大回撤热力图**（三张），
  排版与上面的收益图完全一致（同 8 行、同 16 列），可直接并排对照。
  配色改用 `[-最深, 0]`：**无回撤 = 绿、最深 = 红** —— 回撤全是非正数，
  沿用收益图那种以 0 为中心的发散色带会白白浪费绿色半幅、把色差压掉一半。
  三张图**共用同一个颜色下限**，深浅可比
- `compare_annual_return_dd.csv` / `.md` —— 上面两张热力图的数据
  （年度收益 + 年内最大回撤，三方向 × 8 口径 × 16 年）。CSV 是长表，便于 Excel 透视
- `compare_monthly_<年份>.png` —— **逐月收益热力图**，与上面同样的 8 行，列 = 1~12 月
  （15 年 × 12 月 ≈ 184 列塞不进一张可读的图，故按年拆成多个文件；
  各年文件**共用同一颜色范围**，深浅可比）
- `compare_monthly.csv` —— 逐月收益全量矩阵（8 行 × 184 列），便于自己画图或查数

单文件版共三种，**横轴一律是月份、纵轴一律是策略种类**（8 行）：

| 文件 | 排布 |
| --- | --- |
| `compare_monthly_by_strategy.png` | 一张图：8 策略 × 全部 184 个月，横向拉通（年份用竖分隔线 + 年标签）。**每格是严格的正方形**，所以整图约 23:1 的长条 |
| `compare_monthly_years_panel.png` | 一年一个面板，16 个面板**竖排**（每个面板 8 策略 × 12 月） |
| `compare_monthly_years_row.png` | 一年一个面板，16 个面板**横排**（所有年份在同一行，纵轴只标一次策略名） |

```bash
uv run python compare_directions.py                    # 两种曲线 + 每个利率各一张
uv run python compare_directions.py --curve portfolio   # 只出等权组合那一套
uv run python compare_directions.py --curve signal      # 只出综合信号那一套
uv run python compare_directions.py --only-indicators   # 只要每个利率那几张
uv run python compare_directions.py --only-composite    # 只要综合那两张
uv run python compare_directions.py --rates DR007,R007  # 只出指定利率
uv run python compare_directions.py --log               # 纵轴取对数
```

年度收益热力图可在 `CONFIG` 里调：

```python
"annual_heatmap": True,
"annual_heatmap_rows": "both",       # "direction" / "indicator" / "both"
"annual_return_dd_data": True,       # 导出 年度收益+年内最大回撤 的 CSV/MD
"annual_dd_heatmap": True,           # 三张 年度最大回撤 热力图

"monthly_heatmap": True,
"monthly_direction": "long_short",   # 逐月表看哪个方向（三个方向都出会有 48 个文件）
"monthly_years": None,               # None=全部；也可写 [2024, 2025] 或 "2020-2023"
"monthly_layout": "all",             # "strip" 一张图 / "panel" 年面板竖排 / "row" 年面板横排 / "all"
"monthly_cell_inch": 0.15,           # 一张图那版的单格边长（英寸）；格子是正方形，调大即整图等比变长
```

热力图的收益口径与 `charts/05_annual_returns.png` 一致：**期末净值环比**
（首年 / 首月都从评估起点起算），因此跨年、跨月的跳空都计入新一期，首期不是整期也不失真。
测试里与本项目的 `ratema.charts.annual_returns` 做了逐点交叉验证，
并断言**年内逐月复利必须等于该年的年度收益**（两者同源，偏差 < 1e-12）。

`--indicator` 那张热力图（以及 `compare_by_rate.md`）里，**两个综合口径都在场**，
标签分别是：

| 行标签 | 含义 | 多空双向年化 |
| --- | --- | --- |
| 打分综合 | 五票合成**一个**仓位，每天要么满仓要么空仓 | 3.55% |
| 等权组合 | **五份资金**各跟一个信号后等权平均，仓位比例连续 | 2.75% |
| DR001 … M0017139 | 该利率**单独**的策略（N=1 时两种综合口径等价，故只列一次） | 2.16%~3.09% |
| 基准 买入持有 | 期初买入并持有 | 1.09% |

> 早期版本这一行标签写作「等权综合」，既能读成「等权组合」也能读成「等权打分」，
> 容易让人以为打分综合没有画。现已改为「打分综合」，并把等权组合单独补成一行。

**「综合信号」与「等权组合」的区别**（两张图的曲线不一样，别混用）：

| | 综合信号 | 等权组合 |
| --- | --- | --- |
| 含义 | 五票合成**一个**决策 | **五份资金**各跟一个信号 |
| 仓位 | 每天要么满仓要么空仓 | 连续（持有票数 / N） |
| 曲线 | 更陡、回撤更大 | 更平滑、回撤更小 |

每张图的四条曲线：**多空双向**（主线）、**仅做多**、**仅做空**、**基准 买入持有**。
另外产出 `compare_by_rate.csv` / `.md` —— 按「指标 × 方向」排列的对照表，
可以直接看出哪个利率最适合哪种方向。

这个文件刻意不依赖 `run.py`，参数在它自己的 `CONFIG` 里改，可单独运行。
**单利率图与综合图走的是同一条代码路径**：单利率只是把「只含该指标一个元素」的
列表喂给同一套综合逻辑。N=1 时 `score = ±1`，与「该指标自己的 signal_eff」逐日等价，
所以不需要另写一套逻辑，也自动继承了同一套对齐校验（测试里逐指标 × 逐方向
与 `run_single` 比对，误差为 0）。N=1 时等权组合与综合信号是同一条曲线，
因此单利率只出一套图，不重复出 portfolio。

三条策略曲线共用相同的信号与相同的评估区间（代码里会显式校验对齐并直接报错），
所以图上的差异**只来自交易方向**，能直接看出「做空那一段贡献了多少」。

> **注意**：等权组合是对各指标的净值曲线求平均，而净值曲线本身依赖交易方向，
> 所以每个方向都必须单独跑一遍单指标回测。若三个方向共用同一份回测结果，
> 等权组合会在三个方向下变成同一条曲线——图看起来正常，其实是错的。
> 代码里对此有显式校验，测试也专门盯住这一点。

---

## 2. 策略规则

### 2.1 策略指标

对**每一个**底层利率指标，分别计算 20 日和 120 日简单移动平均（SMA）：

| 条件 | 标记 | 经济含义 |
| --- | --- | --- |
| `MA20 > MA120` | **-1** | 近期资金成本持续高于中长期水平，流动性边际趋紧 → **债券空头** |
| `MA20 < MA120` | **+1** | 短端资金成本低于中长期水平，流动性相对宽松 → **债券多头** |
| `\|MA20 - MA120\| ≤ 重合阈值` | **0** | 两线暂时重合，**延续上一交易日的有效信号** |

输出两个信号列：

- `signal_raw` —— 原始三态信号（-1 / 0 / +1），0 表示当日落入重合区间；
- `signal_eff` —— 有效交易信号（-1 / +1），把 0 用**上一交易日的有效信号**前向填充（规则 3）。

### 2.2 重合度阈值（本项目的核心扩展）

原始规则要求两条均线**严格相等**才算重合。实测数据是浮点数，严格相等几乎不可能发生：
在 3808 个交易日上，`tol=0` 时**重合日数为 0**（见 `output/sweep/pivot_overlap_days_abs.csv`），
规则 3 实际上从未被触发。因此本项目把「重合」定义为一个**可配置的区间**：

```
|MA_short - MA_long| <= tolerance   ->  判定为重合（标记 0，延续前一日有效信号）
```

`tolerance` 由 `--tol-mode` + `--tol` 共同决定：

| `--tol-mode` | 含义 | 示例 | 说明 |
| --- | --- | --- | --- |
| `abs`（默认） | 绝对阈值，单位与利率一致（百分点） | `--tol 0.01` | 即 1bp |
| `bp` | 以基点为单位 | `--tol 1` | 即 1bp = 0.01 个百分点 |
| `rel` | 相对阈值 `\|spread\| / \|MA_long\|` | `--tol 0.01` | 即 1% |
| `std` | 波动自适应：`tol × 滚动标准差(spread)` | `--tol 0.5` | 半个标准差 |
| `q` | 分位数自适应：`tol` 取 0~1 | `--tol 0.05` | 滚动窗口内 5% 分位 |

> 注意 `abs` 与 `bp` 的单位差 100 倍：`--tol-mode abs --tol 0.01` = `--tol-mode bp --tol 1` = 1bp。

**怎么选阈值？** 两条参考：

1. **看分位数**：回测会输出 `output/spread_calibration.csv`，给出每个指标
   `|MA20 - MA120|` 的分位数（单位 bp）。取 `p05` 就表示历史上约 5% 的交易日会被判为重合。
2. **看绩效敏感性**：`ratema sweep` 会扫描一组阈值并输出绩效对比。

### 2.3 多头策略与交易细节

**交易方向**由 `--direction` 决定（默认 `long_only`）：

| 方向 | 信号 +1（看多债券） | 信号 −1（看空债券） |
| --- | --- | --- |
| `long_only`（默认） | 持有多头 | 空仓 |
| `short_only` | 空仓 | **持有空头** |
| `long_short` | 持有多头 | **持有空头**（始终有仓位） |

> 原始策略是「卖出（有持仓时）或继续空仓（无持仓时）」，对应 `long_only`。

**做空的会计口径**（与做多镜像，1 倍名义敞口）：

```
建空：名义 N = 现金/(1+c)，收到 N(1−c)，建仓后净值 = 现金/(1+c)   ← 与建多对称
持有：净值随标的价格反向变动（标的跌 10%，1 倍空头赚 10%）
平空：回购支付 |份额|·P·(1+c)
```

方向切换（多↔空）拆成「先平后建」两笔，两笔成本相加。
**空头亏损无上限**：标的翻倍时 1 倍空头净值归零，继续上涨则转负
（真实交易中会先被强平，本回测不含保证金/强平逻辑）。

- T 日标记 **+1** → T+1 日**买入**（T 日无持仓）或**继续持有**（T 日有持仓）
- T 日标记 **-1** → T+1 日**卖出**（T 日有持仓）或**继续空仓**（T 日无持仓）
- **交易价格**：收盘价
- **交易成本**：默认万分之二（2bp）、**双边收取**（买 2bp + 卖 2bp）
  - `--cost-mode per_side`（默认）：`--cost-bps` 为单边成本，双边合计 2 × cost
  - `--cost-mode round_trip`：`--cost-bps` 为往返合计，单边各收一半
- **空仓不计利息**（闲置资金收益为 0）
- **基准**：买入并持有 `CBA04502.CS`（首日按含成本价建仓）
- **评估窗口（起算口径）**：
  - 起点 = **首个有效信号日**，即 `signal_eff` 首次非空的交易日（跳过 MA120 的 119 个预热日）；
  - 终点 = 数据末日；
  - 策略与基准在同一天、同一初始资金上归一为 1.0，预热期不计入任何绩效指标；
  - 因 T+1 执行，起点当天只可能有信号、不可能有持仓，首笔建仓发生在之后的第一个 `+1` 信号次日。
  - 本样本下起点为 **2011-06-27**（5 个指标一致），首个有效信号为 `-1`，
    故首次建仓推迟到 **2011-08-24**（DR001/R001；DR007/R007 为 08-25，SHIBOR 为 08-22）。
  - 每次回测都会把上述口径写入 `output/report.md` 第 3 节与 `output/evaluation_window.csv`，可逐项审计。

成本按成交金额比例计提：买入份额 `= 现金 / (P × (1+c))`，卖出得款 `= 份额 × P × (1-c)`。
成本在成交当日立即反映到净值上。

---

## 3. 实测结果（默认参数，tol = 0）

评估区间 **2011-06-27 ~ 2026-09-16**（3808 个交易日），成本单边 2bp：

| 指标 | 策略累计 | 策略年化 | 年化波动 | 夏普 | 最大回撤 | 基准年化 | 基准回撤 | 年化超额 | 往返次数 | 胜率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DR001` | 24.54% | 1.46% | 1.69% | 0.87 | **-4.77%** | 1.09% | -9.75% | +0.37% | 48 | 62.5% |
| `R001` | 27.27% | 1.61% | 1.69% | 0.96 | **-4.77%** | 1.09% | -9.75% | +0.52% | 76 | 55.3% |
| `DR007` | **31.87%** | **1.85%** | 1.73% | **1.07** | **-4.92%** | 1.09% | -9.75% | +0.76% | 59 | 61.0% |
| `R007` | 28.51% | 1.67% | 1.78% | 0.94 | -4.92% | 1.09% | -9.75% | +0.58% | 81 | 50.6% |
| `M0017139` | 30.32% | 1.77% | 1.78% | 0.99 | -4.93% | 1.09% | -9.75% | +0.68% | 40 | 72.5% |

结论要点：

- 五条均线全部**跑赢买入持有基准**（年化超额 +0.37% ~ +0.76%）；
- **最大回撤显著更浅**（约 -4.8% vs 基准 -9.75%），是这套择时规则的主要价值来源；
- 交易成本年化拖累约 0.11% ~ 0.22%，`DR007` 表现最好、`R007` 最弱。

### 分年度拆解与胜率口径

除全区间外，报告第 4.1 节、`annual_breakdown.csv` 与 `journal_annual.csv`
都会**按自然年**列示策略与基准的对比。

**胜率有四个口径，含义完全不同，不要混用：**

| 口径 | 定义 | 本样本（5 指标等权，tol=0） |
| --- | --- | --- |
| 策略日胜率 | 策略当日收益 > 0 的天数占比 | **29.89%** |
| 基准日胜率 | 基准当日收益 > 0 的天数占比 | 52.22% |
| 相对胜率 | 策略当日收益**跑赢基准**的天数占比 | 38.07% |
| 交易胜率 | 已平仓往返交易中盈利的比例 | 60.38% |

> ⚠️ **策略日胜率（29.89%）远低于基准（52.22%），这不是 bug**：
> 空仓日收益恰为 0，不计入胜率分子。这类趋势型策略的典型画像是
> **「低胜率 + 高盈亏比」**——靠少数大跌日避开亏损取胜，而非靠多数交易日占优。
> 判断有效性应看**相对胜率、超额收益与回撤**，而不是日胜率。

逐年结果也很说明问题：**16 个年度里有 7 年跑输基准**，但总超额仍有 +10.72%——
因为赢的年份赢得多（2013 年 +3.38%、2017 年 +4.30%），输的年份输得少。
详见 `output/report.md` 第 4.1 节，以及图 `output/charts/09_annual_breakdown.png`
（年度收益对比 / 累计超额 / 逐年胜率 / 逐年回撤，四联图）。

> 分年度交易胜率往往只基于个位数笔交易，噪声很大；
> 图中用**点的大小编码交易笔数**，避免把小样本当成可靠信号。

### 阈值（tol）的影响

`ratema sweep --tol-mode bp` 扫描 **0 / 0.5 / 1 / 2 / 5 / 10 bp**
（等价写法：`--tol-mode abs --tol-grid 0,0.005,0.01,0.02,0.05,0.1`）：

| 阈值 | 重合日占比<br>DR001 / R007 | 交易信号切换<br>DR001 / R007 | 往返次数<br>DR001 / R007 |
| --- | --- | --- | --- |
| 0（严格相等） | 0.00% / 0.00% | 96 / 162 | 48 / 81 |
| 0.5 bp | 2.28% / 2.91% | 78 / 127 | 39 / 63 |
| 1 bp | 4.52% / 5.49% | 76 / 113 | 38 / 56 |
| 2 bp | 8.85% / 9.77% | 74 / 95 | 37 / 47 |
| 5 bp | 22.58% / 20.90% | 62 / 71 | 31 / 35 |
| 10 bp | 39.89% / 32.62% | 43 / 49 | 21 / 24 |

绩效随阈值的变化：

| 指标 | 夏普<br>tol=0 | 夏普<br>5bp | 夏普<br>10bp | 年化<br>tol=0 | 年化<br>5bp | 年化<br>10bp |
| --- | --- | --- | --- | --- | --- | --- |
| `DR001` | 0.87 | 0.98 | **1.26** | 1.46% | 1.64% | **2.12%** |
| `R001` | 0.96 | **1.23** | 1.19 | 1.61% | **2.10%** | 1.94% |
| `DR007` | 1.07 | **1.18** | 1.01 | 1.85% | **2.10%** | 1.75% |
| `R007` | 0.94 | **1.12** | 1.01 | 1.67% | **2.02%** | 1.81% |
| `M0017139` | 0.99 | 0.96 | 0.86 | 1.77% | 1.79% | 1.59% |

**结论**：`tol=0` 时规则 3 从未触发（重合日 0 天），等价于纯粹的「双均线金叉死叉」策略。
把阈值放宽到 **5bp 附近是多数指标的较优区间**——往返次数减少约 35%，
夏普普遍改善 0.05~0.27。再放宽到 10bp 时 `DR001`/`R001` 继续改善，
但 `M0017139` 反而变差（重合日占比已达 55.6%，信号被过度钝化）。
即阈值并非越大越好，这也是本项目提供 `sweep` 的原因。

> ⚠️ **单位陷阱**：`abs` 模式的单位是**百分点**，`bp` 模式的单位是**基点**，
> 两者差 100 倍。`--tol 0.01`（abs）= `--tol 1`（bp）= 1bp。
> 做阈值扫描建议直接用 `--tol-mode bp`。

---

## 4. 命令说明

### `ratema convert`

把文件夹内的 Excel 转成 CSV：

| 文件 | 说明 |
| --- | --- |
| `data/csv/<sheet>_raw.csv` | 忠实还原（表头=中文名，首行数据=指标代码） |
| `data/csv/panel.csv` | **规范宽表**，`date` + 指标代码列，回测默认输入 |
| `data/csv/long.csv` | 长表 `date, series_code, series_name, value` |
| `data/csv/series/<code>.csv` | 每个指标单独一个 CSV |
| `data/csv/columns.json` | 代码 → 中文名映射 |

所有 CSV 以 `utf-8-sig` 编码写出，双击用 Excel 打开中文不乱码。

### `ratema backtest`

```bash
uv run ratema backtest \
  --short 20 --long 120 \
  --tol-mode bp --tol 2 \
  --cost-bps 2 --cost-mode per_side \
  --start 2015-01-01 --end 2025-12-31 \
  --outdir output --tag bp2
```

产出（`output/`，或用 `--tag` 指定子目录）：

| 文件 | 说明 |
| --- | --- |
| `report.md` | 完整中文报告 |
| `summary.csv` | 各指标核心绩效汇总 |
| `metrics.json` | 全部绩效指标与运行参数 |
| `evaluation_window.csv` | 各指标评估窗口与起算口径 |
| `annual_breakdown.csv` | **分年度收益与胜率**（逐指标，含全区间行） |
| `spread_calibration.csv` | 阈值标定（`\|spread\|` 分位数） |
| `equity_curves.csv` | 所有策略 + 基准净值曲线 |
| `equity/<指标>.csv` | 单指标逐日净值 / 回撤 |
| `details/<指标>.csv` | 单指标逐日信号与持仓明细 |
| `trades/<指标>.csv` | 单指标成交流水 |
| `charts/*.png` | 图表（见下节） |

### 图表

`backtest` 默认生成 6~7 张 PNG 到 `output/charts/`（用 `--no-charts` 关闭）：

| 文件 | 内容 |
| --- | --- |
| `01_equity_curves.png` | 5 条策略 + 基准的净值曲线总览 |
| `02_per_series.png` | 每个指标一张：策略 vs 基准 + 超额/落后填充 |
| `03_relative_strength.png` | 策略/基准相对强弱（>1 即跑赢） |
| `04_drawdowns.png` | 回撤对比 |
| `05_annual_returns.png` | 年度收益热力图（行=指标，列=年份） |
| `06_signal_mechanics_<指标>.png` | 信号机理：利率双均线 / spread 与重合带 / 实际仓位 |
| `07_tol_sweep.png` | 不同阈值下的净值（由 `sweep` 生成） |
| `08_composite.png` | 等权综合：净值 + 投票结构 + 仓位对比（由 `composite` 生成） |
| `09_annual_breakdown.png` | **分年度四联图**：年度收益对比 / 累计超额 / 逐年胜率 / 逐年回撤 |

```bash
# 指定用哪个指标、哪个时间段画信号机理图
uv run ratema backtest --signal-series DR001 --signal-window 2020-01-01:2022-12-31
```

### `ratema journal`

**输出每日交易日志** —— 本项目的输出终点。把策略在每个交易日的状态与动作记成日志：

| 文件 | 说明 |
| --- | --- |
| `output/journal/journal.csv` | 逐日全量：信号 / 票数 / 仓位 / 动作 / 行情 / 净值 / 回撤 |
| `output/journal/journal_events.csv` | 动作流水：只含建仓、平仓 |
| `output/journal/journal_annual.csv` | 分年度收益与胜率 |
| `output/journal/journal.md` | 人读日志：概览 + 动作流水 + 最近 N 日 |

```bash
uv run ratema journal --tol-mode bp --tol 5        # 等权综合（默认）
uv run ratema journal --series DR007               # 指定单指标
uv run ratema journal --events-only                # 只看动作日
uv run ratema journal --tail 10                    # 最近 10 个交易日
uv run ratema journal --append output/ledger.csv   # 把当日记录追加/覆盖到台账
```

> `journal` 只记录、不下单、不对账。`--append` 同一天重复运行会**覆盖**该行，
> 因此可以安全地每天跑一次。

### `ratema sweep`

对重合阈值做网格敏感性分析，输出 `output/sweep/sweep_<mode>.{csv,md,json}` 与
`pivot_<metric>_<mode>.csv`。阈值过宽导致全程重合时会记录 `error` 列并继续扫描。

```bash
uv run ratema sweep --tol-mode bp --tol-grid 0,1,2,5,10
uv run ratema sweep --tol-mode q  --tol-grid 0,0.01,0.02,0.05,0.1
uv run ratema sweep --target composite --tol-mode bp   # 扫描等权综合信号
```

---

## 5. 项目结构

```
.
├── pyproject.toml
├── uv.toml                                    # uv 缓存指向项目内
├── .github/workflows/ci.yml                   # CI：lint / types / test / 端到端
├── 中债-10年期国债净价指数与利率数据.xlsx      # 原始数据
├── data/csv/                                  # 转换后的 CSV
├── output/                                    # 全部产出
├── scripts/                                   # 分析与交叉验证脚本
├── src/ratema/
│   ├── io_utils.py     # L0  Excel/CSV 读写与转换
│   ├── indicators.py   # L0  均线 + 重合阈值 + 信号
│   ├── backtest.py     # L0  T+1 执行、双边成本、净值
│   ├── metrics.py      # L0  绩效指标
│   ├── charts.py       # L0  图表（惰性导入 matplotlib）
│   ├── pipeline.py     # L1  单策略编排
│   ├── composite.py    # L1  等权打分综合
│   ├── daily.py        # L1  每日信号快照
│   ├── journal.py      # L1  每日交易日志
│   ├── render.py       # L2  表格 / Markdown 渲染
│   ├── writers.py      # L2  产物落盘
│   ├── parser.py       # L2  命令行参数定义
│   ├── commands/       # L2  各子命令实现
│   └── cli.py          # L2  入口（薄壳）
└── tests/              # 158 个测试
```

## 6. 测试与代码质量

```bash
uv run pytest                     # 158 个单元 / 集成测试
uv run ruff check .               # 静态检查
uv run ruff format --check .      # 格式检查
uv run mypy                       # 类型检查（23 个源文件）
uv run python scripts/cross_check.py   # 引擎 vs 独立向量化实现
```

测试覆盖：重合阈值各模式、前向填充、T+1 执行时滞、成本口径、无未来函数、
净值与基准口径、以及**逐笔引擎与独立向量化实现的一致性**
（真实数据上最大偏差 9.5e-15）。

> `tests/conftest.py` 把 `tmp_path` 覆盖到项目内的 `build/pytest-tmp`，
> 因此在系统临时目录不可写的受限环境里也能正常跑测试。

---

## 7. 常见问题

**Q：`uv` 报 "Failed to initialize cache"？**
A：项目根目录的 `uv.toml` 已把缓存指向项目内的 `.uv-cache`，正常情况下不会遇到。
若你的环境另有策略，可显式覆盖：PowerShell `$env:UV_CACHE_DIR=".uv-cache"`。

**Q：图表里的中文变成方框？**
A：缺中文字体。Windows 自带 Microsoft YaHei 无需处理；Linux 装 `fonts-noto-cjk` 即可。

**Q：提示「全部交易日的 |MA20-MA120| 都落在重合区间内」？**
A：阈值设得太大，信号被完全钝化，无法产生初始信号。调小 `--tol` 即可。
`output/spread_calibration.csv` 里的 `p05_abs_bp` 是很好的起点。

**Q：`abs` 和 `bp` 到底差多少？**
A：100 倍。`--tol 0.01`（abs）= `--tol 1`（bp）= 1bp。扫描阈值建议直接用 `--tol-mode bp`。

---

## 8. 免责声明

本项目仅用于研究与教学，不构成任何投资建议。历史回测结果不代表未来表现。
