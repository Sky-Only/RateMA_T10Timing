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
uv run ratema all

# 3) 看结果
#    output/report.md    完整中文报告
#    output/charts/      图表
```

就这两条命令。`uv` 会在项目内建好隔离环境，依赖版本由 `uv.lock` 锁定，换台机器结果一致。

### 常见任务

| 我想…… | 命令 |
| --- | --- |
| 跑默认回测（阈值 0） | `uv run ratema backtest` |
| 用 5bp 作为重合阈值 | `uv run ratema backtest --tol-mode bp --tol 5` |
| 换均线（如 10/60 日） | `uv run ratema backtest --short 10 --long 60` |
| 只看某段时间 | `uv run ratema backtest --start 2020-01-01 --end 2024-12-31` |
| 改交易成本（如单边 5bp） | `uv run ratema backtest --cost-bps 5` |
| 成本按"往返合计"理解 | `uv run ratema backtest --cost-bps 2 --cost-mode round_trip` |
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
