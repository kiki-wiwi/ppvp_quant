# R5 跨载体对照：CP2 BTC IS 结果与停机记录

本次固定 42 组合、五个载体/窗口内配对统计族及四张图已完成，正式运行退出码为 0。所有族均未通过全部 V1–V6 门禁，主要共同障碍是趋势相对持续多头的增益检验 V3，以及删除贡献最大三个月后的稳健性 V5。结果仅为 BTC 样本内载体比较；不构成自动晋级、样本外有效性或实盘收益结论。

## 1. 基线、提交链与运行状态

分支：`codex/trend-hold-venue-oos-study`。下列完整 SHA 来自本轮登记与 [summary.json](../../reports/trend_hold_venue_round/cp2_btc_is/summary.json) 的 `input_metadata`；本结果提交不在自身文档内写入自引用 SHA。

| 阶段 | 完整提交 SHA |
|---|---|
| R5 已归档结果基线 | `3846971ad9014043cc0371a96c693924b7ca1e61` |
| CP1 实现与合成验证 | `deadf01a4570386e14bbbec41d6e8eadd3345cac` |
| 原 CP2 实现 | `4870310dfa9dd4a51b64aaa149caffaf48de79f9` |
| 原 CP2 单行解锁 | `51a2bc3e15c4a12a87424dc7217278b965c52d81` |
| 输入修订一实现 | `c4c619160bc4c66d643f6bb664b429ed239f6e20` |
| 输入修订一单行解锁 | `ef0d3effc947467fcb91f7cae5a052634a47ae35` |
| 输入修订二／本次实现 | `ef1ad9ace449a13cd79de2a28caf2f32b40fdddb` |
| 本次独立单行解锁 | `90b2f9ddc7aa933e06f26eca14d80aabf96f8c58` |

本次启动 `git status --porcelain` 完全为空，输出目录按干净目录要求新建。运行外层记录为 **2026-11-07 15:21:22.416057 至 16:45:52.203627 UTC**，约 84.5 分钟；账户运行入口 `started_at` 为 `15:21:35.919890 UTC`，结果 `ended_at` 为 `16:45:51.904241 UTC`。这些时间均为 UTC，不是交易所本地时区。

CP1 的 32 个登记文件、142 个历史保护文件及当前 133 个 CP2 登记文件全部通过哈希核验。对 `scope.py` 仅按批准规则在内存还原 `BTC_IS_UNLOCKED=False` 后验证 CP1 哈希；磁盘解锁提交只切换这一个锁。冻结策略公式、参数、成本、种子及统计方法未变。CP1 与历史保护清单内的策略代码、测试保持原字节；获批的 CP2 加载器与登记控制器改动按两次输入修订单独登记。

固定仓库外环境：Python 3.14.4、pandas 3.0.5、NumPy 2.5.4、PyArrow 26.0.1、Numba 0.68.1、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.2.0。最近一次解锁前合成验证 **235 passed，1 deselected**；使用 `-p trend_hold_venue_study.testing_guard -x`，市场文件／网络守卫启用，两道新锁均为 `False`，没有选中测试失败或跳过。

唯一经用户批准排除的历史节点为：

```text
tests/test_trend_hold_engine.py::test_rejects_missing_or_invalid_daily_inputs_and_real_context_before_access
```

该旧 R5 CP1 测试预期历史 `REAL_DATA_UNLOCKED=False`，而 R5 归档基线已经解锁为 `True`。仅通过命令行 `--deselect` 排除，未修改测试或旧锁。完整命令、理由、环境及测试前后冻结状态见 [synthetic_validation.json](../../reports/trend_hold_venue_round/cp2_registration/input_revision_2/synthetic_validation.json)。

## 2. 输入核验与经批准的停止后修订

全部输入限定 BTC，时间严格早于 `2024-01-01 UTC`，资金费在加载入口应用时间谓词。没有读取 VAL、OOS、2024+ 或 ETH；没有用输入修订调整策略或根据策略收益筛选数据。

- **UM 缓存序列化差异**：旧 Parquet 文件字节 SHA 为 `60e4fd7a0cf23cb83b97a71896c30d6e783b8856d4da4d7be1c805b08040ddf4`，当前为 `57479bcbb97a21e58e2347e5c17cc6194a2a6fc1302397d4459592746af39aeb`。规范化内容 SHA 精确一致：`1e26dd7ecfe6ee03dc2814dcb0041ef2d5724ff7018b2b7f20ae6364d99f5a33`。经批准登记当前文件字节与内容双重绑定，未改写旧缓存、manifest 或历史哈希。
- **SPOT close_time**：严格登记 8 条原始时间字段例外，其中 5 条价格有效、3 条原已为 `bad` 占位；不改变 OHLCV，不增加维护标记，可用时间仍由 `open_time + 1min` 决定。
- **CM 原生起点与资金费**：首个原生分钟 `2020-08-11 07:02 UTC`，首个完整日为 `2020-08-12`。规范资金费表保留 3,714 条，仅 Context 排除早于原生起点的 `2020-08-11 00:00` 一条，实际 Context 为 3,713 条；不删除规范原始记录。
- **CM 资金费来源拼接**：18 个可用 Vision 月包的 1,593 条记录与已保存官方 dapi 记录按 Decimal 精确交叉核对；仅补入批准的 54 条月末 `00/08/16 UTC` 结算，令该重叠段完整为 1,647 条。Common 窗口合计 3,651 次结算。完整规范表总量与这一重叠段计数不同；41 页 API 的请求参数、时间与哈希继续保留。
- **CM 日包补齐**：官方 `2023-08-28` 至 `2023-08-31` 四个日包各补 1,440 分钟，共 5,760 分钟；原月包字节不动。其余缺失按登记规则确定性标记并构造连续合法价格网格，遇整日无有效分钟仍应停机。

最终输入计数来自 `summary.json`，`bad` 与 `archive_gap` 不是互斥分类：

| Context | 分钟数 | bad | archive_gap | 资金费事件 |
|---|---:|---:|---:|---:|
| UM | 2,103,840 | 144 | 10 | 4,383 |
| SPOT | 2,103,840 | 2,536 | 2,325 | 0 |
| CM | 1,782,298 | 10,685 | 0 | 3,713 |

本次输入绑定文件为 [input_binding.json](../../reports/trend_hold_venue_round/cp2_registration/input_revision_2/input_binding.json)，SHA256 `7bf18a53847089249ce623e27e0f974081912518d5417aced081f98f67cf38cc`。完整原始包与缓存哈希保留在登记清单及运行输入元数据中；未用重新冻结覆盖不一致。

此前遇到下载、缺失结算、输入时标及缓存字节问题均先停止，取得针对性批准后再通过独立实现／解锁提交恢复。历史原现场与哈希索引归档在以下目录；不是本次成功运行的局部补跑：

- `reports/trend_hold_venue_round/cp2_registration/validation_history/vision_gap_stop/`
- `reports/trend_hold_venue_round/cp2_registration/validation_history/vision_month_end_stop/`
- `reports/trend_hold_venue_round/cp2_registration/validation_history/um_cache_serialization/`
- `reports/trend_hold_venue_round/cp2_registration/input_revision_1/validation_history/attempt_1/`
- `reports/trend_hold_venue_round/cp2_registration/input_revision_2/validation_history/preunlock_preflight_2/`（含嵌套 `validation_history/attempt_2/` 与解锁前输入预检证据）。

## 3. 实际执行范围

| 窗口／任务 | UTC 时间范围（右端不含） | 天／月 | 账户数 |
|---|---|---:|---:|
| IS-Full：V1、P_USDM、V2、P_SPOT × 三成本 | `[2020-03-01, 2024-01-01)` | 1,401／46 | 12 |
| IS-Common：全部八版本 × 三成本 | `[2020-09-01, 2024-01-01)` | 1,217／40 | 24 |
| Common 整张合约：V3、P_CM、V4，main_17 | 同上 | 1,217／40 | 3 |
| CM 原生自信号：V3、P_CM、V4，main_17 | `[2020-11-01, 2024-01-01)` | 1,156／38 | 3 |
| **合计** | | | **42** |

三成本为 `gross_0 / main_17 / harsh_34`。Common 使用自 `2020-03-01` 生成的规范 UM 信号，并继承 `2020-08-31` 周一预算；各评估窗口独立以 100,000 USD/USDT 账户启动。整张合约按 100 USD/张执行。未额外增加同为 38 个月的 UM 参考账户，也未扩大到 CP3。

## 4. 主成本核心结果

现金闭环是主比较口径：终值包含末端平合约、现货卖出滑点及手续费；币数列保留卖出现货换回 USDT 前的最终币余额。最大回撤来自有效分钟净值，表内百分数仅由已保存指标换算显示。`V4_CM_CoinAccum` 的目标敞口为 `1 + gL`，含额外抵押币价格暴露，不能与同目标敞口的 V1/V2/V3 作风险调整前的胜负排名。

### IS-Full，main_17

| 版本 | 现金终值 USD | 累计收益 | CAGR | Sharpe | 最大回撤 | 净赚币率 | 净资金费 USD | 现金裁剪次数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `V1_USDM` | 232,978.69 | 132.98% | 24.65% | 1.281 | 33.69% | — | -27,957.95 | 0 |
| `P_USDM` | 194,589.23 | 94.59% | 18.94% | 0.824 | 42.90% | — | -31,507.42 | 0 |
| `V2_SPOT` | 273,023.40 | 173.02% | 29.91% | 1.510 | 32.56% | — | +0.00 | 2 |
| `P_SPOT` | 235,961.44 | 135.96% | 25.06% | 1.030 | 41.53% | — | +0.00 | 2 |

### IS-Common，main_17

| 版本 | 现金终值 USD | 累计收益 | CAGR | Sharpe | 最大回撤 | 净赚币率 | 净资金费 USD | 现金裁剪次数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `V1_USDM` | 193,211.93 | 93.21% | 21.84% | 1.188 | 33.69% | — | -20,432.90 | 0 |
| `P_USDM` | 168,288.31 | 68.29% | 16.90% | 0.771 | 42.90% | — | -24,098.48 | 0 |
| `V2_SPOT` | 220,362.81 | 120.36% | 26.74% | 1.411 | 32.56% | — | +0.00 | 2 |
| `P_SPOT` | 197,555.97 | 97.56% | 22.65% | 0.975 | 41.53% | — | +0.00 | 2 |
| `V3_CM_DeltaMatched` | 284,112.52 | 184.11% | 36.78% | 1.838 | 30.23% | -21.59% | +48,064.99 | 0 |
| `P_CM_DeltaMatched` | 250,791.43 | 150.79% | 31.75% | 1.277 | 40.31% | -30.79% | +44,679.61 | 0 |
| `V4_CM_CoinAccum` | 532,967.86 | 432.97% | 65.18% | 1.036 | 85.11% | 47.09% | -49,949.71 | 0 |
| `HODL_1x_Coin` | 362,347.97 | 262.35% | 47.13% | 0.922 | 77.52% | 0.00% | +0.00 | 0 |

四个币账户严格共享初始 `8.575474414806393 BTC`。最终 V3、P_CM、V4、HODL 分别为 `6.723922486498744`、`5.935331948666443`、`12.613434013953784`、`8.575474414806393 BTC`。V3 的美元收益较高同时币数下降 21.59%；V4 净赚币 47.09%，但最大回撤 85.11%，高于 HODL 的 77.52%。这些不同目标不可混为一个“最优载体”结论。

### Common 三成本对照

| Common 版本 | gross_0 收益 | main_17 收益 | harsh_34 收益 | main_17 / harsh_34 净赚币率 |
|---|---:|---:|---:|---:|
| `V1_USDM` | 125.02% | 93.21% | 89.38% | — / — |
| `P_USDM` | 100.04% | 68.29% | 66.36% | — / — |
| `V2_SPOT` | 124.83% | 120.36% | 115.98% | — / — |
| `P_SPOT` | 99.85% | 97.56% | 95.29% | — / — |
| `V3_CM_DeltaMatched` | 124.37% | 184.11% | 177.95% | -21.59% / -23.16% |
| `P_CM_DeltaMatched` | 100.10% | 150.79% | 147.45% | -30.79% / -31.59% |
| `V4_CM_CoinAccum` | 504.65% | 432.97% | 420.91% | 47.09% / 44.00% |
| `HODL_1x_Coin` | 262.96% | 262.35% | 261.73% | 0.00% / 0.00% |

`gross_0` 关闭资金费；因此 gross → main 同时变化交易摩擦与资金费，不能解释为纯手续费差。UM／SPOT／CM 同敞口趋势账户的 gross 累计收益约 125%，主成本后分化明显。CM 套保的正资金费贡献使 V3 的 main 终值高于 gross，不能据此声称增加成本提高收益。`harsh_34` 下收益与 V6 仍为正，成本不是本轮门禁失败的唯一原因。全 42 组的费用、滑点、换币摩擦、波动率及币／美元估值见 [comparison.csv](../../reports/trend_hold_venue_round/cp2_btc_is/comparison.csv)。

## 5. 逐期归因、R5 对账及载体差异

[原 R5 精确对账](../../reports/trend_hold_venue_round/cp2_btc_is/r5_exact_reconciliation.json) 的六组 Full UM 账户（T/P × 三成本）全部通过：各 1,401 天每日净值、日收益及终值与 `3846971` 归档结果逐项完全一致。

[reconciliation.csv](../../reports/trend_hold_venue_round/cp2_btc_is/reconciliation.csv) 记录的最大绝对误差为：现金闭环 `3.4924596548080444e-10 USD`；CM 内生美元归因 `2.6505726415425997e-26 USD`；币归因 `9.795279697153482e-31 BTC`；钱包核对 `1.7266211685076596e-15 BTC`。满足冻结的美元 `<1e-9` 及钱包币量容差。动态持币价差、当期合约盯市、资金费、交易摩擦、起止换币均已逐期入账；未用固定初始币数的静态归因。

全部 42 组 `liquidated=False`，零下界债务豁免调整均为 0；这只表示本次样本未触发压力强平。最大现货浮点夹零调整为 `3.637978807091713e-11 USD`，单独登记而不计作市场盈利。压力触线残币与开盘穿仓夹零两条路径仍由已通过的合成测试覆盖。

Common main 的描述性跨载体桥接如下（相对同信号 UM，资金费为在各事件现货参考价下的带符号美元折算）：

| 比较 | 现金终值差 USD | 其中资金费差 USD | 期末基差项 USD | 剩余价格路径项 USD |
|---|---:|---:|---:|---:|
| V2 − V1 | +27,150.88 | +20,432.90 | 0.00 | +7,000.37 |
| V3 − V1 | +90,900.59 | +68,497.89 | −146.72 | +23,889.11 |
| P_SPOT − P_USDM | +29,267.66 | +24,098.48 | 0.00 | +5,344.96 |
| P_CM − P_USDM | +82,503.12 | +68,778.09 | −129.51 | +14,702.89 |

表中列出的分项不是全部加总项；[crossvenue.csv](../../reports/trend_hold_venue_round/cp2_btc_is/crossvenue.csv) 另列手续费、滑点、换币与舍入调整，完整桥接误差均为 0。V3−V1 的手续费、滑点及换币差分别为 −602.02、−438.77、−298.91 USD。残差包含价格路径、周度调整、现金裁剪及复利效应，期末基差桥接也是描述性分解，不能据此识别独立因果 Alpha。

## 6. 五个配对统计族与 V1–V6

定义 `T` 为该载体趋势账户日收益，`P` 为相同事前风险预算的持续多头，`D=T−P`。沿用受约束按月聚类 Webb 检验：99,999 次、有限模拟分母 100,000、种子 `2026111001`；自然月配对块区间 10,000 次、种子 `2026111002`。同族保持配对月与抽样顺序。每个载体／窗口仅对 T、D 两个假设作 Holm 校正，族大小为 2；没有宣称跨五族全局多重控制，历史 IS 选择仍是限制。

| 统计族 | T 原始 p | T Holm p | D 原始／Holm p | V1 | V2 | V3 | V4 | V5 | V6 |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full UM | 0.02276 | 0.04552 | 0.32984 | 通过 | 通过 | 未过 | 通过 | 未过 | 通过 |
| Full SPOT | 0.01053 | 0.02106 | 0.37248 | 通过 | 通过 | 未过 | 通过 | 未过 | 通过 |
| Common UM | 0.05418 | 0.10836 | 0.36232 | 通过 | 未过 | 未过 | 未过 | 未过 | 通过 |
| Common SPOT | 0.03001 | 0.06002 | 0.40228 | 通过 | 未过 | 未过 | 未过 | 未过 | 通过 |
| Common CM DeltaMatched | 0.00916 | 0.01832 | 0.37963 | 通过 | 通过 | 未过 | 未过 | 未过 | 通过 |

这里 V1–V6 是门禁编号，与版本名 V1_USDM 等不同：V1 覆盖、V2 T 为正且检验通过、V3 配对增益通过、V4 两半稳定、V5 删三个月稳健、V6 恶劣成本。完整细项见各族 [statistics](../../reports/trend_hold_venue_round/cp2_btc_is/statistics/) 下 `tests.csv / gates.csv / intervals.csv / metadata.json`。

区间为已保存 95% 月块百分位区间，以下仅将日均收益换算为 bps／日：

| 统计族 | T 日均值 95% CI（bps） | D 日均值 95% CI（bps） | T Sharpe 95% CI | V5 删月后 D 均值（bps／日） |
|---|---:|---:|---:|---:|
| Full UM | [0.652, 12.941] | [−2.944, 4.995] | [0.140, 2.349] | −1.517 |
| Full SPOT | [1.656, 14.234] | [−3.163, 4.742] | [0.357, 2.585] | −1.737 |
| Common UM | [−0.633, 13.024] | [−3.304, 5.181] | [−0.147, 2.429] | −1.851 |
| Common SPOT | [0.334, 14.169] | [−3.525, 4.907] | [0.075, 2.655] | −2.050 |
| Common CM DeltaMatched | [2.143, 16.508] | [−3.411, 5.031] | [0.475, 3.108] | −1.924 |

五族 V1 均通过；Full 两半 on/off 分别为 420/251、329/401 天，Common 为 315/172、329/401 天，均超过每状态 60 天。没有覆盖不足的门禁失败，但 40／46 个自然月仍限制检验精度及跨月相关性的解释。

所有 D 的区间跨 0，Holm p 为 0.32984–0.40228，未证明趋势开关超越相同风险预算持续多头。V5 在 Full 删除 `2022-06、2020-03、2023-08`（剩 1,309 天），在 Common 删除 `2022-06、2023-08、2022-01`（剩 1,125 天）后，五族 D 均转负。Common 的 V4 也因前半段 D 为负而失败。共同障碍是增益不显著且依赖少数月份，不能归结为仅成本侵蚀或仅样本覆盖不足。

全部十项检验 `mc_boundary_sensitive=False`，均无强制 p=1 的退化原因，零 bootstrap 标准误次数为 0；区间各 10,000 次均有效、无丢弃。唯一“区间支持为正但 Holm 不过”的披露项是 **Common SPOT 的 T**：均值 CI 下限 +0.334 bps／日，原始 p=0.03001，Holm p=0.06002；仍依冻结门禁判为不通过，不改用区间放行，也不增加抽样或重试。

## 7. 自信号、整张合约与四张图

CM 自信号 1,156 天与规范 UM 的开关一致率 **100%**。预算差（CM−UM）日均 −0.0009334494、周均 −0.0011875319，最大绝对差 0.0501407465（杠杆预算约 5.0141 个百分点）。这不代表预算完全相同。自信号账户自 `2020-11-01` 独立启动，未增跑同起点 UM 账户，因此不得把它与 Common 终值之差解释为纯信号影响。详见 [self_signal_summary.json](../../reports/trend_hold_venue_round/cp2_btc_is/self_signal_summary.json)。

整张合约相对连续面值的 Common main 诊断：

| 版本 | 现金终值差 USD | 净赚币率差（百分点） |
|---|---:|---:|
| V3_CM_DeltaMatched | +209.63 | +0.05785 |
| P_CM_DeltaMatched | +173.02 | +0.04775 |
| V4_CM_CoinAccum | −178.30 | −0.04921 |

幅度均在约 ±210 USD，未据此调参、选择取整方式或改变门禁。原始数值见 [rounding.csv](../../reports/trend_hold_venue_round/cp2_btc_is/rounding.csv)。

1. [图 1：净值与币余额](../../reports/trend_hold_venue_round/cp2_btc_is/figure_1_equity_and_coin.png)。Full／Common 分别显示同载体 T/P；Common CM DeltaMatched 的收益与回撤表现优于 UM/现货的描述性点估计。V4 与 HODL 独立成图，净赚币并不消除 85.11% 最大回撤风险；末端现金点包含实际换币摩擦。
2. [图 2：成本阶梯](../../reports/trend_hold_venue_round/cp2_btc_is/figure_2_cost_steps.png)。同敞口 gross 回报接近，资金费开启后形成明显载体差异。harsh 下 V3 仍收益 +177.95%、V4 净赚币 +44.00%，但不能以 gross/main 差当作单纯手续费效应。
3. [图 3：资金费、摩擦与门禁](../../reports/trend_hold_venue_round/cp2_btc_is/figure_3_funding_friction_and_gates.png)。Common main 趋势账户 UM 付资金费 20,432.90 USD、现货为 0、CM DeltaMatched 净收 48,064.99 USD；五族 V3/V5 全部失败，载体收益差不等于趋势增益已获证明。
4. [图 4：自信号与取整](../../reports/trend_hold_venue_round/cp2_btc_is/figure_4_signal_and_rounding_diagnostics.png)。方向开关完全一致但预算有差；整张合约只形成上述较小终值变化，不用作参数挑选。

图口径与限制另见 [figure_captions.md](../../reports/trend_hold_venue_round/cp2_btc_is/figure_captions.md)。

## 8. 产物交付与阶段停止

正式运行产物为 395 个文件，约 5.9564 GiB；包含 42 个账户的分钟／逐日账本、订单与决策、资金费、动态归因、压力事件及元数据，五族完整统计表与固定抽样矩阵、输入核验、R5 对账、四张图及运行日志。完整路径、原始字节大小与 SHA256 以 [delivery_hashes.json](../../reports/trend_hold_venue_round/cp2_btc_is/delivery_hashes.json) 为交付核验依据。

84 份大账本约 5.7692 GiB 按 Git LFS 方式交付：Git 保存内容寻址指针，完整对象单独上传；已生成结果字节与登记哈希不改变。LFS 对象清单及上传核验日志与交付清单配套记录。全部 84 个 LFS 对象已由服务器逐项校验，上传过程零自动重试；原始 395 个运行产物的字节与 SHA256 在发布后再次核对一致。完整结果提交 SHA 与远端分支状态由提交后的最终核验汇报给出。

本次正式运行始终 `BTC_IS_UNLOCKED=True`、`POST_IS_AND_ETH_UNLOCKED=False`；后者全程未改动。`automatic_promotion=False`。本阶段完成后停在 CP2，等待用户审查；不进入 CP3，不运行任何 2024+、VAL/OOS 或 ETH 验证。
