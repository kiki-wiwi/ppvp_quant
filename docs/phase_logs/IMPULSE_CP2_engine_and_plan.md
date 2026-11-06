# 脉冲研究 Checkpoint 2 阶段记录：多空双向撮合引擎、因果特征与预注册计划冻结

记录日期：2026-11-06（UTC）。分支 `claude/laughing-pasteur-0byag2`；提交号见第 2 节。

本阶段遵守用户 Checkpoint 2 裁定第五条的边界。真实行情没有进入撮合引擎，没有计算任何真实逐笔盈亏、净值曲线或收益统计，也没有做任何参数筛选。真实 IS 数据只用于触发前因果特征的回归：ATR_15m、基准成交量、S6 箱体状态、10 个版本的入场信号与资格排除计数、冻结的止损距离，以及由止损距离与成本常数推出的计划名义杠杆。没有下载、读取或运行任何 VAL、OOS、ETH 的数据或绩效，B 组全时段附注只登记为第 3、4 阶段的附注任务，本阶段没有计算。

## 1. 结论摘要

Checkpoint 2 的交付物已完成：多空双向 1m 撮合与记账引擎、因果特征与 10 个冻结版本的入场输入、预注册实验计划，以及手算 / 合成测试。`test_impulse_cp2.py` 的 99 个测试全部通过（其中 97 个只用合成数据，2 个在真实数据上做触发前特征的因果回归与锁定检查），Checkpoint 1 的测试照常通过。撮合引擎从未在真实数据上调用；真实 IS 数据只进入了触发前特征的回归（第 5 节）。

撮合与记账的正确性经三轮独立复核。第一轮在统计设计上有五条重要发现，在引擎中发现了会静默出错的规则缺陷：资金费最多只收 8 次；停机延续到数据末端时会编造离场；独立事件模拟不拒绝非法入场。以上均已修正。第二轮没有发现撮合或记账错误：按修订后计划重写的盲引擎在 2,000 次合成组合回测、约 122 万笔交易上与引擎逐位一致；两位互不相知的实现者按计划第 8 节各自写出统计实现，在全部门禁 p 值、抽取哈希与放宽计数上逐位一致。它的发现是输入校验缺口、测试缺口与计划措辞。第三轮针对第二轮的修正，8 条非细节发现全部是次要的，同样已处理。三轮合计 74 条非细节发现经另一位复核者以“证伪”为目标独立复现，没有一条被推翻。修正之后重跑变异测试，第二轮的全部违规变异与针对新检查的变异都会使测试失败。

需要用户在审批中逐项表态的是第 8 节列出的 14 项复核后修订（第 13、14 项只是判定文字与定义的补充）。其中三项改变统计判定方法：门禁 G2、G3 与 F3 改以按月聚类的野自助 t 检验 p 值判定（百分位 95% 区间照常报告但不作门禁，这偏离了裁定“主口径维持按月聚类 95% 置信区间”的字面表述）；夏普的区间与 p 值改为按自然月块重抽样；F5 改为双侧 0.025。依据都是合成数据上的模拟，没有使用任何真实交易结果。

触发前特征有两点直接影响第 3 阶段结果的解读。第一，0.15% 的距离下限在全部版本中都没有生效，2.0x 上限在主成本下也没有生效，主配置的仓位完全由止损距离决定，A3（SL_C）的计划名义杠杆中位数约为 A1（SL_A）的 2.6 倍。第二，A 组的可成交笔数高度集中在 2021 年，按月份计的有效样本约 11 个月，A 组门禁与 A 对 B 的箱体比较的统计功效因此受限。

本阶段没有未解决的阻塞项；待决事项见第 9 节。

## 2. 交付物与冻结标识

| 文件 | 内容 |
|---|---|
| `impulse_study/engine.py` | 多空双向 1m 撮合与记账（numba）：t 收盘冻结数量，入场开盘跳空、持仓跳空与盘中止损，末分钟止损优先，量能延续判别，停机强制离场，资金费，冷却，逐分钟盯市，台账，输入校验；真实行情锁 |
| `impulse_study/features.py` | ATR_15m(14) 自然整刻棒、基准成交量中位数、脉冲起点极值、10 个版本的逐分钟入场输入与两种资格排除计数、S6 箱体构造入口 `s6_box_ok`；上下文构造时的网格、资金费与漏记分钟校验 |
| `impulse_study/versions.py` | 10 个冻结版本、4 种成本、2 种仓位、全部常数、A 组箱体口径与表 5b 漏记分钟；`frozen_key()` |
| `impulse_study/synthetic.py` | 合成数据（带合成标记的网格、随机网格、手工信号输入） |
| `test_impulse_cp2.py` | 手算 / 合成测试与真实数据上的两项因果特征与锁定回归 |
| `tools/run_impulse_cp2_features.py` | 真实 IS 的触发前特征 → `reports/impulse_vs_box_round/cp2/cp2_feature_checks.json` |
| `tools/bench_impulse_engine.py` | 2,103,840 根合成 1m 的耗时基准 → `reports/impulse_vs_box_round/cp2/bench_engine.json` |
| `reports/impulse_vs_box_round/experiment_plan_impulse.md` | 预注册实验计划（全部参数、种子、门禁与判定规则；第 13 节为冻结标识与代码 SHA256） |

冻结标识：`versions.frozen_key()` = `fc0d37b9d198ff85`；10 个版本键见计划第 7 节（与复核前相同）。27 个冻结文件的 SHA256 见计划第 13 节，`cp2_feature_checks.json` 记录的 `code_sha256` 与之逐项相同（特征脚本在运行前核对这一点）。A 组箱体在真实数据上的逐分钟 `box_ok` 的 SHA256 为 `305805e09a15b8b14439bf594a8d3ea2566b70103f012a128ccb3bbfe658560d`，第 3 阶段的运行脚本须与之核对。

本阶段的提交（分支 `claude/laughing-pasteur-0byag2`，均已推送）：

| 提交 | 内容 |
|---|---|
| `ef4fa32` | 复核前的引擎、特征、版本、测试、工具与计划草案 |
| `6096f3e` | 第一轮复核的修正 |
| `eedb4fe` | 计划补充：第 4 阶段敏感性代码在解锁之前提交 |
| `58117a6` | 第二、三轮复核的修正：输入校验、测试、计划修订与第 13 节 SHA256 表。冻结的代码与计划以此为准 |
| `74522d4` | 在 `58117a6` 的干净工作树上重新生成的 `cp2_feature_checks.json`；耗时基准脚本改为记录提交号（该脚本不在冻结范围内，冻结文件与 `58117a6` 逐字节相同） |
| 本记录所在的提交 | 在 `74522d4` 的干净工作树上重新生成的 `bench_engine.json` 与本记录 |

## 3. 实现要点与本阶段确定的约定

引擎逐条实现计划第 4–5 节，下面只记录实现中需要用户知道的约定，以及为了让规则可执行而在本阶段写定的细节（都已写进计划，并由测试固定）。

数量与成交。订单在触发 K 线 t 收盘时冻结：E_ref = C_t，以损定仓 q = min(0.5%·Equity_t / (max(|E_ref − SL|, 0.15%·E_ref) + c_rt·E_ref), 2.0·Equity_t / E_ref)，等名义 q = 1.0·Equity_t / E_ref。下一根 t+1 开盘按 O_{t+1}·(1 + s·slip_entry) 成交，数量不随这个开盘价回改；0.15% 下限只进入数量的分母，不移动止损价。台账的 `cap_bound` 记录 2.0x 上限严格小于以损定仓数量，`floor_bound` 记录止损距离严格小于 0.15%·E_ref（下限在分母中生效，不论上限是否同时约束；在 14 / 17 / 34bp 下下限生效时上限不会约束，两者只在零成本情景下可能同时为真），恰好相等都记为假。

止损与时间离场。入场开盘已越过止损时保留开仓、随即按开盘价平仓，两边计费；此后持仓分钟开盘跳空越过止损按该开盘价离场，盘中触及按止损价离场；第 1、2 根合成棒的末分钟（t+15、t+30）内若盘中触及止损，一律按止损成交（保守最坏路径），这是 1m 数据下唯一需要的分钟内次序假设。第 1 根满时按 v1 ≥ 1.3 且 k1 ≥ 0.10 决定是否持有第 2 根；B4 也照常计算这一检验，但只作诊断（B4 的 `cont = 1` 不表示实际延续）。`cont` 取 1、0、−1，−1 表示在 t+15 收盘之前已离场，或停机覆盖 t+15 而按规则不做判别。

停机。持仓中遇到维护占位分钟时，在停机后第一根有效分钟的开盘强制离场，标签恒为 `halt_forced_exit`（即使该开盘已越过止损），持仓可因此超过 30 分钟。停机延续到数据末端时计划没有定义离场，引擎直接报错，而不是编造离场价；IS 内不会发生。

资金费。结算时点 τ 满足“入场时刻 < τ ≤ 离场时刻”才收付，金额 −s·q·P_τ·r_τ，P_τ 为 τ 开盘那一分钟的开盘价（停机期间取占位分钟的开盘价，即停机前最后收盘价），时间离场恰在 τ 收盘时取该分钟收盘价；一笔交易内的结算次数不设上限。结算记入以 τ 为开盘的那一分钟的盯市净值。本研究用 1m 成交价近似交易所实际使用的标记价格。

冷却与净值。离场时刻对开盘类离场取该分钟开盘、对盘中止损与时间离场取该分钟收盘；下一个可接受的触发索引 ≥ 离场时刻索引 + 14（即入场时刻 ≥ 离场时刻 + 15 分钟）。逐分钟盯市，日净值取每个 UTC 日最后一分钟。没有成交的版本返回列与类型完整的空台账。

输入校验与真实行情锁。`features.make_context` 只接受带时区的索引（统一转换为 UTC）、严格连续且无重复的分钟网格，以及逐分钟的 `box_ok`（复制一份，不与调用方共用），资金费时间戳不得重复、费率必须有限，网格范围内的结算与漏记分钟必须恰在分钟开盘上。`run_portfolio` 只接受由同一个上下文对象构造、且构造之后该上下文的交易窗口、该版本的触发条件与（A 组）箱体没有被原地改动的版本输入（`VersionInputs.context is ctx`，且内容摘要 `VersionInputs.input_key` 与 `features.input_key(ctx, 版本)` 相同；换窗口、换触发条件或换箱体的上下文都必须重建版本输入，计划还禁止第 3、4 阶段就地改写已构造的上下文），各数组为一维且与网格等长，`entry_dir` 整列只能取 −1、0、+1，交易窗口最后一个可触发分钟之后至少还有 30 分钟数据；每个信号或独立事件都要求方向为 ±1、入场 K 线有效、止损有限且在亏损一侧、基准成交量有定义；`simulate_events` 另要求事件在交易窗口内、无维护污染、其后至少 30 分钟数据、索引为整数、各输入等长。`engine.REAL_DATA_UNLOCKED = False`：`source` 不是 `'synthetic'` 的上下文一进入 `run_portfolio` 或 `simulate_events` 就抛出 `EngineLockedError`，而 `'synthetic'` 标签只在网格带 `synthetic.grid_from_arrays` 写入的标记时才被 `make_context` 接受。这个锁防的是误用：`dataclasses.replace` 仍可显式改写标签，Python 中无法阻止有意改写。解锁之后，真实上下文还必须恰在网格内全部 UTC 00:00 / 08:00 / 16:00 分钟带有资金费结算（IS 全网格 4,383 次，不多不少），并恰好标记冻结的 10 个漏记分钟。

特征。ATR_15m(14) 按 UTC 自然整刻 15m 桶只聚合有效分钟（部分桶用有效分钟、收盘取最后一根有效分钟；整桶停机时不计 TR、ATR 沿用上一值），前 14 个有效 TR 的均值初始化，触发 K 线 T 只读在 T + 1min 之前收盘的桶，IS 内首值在 2020-01-01 03:30 可用。基准成交量为 [T − 299min, T + 1min) 内 20 个不重叠 15 分钟窗口成交量和的中位数。W 只看本版本自身阈值口径、信号所依据的脉冲方向的指标。A 组箱体只能经 `features.s6_box_ok` 以冻结口径 `S6_reentry_above_poc` 构造。资格排除同时给出逐项计数与按计划第 2.4 节顺序的互斥计数，后者满足“窗口内自身方向触发分钟数 = 入场信号数 + 各项互斥排除之和”，每次构造版本输入时核对。

## 4. 测试结果

`python -m pytest -q test_impulse_cp2.py`：在 `58117a6` 的代码上 99 个测试全部通过（31.5 秒）。其中 97 个只用合成数据，2 个使用真实数据：`test_real_data_features_are_causal_under_prefix_truncation` 与 `test_real_data_context_is_locked_out_of_the_engine`，只做触发前特征的因果回归与锁定检查。`python -m pytest -q test_impulse_cp1.py -k "not adjudication and not download"`：20 个全部通过，按惯例不选需要联网下载的 2 个。在最终代码上，97 个合成测试按文件逆序、用冷的 numba 缓存运行时全部通过（10.7 秒），禁用 numba、走纯 Python 路径时也全部通过（63.8 秒）。第三轮测试审查在第二轮修正后的版本上，还把第二轮新增或改动的每个测试单独运行了一遍，各用一份冷缓存，结果都通过。

有一个测试（`test_unlocked_entry_points_run_the_real_input_checks`）在测试进程内用 monkeypatch 临时把锁置为真，用来核对解锁后的两个入口确实先做输入完整性检查。它用的是带 `'binance_is'` 标签的合成网格，在任何撮合之前就报错；提交中的锁值始终是 `False`。

期望值都在测试中按公式手算给出（价格、数量、费用、资金费与净值逐项写出），撮合只用合成数据。用户要求的 9 个测试面与对应测试如下（测试名省略前缀 `test_`；参数化测试按函数计一次）。

| 测试面 | 对应测试 |
|---|---|
| 1. 多空镜像（盈亏、滑点方向、止损触发方向、资金费符号） | `mirror_symmetry_long_vs_short_on_reflected_path`、`mirror_symmetry_with_gap_and_intrabar_stops`、`fees_and_slippage_directions_long_and_short`、`funding_sign_long_pays_short_receives`、`settlement_inside_halt_has_the_side_sign`、`short_mark_to_market_equity`、`short_gap_stop_during_holding` |
| 2. ATR_15m（自然整刻边界、前 14 个有效 TR 初始化、初始化前不交易、部分桶、整桶停机） | `atr15_initialization_and_wilder`、`atr15_reads_only_closed_natural_buckets`、`atr15_partial_bucket_uses_valid_minutes_only`、`atr15_partial_bucket_close_is_last_valid_close`、`atr15_fully_bad_bucket_holds_value_and_previous_close`、`atr15_init_skips_fully_bad_buckets`、`no_trade_before_atr15_initialized` |
| 3. S6 二段支撑与日线 as_of 边界 | `s6_requires_exit_above_poc_after_touch`、`s6_state_used_by_box_versions_and_asof_boundary`、`s6_bounce_level_and_strict_poc`、`s6_box_ok_uses_the_frozen_s6_variant`、`s6_box_ok_forwards_the_spot_warmup`；真实数据上的 `real_data_features_are_causal_under_prefix_truncation` |
| 4. W 按自身口径与方向选择、L_origin / H_origin 止损 | `w_selection_uses_own_rule_and_direction`、`sl_c_and_short_origin`、`sl_a_and_a4_flush_long`、`origin_close_term_binds`、`ledger_category_w_and_atr_are_read_at_the_trigger_minute` |
| 5. 双向版本跳过方向不明分钟 | `ambiguous_minutes_skipped_only_in_bidirectional_versions`、`exclusive_exclusion_counts_reconcile_with_signals` |
| 6. 入场开盘跳空（两边计费、离场不利滑点）与同分钟止损优先 | `gap_through_stop_at_entry_open_long_and_short`、`gap_and_intrabar_stops_during_holding`、`stop_has_priority_over_time_exit_in_last_minute_of_bar1_and_bar2`、`mae_mfe_stop_minute_counts_only_open_and_stop_price`、`equality_boundaries_of_stops_continuation_and_flags`、`ledger_times_follow_the_exit_time_definition` |
| 7. 费用与滑点；q 在 t 按 E_ref 冻结；下限不移动 SL；2.0x 上限按 E_ref；1.0x 等名义 | `quantity_frozen_at_signal_close_not_entry_open`、`min_distance_floor_limits_quantity_but_not_the_stop`、`leverage_cap_and_equal_notional`、`caps_and_notional_use_eref_not_entry_open_and_are_flagged`、`cap_and_floor_flags_are_strict_inequalities`、`second_trade_sized_from_current_equity`、`invalid_stop_side_is_dropped`、`trade_list_identical_across_costs_and_sizings_and_ledger_reconciles` |
| 8. 停机后第一个有效开盘离场；表 5b 漏记分钟标记 | `halt_during_holding_forces_exit_at_first_valid_open`、`halt_and_time_exit_conventions`、`halt_covering_t15_records_no_continuation_test`、`trade_reaching_end_of_data_raises`、`archive_gap_flag_on_holding_window`、`archive_gap_flags_boundaries`、`contamination_and_bad_entry_bar_void_the_signal` |
| 9. 冷却、单仓、窗口起点重置、IS_END − 31min 末端、前缀截断因果性 | `cooldown_and_single_position`、`window_start_reset_and_end_boundary`、`event_simulation_requires_30_minutes_after_the_event`、`prefix_truncation_engine_synthetic`；真实数据上的 `real_data_features_are_causal_under_prefix_truncation` |

其余测试覆盖量能延续判别（`continuation_rules`、`cont_records_bar1_test_outcome_and_v1_k1_independently`）、基准成交量（`baseline_window_count_and_median`、`baseline_undefined_excludes_and_is_counted`、`engine_rejects_signal_with_undefined_baseline`）、资金费的时点与次数（`funding_timing_conventions`、`funding_any_number_of_settlements_per_trade`、`settlement_enters_equity_of_the_minute_it_opens`、`funding_at_or_after_grid_end_is_not_read`、`n_funding_mapped_counts_only_settlements_on_the_grid`、`funding_last_boundary_column`）、台账与净值（`bars_held_and_minutes_held`、`zero_trade_run_returns_full_ledger_schema`、`daily_equity_is_the_last_minute_of_each_utc_day`、`daily_equity_last_minute_differs_from_min_max_first`、`ledger_column_types`、`exclusive_exclusion_order_for_each_adjacent_pair`、`n_signals_in_window_counts_only_in_window_signals`）、输入校验（`run_portfolio_and_event_simulation_reject_invalid_inputs`、`event_simulation_rejects_side_other_than_plus_or_minus_one`、`run_portfolio_rejects_version_inputs_not_aligned_with_the_grid`、`run_portfolio_rejects_inputs_built_on_another_context`、`run_portfolio_rejects_inputs_after_in_place_context_changes`、`make_context_requires_a_complete_unique_minute_grid`、`make_context_requires_a_timezone_aware_index_and_uses_utc`、`make_context_copies_box_ok`、`context_rejects_bad_funding_and_off_grid_gap_minutes`、`defensive_checks_of_private_helpers`）、numba 与纯 Python 路径逐位一致及独立事件模拟与组合逐笔一致（`numba_and_python_paths_agree`、`event_simulation_matches_portfolio_per_trade_returns`）、真实行情锁与解锁条件（`engine_refuses_real_data_in_checkpoint2`、`synthetic_label_requires_synthetic_grid`、`unlocked_real_context_requires_funding_and_frozen_gap_minutes`、`unlocked_funding_check_compares_settlement_minutes_not_only_their_count`、`funding_supplied_flag_and_grid_without_settlement_minutes`、`unlocked_entry_points_run_the_real_input_checks`，以及真实数据上的 `real_data_context_is_locked_out_of_the_engine`），和冻结标识（`frozen_identifiers_and_constants`、`frozen_versions_inheritance`）。

## 5. 真实数据上的触发前特征回归

`python tools/run_impulse_cp2_features.py` 在 `58117a6` 的干净工作树上运行 44 秒，结果文件记录了提交号，未提交文件列表为空。脚本只读取 IS 的永续 1m 与 2019 年的现货日线，不调用撮合引擎。运行前它核对两件事：27 个冻结文件的 SHA256 与计划第 13 节逐项相同；`versions.ARCHIVE_GAP_MINUTES` 与 Checkpoint 1 表 5b 相同。

主窗口共 2,017,410 个触发分钟（2020-03-01 00:00 至 2023-12-31 23:29），没有一个分钟的 ATR_15m 未就绪。ATR_15m 的首值在 2020-01-01 03:30 可用；IS 的 140,256 个自然 15m 桶中，6 个整桶停机，7 个部分停机。主窗口内 ATR_15m / C_T 的中位数为 40.4bp，5% 与 95% 分位分别为 13.6bp 与 104.7bp。S6 的 1,401 个 as_of 日中有 818 个满足二段条件，主窗口内 18.4% 的分钟处在箱体条件下。上下文标记的漏记分钟恰为表 5b 的 10 个。

计划附录 A 列出了 10 个版本的入场信号数、两种排除计数、触发类别、W 占比、按持仓日程推算的可成交笔数、止损距离与计划名义杠杆；这 20 行数字与本次结果逐字相同（以脚本逐行比对）。两种排除计数在全部 10 个版本中相同。非零的只有 A 组的箱体排除和双向版本的方向不明跳过；维护污染、ATR 未就绪、入场 K 线为 `bad`、基准成交量无定义与止损无效这五项都是 0。

前缀截断核验在 2020-03-12 13:37、2021-06-26 13:37、2022-11-09 00:00、2023-12-31 23:59 四个截断点上进行：ATR_15m、基准成交量、S6 箱体条件，以及 10 个版本的入场方向、W、触发类别与止损价都逐位不变。

## 6. 耗时基准

`python tools/bench_impulse_engine.py` 在 `74522d4` 的干净工作树上运行，环境为 4 核 x86_64、Python 3.13.18、numba 开启，磁盘上的 numba 缓存已经预热。合成网格为 2,103,840 根 1m，分钟网格、主窗口与末端几何都与 IS 相同，含 3 段停机与 8 小时一次的资金费。各步耗时：生成合成网格 2.7 秒；构造上下文 2.4 秒；构造 10 个版本输入 5.2 秒；单次组合回测 0.10–0.18 秒；10 个版本 × 4 种成本 × 2 种仓位共 80 次组合回测 9.0 秒；1,680,900 个匹配随机独立事件 2.2 秒。缓存是冷的时候，首次调用另需约 2 秒编译（第二轮代码审查测得）。

第三轮测得两项新增检查的开销：版本输入的核对含内容摘要，每次约 6 毫秒；解锁后的资金费核对每次约 13 毫秒。两者相对单次回测的耗时都很小。基准只用合成数据，不能代表第 3 阶段统计部分的耗时；野自助法按 99,999 × G 生成权重，尚未在真实规模上计时。

## 7. 独立复核与修正

本阶段做了三轮独立复核，都只使用合成数据与 Checkpoint 1、2 的触发前汇总文件。盲重算者只读计划、不读本包代码；每条非细节发现都再交给另一位以“证伪”为目标的复核者独立复现，下面的数字都是复现之后的结果。

第一轮（针对 `ef4fa32`）。五项工作并行：只读计划、不读本包代码的盲重算引擎（约 1,096 次合成组合回测、约 17.3 万笔交易）与盲重算特征，规格符合性审查（其中按裁定原文写的参考实现在 360 次合成组合回测上与引擎的逐笔相对差 ≤ 4e-14），对抗性代码审查，以及统计设计审查。共报告 60 条发现，其中 36 条非细节发现经复现：20 条完全成立、16 条部分成立、没有被推翻的；复现后定级为 5 条重要、20 条次要、11 条细节，另有 24 条细节未单独复现。5 条重要发现都在统计设计上：夏普门禁与判定 (c)、(d) 原用 10 天块的平稳自助法，与“按月聚类为主口径”不一致，且在存在月度效应时明显偏松（两条发现）；按月聚类百分位 p 值在 A 组的月度集中度下偏松，在 Holm 第一步约为名义水平的 2–3 倍；统计与抽样代码不在冻结范围之内，第 12 节的兜底规则覆盖不到；判定 (a) 第 2 条的结论标签可能与事实不符，两个配对方向相反时也会被报为正贡献。代码方面的发现都是次要的，包括：一笔交易内资金费最多只收 8 次、超出部分静默丢弃（IS 的 8 小时结算间隔下不会触发，但规则错误）；`simulate_events` 不拒绝入场 K 线为停机占位、维护污染或在窗口之外的事件；停机延续到数据末端时引擎会编造一个离场。变异测试显示 16 个违规代码改动（例如把数量改按初始资金计算、基准窗口错位一分钟、去掉起点极值中的收盘项、去掉空头持仓中的跳空止损）不会使任何测试失败。修正在 `6096f3e` 与 `eedb4fe`：引擎取消结算次数上限，数据末端改为报错，补齐输入校验、空台账结构、`cap_bound` / `floor_bound`、v1 与 k1 分开记录、`minutes_held`；上下文改为不可重新赋值并核对合成标记、资金费与漏记分钟；A 组箱体口径与 10 个漏记分钟并入冻结物（frozen_key 因此改变）；资格排除增加互斥计数；新增 27 个测试；计划按第 8 节所列修订（第 0 节第 6 项）改写。

第二轮（针对 `eedb4fe`）。五项工作：变异测试（89 个变异：源自第一轮的 37 个、针对第一轮修正的 42 个与另加的 10 个；结果 73 个被杀死、16 个存活，第一轮存活的 16 个全部被杀死，新存活的 16 个中 3 个是经差分运行确认的等价变异，13 个对应 12 处测试缺口）；按修订后的计划重写的盲引擎与引擎逐位比较；两位互不相知的实现者各自按计划第 8 节实现统计并逐位比较；计划审查；修正部分的代码审查。共报告 57 条发现，30 条非细节发现经复现：25 条完全成立、5 条部分成立、没有被推翻的；复现后定级为 21 条次要、9 条细节，没有重要发现；另有 27 条细节。没有发现撮合或记账错误：盲引擎在 24 个随机网格的 1,920 次组合回测（352,680 笔交易、5,300 万个净值分钟、194 万个独立事件）与 IS 分钟几何上的 80 次回测（866,840 笔）上与引擎逐位一致；修正前后的引擎在资金费上限不起作用的输入上逐位相同。两份统计实现在 96 个输出量中有 91 个逐位一致，其余 5 个是报告均值的最后一位（求和顺序不同，相对差 ≤ 7.9e-16），所有门禁 p 值、抽取哈希与放宽计数相同。第二轮的发现集中在三类。一是输入校验的缺口：版本输入的数组长度短于网格时，numba 内核会越界读写（可能得到无意义的成交或使进程崩溃）；在另一个窗口或另一组触发条件上构造的版本输入会被静默接受；解锁后的资金费检查接受空的或不完整的序列；不完整或有重复分钟的网格会被接受。二是 13 个存活变异对应的测试缺口（如空头在停机中的结算符号、事件方向检查、基准未定义检查、事件 30 分钟尾部检查、`s6_box_ok` 的口径接线、台账时间、日净值、台账的类别与 ATR 列）。三是计划文字：“主窗口”的两种读法会改变 a_all 与 G3 的输入，统计量无定义时的 p 值规则、放宽计数口径、§9.3 的“当且仅当”句、B 组附注与越界核对相冲突、既有统计模块没有按 SHA256 冻结、种子表缺行，以及 B = 9,999 时 Holm 第一步附近约 20% 的蒙特卡洛相对误差。

第二轮之后的修正（与第三轮的修正一起在 `58117a6` 提交）：`run_portfolio` 要求版本输入由同一个上下文对象构造、各数组一维且与网格等长（代码审查建议用内容哈希绑定，这里先改用对象同一性，任何其他上下文对象都会报错而不是静默运行；第三轮指出它发现不了就地改写之后，又加上了内容摘要核对）；解锁后的资金费检查改为恰在网格内全部 UTC 00 / 08 / 16 时整有结算；`make_context` 拒绝不完整、有重复或乱序的分钟网格与长度不符的 `box_ok`；新增与改写 21 个测试；耗时基准的候选池加上 ATR > 0；冻结代码清单扩大到 `impulse_study` 与第 8 节所用统计模块的全部仓库内导入文件以及 `test_impulse_cp2.py`；计划按第 8 节所列修订改写约 40 处。随后对第二轮修正后的工作树重跑全部 140 个变异（第二轮的 89 个、代码审查的 41 个、针对本轮修正的 10 个）：136 个被杀死；2 个存活，分别是 `cont` 列的整数类型与 `funding_supplied` 标志，二者都没有被测试钉住；另有 2 个因针对的代码已被替换而失效，替代它们的变异全部被杀死。

第三轮（针对第二轮修正后的工作树）。三项审查并行。代码审查在 IS 规模的合成数据上比较了 `eedb4fe` 与修正后代码的 408 个输出哈希，全部相同，并比较了 numba 与纯 Python 路径。测试审查逐个手算新测试的期望值，并抽查变异。计划审查核对第二轮的全部计划修改、全文一致性、种子表与裁定符合性。共报告 27 条发现，8 条非细节发现经复现全部成立，都是次要，另有 19 条细节。主要内容如下：对象同一性检查发现不了构造版本输入之后对同一个上下文的就地改写；资金费核对的测试只区分结算次数、不区分结算分钟；没有测试证明解锁后的两个入口确实先做输入完整性检查；`s6_box_ok` 对现货日线预热的转发没有被测试钉住；计划第 11 节第 2 项把 t+30 末分钟止损的 `cont` 写错；第 0 节第 6 项的修订清单不全。这些已在 `58117a6` 中全部处理：引擎在对象同一性之外再核对交易窗口、触发条件与箱体的内容摘要（`features.input_key`，IS 规模每次约 6 毫秒），计划禁止第 3、4 阶段就地改写已构造的上下文。按细节建议，另要求网格索引带时区并统一为 UTC，复制 `box_ok`，并把 `impulse_study/qa.py` 纳入冻结清单。本轮共新增 11 个测试。之后对本轮修正与此前存活的两个变异再做变异测试，20 个变异全部被杀死。

## 8. 复核后的方法修订（须在 Checkpoint 2 审批中确认）

下列修订都在复核之后、任何真实逐笔结果出现之前做出，依据只有合成数据上的模拟（月度笔数取自 Checkpoint 1 表 12 与附录 A 的触发前计数）与对计划文字的审查。计划第 0 节第 6 项逐项列出了它们，并写明：任一项未获确认时第 3 阶段不得开始，该项按用户的裁定改写计划并重新冻结。第 1、2、3 项改变了统计判定方法，其中第 1 项偏离裁定的字面表述，需要用户逐项表态；第 13、14 项只是判定文字与定义的补充。

1. 门禁 p 值（计划第 8.2 节）。裁定是“统计检验主口径维持按月聚类 95% 置信区间”。复核发现，在 A 组有效月份数约 11 的月度集中度下，按月聚类百分位区间系统性偏窄：在名义单侧 2.5% 下，以“区间不含 0”判定的实际拒绝率在前两轮复核的多种合成设定中约为 3.1%–5.3%，在 Holm 第一步（0.0025）约为名义水平的 2–3 倍。计划因此改为以同样按月聚类的受约束野自助 t 检验（Webb 六点权重）的 p 值作 G2、G3 与 F3 的门禁，同样设定下的实际水平约为 2.0%–3.4%；百分位 95% 区间照常报告，但不作门禁。第二轮还发现两者的不一致基本是单向的（区间不含 0 而门禁不通过），多数在边界附近但不限于边界附近。这是对主口径实现方式的改动：仍按月聚类，但门禁不再直接读区间。若用户坚持以区间作门禁，可以改回，但须接受上述偏松的水平。重抽样次数 B 从 9,999 提高到 99,999，因为 Holm 第一步阈值附近的蒙特卡洛标准误在 9,999 次时约为 0.0005（相对 20%），99,999 次时约为 0.00016。
2. 夏普的区间与 p 值（第 8.4 节）。原计划用平均块长 10 天的平稳自助法；在合成模拟中它在没有月度效应时约为 2.8%，随月度效应增大升至约 5%–11%，且与“按月聚类为主口径”不一致。改为按自然月块重抽样（46 个月，5,000 次），实际水平约 3.5%–4.7%，仍略偏松，第 4 阶段的结论须注明；平稳自助法改为辅助。
3. F5 的水平（第 8.5、9.5 节）。唯一胜出的候选是按夏普点估计选出的，原规则在选出方向上做单侧 0.025，使两个真实夏普相同的版本被错判出胜者的比例约为 6.2%–7.5%；改为双侧 0.025 后约为 3.7%–4.4%（按月块重抽样本身偏松，检验准确时约为 2.5%）。
4. 判定 (a)（第 9.2 节）。新增第一条“两个配对方向相反，箱体效果依赖止损口径”；第三条改为对每个正方向显著的配对逐项写明未满足的条件，不再在箱体版本通过 G2 时误写“箱体版本本身不能覆盖 17bp 摩擦”。
5. 判定 (b)（第 9.3 节）。只有 B5 满足 G1–G3 时，B5 还须通过 G8z（因果 σ_ref 敏感性下 net_bps 均值 > 0）结论才成立；原“当且仅当”句与这一例外矛盾，已改写为单一条件。
6. 退化与平局规则（第 8.7 节）。含交易的自然月少于 12 个时 p 值记为 1、门禁不通过；se、se_D 为 0 或配对中一方没有交易时 p 值记为 1 并照常进入 Holm 校正；以及夏普并列与 G7 的平局规则。
7. 匹配随机的细则（第 8.3 节）。ATR 档的分层（a_all 为第 1.3 节主窗口内全部非 `bad` 且 ATR 已就绪分钟）、候选池（ATR_15m > 0，Z 口径版本另要求 σ7d 有效）、逐笔抽样顺序、四级放宽与六类互斥计数、缺失处理、随机入场止损距离 d_i 的计算、随机化零分布与平衡诊断。
8. 种子表（第 8.6 节）。新增按日聚类、图 4 对照图与 B 组附注野自助 p 值的种子行；辅助平稳自助法用主检验种子 + 5000；第 4 阶段敏感性用 SEED_BASE + 10000·k + 主检验偏移。全部偏移互不相同（第三轮复核另行枚举核对：2,226 至 3,598 个偏移，视规则的宽窄读法而定，均无重复，也不与两个辅助函数的默认种子重合）。
9. 第 4 阶段清单（第 11 节第 1、2、3、5、8、9 项）。新增两条对账恒等式；路径歧义一项写明 t+15 与 t+30 末分钟止损的 `cont` 取值；剔除数据缺陷交易后报告的统计量；G8z 的因果 σ_ref 定义为每个 UTC 日开始时的扩展中位数；第 8 项以平稳自助法而不是按日聚类作为 G4、F4、F5 的辅助（裁定要求“登记按日聚类置信区间作为辅助报告”，逐笔检验照做；以日收益为单位的检验，独立同分布的按日重抽样会忽略日收益的序列相关，故以 10 天块的平稳自助法代替，这一替代需要确认）；附注的组合指标、聚类月份与持仓时间分母。
10. 第 3 阶段的提交顺序与变更规则（第 10、12 节）。代码分两个提交（先只用合成数据测试的统计、图表与第 4 阶段敏感性代码，再单独翻转锁），运行产出另成第三个提交；任何代码与计划文字不一致或运行报错都须停止并报用户批准；解锁之后只允许第 11 节第 10 项的盲重算与用户在 Checkpoint 3 明确要求、标注为“事后追加（用户要求）”的分析。
11. 冻结标识（第 13 节）。frozen_key 由 `0b62edf856faafbc` 变为 `fc0d37b9d198ff85`，因为 A 组箱体口径与 10 个漏记分钟并入了冻结物；10 个版本键不变。冻结代码按 SHA256 登记，范围是 `impulse_study` 包内全部模块、它们与第 8 节所用统计模块在 Python 中实际导入的全部仓库内文件，以及 `test_impulse_cp2.py`，共 27 个文件。
12. 撮合、记账、输入校验与越界核对约定（第 5 节、第 8.1 节）。停机标签优先、数据末端报错、资金费不设次数上限、标志的严格不等式、`cont` 的取值、台账诊断列、输入校验（含版本输入与上下文的绑定）、第 3、4 阶段不得就地改写已构造的上下文，以及 B 组附注的越界范围与持仓时间占比的分母。
13. 判定文字的补充（第 9.4、9.5 节）。“显著为正”的定义；F4 的双侧 0.05 有意保留的说明；“成本敏感的候选”标注不论 P 是否为空都适用且 A5、B5 另须通过 G8z；A 组与 B 组版本之间的比较须同时报告逐笔 net_bps、net_R 与成交笔数。
14. 只作定义澄清、报告要求或冻结范围说明、不改变任何门禁或判定的补充，逐项列在计划第 0 节第 6 项第（14）条。

## 9. 仍未解决的阻塞项与待决事项

没有阻塞第 2 阶段交付物的未解决缺陷。以下事项需要用户在审批中决定或知悉。

1. 第 8 节的 14 项复核后修订须逐项确认（第 13、14 项只是文字补充）；其中第 1 项（门禁改用野自助 p 值、百分位区间只作报告）偏离裁定的字面表述，第 9 项（以日收益为单位的检验以平稳自助法而不是按日聚类作辅助）是对裁定“登记按日聚类置信区间作为辅助报告”的替代。若不批准，计划须改回相应条文后重新冻结，再进入第 3 阶段。
2. G6 要求 harsh_34bp 下 net_bps 均值的点估计 > 0，相当于逐笔毛收益约超过 34bp 加资金费，是有意从严的门槛；它决定“成本敏感的候选”与可申请版本的界线，请确认这一严格程度。
3. 资金费用 1m 成交价近似交易所实际使用的标记价格（计划第 5.6、5.9 节）。结算间隔为 8 小时，持仓一般不超过 30 分钟，单笔至多遇到一次结算（停机延长持仓时除外）；两种价格的差别只进入结算金额中的价格因子，约为 q·r·|P_成交 − P_标记|，量级远小于单笔费用，但未经真实数据核对。
4. 分钟内次序只做一个假设：合成棒末分钟内盘中触及止损时按止损成交（保守最坏路径）。乐观路径只作为第 4 阶段的敏感性（第 11 节第 2 项）。
5. F1、F2 的 Holm 校正把 A、B 两组的 10 个版本放在同一族内，A 组的原始 p 值会改变 B 组版本的校正倍数（计划第 9.3 节已事先接受这一点），请确认。
6. 日夏普随交易频率上升，A 组与 B 组之间以夏普作的比较（F4 的族内比较除外）须同时看逐笔 net_bps、net_R 与成交笔数；计划第 9.5 节已作此要求。
7. A 组按月份计的有效样本约 11 个月；即使改用野自助 t 检验与按月块重抽样，后者在合成模拟中仍略偏松（名义单侧 2.5% 下约 3.5%–4.7%），靠近阈值的结论须结合这一点解读。
8. 第 3 阶段的统计代码、图表代码与第 4 阶段敏感性代码按设计尚未编写；按计划第 10 节，它们在第 3 阶段第一个提交中实现，只用合成数据测试，之后才单独提交解锁。
9. 解锁之后，两个只适用于第 2 阶段的锁定测试会自动跳过（计划第 10 节），其余测试须照常通过。
10. 第 3 阶段的运行脚本须注意两点实现约束：版本输入只能与构造它的那个上下文对象一起撮合，构造之后不得就地改写该上下文；B 组附注、G8z 与各箱体敏感性都须在各自新建的上下文上重建版本输入（若用多进程，上下文与版本输入须一起传递）；资金费的结算时间须先取整到分钟再传入 `make_context`，引擎在解锁后核对结算恰在网格内全部 UTC 00 / 08 / 16 时整。

## 10. 复现

在仓库根目录、干净工作树上：

```bash
python -m pytest -q test_impulse_cp2.py
```

```bash
python -m pytest -q test_impulse_cp1.py -k "not adjudication and not download"
```

```bash
python tools/run_impulse_cp2_features.py
```

```bash
python tools/bench_impulse_engine.py
```

真实数据上的两项测试与特征脚本需要 1m 缓存（`python -m impulse_study.data_1m --rebuild` 按 manifest 锁定的哈希重建）；缓存不存在时这两项测试自动跳过，其余测试只用合成数据。特征脚本只读取 IS 的永续 1m 与 2019 年的现货日线，不调用撮合引擎，并把所用代码的 SHA256 写入结果文件，可与计划第 13 节逐项比对。耗时基准只用合成数据，绝对耗时随机器不同而变化。
