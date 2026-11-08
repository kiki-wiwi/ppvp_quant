# 研究地图与历史证据边界

本文件供只能访问 GitHub 的外部顾问逐轮审查。研究终点固定为 [21eee8746452d5d46b78f5ef8c50c024ebd21779](https://github.com/kiki-wiwi/ppvp_quant/tree/21eee8746452d5d46b78f5ef8c50c024ebd21779)，分支为 `codex/trend-hold-venue-oos-study`。本次新增审计导航与原件副本不改变此前的策略、输入、统计和结果。本文只整理已有证据，不报告新回测或新统计结果。

相对链接指向本仓库现有文件。审查历史状态时，应结合下方提交链接查看当时版本；旧 README、登记里的锁状态和未授权字段属于对应阶段的原文，不能单独代替后续解锁提交。

## 1. 建议阅读顺序

1. 先读各轮最终阶段报告，确认研究问题、数据窗口、比较对象和失败门禁。
2. 再读该轮预先登记的计划、实现说明、范围锁与统计代码，区分事前规定和结果后的描述。
3. 按“实现／登记 → 独立解锁 → 结果归档”的提交顺序检查差异；发生输入修订的轮次还需读完整修订链。
4. 把报告数字追溯到已保存的比较表、账户账本、统计元数据与哈希清单。哈希一致只证明对应字节一致，不单独证明策略或统计方法正确。
5. 最后核对本文列出的历史缺档和解释边界。数据取得、LFS、环境与字节核验的操作说明由本审计目录其他指南提供；本文不提供未经验证的重跑命令。

## 2. 全部研究入口

### Phase 0–2：初始 PPVP 策略、引擎与基线

- 阶段报告：[数据工程](../../docs/phase_logs/PHASE0_data_engineering.md)、[引擎与防未来函数审查](../../docs/phase_logs/PHASE1_engine.md)、[BTC 基线与后续判定规则](../../docs/phase_logs/PHASE2_baseline.md)。
- 代码：[PPVP 引擎](../../ppvp_engine.py)、[状态机](../../strategy_state_machine.py)、[回测器](../../backtester.py)、[研究入口](../../pipeline.py)、[指标](../../metrics.py)、[归因](../../attribution.py)、[数据加载](../../data_loader.py)。
- 证据：[数据 manifest](../../data/manifest.json)、[基线 JSON](../phase2_baseline.json)、[表格](../phase2_tables.md)、[74 个 IS 复核变体结果](../phase2_review_checks.json)、[复核脚本](../../tools/phase2_review_checks.py)。
- 顾问重点：4H 箱体与 1H 决策的可用时点；同根止损/止盈路径；资金费、成本与状态机边界；ETH 留出及 OOS 限制；Phase 2 在同一 IS 上已经查看的变体范围。74 个变体是历史探索暴露，不能当作独立确认性证据。

### 原 PPVP 审计轮：B_original、B_audited 与九版本实验

- 主文档：[实现审计](../audit_round/audit_report.md)、[决策报告](../audit_round/decision_report.md)、[预登记实验计划](../audit_round/experiment_plan.md)、[后续失败诊断](../audit_round/codex_gpt61sol_high_diagnosis.md)。
- 代码：[ppvp_audit](../../ppvp_audit/)。证据：[冻结基线](../audit_round/baseline_manifest.json)、[实验结果](../audit_round/experiments/results.json)、[实验账本](../audit_round/experiment_ledger.csv)、[审计原脚本与输出说明](../audit_round/audit_evidence/README.md)。
- 顾问重点：B_original 与 B_audited 的规则差异；研究从 1m 改为 1H 最坏路径的授权与影响；九版本事前计划是否先于运行；成交机会、实际交易、随机对照与组合收益的分母是否一致。该轮审计证据不等于更早 Phase 1 四轮审查的全部原始现场。

### Impulse vs Box：CP1 数据、CP2 引擎、CP3 正式结果

- 报告：[CP1 数据](../../docs/phase_logs/IMPULSE_CP1_data.md)、[CP2 引擎与计划](../../docs/phase_logs/IMPULSE_CP2_engine_and_plan.md)、[CP3 结果](../../docs/phase_logs/IMPULSE_CP3_results.md)。
- 规格：[experiment_plan_impulse.md](../impulse_vs_box_round/experiment_plan_impulse.md)。代码：[impulse_study](../../impulse_study/)。
- 证据：[1m manifest](../../data/manifest_impulse_1m.json)、[CP1 输出](../impulse_vs_box_round/cp1/)、[CP2 特征与基准输出](../impulse_vs_box_round/cp2/)、[CP3 原始结果](../impulse_vs_box_round/cp3/)、[80 主组合比较表](../impulse_vs_box_round/cp3/all80_comparison.csv)。CP3 的登记 JSON 还应查看 [a89814f 提交说明](https://github.com/kiki-wiwi/ppvp_quant/commit/a89814f)。
- 新增原件副本：[停止记录](evidence/impulse/STOPPED.json)、[控制台](evidence/impulse/run.txt)、[启动记录](evidence/impulse/run_start.json)、[原路径与哈希映射](evidence/impulse/source_map.json)。这三件记录的是成功撮合之前的输入核验停机，不是成功运行的统计输出。
- 顾问重点：触发频率与策略收益的区分；80 主组合、匹配随机、B 组附注与 G8z 的统计角色；pandas 版本导致内容哈希差异时是否先停机；重启前是否保持冻结代码和 manifest。CP4 代码存在不表示 CP4 终审已执行。

### Channel vs VAH：普通通道与滚动 VAH

- 报告：[正式结果](../../docs/phase_logs/BREAKOUT_IS_results.md)、[实现与合成验证](../../docs/phase_logs/BREAKOUT_implementation_and_synthetic.md)。
- 规格：[approved_plan.md](../channel_vs_vah_round/approved_plan.md)。代码：[breakout_study](../../breakout_study/)。
- 证据：[结果索引](../channel_vs_vah_round/results/README.md)、[注册](../channel_vs_vah_round/registration.json)、[实现哈希](../channel_vs_vah_round/implementation_hashes.json)、[运行凭证](../channel_vs_vah_round/results/provenance/)。
- 顾问重点：两套信号的共同执行条件、阈值形成时点、六组合和冻结检验族；`metadata` 的 `candidate_eligible` 与最终门禁的区别；`n_signals` 字段是输入小时行数。旧登记里的 `real_data_authorized=false` 是实现阶段原文。

### Trend Dip Limit：趋势内回撤限价与市价对照

- 报告：[正式结果](../../docs/phase_logs/DIP_LIMIT_IS_results.md)、[实现与合成验证](../../docs/phase_logs/DIP_LIMIT_implementation_and_synthetic.md)。
- 规格：[approved_plan.md](../trend_dip_limit_round/approved_plan.md)、[implementation_plan.md](../trend_dip_limit_round/implementation_plan.md)。代码：[dip_limit_study](../../dip_limit_study/)。
- 证据：[结果索引](../trend_dip_limit_round/results/README.md)、[注册](../trend_dip_limit_round/registration.json)、[实现哈希](../trend_dip_limit_round/implementation_hashes.json)、[运行凭证](../trend_dip_limit_round/results/provenance/)、[交付哈希](../trend_dip_limit_round/results/delivery_hashes.json)。
- 顾问重点：父信号与实际成交事件的对应、限价未成交、六主组合与两个压力组合、成交选择和机会成本；事件统计与日历账户统计不可互换。正式阶段部分旧锁断言被排除的原因应与阶段报告核对。

### Channel vs Matched Random：无过滤通道与匹配随机

- 报告：[正式结果](../../docs/phase_logs/CHANNEL_RANDOM_results.md)、[实现与合成验证](../../docs/phase_logs/CHANNEL_RANDOM_implementation_and_synthetic.md)。
- 规格：[approved_plan.md](../channel_random_round/approved_plan.md)。代码：[channel_random_study](../../channel_random_study/)。
- 证据：[注册](../channel_random_round/registration.json)、[环境登记](../channel_random_round/environment.json)、[结果目录](../channel_random_round/results/)、[原始输出哈希](../channel_random_round/results/output_hashes.json)、[运行凭证](../channel_random_round/results/provenance/)。
- 顾问重点：一次生成的 200 条随机时间表、分层饱和、父机会与真实成交数不同的处理；Rbar 同日日收益平均并非真实可交易账户，也不同于随机中位路径；有限随机复制误差与市场不确定性不能混同。

### R5：日线趋势持有，T 对持续多头 P

- 报告：[正式结果](../../docs/phase_logs/TREND_HOLD_results.md)、[实现与合成验证](../../docs/phase_logs/TREND_HOLD_implementation_and_synthetic.md)。
- 规格：[approved_plan.md](../trend_hold_round/approved_plan.md)、[implementation_notes.md](../trend_hold_round/implementation_notes.md)。代码：[trend_hold_study](../../trend_hold_study/)。
- 证据：[注册](../trend_hold_round/registration.json)、[实现哈希](../trend_hold_round/implementation_hashes.json)、[结果目录](../trend_hold_round/results/)、[比较表](../trend_hold_round/results/comparison.csv)、[门禁](../trend_hold_round/results/gates.csv)、[统计元数据](../trend_hold_round/results/statistical_metadata.json)、[原始输出哈希](../trend_hold_round/results/output_hashes.json)。
- 新增补档：[R5 八份原件与映射](evidence/r5/)、[source_map.json](evidence/r5/source_map.json)。它们补齐旧结果报告仅以 `/tmp/trend_hold_formal__ue3kg42/` 路径和 SHA256 引用的来源预检、输入预检、干净解锁、正式进程与四次发布停机记录。
- 顾问重点：SMA60 与波动预算的因果时点、周一调整、相同事前预算与不同实际暴露、两项检验的 Holm 族、极端月份集中性。原结果 V3/V5 未通过，不能用较小回撤或后续载体收益替代冻结的相对增量门禁。

### 跨载体 CP1：冻结会计与合成验证

- 报告：[CP1 阶段日志](../../docs/phase_logs/TREND_HOLD_VENUE_CP1_implementation_and_synthetic.md)。
- 权威规格：[approved_spec.md](../trend_hold_venue_round/approved_spec.md)、[boundary_addendum.md](../trend_hold_venue_round/boundary_addendum.md)、[implementation_plan.md](../trend_hold_venue_round/implementation_plan.md)。应结合后续明确登记的修订阅读原规格。
- 代码：[trend_hold_venue_study](../../trend_hold_venue_study/)。证据：[注册](../trend_hold_venue_round/registration.json)、[实现哈希](../trend_hold_venue_round/implementation_hashes.json)、[合成验证](../trend_hold_venue_round/synthetic_validation.json)、[首次失败证据](../trend_hold_venue_round/validation_history/attempt_1_stopped/)。
- 顾问重点：U 本位、现货、币本位套保及增币账户的精确余额式；现金裁剪、同边界旧仓资金费、压力触线后残币、零下界调整、动态归因；18 组合对旧 R5 的逐分钟对照。CP1 没有真实策略收益。

### 跨载体 CP2：BTC IS，42 账户

- 报告：[CP2 正式结果](../../docs/phase_logs/TREND_HOLD_VENUE_CP2_btc_is_results.md)。
- 登记：[实现说明](../trend_hold_venue_round/cp2_registration/implementation_notes.md)、[初始登记目录](../trend_hold_venue_round/cp2_registration/)、[输入修订一](../trend_hold_venue_round/cp2_registration/input_revision_1/revision_notes.md)、[输入修订二](../trend_hold_venue_round/cp2_registration/input_revision_2/revision_notes.md)、[最终输入绑定](../trend_hold_venue_round/cp2_registration/input_revision_2/input_binding.json)。
- 代码：[数据加载](../../trend_hold_venue_study/data.py)、[资金费来源](../../trend_hold_venue_study/funding_sources.py)、[阶段控制](../../trend_hold_venue_study/cp2_control.py)、[运行编排](../../trend_hold_venue_study/runtime.py)。
- 结果：[目录](../trend_hold_venue_round/cp2_btc_is/)、[比较表](../trend_hold_venue_round/cp2_btc_is/comparison.csv)、[R5 精确对账](../trend_hold_venue_round/cp2_btc_is/r5_exact_reconciliation.json)、[交付哈希](../trend_hold_venue_round/cp2_btc_is/delivery_hashes.json)、[图注](../trend_hold_venue_round/cp2_btc_is/figure_captions.md)。
- 顾问重点：IS-Full、IS-Common、整张合约和 CM 自信号窗口不可混比；缓存文件字节与内容哈希的区别；SPOT 原生时标例外、CM 上线边界与月末资金费补齐；六组 Full UM 与 R5 完全相同的证据。所有统计族未通过全部门禁。

### 跨载体 CP3：BTC Post-IS 与 ETH，一次性描述性终局

- 报告：[CP3 终局结果](../../docs/phase_logs/TREND_HOLD_VENUE_OOS_results.md)。
- 登记：[实现及修订计划](../trend_hold_venue_round/cp3_registration/implementation_plan.md)、[输入清单](../trend_hold_venue_round/cp3_registration/venue_cp3_inputs.json)、[实现哈希](../trend_hold_venue_round/cp3_registration/implementation_hashes.json)、[合成验证及完整历史命令](../trend_hold_venue_round/cp3_registration/synthetic_validation.json)、[停止和诊断历史](../trend_hold_venue_round/cp3_registration/validation_history/)。
- 代码：[窗口适配](../../trend_hold_venue_study/cp3_windows.py)、[输入](../../trend_hold_venue_study/cp3_data.py)、[运行编排](../../trend_hold_venue_study/cp3_runtime.py)、[分析](../../trend_hold_venue_study/cp3_analysis.py)、[阶段控制](../../trend_hold_venue_study/cp3_control.py)。
- 结果：[目录](../trend_hold_venue_round/cp3_post_is_and_eth/)、[交付哈希](../trend_hold_venue_round/cp3_post_is_and_eth/delivery_hashes.json)、[执行与发布凭证](../trend_hold_venue_round/cp3_post_is_and_eth/delivery_validation/)、[图注](../trend_hold_venue_round/cp3_post_is_and_eth/figure_captions.md)。
- 顾问重点：最终 Post-IS 为 `[2024-01-01, 2026-06-01)`，VAL/OOS 在 2025-07-01 连续切分、不重置；87 账户与 21 描述性配对统计族；BTC IS 只复用 CP2 结果；ETH 原生上线边界；V4 额外抵押币风险。族内两项 Holm 不等于全部研究/全部家族的全局多重性控制。原 R5 失败没有因本阶段而自动晋级。

## 3. 关键提交链

下表链接为 Git 提交对象；短 SHA 是显示标签，不能用文件修改时间代替提交先后。输入修订是受记录的历史链的一部分，不应只展示最后一次成功。

| 阶段 | 关键提交（按顺序） |
|---|---|
| Phase 0、Phase 1、Phase 2 收尾 | [cb95e21](https://github.com/kiki-wiwi/ppvp_quant/commit/cb95e21) → [d1bc53c](https://github.com/kiki-wiwi/ppvp_quant/commit/d1bc53c) → [04724f0](https://github.com/kiki-wiwi/ppvp_quant/commit/04724f0) |
| 原审计轮 | [382bc9c 冻结](https://github.com/kiki-wiwi/ppvp_quant/commit/382bc9c) → [b871c0a 预登记](https://github.com/kiki-wiwi/ppvp_quant/commit/b871c0a) → [aa1ecb4 补充计划](https://github.com/kiki-wiwi/ppvp_quant/commit/aa1ecb4) → [88607ed 运行](https://github.com/kiki-wiwi/ppvp_quant/commit/88607ed) → [ab577ba 报告修订](https://github.com/kiki-wiwi/ppvp_quant/commit/ab577ba) → [64ca64b 诊断](https://github.com/kiki-wiwi/ppvp_quant/commit/64ca64b) → [2f66ea1 合并](https://github.com/kiki-wiwi/ppvp_quant/commit/2f66ea1) |
| Impulse CP1/CP2 | [555258a CP1](https://github.com/kiki-wiwi/ppvp_quant/commit/555258a) → [b4e3fc1 CP2](https://github.com/kiki-wiwi/ppvp_quant/commit/b4e3fc1) |
| Impulse CP3 | [d029757 实现](https://github.com/kiki-wiwi/ppvp_quant/commit/d029757) → [a89814f 登记解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/a89814f) → [cee29ef 结果](https://github.com/kiki-wiwi/ppvp_quant/commit/cee29ef) |
| Channel/VAH | [570da83 实现](https://github.com/kiki-wiwi/ppvp_quant/commit/570da83) → [5f7b2e6 解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/5f7b2e6) → [6777532 结果](https://github.com/kiki-wiwi/ppvp_quant/commit/6777532) |
| Dip Limit | [e9dcc33 实现](https://github.com/kiki-wiwi/ppvp_quant/commit/e9dcc33) → [66cd0b2 解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/66cd0b2) → [366c592 结果](https://github.com/kiki-wiwi/ppvp_quant/commit/366c592) |
| Channel/Random | [5c13826 实现](https://github.com/kiki-wiwi/ppvp_quant/commit/5c13826) → [e934a8a 解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/e934a8a) → [35490be 结果](https://github.com/kiki-wiwi/ppvp_quant/commit/35490be) |
| R5 | [cda559d 实现](https://github.com/kiki-wiwi/ppvp_quant/commit/cda559d) → [fc04caa 解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/fc04caa) → [3846971 结果](https://github.com/kiki-wiwi/ppvp_quant/commit/3846971) |
| 跨载体 CP1 | [deadf01](https://github.com/kiki-wiwi/ppvp_quant/commit/deadf01) |
| 跨载体 CP2 初次实现及输入修订一 | [4870310 实现](https://github.com/kiki-wiwi/ppvp_quant/commit/4870310) → [51a2bc3 解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/51a2bc3) → [c4c6191 修订一](https://github.com/kiki-wiwi/ppvp_quant/commit/c4c6191) → [ef0d3ef 独立解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/ef0d3ef) |
| 跨载体 CP2 输入修订二及结果 | [ef1ad9a 修订二](https://github.com/kiki-wiwi/ppvp_quant/commit/ef1ad9a) → [90b2f9d 独立解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/90b2f9d) → [f0c21ac 结果](https://github.com/kiki-wiwi/ppvp_quant/commit/f0c21ac) |
| 跨载体 CP3 | [4d3d103 实现及输入登记](https://github.com/kiki-wiwi/ppvp_quant/commit/4d3d103) → [05fe6f1 独立解锁](https://github.com/kiki-wiwi/ppvp_quant/commit/05fe6f1) → [21eee87 结果](https://github.com/kiki-wiwi/ppvp_quant/commit/21eee87) |

## 4. 已补档与仍然存在的历史缺项

| 项目 | GitHub 当前可读证据与限制 |
|---|---|
| R5 的八份外部运行/发布记录 | 本审计新增 [evidence/r5](evidence/r5/) 原字节副本；[source_map.json](evidence/r5/source_map.json) 对应旧报告公布的 SHA256，原件和副本逐项一致。旧结果目录与旧哈希清单未改。 |
| Impulse CP3 三份输入停机记录 | 本审计新增 [evidence/impulse](evidence/impulse/) 原字节副本；[source_map.json](evidence/impulse/source_map.json) 对应旧报告公布的 SHA256。它们说明成功运行前的停止与环境恢复过程。 |
| Phase 1 四轮审查原始现场 | [PHASE1_engine.md](../../docs/phase_logs/PHASE1_engine.md) 末尾明确说明：原始发现、复核者脚本、行为对照与归因脚本在会话临时目录执行，没有入库。报告保留结论，但没有可定位的精确原件路径，不能承诺从 GitHub 恢复这部分现场。 |
| Phase 2 两个撮合规则核对 p 值 | [PHASE2_baseline.md](../../docs/phase_logs/PHASE2_baseline.md) 开头及第 5 节说明相关复核脚本未入库且结果依赖随机种子。另有仅引用结论的临时规则核对脚本；已归档的 74 变体脚本不能补证这些未归档核对。 |
| 原审计轮中间文件 | [audit_evidence/README.md](../audit_round/audit_evidence/README.md) 说明只保留部分文本/表格格式，`.pkl`、`.npy` 中间产物及 numba 缓存未入库；脚本还包含当时的临时目录。此处仅披露边界，不将“可重新生成”当作已核验事实。 |
| 各轮大型合成虚构输出 | Channel/VAH、Dip Limit、Channel/Random 的阶段日志说明部分合成台账、图和补充验证产物留在临时目录。已归档测试日志及哈希不等于全部合成文件已上传；这些文件也不是正式行情收益证据。 |
| CP3 全输入可用性诊断 | [归档目录](../trend_hold_venue_round/cp3_registration/validation_history/full_input_diagnosis_20261108/) 提供 8 份摘要、表格和索引；[evidence_hashes.json](../trend_hold_venue_round/cp3_registration/validation_history/full_input_diagnosis_20261108/evidence_hashes.json) 列出 2,770 个外部文件，原始现场位于 `/mnt/ppvp-backtests/diagnostics/cp3_availability_20261108/`。原结果提交只有摘要；本次审计交付已将这 2,770 项的原字节补入 [输入分卷与路径索引](inputs/index.json)，顾问可从 GitHub 取得，详情见 [HANDOFF.md](HANDOFF.md)。 |
| CP3 其他停止现场 | [validation_history](../trend_hold_venue_round/cp3_registration/validation_history/) 已保存停止摘要、部分原始日志和哈希；[终局报告](../../docs/phase_logs/TREND_HOLD_VENUE_OOS_results.md) 仍引用专卷外部现场。不要将“保留现场”改写为“所有现场文件均在 Git”。 |
| Channel/VAH、Dip Limit、Channel/Random 正式凭证 | 必要预检、启动与控制台记录已复制至各轮 `results/provenance/`。报告同时提及外部临时副本，其存在不意味着正式凭证只能从本机访问。 |

补档是现存原件的原字节复制；没有根据结果重新编造日志，没有填补不存在的历史证据。`source_map.json` 的新归档日期不改变原件所记录的事件日期。其他新增数据交付文件的范围以本审计目录的数据指南及其清单为准。

## 5. 阶段锁与历史测试记录的正确解释

最终研究树已处于归档终止态。旧测试及运行器的前置条件包含当时的锁值、直接父提交、输出目录不存在及允许新增文件范围；它们并非设计为在终局 HEAD 上逐轮通用执行。

最直接的例子是 [tests/test_trend_venue_cp3_control.py](../../tests/test_trend_venue_cp3_control.py)：`test_current_cp3_preunlock_baseline_and_bounded_authorization` 明确要求 `BTC_IS_UNLOCKED=True`、`POST_IS_AND_ETH_UNLOCKED=False`；终局 [scope.py](../../trend_hold_venue_study/scope.py) 已按授权将两个锁都置为 True。因此 CP3 登记的“325 passed，3 deselected”属于实现提交的解锁前状态，不是对终局 HEAD 的全量 pytest 通过承诺。

同理，R5、CP1、CP2 的历史测试排除项、失败现场与锁状态必须按各自日志读取。不能为获得绿色结果而改动冻结测试、重关锁、删除输出、增加未经登记的排除项，或把预期的阶段拒绝报告成原结果损坏。

审查当前归档首先使用 Git 元数据、现存源文本、原始输出、凭证和独立字节核验。原有正式 CLI（包括阶段专用的 `--check`）不是本交付推荐的审计入口。本文没有执行任何旧测试、回测、真实特征或统计重新计算。
