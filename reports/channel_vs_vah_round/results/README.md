# BTC IS 六组合结果

一次正式运行成功；**P_channel 证据不足，V_vah 非候选，无合格相对赢家，按预注册停止本轮假设**。主成本总收益分别 +2.616% / +10.268%，相对差 Holm p=0.68926，不能认定 VAH 增量成立。

- [阶段报告与所有门禁](../../../docs/phase_logs/BREAKOUT_IS_results.md)
- [六组合对比](comparison.csv)、[年度/半段/事件月](periods.csv)、[完整统计与输入记录](summary.json)
- [净值与回撤](figure_1_equity_drawdown.png)、[年度净期望与成交数](figure_2_years.png)、[成本对比](figure_3_costs.png)
- 每个组合的 `trades.csv.gz`、`decisions.csv.gz`、`stop_updates.csv.gz`、`minute_equity.parquet`、`daily.csv`、`metadata.json` 均为原始运行输出；另附两组 `signals.csv.gz`、共同特征与参考窗口快照。
- [原始输出 SHA256](provenance/original_output_hashes.json)、[正式单次运行状态](provenance/formal_run_state.json)、[输入预检](provenance/preflight.json)、[解锁提交](provenance/unlock_commit.json)、[运行日志](run.log)。

运行基线：`570da8356464af5ae678cd8a5e5e6e492e1e4f2d`；干净启动的解锁提交：`5f7b2e6fe24aa41b64035b98f07c9116341fd035`。旧登记中“仅实现、未批准真实数据”的字段作为历史原文保留，本次正式权限来自后续用户批准和登记解锁提交。执行 metadata 的 `candidate_eligible` 不代表最终门禁通过，`n_signals` 是输入小时行数。

本次只读取 BTC 永续 IS，未读取现货、VAL、OOS、ETH。没有额外回测、重采样、调参或独立验证。
