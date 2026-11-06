# 方向二：BTC IS正式结果

结论：M和L均因样本不足优先判证据不足，同时毛/净收益与其他盈利门禁不通过；没有合格候选，停止本轮。

实现 `e9dcc33d93225af314028668ff4200ad2bf81f36` → 解锁 `66cd0b2866c603f8d88574e91b77e55f31cc2025` → 本结果提交。仅一次正式调用，六主组合＋两个限价执行压力＋83个独立父事件，无补跑。

- [完整阶段报告](../../../docs/phase_logs/DIP_LIMIT_IS_results.md)
- [八组合核心表](comparison.csv)、[年度/半段与风险月](periods.csv)、[完整统计/门禁](summary.json)
- [事件归因](execution_attribution.json)、[执行风险与成交集合差异](execution_disclosures.json)
- [图1净值和回撤](figure_1_equity_drawdown.png)、[图2成交及机会成本](figure_2_fills_opportunity_cost.png)、[图3成本和执行压力](figure_3_cost_execution_stress.png)
- [原始运行输出SHA256](provenance/original_output_hashes.json)、[输入核验](provenance/preflight.json)、[启动与退出记录](provenance/formal_attempt.json)

每个组合都有trades/orders/decisions CSV.gz、分钟净值parquet、日净值CSV和metadata。父信号/小时特征/高周期引用快照及独立事件账本共同保留。日历范围严格为BTC IS，未读VAL/OOS/ETH。
