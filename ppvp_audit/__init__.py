"""
ppvp_audit — 基线审计与有限机制实验轮次（BTC IS only）的研究代码。

本轮的研究边界（用户指令，见 reports/audit_round/audit_report.md 第 1 节）：
* 只研究 BTCUSDT 永续的 IS 区间 [2020-01-01 00:00 UTC, 2024-01-01 00:00 UTC)。
* 不运行、不查看、不据以选择任何 VAL、OOS 或 ETH 结果。所有入口都经 ``common.is_run`` /
  ``common.load_is_market`` 读取数据，数据在送入引擎之前就截断在 IS_END（2024-01-01）之前，
  VAL / OOS 的 K 线根本不进入计算；每个入口另有断言，任何对象延伸到 IS_END 之后都直接报错。
* 指标预热只使用当时已存在的历史数据：日线 SMA 等慢速价格指标在 2020 年之前用 Binance 现货日线收盘预热
  （仅价格，不含成交量），成交量分布只用永续成交量。
"""
