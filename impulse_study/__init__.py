"""
impulse_study — “日线箱体二段反弹 + 时间价格脉冲” vs “纯时间价格脉冲”双轨预注册研究（独立策略模块）。

本包与上一轮 PPVP 策略代码（ppvp_engine / strategy_state_machine / backtester / ppvp_audit）解耦：
只复用 data_loader 的归档下载与 SHA256 校验、ppvp_engine 的 Wilder ATR 与价值区算法、config 的样本切分常量。

范围（全部阶段）：BTCUSDT USDⓈ-M 永续，IS [2020-01-01, 2024-01-01) UTC，外加 2020 年之前的现货预热数据。
VAL、OOS 与 ETH 的任何数据都不下载、不读取（scope.py 中的断言在每个读取入口执行）。

模块
----
scope.py     研究范围常量与越界断言
data_1m.py   1m / 15m / 1d 官方归档的下载、校验、裁决、缓存与 manifest（Checkpoint 1）
qa.py        1m 数据完整性与跨周期交叉对齐核验（Checkpoint 1）
daily.py     UTC 日线、ATR_1D、滚动 60 日成交量分布与“二段支撑区”状态（因果，Checkpoint 1）
signals.py   时间价格脉冲触发器与纯频次统计（不计算任何触发后价格或盈亏，Checkpoint 1）
"""
