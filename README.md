# ppvp_quant

4H 枢轴点成交量分布（Pivot Points Volume Profile, PPVP）箱体识别 + 1H 箱体底部二次确认看涨 K 线做多策略的研究、回测与夏普优化系统。标的为 Binance USDⓈ-M 永续 `BTCUSDT`（主研究）与 `ETHUSDT`（跨品种零改参验证）。数据区间为 2020-01-01 至 2026-11-01（UTC）。研究区间（IS 2020–2023、VAL 2024-01-01 至 2025-06-30）止于 2025-07-01，此后至 2026-11-01 为 OOS，在参数锁定前封存。

## 阶段记录

每个阶段完成后，代码与阶段记录会一并提交，便于逐阶段回溯审查。

| 阶段 | 内容 | 记录 |
|---|---|---|
| Phase 0 | 数据工程：官方归档下载、SHA256 校验、清洗补缺、交叉校验、Parquet 缓存 | [docs/phase_logs/PHASE0_data_engineering.md](docs/phase_logs/PHASE0_data_engineering.md) |
| Phase 1 | PPVP 引擎、1H 状态机、事件驱动回测器、日度指标、OOS 锁；防未来函数闸门（BTC/ETH 截断逐位比对、执行不变量、切换点内核截断、突变测试）与四轮对抗式审计修复 | [docs/phase_logs/PHASE1_engine.md](docs/phase_logs/PHASE1_engine.md) |
| Phase 2 | BTC 基线回测（IS / VAL）、成本对照、随机入场零假设、IS 亏损与回撤归因、Phase 3 假设清单；ETH 留出锁 | [docs/phase_logs/PHASE2_baseline.md](docs/phase_logs/PHASE2_baseline.md) |

## 快速开始

```bash
pip install -r requirements.txt
```

```bash
python data_loader.py --verify
```

```bash
python -m pytest test_lookahead.py test_audit_regressions.py test_attribution.py -v -s
```

```bash
python tools/run_phase2.py
```

```bash
python tools/mutation_test.py
```

研究阶段请只通过 `pipeline.run()` 运行回测：它默认把数据截断在 2025-07-01 之前。ETH 是跨品种零改参验证集（`config.HOLDOUT_SYMBOLS`），参数锁定前 `pipeline.run` 只允许它做 `segments=()` 的结构性运行：只返回箱体与信号（`result` 为 None），不运行回测，也不出任何分段绩效。`pipeline` 的入口与 `metrics` 的全部指标函数在未显式传入 `allow_oos=True` 时拒绝读取或统计 OOS 区间；这个参数只是约定，只应在 Phase 4 参数锁定后使用。`data_loader` 的原始数据读取与引擎函数（`build_boxes`、`generate_signals`、`run_backtest`）不加锁，研究代码不得绕过 `pipeline` 直接用它们取数。
