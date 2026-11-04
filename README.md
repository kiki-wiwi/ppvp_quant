# ppvp_quant

4H 枢轴点成交量分布（Pivot Points Volume Profile, PPVP）箱体识别 + 1H 箱体底部二次确认看涨 K 线做多策略的研究、回测与夏普优化系统。标的为 Binance USDⓈ-M 永续 `BTCUSDT`（主研究）与 `ETHUSDT`（跨品种零改参验证），研究区间 2020-01-01 至 2026-11-01（UTC）。

## 阶段记录

每个阶段完成后，代码与阶段记录会一并提交，便于逐阶段回溯审查。

| 阶段 | 内容 | 记录 |
|---|---|---|
| Phase 0 | 数据工程：官方归档下载、SHA256 校验、清洗补缺、交叉校验、Parquet 缓存 | [docs/phase_logs/PHASE0_data_engineering.md](docs/phase_logs/PHASE0_data_engineering.md) |
| Phase 1 | PPVP 引擎、1H 状态机、事件驱动回测器、日度指标、OOS 锁；防未来函数闸门（BTC/ETH 截断逐位比对、突变测试）与对抗式审计修复 | [docs/phase_logs/PHASE1_engine.md](docs/phase_logs/PHASE1_engine.md) |

## 快速开始

```bash
pip install -r requirements.txt
```

```bash
python data_loader.py --verify
```

```bash
python -m pytest test_lookahead.py test_audit_regressions.py -v -s
```

研究阶段请只通过 `pipeline.run()` 运行回测：它默认把数据截断在 2025-07-01 之前，OOS 区间在参数锁定（Phase 4）之前不可读取。
