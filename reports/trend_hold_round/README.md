# 日线趋势持有研究：实现与合成验证阶段

用户已批准 `approved_plan.md` 原文，并授权本阶段实现；原文草案状态行按原字节保留，当前授权以本登记与阶段日志为准。

基线 `35490be840ca49ec0d7172e4356ae556df94f640`；分支 `codex/daily-trend-hold-study`。独立模块 `trend_hold_study`，真实锁默认 False。旧研究及其已解锁标记不提供本研究权限。

- 默认 `tools/run_trend_hold_study.py --check` 仅核验源文件、登记与环境，不读行情。
- 合成验证使用 `python -m pytest -p trend_hold_study.testing_guard tests/test_trend_hold_*.py`，插件禁止市场文件和网络访问。
- `--synthetic --output <仓库外新目录>` 只能使用人工全日历数据，结果不代表真实BTC表现。
- `--real` 必须先另获用户正式运行批准，并有单行解锁提交及干净工作树；不能复用已有结果目录，不自动重试。

执行、统计、三张图和六组合输出均已实现，实际行情/输入缓存核验留待获批正式阶段。批准本实现或计划均不授权VAL/OOS/ETH、旧CP4或实盘。

完整说明见 `docs/phase_logs/TREND_HOLD_implementation_and_synthetic.md`；源文件哈希见 `implementation_hashes.json`。该清单只允许内存归一新scope的唯一解锁行，不改变其他文件字节。
