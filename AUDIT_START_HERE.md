# 外部独立审计：从这里开始

本仓库包含多轮量化策略研究及其失败记录。**请审计整条研究链，不只看最终净值图。** 本入口新增于 2026-11-08，研究结果锚定提交为 [`21eee8746452d5d46b78f5ef8c50c024ebd21779`](https://github.com/kiki-wiwi/ppvp_quant/tree/21eee8746452d5d46b78f5ef8c50c024ebd21779)。本次交付只补材料、导航和字节校验工具，没有重跑任何回测、检验或图表，没有修改旧代码、计划、哈希或结果。

**请使用分支 `codex/trend-hold-venue-oos-study` 或委托人提供的审计交付提交；仓库默认 `main` 和旧 README 不代表最新研究状态。**

## 建议先读

1. [完整研究地图、计划、代码和提交链](reports/external_audit_20261108/RESEARCH_MAP.md)：从原 PPVP、脉冲、突破、限价承接、匹配随机到 R5 和跨载体 CP1–3。
2. [取得数据与只读核验](reports/external_audit_20261108/HANDOFF.md)：Git LFS、输入分卷、运行环境、历史测试的正确阶段及证据限制。
3. [最终跨载体报告](docs/phase_logs/TREND_HOLD_VENUE_OOS_results.md)、[BTC IS 跨载体报告](docs/phase_logs/TREND_HOLD_VENUE_CP2_btc_is_results.md)、[原 R5 结果](docs/phase_logs/TREND_HOLD_results.md)。
4. [审计问题清单](reports/external_audit_20261108/AUDIT_BRIEF.md)：供顾问独立判断，不预设策略应当通过。

## 数据在哪里

- 原有报告、逐笔/每日/分钟账本、随机对照、统计矩阵及图片保留原位置；[快照清单](reports/external_audit_20261108/snapshot.json)覆盖 **4,930 个既有文件，20,413,969,282 字节**。
- 其中 **258 个 Git LFS 对象**是完整分钟账本和归因数据；网页上的 LFS 指针不是账本本体。
- 新增 [输入目录](reports/external_audit_20261108/inputs/)包含 **3 个 LFS tar 分卷**，共 **1,560,094,720 字节**；[索引](reports/external_audit_20261108/inputs/index.json)登记 **3,651 个去重文件对象**及其原路径、历史清单、SHA256。正式输入缓存、原始 ZIP/CHECKSUM、原始诊断证据均可从本仓库取得，无需访问本机 `/tmp` 或 `/mnt`。
- 早期 673 个已登记 ZIP 在本次按原 URL 重新取得，全部匹配历史冻结 SHA256；[恢复记录](reports/external_audit_20261108/evidence/historical_zip_recovery.json)区分此次取得时间与原研究时间。
- [审计增补交付清单](reports/external_audit_20261108/delivery.json)登记新增文件；它不取代任何旧 manifest。

## 必须保留的解释边界

最终 CP3 是**用户授权的一次性描述性样本外/跨载体评估**。原 R5 未通过 IS 的 V3/V5，后续较好的净值或回撤不能抹去这一事实。BTC/ETH Post-IS 实际截止 **2026-06-01 UTC**：VAL 18 个月、OOS 11 个月；长跨度参考净值由两段独立账户连乘，不能称为连续账户。

历史证据并非毫无缺口：早期 OKX 补洞原始响应、部分早期临时审计脚本、836 份当年 CHECKSUM 响应原文和 3 个旧 CP1 派生缓存没有保留原件。其具体影响、替代证据与可独立重建的输入见 [交付说明](reports/external_audit_20261108/HANDOFF.md)。**哈希全匹配只证明材料字节一致，不证明策略有效或统计正确。**
