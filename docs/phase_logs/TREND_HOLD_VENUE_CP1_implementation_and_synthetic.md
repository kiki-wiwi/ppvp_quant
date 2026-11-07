# 跨载体日线趋势持有研究 CP1：实现与合成验证

状态：CP1 实现与适用合成验证完成；仅本阶段交付，停止等待 CP2 审查。

## 基线与授权

- 分支：`codex/trend-hold-venue-oos-study`。
- R5 基线及已发布结果提交：`3846971ad9014043cc0371a96c693924b7ca1e61`。此前 R5 解锁提交为 `fc04caaf7463555c87cf30ad032f07adf0d1b0c2`，实现提交为 `cda559de7fe510a61f74dd5315eb25e1692d16ca`。
- 已完成原生 Git SSH 认证和 R5 归档推送；本次从该基线独立派生，只新增本轮文件。
- 权威规格：`reports/trend_hold_venue_round/approved_spec.md` 原文，以及 `boundary_addendum.md` 四项定稿与后续明确批准的零下界调整。旧策略参数、种子、代码、测试、manifest、冻结计划和归档结果未改动。
- 原 R5 未通过 V3/V5；未来样本外评估属于用户授权的描述性评估，不构成自动晋级。本阶段没有读取任何真实行情或计算真实策略收益。

## 范围锁与代码交付

新模块 `trend_hold_venue_study/` 的 `BTC_IS_UNLOCKED=False`、`POST_IS_AND_ETH_UNLOCKED=False` 均保留物理独立布尔开关。旧 R5 已解锁状态未被改写。

实现了不可变上下文与因果日线特征、独立线性执行器、无借贷现货现金裁剪、币本位套保/增币/HODL账户、维护取消重试、首日估算币数冻结订单、资金费旧仓结算、压力残币终止态、两套美元估值、动态归因与显式零下界调整。

统计入口保留 99,999 次 Webb、10,000 次整月重抽样和原种子，配对共享抽样，Holm 族大小为 2。实现七项绩效指标及连续 VAL/OOS 切片；这些功能本轮只以合成数组验证。

## 验证结果

- **129 passed，1 deselected，0 failed，0 skipped**；其中新增测试 **78** 项，适用旧 R5 回归 **51** 项。
- 新增 A–E 全部通过：A 锁美元；B 25,125 触线与跳空夹零；C 首单/资金费/滑点/钱包与归因；D 现金裁剪；E **18 组**与冻结 R5 逐分钟、订单及资金费账本完全一致回归。
- 两道新锁、前缀因果、维护重试、固定种子配对、零权益收益与连续切片等新增测试均通过。
- pytest 使用 `-p trend_hold_venue_study.testing_guard -x`：在测试收集前阻止行情文件与网络访问。
- 固定解释器：`/home/ubuntu/Documents/Codex/2026-11-06/github-ppvp-quant-agent-2-checkpoint/work/diag_env/pandas3.0.5_py314/bin/python`。Python 3.14.4、pandas 3.0.5、NumPy 2.5.4、pyarrow 26.0.1；完整版本在验证 JSON。
- 测试耗时 5.408 秒（含启动）。恢复前所有代码/测试 SHA256 与首次停机快照相同；验证期间保持一致。
- 142 个既有保护源文件逐字节哈希核验通过，既有跟踪文件相对基线无差异。

唯一排除节点：

```text
tests/test_trend_hold_engine.py::test_rejects_missing_or_invalid_daily_inputs_and_real_context_before_access
```

该旧阶段测试要求旧 `REAL_DATA_UNLOCKED=False`；但 R5 已归档解锁为 `True`，其末尾断言因此收到上下文类型错误而不是锁定错误。首次执行在此停止（85 通过、1 失败），未自行修复或重跑。用户随后明确批准仅用命令行 `--deselect` 排除此唯一节点。第二次执行保持全部源代码与测试原字节不变，执行其余全部选定测试；完整命令、排除节点、原因与版本已写入 `synthetic_validation.json` 及 `synthetic_pytest.txt`。

## 停机记录归档与文件登记

首次停机的三个文件完整移动到本轮报告下的 `validation_history/attempt_1_stopped/`，没有丢弃原始失败证据；顶层验证日志现表示获批恢复后的通过结果。

| 保留文件 | 原始及归档 SHA256 |
|---|---|
| `reports/trend_hold_venue_round/validation_history/attempt_1_stopped/synthetic_pytest.txt` | `4135b3e1b47ee368f1f2a0bb8ab6859f8af9b81aec170c9f3b51d7adc46f6c31` |
| `reports/trend_hold_venue_round/validation_history/attempt_1_stopped/synthetic_validation.json` | `900a90d2c7d82e3ce5001990c4939882ba94f428b0b95829889198c746700b1b` |
| `reports/trend_hold_venue_round/validation_history/attempt_1_stopped/STOPPED.json` | `900a90d2c7d82e3ce5001990c4939882ba94f428b0b95829889198c746700b1b` |

`registration.json` 登记基线、授权、固定环境与 142 个旧保护文件。`implementation_hashes.json` 登记本轮新增源码、测试、规格、日志及本阶段日志的 SHA256；仅它自身不递归哈希，自身由 Git 提交保护。未来解锁仅允许在核验时内存还原新 scope 中两条布尔锁，其余字节必须保持不变。

本轮外部工作脚本仅位于已知 `/tmp/trend_hold_formal__ue3kg42/venue_cp1/`；仓库新增文件全部列入实现登记和本次提交，不留未登记的非忽略临时文件。

## 剩余事项与停止边界

CP1 未提供真实数据加载器、正式批量执行/导出及图表入口。CP2 前仍须准备各载体原始包、资金费与缓存登记及这些运行入口，并另获 BTC IS 阶段批准；仅切换锁还不足以直接运行完整真实批次。全窗口真实性能及真实对账尚未验证。

不读取或运行真实 BTC IS、VAL/OOS/ETH，不进入 CP2/CP3。交付本实现与证据提交后停机。
