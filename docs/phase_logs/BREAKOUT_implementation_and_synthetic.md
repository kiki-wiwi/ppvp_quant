# Channel / VAH：实现与合成验证阶段

## 授权与基线

用户批准最小预注册计划，并授权独立分支登记、独立模块和合成验证；尚未批准真实行情读取或正式回测。本阶段全部行为限定在此范围。

- 分支：`codex/channel-vah-study`。
- 基线：`cee29ef2c9356f4d23da1f94ec0dc44fe2cd30b6`。
- 批准稿：`reports/channel_vs_vah_round/approved_plan.md`，逐字复制自用户审查的仓库外草案。
- 批准稿 SHA256：`fb661ffc51245092b209a53073e4e57340d335bd5333cc37595af11b6e3075af`。
- 新模块的 `breakout_study.scope.REAL_DATA_UNLOCKED = False`；没有调用真实读取入口进行行情读取。
- 既有冻结计划、27个冻结文件、数据 manifest 及其他既有文件均未修改。按旧计划对旧引擎解锁行仅作内存归一后，27项哈希全部一致；旧 `frozen_key = fc0d37b9d198ff85`。

本次提交号由 Git 提交本身及交付消息记录，避免在文件内循环记录自身 SHA。新增文件哈希登记于 `reports/channel_vs_vah_round/implementation_hashes.json`，清单不包含自身。

## 实现

新增独立的范围与来源锁、不可变分钟上下文、闭合 1H/4H 特征、两种突破线、分钟撮合与资金费账本、月聚类统计/Holm/N1–N6、报告表和3图、默认仅验证的运行入口。两个版本和三个成本档严格固定，无参数网格或种子覆盖。

真实加载代码已实现但保持锁定。未来授权后只允许读取固定 BTCUSDT 永续缓存与资金费文件；两个入口都在读取时施加 IS 时间谓词，不读取现货、VAL、OOS 或 ETH，不自动下载或改写缓存。合成验证通过替代读取函数提供内存生成帧检查谓词、资金费网格和内容哈希；没有临时把模块的真实解锁常量改为 True。

完整合成集成验证使用 2020–2023 日历上的程序生成价格，执行六个固定组合及完整 99,999 次 Webb / 10,000 次月区间，验证台账、统计、门禁、3张图和异常停止。这些输出不代表真实策略收益，不作为投资结论；只存放在 pytest 临时目录，未入库为研究结果。

## 验证证据

测试解释器位于仓库之外：Python 3.14.4、pandas 3.0.5、NumPy 2.5.4、pyarrow 26.0.1、Numba 0.68.1、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.2.0。未修改既有 requirements.txt。

各次正式验证均加载 `-p breakout_study.testing_guard`，阻止行情目录访问与网络连接。新功能先以合成反例验证，再实现或修正。未执行裸 pytest 全量发现，以免收集旧轮真实行情测试。

| 运行 | 实际结果 | 范围 |
|---|---|---|
| 新核心模块 + 相关旧回归 | 195 passed，1 skipped，2 deselected，20 subtests passed，46.79s | 新 scope/data/features/engine/statistics、CLI锁；CP2合成、CP3统计合成、两个CP1特征回归 |
| 完整六组合合成集成 | 1 passed，168.22s | 完整IS形状的虚构分钟价格，六组合、统计及3张图 |
| 后续受影响边界回归 | 24 passed，1 deselected，1.56s | 范围/加载器/报告/运行器；与首行有18项重叠，不重复累计 |
| 最后归档缺口UTC类型回归 | 4 passed，0.10s | 加载器测试；其中3项与前表重叠，新增1项 |

合计 **203个不同测试通过**：新增研究 **87项**、相关既有合成回归 **116项**；另有20个通过的 subtests。计数按测试身份去重，不把针对修正的重复运行重复累计。

唯一跳过项：旧 `test_engine_refuses_real_data_in_checkpoint2`，只适用于旧 CP2 未解锁状态；旧研究已在基线上解锁，按其原有标记跳过。

明确排除而未运行：
- `test_real_data_features_are_causal_under_prefix_truncation`
- `test_real_data_context_is_locked_out_of_the_engine`

主体命令（解释器须使用上述隔离环境）：

```text
python -m pytest -p breakout_study.testing_guard \
  test_breakout_scope.py test_breakout_data.py test_breakout_features.py \
  test_breakout_engine.py test_breakout_statistics.py \
  test_breakout_runtime.py::test_runner_defaults_to_verification_and_never_unlocks \
  test_impulse_cp2.py test_impulse_cp3_statistics.py \
  test_impulse_cp1.py::test_profile_levels_match_ppvp_engine \
  test_impulse_cp1.py::test_daily_from_1m_available_time_and_partial_day \
  -k 'not test_real_data_features_are_causal_under_prefix_truncation and not test_real_data_context_is_locked_out_of_the_engine' -q

python -m pytest -p breakout_study.testing_guard \
  test_breakout_runtime.py::test_full_synthetic_six_combination_pipeline -q -s

python -m pytest -p breakout_study.testing_guard \
  test_breakout_data.py test_breakout_scope.py test_breakout_reporting.py test_breakout_runtime.py \
  -k 'not test_full_synthetic_six_combination_pipeline' -q
```

## 实现中解决的问题

全部发生于新增代码和合成夹具阶段：修正 Pandas ns/us 夹具单位、合成自然4H边界对齐及手算测试预期；新读取入口把既有无时区归档缺口字符串显式解释为 UTC ns。这些修正未改变研究公式或旧文件。

集成检查修正：保留撮合器的永久非正净值状态，即使日末恢复仍不得获候选资格；任一成本组合（包括 gross_0）失败都传递到版本排除。未来授权正式运行的冻结/输入校验异常也必须写 STOPPED 记录，不自动修复或重跑。相关失败路径均以合成反例验证。

本阶段没有真实冻结校验失败；模拟篡改测试只替换读取返回值或使用临时文件，不修改仓库冻结资产。

## 剩余事项与停止

没有已知阻塞合成实现的问题。尚未验证真实缓存可用性、真实输入哈希、真实触发频率、真实绩效或真实运行资源消耗，因为这些都不属于本阶段授权。托管工作区未复制被忽略的真实分钟缓存。

下一步须用户明确批准正式运行后，才可准备并核验登记的 BTC IS 缓存、登记本模块单行解锁、确认新代码/旧冻结资产及干净工作树，再执行六组合。若输入或冻结核验失败，停止并保留日志，不能改 manifest 或自行重跑。本次不进入旧研究 CP4，不执行 VAL/OOS/ETH。
