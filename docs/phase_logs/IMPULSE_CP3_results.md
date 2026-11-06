# IMPULSE CP3 results

本次仅为 CP3 结果。CP4 终审未完成；未运行其他 CP4 敏感性、独立盲复算或 VAL / OOS / ETH。

CP2 基线：`b4e3fc19f7f8ece6177a8460983d68b93bd481c3`。
实现提交：`d0297578f9d2f6604d8676b1db3fdfa550f696f9`。
登记与解锁提交：`a89814fb2f7fe3b8b5bf4467202083b60838ee63`。

运行起始工作树：干净，dirty_files=[]。
frozen_key：`fc0d37b9d198ff85`。27 个冻结哈希核对通过；engine.py 在内存还原为 False 后核验。

实际执行：10 版本 × 4 成本 × 2 仓位 = 80 次组合；10 版本 × 4 成本匹配随机（每笔100个），
每版本5000次辅助随机化；B1–B5全时段主成本/风险仓位附注；A5/B5因果σ_ref门禁；第9节统计；图1–4及图4独立起点模拟。
G2/G3/F3 Webb B=99,999；逐笔区间B=10,000；自然月夏普块B=5,000；图表区间B=2,000。

输入：BTCUSDT IS；资金费在加载入口以 funding_time < 2024-01-01 及 >=2020-01-01 谓词限制，并取整到分钟。
撮合前确认：4383次结算，10个冻结漏记分钟；
S6 box_ok SHA256=`305805e09a15b8b14439bf594a8d3ea2566b70103f012a128ccb3bbfe658560d`，与CP2一致。

A5/B5为“全IS波动标定的稳健性版本”，不得解释为IS内完全无前视；A5与B5不是箱体单变量对照。

## 文件登记

|文件|SHA256|
|---|---|
|`impulse_study/cp3_figures.py`|`2146c46c802da9799777632e03727f0693155082cb6af430ae212b35b00c7f5b`|
|`impulse_study/cp3_matching.py`|`f48aed7bc469ded272a2405dbf92c27f70b4cb5d0bb7cb921082900b10f5df1d`|
|`impulse_study/cp3_runtime.py`|`a030c75f2111d8d0804602e81f125a7df89dbd315c023240ab45bc4181cbbce4`|
|`impulse_study/cp3_sensitivity.py`|`e59c8123791d306e515c4be1d24de9f0eda79ff3cb0234a6ca845ce6178dd22a`|
|`impulse_study/cp3_statistics.py`|`28832435d5497ac3cff6ff9003088eef421effc73477fc2dc0952919d2cc31c1`|
|`test_impulse_cp3_integration.py`|`70cde7170e1710de98dcd23f8e06f67097082ca2702d40928d9e545cc06f628e`|
|`test_impulse_cp3_matching.py`|`2a1d3254a4e5619431ca9877fd725851d642f60a56c2de9bcff40a66536e8984`|
|`test_impulse_cp3_runtime_figures.py`|`c5e4adaa765727416033c099d6aad5359725a17ae70381a8d0be524f8331bf84`|
|`test_impulse_cp3_sensitivity.py`|`702fde6d1eed12b0464eb9234cb77d965f3e9639539f4c7e6c5730f5f020abc3`|
|`test_impulse_cp3_statistics.py`|`82bb7328a1b8b616e83b042a6f8c83c22fb0e9d031556c2660bc2b9ca6b25fb9`|
|`tools/run_impulse_cp3.py`|`68e1f36d6837dc4da30c88fb66b59ecae88d48e940929cda8c9bf5a7cdff467d`|
|`tools/run_impulse_cp1.py`|`103aa26b9356c0d85c55beea0fa36fe87b4d313de9d4bb109fdd6f7fd2df6c2e`|
|`data/manifest_impulse_1m.json`|`7e0cb4af7c8a377a530aee4910d20e3339dc50db7c0734f6c21fea3b627d3666`|
|`reports/impulse_vs_box_round/cp2/cp2_feature_checks.json`|`abbfbf4f46f199a434617bfd0a76d1a9d4bd59f08a15f3a17b85262788cc3d43`|
|`reports/impulse_vs_box_round/experiment_plan_impulse.md`|`1103cfcdece166b24adfadec27b3cfbf54e05a5eb7277733b7304dd9635b0519`|
|`requirements.txt`|`cf19cfdc6bfcb534525b5f3e83218e1a0de53d2765f4e6b89e2b73bf45a2b3e5`|

## 冻结文件核验

|文件|规范化SHA256|
|---|---|
|`impulse_study/__init__.py`|`638e5da61906c4d4454c5af0c18b5bc99d7d7a0862c695124c79c3183e23f747`|
|`impulse_study/engine.py`|`bb45fc65f18b784df0207fae59f8fb172d73b6942a92e9e58de9f681242dbfad`|
|`impulse_study/features.py`|`345870dd82d17b813f45e42c6525349791e2cbe00e082653aa74168186901d21`|
|`impulse_study/versions.py`|`a95d63037379f83765591a4f5eda3d11f52d3f128f02b347836cd82f1109c76a`|
|`impulse_study/synthetic.py`|`c55ead6d2fcc6bc00cf87a8b44d40512f205cee562c784458ea7a4cbee6bf147`|
|`impulse_study/signals.py`|`2fb18affd57d5be3b79839c0e06639cb03a4b1532f6a4fc155c27bb68d2fa0e3`|
|`impulse_study/daily.py`|`79272474bcb6c08d6e3f5485a4ec743fd16481cc6e424b49f67209b4872803d1`|
|`impulse_study/data_1m.py`|`c0123f787a0b9f8b353108179648ed580cebb148a8fb068126c3e5f21e40fe84`|
|`impulse_study/scope.py`|`ecc6b5d31f572a07488af2fbd04e96283ec3439f1715ebc98b66bcd7b36be9dc`|
|`impulse_study/qa.py`|`6ecb9553768eedfda060c60762644c6b6d44f70d0d9849d7454333376960ce29`|
|`numba_compat.py`|`fadd3ecc7e5b9f91637e513db77e673a24d4b5a6dd8de191a57729b28f8e28de`|
|`ppvp_engine.py`|`6c498886be2731c96b3e8b1e50394859788fda8e17555c540120ee038cd0a2b2`|
|`config.py`|`858ff5a924a9555c950a96d30054642144dbe4a1e2d6f055e769540879beb429`|
|`data_loader.py`|`92f88c5aabf4340c6bf186fb52cf483f20e5d399edd36da9a295467c720598c1`|
|`ppvp_audit/__init__.py`|`365d600d3f7064c8e12527c379886db839edb78a0f1187df9f0bba9bf38109d7`|
|`ppvp_audit/nulls.py`|`cd47c6ca9da7a2cf51ff02a76a10d5f07f7cda9eb3235002fb3c20356226c46b`|
|`ppvp_audit/experiments.py`|`af5ddba064faf9320ec33a014394cae78828a63ee9d8a68273a648c5e6a71415`|
|`ppvp_audit/common.py`|`1ac48d73df21389ebbfeb830158de96015c508ed8a34d2a8ddc263b0e9697eeb`|
|`ppvp_audit/engine_path.py`|`79c5a9d041b194bde16c951ae11ae4f8c5b1c0cb375dd213855e7d396791d8e6`|
|`ppvp_audit/event_sim.py`|`6b57c86ed402b24c24f88e70c8f0fb8a22daa14a8841f076ec36b16e44ee7cfa`|
|`ppvp_audit/variants.py`|`288e0f9b953aff7cd3baedb8d57dd06837403299c4b26d7daa51652f8340b339`|
|`attribution.py`|`1c7165769eef97fb86d1ef010e4dc3cdeef381f4ab52290bd7cfffe488c7357c`|
|`metrics.py`|`13d16204f0724ff54e5a42f2ed3c1d8fd7a7661cd2955b949f17e6b5ef86e52b`|
|`backtester.py`|`6dd67439db79fe983bad7a3343bc7dcef118e012d7560499ced91756a8177568`|
|`pipeline.py`|`0dfcf1cf61161e449c747a5cf2f6b904bb7b04fbb73194b1e0ee100e74e9e6d7`|
|`strategy_state_machine.py`|`790ad492a9c96b2188148e0de61f53cce9cbca19fb31ab86eb4690e4cffb3e2a`|
|`test_impulse_cp2.py`|`c9252c84fa769eda305385978bab8aba2935b2aded786e1b8bb683928fc774a4`|

## 解锁前确定的实现细节

### impulse_study.cp3_runtime

CP3 run controls and narrow input entry points (§10, §12).

Pre-unlock implementation choices: missing caches are rebuilt only for perp_1m
and spot_1d_warmup from the existing pinned manifest. CP4-only market archives
are not fetched. Output is create-once; failures leave all partial output and a
STOPPED record. Registration is JSON embedded in the unlock commit message.
No retries of the study, selective resume, seed options, or reduced draws exist.

### impulse_study.cp3_statistics

CP3 statistics implementing frozen plan §§8–9 (no market loading).

Implementation choices registered before unlock:
* Undefined numbers are None in reports; every undefined gate p is 1.
* Cluster sums use pandas' sorted group aggregation, as the frozen helper does.
  Missing matched excess is excluded; nonfinite ordinary trade metrics raise.
* Three paired D metrics reuse one CI matrix and one Webb matrix per seed.
* Webb weights are drawn in one exact (99999,G) call; derived arithmetic is
  processed in batches of 1024 rows without additional random draws.
* Monthly Sharpe draws concatenate natural-month daily arrays in sampled order,
  exactly as §8.4 specifies. Daily indices must contain the full frozen window.
* Max drawdown is daily-close drawdown including the initial-capital anchor,
  consistent with the repository metrics convention; minute drawdown is an
  additional descriptive item when a minute equity series is available.
* Stable Holm ties keep the fixed input order; frozen gates use adjusted p.
* Stationary auxiliary diagnostics reuse the frozen attribution core helpers
  with exactly one draw matrix per seed; their results are checked against
  the public helpers on synthetic inputs. This also counts discarded draws
  without replaying a seed or changing the frozen public helpers.
* G7 tied month totals use ascending month order. F4/F5 point ties use <1e-12.
The CP4 auxiliary functions are defined here but are not invoked by CP3.

### impulse_study.cp3_matching

Frozen CP3 §8.3 matched random-entry controls.

Pre-unlock implementation choices (no statistical parameters are configurable):
* A valid sigma7d is finite, matching the frozen signal implementation.
* Balance compares matched originals with their 100 draws per trade. It reports
  count, mean, population SD, min/5/25/50/75/95/max, and 24 hour counts.
* If no trade has candidates, the auxiliary randomization p is 1, its observed
  mean is undefined and its null-mean array is empty; G3's empty-sample handling
  belongs to cp3_statistics. Missing trades consume no random numbers.
* Pool lookup uses stable sorting by integer keys; every returned pool retains
  ascending minute-index order. The one RNG per procedure is consumed exactly
  in frozen row-major order; simulation does not consume random numbers.

### impulse_study.cp3_sensitivity

Pre-unlock implementations of frozen plan section 11(1--9).

This module never loads market data, mutates a Context, or changes the frozen
engine. Its default CP3 entry point runs only G8z and the B full-period appendix.
CP4 is an explicit separate entry point, whose statistics are supplied as hooks.

Choices not further specified by the plan, fixed before any real result:
* Empty expanding histories produce NaN (no Z trigger); valid sigma includes
  finite zero values and all minutes, including contaminated/bad minutes.
  Two heaps maintain the exact nanmedian in O(n log n), sampled BEFORE each UTC
  day's values are inserted. The lower bound is 2020-01-01 00:15, inclusive.
  New temporal helpers normalize aware indices to UTC nanoseconds before integer
  comparisons, as approved after the microsecond synthetic-fixture failure.
* Defect removal removes the UNION of the three flags. Surviving trades retain
  their original quantity and schedule; skipped opportunities are not reopened.
  Their fees/funding/marked equity are rebuilt from the retained ledger.
* Optimistic and natural-bar reruns use main cost/risk sizing for section 11's
  descriptive comparisons. Defect removal also supplies the harsh-cost ledger
  required by G6; section 6 exposes all 80 frozen configurations unchanged.
* Stop/close conflicts count simultaneous eligible exits, including an equal
  reference price; a separate economic_result_changes flag identifies a changed
  per-unit net result (including a newly payable close-boundary funding charge).
* Natural-bar baseline sums bad-minute volume as zero, just like the main
  baseline. Complete natural bars include minutes before entry, while MAE/MFE
  and P&L start at entry. Optimistic closes use the whole exit-minute H/L for
  MAE/MFE because the position now survives through that minute's close.
* Full-period appendix and box variants rebuild Context with make_context;
  only causal Z uses replace(new cond), as explicitly required in section 5.8.
  Fresh reconstruction preserves a supplied empty funding series distinctly.
* The alternative matching/accounting kernel below is copied from the frozen
  engine; only holding boundaries, continuation inputs and stop/close priority
  differ. Ledger equity reconstruction is a separate cash-event calculation.

### impulse_study.cp3_figures

Frozen §10 figures; descriptive intervals use 2,000 shared month draws.

Pre-unlock choices: for each figure/version, one ascending union of entry months
and one draw matrix serves every horizon/category/metric. Empty replicates are
discarded without replacement. Fewer than 12 active months gives an undefined
interval (§8.7). Onsets are transitions in each direction's eligible entry
signal (VersionInputs.entry_dir); they are simulated independently, without
portfolio position/cooldown. Histogram bins are common within each panel,
cover the entire observed range, and use numpy's automatic bin selection.

## 测试与环境

```json
{
  "command": [
    "/home/ubuntu/Documents/Codex/2026-11-06/github-ppvp-quant-agent-2-checkpoint/work/diag_env/pandas3.0.5_py314/bin/python",
    "-m",
    "pytest",
    "-x",
    "-q",
    "test_impulse_cp1.py",
    "test_impulse_cp2.py",
    "test_impulse_cp3_integration.py",
    "test_impulse_cp3_matching.py",
    "test_impulse_cp3_runtime_figures.py",
    "test_impulse_cp3_sensitivity.py",
    "test_impulse_cp3_statistics.py",
    "-k",
    "not test_real_data_cache_hash_and_headline_counts and not test_real_data_offline_rebuild_from_manifest"
  ],
  "exit_code": 0,
  "tail": [
    "........................................................................ [ 38%]",
    ".......s......................................s......................... [ 77%]",
    "..........................................                               [100%]",
    "184 passed, 2 skipped, 2 deselected in 117.74s (0:01:57)"
  ]
}
```

```json
{
  "python": "3.14.4",
  "platform": "Linux-7.0.0-1015-oracle-aarch64-with-glibc2.43",
  "numpy": "2.5.4",
  "pandas": "3.0.5",
  "numba": "0.68.1",
  "scipy": "1.18.1",
  "pyarrow": "26.0.1",
  "matplotlib": "3.11.2",
  "pytest": "9.2.0"
}
```

## 结果

完整主配置表、门禁、多重校正、蒙特卡洛边界披露见 `reports/impulse_vs_box_round/cp3/summary.json` 与 `main_comparison.csv`。
四张图及逐点数据均在同目录。以上结果不构成 CP4 终审或后续验证授权。

## 撮合前环境对齐与停机日志归档

首次运行于输入核验阶段因 pandas 3.1.1 解析 spot_1d_warmup 的内容哈希与冻结 manifest 不一致而停止，未进入撮合、未产生真实交易结果。经用户批准的只读诊断，四个 2019 官方现货日线 zip 原始 SHA256 均一致；仅 quote_volume 的57个值有末位差异（56个1 ULP、1个2 ULP），索引、列类型与 OHLC 逐位一致，S6 box_ok_sha256 与 CP2 完全一致。

用户二次批准后，采用仓库外 pandas 3.0.5 隔离环境，保持 Python 3.14.4、numpy 2.5.4、pyarrow 26.0.1 及其他运行依赖版本不变。原代码重建的 spot 内容哈希精确恢复为 `8012353b7861d5fa7981a302937b46f3f9399d6c90b0c63c334e7eec75a7de1f`。未修改代码、manifest、冻结哈希或 b4e3fc1 → d0297578 → a89814fb 提交链。

隔离解释器：`/home/ubuntu/Documents/Codex/2026-11-06/github-ppvp-quant-agent-2-checkpoint/work/diag_env/pandas3.0.5_py314/bin/python`。

上次3个未跟踪停机日志于 2026-11-06T07:47:22.946957+00:00 完整移动至仓库外 `/home/ubuntu/Documents/Codex/2026-11-06/github-ppvp-quant-agent-2-checkpoint/work/cp3_pre_match_stop_logs_20261106T065250Z`，逐文件 SHA256 如下；归档未丢弃或改写原日志。

|文件|SHA256|
|---|---|
|`STOPPED.json`|`11305d1d81866915919147515ca9912d108af208483d1c78a898f4dc1c4a139e`|
|`run.txt`|`6773957c9aabb1a14048f9eef2384a135170cda4eb98af64da69785e07297694`|
|`run_start.json`|`89422feb13b5d9fe6cbd8f48814e8a251407aaa1be953db9506112648167af56`|

归档后 git status --porcelain 完全为空且 cp3 输出目录不存在。正式运行由该干净 a89814fb 工作树直接执行原 tools/run_impulse_cp3.py，从输入核验完整重启，没有局部续跑、修改种子或缩减配置。184项测试通过，2项CP2专用锁定测试跳过，2项CP1真实缓存构建/复核测试排除；其余规定测试通过。

正式运行 UTC 2026-11-06 07:48:24.024832 至 07:52:47.804242，耗时263.779秒；成功完成80次主组合、40组版本/成本匹配随机、5个B组附注、2个G8z、第9节统计及四图（含独立起点模拟），过程无异常。10版本G1通过，G2–G7均未通过；A5/B5的G8z未通过；无申请资格版本、无成本敏感候选。CP4终审尚未完成，正式脚本已停止。
