# GitHub 审计交付说明

## 1. 只访问 GitHub 如何取得完整材料

仓库公开地址：<https://github.com/kiki-wiwi/ppvp_quant>。请向委托人取得本审计交付的完整提交 SHA，以该 SHA 固定审计快照。研究锚点为 `21eee8746452d5d46b78f5ef8c50c024ebd21779`；后续审计提交只新增材料。

有 Git、Git LFS 和 Python 3.10+ 的机器可执行以下读取步骤。建议留出 **60 GiB** 空间：普通文件、LFS 对象库和工作树可能各占一份；输入分卷解包还需要约 1.56 GB。无需 Binance 账户、交易权限、研究机器 SSH 或本机绝对路径。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch codex/trend-hold-venue-oos-study https://github.com/kiki-wiwi/ppvp_quant.git
cd ppvp_quant
# 如委托人指定完整审计 SHA，先 git checkout --detach 该SHA。
git lfs install --local
git lfs fetch origin HEAD
git lfs checkout
python3 reports/external_audit_20261108/verify.py
```

最后一条只用 Python 标准库，读取字节、SHA256、文件大小与 tar 成员，不联网、不导入策略、不计算任何收益、不写入文件。正常完整结果应为 `status: PASS`。若只想先读报告，可加 `--metadata-only`，但返回的 `PARTIAL` **不等于完整核验通过**。GitHub 的“Download ZIP”是否包含 LFS 本体取决于仓库/下载方式，不能仅凭压缩包已下载就认为拿到了所有账本；缺失 LFS 本体时校验器会失败。

审计工具自身合成测试：

```bash
python3 -B reports/external_audit_20261108/test_verify.py
```

这不是策略测试，也不触发真实行情读取。原策略的各阶段合成测试记录仍在相应注册目录。

## 2. 输入分卷与路径映射

[inputs/index.json](inputs/index.json)包含：

- `packs`：3 个分卷的仓库相对路径、原始字节大小和 SHA256。
- `objects`：每个 tar 成员为 `objects/<SHA256>`，原字节去重保存，给出分卷、成员、大小和历史别名。
- `references`：5,068 条已解析引用，保留原路径、逻辑路径、注册来源、文件 SHA。若已有同字节文件在仓库，`repository_file` 指向它；否则按 SHA 从分卷取得。别名仅用于查找，**不是可无条件执行的覆盖目标**。
- `missing_historical_references`：未保存的历史原件；不伪造响应或将新派生文件冒充旧原件。
- `historical_container_variants`：旧 BTC UM 1m Parquet 的 1 个已批准容器字节变体；旧哈希未改，当前接受的容器 SHA 与一致的内容哈希分别登记。

如需用自己的分析程序读取输入，可在完整校验 PASS 后解包到新目录，保持原研究工作树不变：

```bash
mkdir audit-inputs
for pack in reports/external_audit_20261108/inputs/inputs-*.tar; do
  tar -xf "$pack" -C audit-inputs
done
```

示例：定位已登记 BTC 1m 缓存，不加载数据：

```python
import json
from pathlib import Path
index = json.loads(Path('reports/external_audit_20261108/inputs/index.json').read_text())
for ref in index['references']:
    if ref['logical_path'] == 'data/processed_1m/BTCUSDT_perp_1m_IS.parquet':
        print(ref.get('repository_file', 'audit-inputs/objects/' + ref['sha256']))
        print(ref['sha256'], ref['registrations'])
```

所有 836 条早期 ZIP 引用均有对应字节：163 条可复用现存登记源，673 条在本次重新取得并匹配历史 SHA。正式 CP2 的 111 个 ZIP/CHECKSUM 对、CP3 的 479 个 ZIP/CHECKSUM 对、全部正式回放缓存与 2,770 项原始输入诊断证据也在分卷/既有文件索引中。诊断中包含后来放弃的 2026-06 之后的归档调查，**它们属于历史诊断证据，不是正式 CP3 输入**；实际运行输入边界以 `venue_cp3_inputs.json` 和结果的 `input_verification.json` 为准。没有在本次恢复或执行新样本回测。

原数据清单中的 `/tmp`、`/mnt` 和研究机绝对路径作为历史事实保留。顾问通过新索引查字节，不必创建同名系统目录。R5、Impulse 的外部停止/发布凭证另直接补在 [evidence](evidence/)；CP3 最后 Git 推送记录亦补入，避免自引用最终结果提交 SHA。

## 3. 环境与“复核材料”不等于“原命令随处重跑”

[environment.json](environment.json)记录此次与正式运行相同的固定解释器及已安装包；[requirements-audit.lock.txt](requirements-audit.lock.txt)固定其所有包版本，不含下载地址或凭据。正式核心版本为 Python 3.14.4、pandas 3.0.5、NumPy 2.5.4、PyArrow 26.0.1、Numba 0.68.1、SciPy 1.18.1、Matplotlib 3.11.2、pytest 9.2.0。平台为 Linux aarch64。它是环境规格记录，不是完整离线 Python/操作系统镜像；字节核验与阅读 CSV 无需安装这些科学计算包。

旧根 `requirements.txt` 的最低版本约束不足以保证逐位重现；此前 pandas 解析差异与 PyArrow 容器差异已在旧日志披露。这里没有更换正式环境、重建缓存或重新计算内容哈希。

不要在最终归档 HEAD 上直接执行旧 `--check`、`--real` 或全项目 `pytest` 并把拒绝视为结果损坏。正式运行器刻意要求对应**独立解锁提交、干净工作树、全新输出目录、固定新卷 UUID/挂载路径及空间**；它不是可移植回放程序。测试中的某些断言亦属于解锁前状态。若顾问计划独立复算，先在隔离副本确定具体历史提交、环境和入口，将自己的适配及输出明确标为独立复算，绝不能覆盖本仓库原结果或放宽冻结门禁。

CP3 原合成验证位于实现提交 `4d3d103e6614e98de1bdf15cb17a8619d4e74ad7`，BTC 锁 True、POST 锁 False。其记录为 325 passed、3 deselected；精确排除的是：

- `tests/test_trend_hold_engine.py::test_rejects_missing_or_invalid_daily_inputs_and_real_context_before_access`
- `tests/test_trend_venue_scope_features.py::test_two_real_locks_and_provenance`
- `tests/test_trend_venue_cp2_runtime.py::test_runtime_rejects_post_is_eth_and_real_locked_before_feature_work`

原命令、原因与输出见 [synthetic_validation.json](../trend_hold_venue_round/cp3_registration/synthetic_validation.json)、[synthetic_pytest.txt](../trend_hold_venue_round/cp3_registration/synthetic_pytest.txt)。现在两道锁都已 True；没有修改测试来伪造最终 HEAD 全绿。

## 4. 未能补造的历史原件与影响

| 原件 | 状态与审计影响 |
|---|---|
| 836 份早期 `.CHECKSUM` HTTP 响应原文 | 旧下载器只保存 ZIP 哈希与校验结论，未保存响应字节。ZIP 原字节现在全部可得且与冻结哈希匹配；不能把本次恢复记录冒充当年的响应证据。CP2/CP3 的已保存 CHECKSUM 原件另完整提供。 |
| 旧 `BTCUSDT_perp_15m_official_IS.parquet`、`BTCUSDT_perp_1d_official_IS.parquet`、`BTCUSDT_spot_1m_warmup_2019.parquet` | 原容器不在已登记已知目录中。分别用于旧 Impulse CP1 交叉核验和 S4 敏感性计数；对应官方原始 ZIP、解析代码和冻结内容哈希已提供，顾问可独立重建核对。它们不属于 R5/跨载体正式回放缓存缺失。 |
| Phase 0 OKX 补洞原始 API 响应 | 原代码未落盘、未登记原始响应 SHA，无法事后恢复当时响应的证据身份。已跟踪的最终 1h 缓存、填补标记、规则和报告仍在；补洞发生于旧 2026-10-28 数据，正式 CP3 已在 2026-06-01 截止，未用这些行。不能声称全部旧数据均是 Binance 原生行情。 |
| Phase 1 四轮临时审计脚本、Phase 2 若干临时核对脚本 | 旧日志明确未入库，无法提供其原始执行证据；见 [研究地图](RESEARCH_MAP.md)。当前冻结源码、已有测试与结果可供顾问独立验证，但不能代替缺失原脚本的历史身份。 |
| BTC UM 1m 历史 Parquet 字节变体 | 旧 `60e4fd…` 与当前 `57479b…` 为已批准 PyArrow 容器差异，冻结 `content_sha256 = 1e26dd7ecfe6ee03dc2814dcb0041ef2d5724ff7018b2b7f20ae6364d99f5a33` 一致。以 CP2 input_revision_2 的授权与预检为证，不改旧 manifest。 |

这些限制会影响“当时每一步原始现场都完整可追溯”的主张，必须保留在顾问最终审计报告中。这里不声称已取得独立审计结论，也不声称仅凭打包和 SHA 校验就通过了策略审计。
