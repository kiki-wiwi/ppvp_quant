# 独立审计记录：复核路径

本目录是对 `kiki-wiwi/ppvp_quant` 审计提交 `b87e167f93e6bec0ef5def53007894c6014ce3e2`（研究锚点 `21eee8746452d5d46b78f5ef8c50c024ebd21779`，分支 `codex/trend-hold-venue-oos-study`）所做独立审计的完整记录。审计结论见 [AUDIT_REPORT.md](AUDIT_REPORT.md)。

本目录只新增文件，不修改被审计分支上的任何代码、计划、哈希或结果。`scripts/` 中的复算程序不导入被审计仓库的任何模块；R5 与跨载体引擎依据 `approved_plan.md`、`approved_spec.md` 与 `boundary_addendum.md` 的文字独立编写。`outputs/` 保存每个脚本的原始输出，报告中的每个数字都可在其中找到。`subaudit_scripts/` 保存两个辅助审计流程（早期各轮复算、提交链核查）使用的只读脚本与 git 日志摘录。

## 目录

`scripts/` 内各脚本的用途如下：`verify_snapshot.py` 核对快照清单与锚点提交 Git 树、LFS 指针和三个输入分卷；`check_live_checksums.py` 抓取 Binance 官方 CHECKSUM 并与 1,326 个登记 ZIP 哈希比对；`rebuild_btc_1m.py`、`rebuild_funding.py`、`rebuild_venue.py`、`check_funding.py` 从官方原始归档重建缓存并逐值比较；`r5_independent.py`、`r5_stats.py` 独立复算 R5 六组合与统计；`venue_independent.py` 独立复算跨载体主信号账户；`cp3_stats.py` 重算 CP3 二十一个配对统计族；`r5_timing.py`、`sensitivity_window.py`、`extra_diagnostics.py` 是审计方的事后敏感性分析，明确不替代任何冻结门禁。

## 环境

复算使用 Python 3.13.18、pandas 3.0.5、NumPy 2.5.4、PyArrow 26.0.1、SciPy 1.18.1（Linux x86_64）。原研究登记环境为 Python 3.14.4、Linux aarch64；除解释器与平台外，核心包版本一致。全部复算结果与存档逐位一致或仅有 1e-14 量级的浮点差异。

## 复核步骤

以下命令在一个空工作目录中执行。`audit/` 是固定在审计提交的只读工作树，其余目录存放从分卷提取的输入对象；配置文件 `scripts/cfg_*.json` 中的相对路径即按此布局书写。只需拉取三个输入分卷的 LFS 本体（约 1.56 GB），不需要下载约 18.6 GB 的分钟账本 LFS 对象。

```bash
git clone https://github.com/kiki-wiwi/ppvp_quant.git repo
```

```bash
git -C repo fetch origin codex/trend-hold-venue-oos-study claude/modest-maxwell-kfgsug
```

```bash
git -C repo worktree add --detach ../audit b87e167f93e6bec0ef5def53007894c6014ce3e2
```

```bash
git -C audit lfs install --local && git -C audit lfs pull --include="reports/external_audit_20261108/inputs/inputs-00*.tar"
```

```bash
git -C repo archive origin/claude/modest-maxwell-kfgsug reports/independent_audit_b87e167 | tar -x --strip-components=1
```

```bash
python3 -m venv venv && venv/bin/pip install pandas==3.0.5 numpy==2.5.4 pyarrow==26.0.1 scipy==1.18.1
```

```bash
python3 -I independent_audit_b87e167/scripts/verify_snapshot.py audit 21eee8746452d5d46b78f5ef8c50c024ebd21779
```

```bash
for w in want_r5 want_funding; do python3 -I independent_audit_b87e167/scripts/extract_objects.py audit inputs_r5 independent_audit_b87e167/scripts/$w.json; done
```

```bash
python3 -I independent_audit_b87e167/scripts/extract_objects.py audit inputs_venue independent_audit_b87e167/scripts/want_venue.json
```

```bash
mkdir -p inputs_all && for p in audit/reports/external_audit_20261108/inputs/inputs-00*.tar; do tar -xf "$p" -C inputs_all; done
```

```bash
cp independent_audit_b87e167/scripts/cfg_*.json . && venv/bin/python -I independent_audit_b87e167/scripts/r5_independent.py inputs_r5/57479bcbb97a21e58e2347e5c17cc6194a2a6fc1302397d4459592746af39aeb audit/data/processed/BTCUSDT_funding.parquet r5_out
```

```bash
venv/bin/python -I independent_audit_b87e167/scripts/r5_stats.py r5_out/daily_returns.csv audit/reports/trend_hold_round/results r5_out/features.csv
```

```bash
for w in btc_common btc_post eth_common eth_post; do venv/bin/python -I independent_audit_b87e167/scripts/venue_independent.py cfg_$w.json venue_out; done
```

```bash
venv/bin/python -I independent_audit_b87e167/scripts/cp3_stats.py audit/reports/trend_hold_venue_round/cp3_post_is_and_eth/accounts audit/reports/trend_hold_venue_round/cp3_post_is_and_eth/statistics
```

其余脚本的参数写在各自文件首部的文档字符串中。`check_live_checksums.py` 需要能访问 `data.binance.vision`；在需要代理证书的环境中可通过环境变量 `AUDIT_CA_BUNDLE` 指定证书文件。`sensitivity_window.py` 需要另行从官方站点下载 BTCUSDT、ETHUSDT 现货 1m 的 2026-06 至 2026-10 月包及其 CHECKSUM 并放入同一目录。
