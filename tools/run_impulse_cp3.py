#!/usr/bin/env python3
"""One-shot CP3 runner. No resume/reduced-configuration/seed override exists.

Only real work here: 80 main combinations, 40 matched cost controls and their
auxiliary randomization nulls, five main-cost/risk B full-period appendices,
two causal-sigma G8z runs, §9 statistics, and figures 1–4/onset simulations.
All other CP4 sensitivity and independent recomputation remain unexecuted.
New statistical/plot/implementation choices are documented in new module
headers and copied into the final phase log by this runner before commit 3.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import inspect
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from impulse_study import engine as E, features as F, versions as VS
from impulse_study import cp3_runtime as R

OUT = ROOT / 'reports/impulse_vs_box_round/cp3'


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return jsonable(value.tolist())
    if isinstance(value, pd.DataFrame):
        return jsonable(value.to_dict(orient='records'))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save_result(result, prefix):
    result.trades.to_csv(OUT / f'{prefix}_trades.csv.gz', index=False, compression='gzip')
    result.daily_equity.rename('equity').to_csv(OUT / f'{prefix}_daily_equity.csv', index_label='utc_day')


def unlocked_tests():
    files = ['test_impulse_cp1.py', 'test_impulse_cp2.py'] + sorted(p.name for p in ROOT.glob('test_impulse_cp3*.py'))
    command = [sys.executable, '-m', 'pytest', '-x', '-q', *files, '-k',
               'not test_real_data_cache_hash_and_headline_counts and not test_real_data_offline_rebuild_from_manifest']
    with (OUT / 'post_unlock_tests.txt').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('post-unlock tests failed; see post_unlock_tests.txt; no study continuation')
    return {'command': command, 'exit_code': result.returncode,
            'tail': (OUT / 'post_unlock_tests.txt').read_text().splitlines()[-5:]}


def phase_log(meta, summary):
    from impulse_study import cp3_statistics, cp3_matching, cp3_sensitivity, cp3_figures
    reg = meta['registration']
    lines = ['# IMPULSE CP3 results', '',
             '本次仅为 CP3 结果。CP4 终审未完成；未运行其他 CP4 敏感性、独立盲复算或 VAL / OOS / ETH。', '',
             f"CP2 基线：`{R.BASELINE}`。",
             f"实现提交：`{meta['implementation_commit']}`。",
             f"登记与解锁提交：`{meta['unlock_commit']}`。", '',
             f"运行起始工作树：干净，dirty_files={meta['dirty_files']}。",
             f"frozen_key：`{meta['frozen_key']}`。27 个冻结哈希核对通过；engine.py 在内存还原为 False 后核验。", '',
             '实际执行：10 版本 × 4 成本 × 2 仓位 = 80 次组合；10 版本 × 4 成本匹配随机（每笔100个），',
             '每版本5000次辅助随机化；B1–B5全时段主成本/风险仓位附注；A5/B5因果σ_ref门禁；第9节统计；图1–4及图4独立起点模拟。',
             'G2/G3/F3 Webb B=99,999；逐笔区间B=10,000；自然月夏普块B=5,000；图表区间B=2,000。', '',
             '输入：BTCUSDT IS；资金费在加载入口以 funding_time < 2024-01-01 及 >=2020-01-01 谓词限制，并取整到分钟。',
             f"撮合前确认：{meta['inputs']['funding_count']}次结算，{meta['inputs']['archive_gap_count']}个冻结漏记分钟；",
             f"S6 box_ok SHA256=`{meta['inputs']['box_ok_sha256']}`，与CP2一致。", '',
             'A5/B5为“全IS波动标定的稳健性版本”，不得解释为IS内完全无前视；A5与B5不是箱体单变量对照。', '',
             '## 文件登记', '', '|文件|SHA256|', '|---|---|']
    for section in ('new_file_sha256', 'existing_dependency_sha256'):
        for path, sha in reg[section].items():
            lines.append(f'|`{path}`|`{sha}`|')
    lines += ['', '## 冻结文件核验', '', '|文件|规范化SHA256|', '|---|---|']
    lines += [f'|`{p}`|`{s}`|' for p,s in meta['frozen_sha256_normalized'].items()]
    lines += ['', '## 解锁前确定的实现细节', '']
    for mod in (R, cp3_statistics, cp3_matching, cp3_sensitivity, cp3_figures):
        lines += [f'### {mod.__name__}', '', inspect.getdoc(mod) or '', '']
    lines += ['## 测试与环境', '', '```json', json.dumps(jsonable(meta['tests']), ensure_ascii=False, indent=2), '```', '',
              '```json', json.dumps(meta['environment'], indent=2), '```', '',
              '## 结果', '',
              '完整主配置表、门禁、多重校正、蒙特卡洛边界披露见 `reports/impulse_vs_box_round/cp3/summary.json` 与 `main_comparison.csv`。',
              '四张图及逐点数据均在同目录。以上结果不构成 CP4 终审或后续验证授权。', '']
    path = ROOT / 'docs/phase_logs/IMPULSE_CP3_results.md'
    with path.open('x') as handle:
        handle.write('\n'.join(lines))


def main():
    meta = R.verify_unlock()  # clean state captured before creating any output
    OUT.mkdir(exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)sZ %(message)s',
                        handlers=[logging.FileHandler(OUT / 'run.txt'), logging.StreamHandler(sys.stdout)])
    logging.Formatter.converter = time.gmtime
    started = time.time()
    meta['started_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
    meta['environment'] = {'python': platform.python_version(), 'platform': platform.platform(),
                           **{k: importlib.metadata.version(k) for k in ('numpy','pandas','numba','scipy','pyarrow','matplotlib','pytest')}}
    write_json(OUT / 'run_start.json', meta)
    try:
        from impulse_study import cp3_statistics as S, cp3_matching as M, cp3_sensitivity as X, cp3_figures as P
        logging.info('Input loading and frozen pre-match checks')
        ctx, checks = R.load_real_context()
        meta['inputs'] = checks
        write_json(OUT / 'input_checks.json', checks)
        logging.info('Post-unlock regression including required CP2 skips')
        meta['tests'] = unlocked_tests()
        initial_fingerprint = X.context_fingerprint(ctx)
        main_results, harsh_results, ledgers = {}, {}, {}
        all_metrics, reconciliations = [], []
        for name, spec in VS.VERSIONS.items():
            vin = F.build_version_inputs(ctx, spec)
            ledgers[name] = {}
            identity = None
            for cname, cost in VS.COSTS.items():
                for sname, sizing in VS.SIZINGS.items():
                    logging.info('Portfolio %s / %s / %s', name, cname, sname)
                    result = E.run_portfolio(ctx, vin, cost, sizing)
                    S.assert_statistics_scope(trades=result.trades)
                    current = result.trades[['trigger_idx','entry_idx','exit_bar','side','reason','sl','e_ref']].reset_index(drop=True)
                    if identity is None:
                        identity = current
                    elif not identity.equals(current):
                        raise AssertionError('cost/sizing changed the frozen trade schedule')
                    if not np.isclose(result.trades.net_pnl.sum(), result.meta['final_equity'] - VS.INITIAL_CAPITAL, rtol=1e-12, atol=1e-7):
                        raise AssertionError('ledger final equity mismatch')
                    save_result(result, f'{name}__{cname}__{sname}')
                    metrics = S.portfolio_metrics(result)
                    all_metrics.append({'version':name,'cost':cname,'sizing':sname,**metrics})
                    if sname == VS.MAIN_SIZING:
                        ledgers[name][cname] = result.trades
                        if cname == VS.MAIN_COST:
                            main_results[name] = result
                        elif cname == 'harsh_34bp':
                            harsh_results[name] = result
                    if cname == VS.MAIN_COST and sname == VS.MAIN_SIZING:
                        n_sig = int((vin.entry_dir != 0).sum())
                        n_tr = len(result.trades)
                        exclusions = vin.skipped_exclusive
                        reconciliations.append({'version':name,'direction_trigger_minutes':n_sig + sum(exclusions.values()),
                            'entry_signals':n_sig,'trades':n_tr,'position_or_cooldown_skips':n_sig - n_tr,
                            **{'exclusive_'+k:v for k,v in exclusions.items()},
                            **{'nonexclusive_'+k:v for k,v in vin.skipped.items()}})
        pd.DataFrame(all_metrics).to_csv(OUT / 'all80_comparison.csv', index=False)
        pd.DataFrame(reconciliations).to_csv(OUT / 'signal_trade_reconciliation.csv', index=False)
        excess, matching_summary = {}, {}
        for name, spec in VS.VERSIONS.items():
            logging.info('Matched controls %s: K=100, four costs, null B=5000', name)
            match = M.run_matched_controls(ctx, spec, ledgers[name])
            match.records.to_csv(OUT / f'matched_{name}.csv.gz', index=False, compression='gzip')
            np.savez_compressed(OUT / f'matched_samples_{name}.npz', sample_indices=match.sample_indices,
                                **match.random_net_bps)
            extra = main_results[name].trades.copy()
            records = match.records.set_index('trigger_idx')
            extra['excess_bps'] = records.loc[extra.trigger_idx, 'excess_main_17bp'].to_numpy()
            excess[name] = extra
            matching_summary[name] = {'relaxation_counts':match.relaxation_counts,'balance':match.balance,
                                       'quantiles':match.quantiles, 'candidate_count':match.candidate_count,
                                       'randomization':match.randomization}
            write_json(OUT / f'matched_summary_{name}.json', matching_summary[name])
        logging.info('Only CP3-authorized sensitivities: G8z and B full-period appendix')
        special = X.run_cp3_sensitivities(ctx)
        for name, result in special['causal_results'].items():
            save_result(result, f'G8z__{name}')
        appendix = {}
        for name, result in special['full_period_results'].items():
            save_result(result, f'appendix__{name}')
            v = list(VS.VERSIONS).index(name)
            appendix[name] = {'metrics':S.portfolio_metrics(result, appendix=True),
                'net_bps_month':S.cluster_mean(result.trades, ci_seed=VS.SEED_BASE+3500+v,
                                                p_seed=VS.SEED_BASE+3550+v, appendix=True)}
        appendix_comparable = {}
        for name, result in special['full_period_results'].items():
            aligned = result.trades.loc[result.trades.trigger_time >= pd.Timestamp('2020-01-07 15:51', tz='UTC')]
            appendix_comparable[name] = {'n': len(aligned), **{metric: S.cluster_point_mean(aligned, metric, appendix=True)
                for metric in ('gross_bps', 'net_bps', 'net_R')}}
        write_json(OUT / 'B_full_period_appendix.json', {'per_version_full_period': appendix,
            'pct_z_aligned_trade_comparison': appendix_comparable,
            'alignment_trigger_start_utc': '2020-01-07 15:51',
            'note': 'Full-period portfolio metrics have different feature-ready starts; compare percentage and Z trade metrics only in the aligned table. No matched controls; no gate entry.'})
        logging.info('Frozen statistics and gates')
        summary = S.evaluate_study(main_results, harsh_results, excess, special['causal_results'], context=ctx)
        summary['B_full_period_appendix'] = {'per_version_full_period': appendix, 'pct_z_aligned_trade_comparison': appendix_comparable}
        summary['matching'] = matching_summary
        logging.info('Figures 1-4 and independent onset controls')
        meta['figure_rows'] = P.make_figures(ctx, main_results, OUT)
        if X.context_fingerprint(ctx) != initial_fingerprint:
            raise AssertionError('main context mutated during CP3')
        R.frozen_hashes()
        reg = meta['registration']
        for section in ('new_file_sha256', 'existing_dependency_sha256'):
            for name, expected in reg[section].items():
                if R.sha256(ROOT / name) != expected:
                    raise AssertionError(f'registered file changed during run: {name}')
        table = []
        for name, entry in summary['versions'].items():
            row = {'version':name, **entry['metrics'], **entry['gates']}
            row.update({'G2_raw_p':entry['G2']['p'], 'G3_raw_p':entry['G3']['p']})
            table.append(row)
        pd.DataFrame(table).to_csv(OUT / 'main_comparison.csv', index=False)
        meta['finished_utc'] = pd.Timestamp.now(tz='UTC').isoformat()
        meta['runtime_seconds'] = time.time()-started
        meta['scope_counts'] = {'main_portfolios':80,'matched_version_costs':40,'B_appendix':5,'G8z':2,
                                'CP4_other_sensitivities':0,'VAL_OOS_ETH':0}
        summary['provenance'] = meta
        summary['status'] = 'CP3 completed; CP4 final review not performed'
        write_json(OUT / 'summary.json', summary)
        phase_log(meta, summary)
        logging.info('CP3 complete. Stop: no CP4 execution.')
        return summary
    except BaseException as exc:
        logging.exception('STOP: no repair, partial rerun, resume or further execution permitted')
        write_json(OUT / 'STOPPED.json', {'type':type(exc).__name__,'message':str(exc),
                    'traceback':traceback.format_exc(),'started':meta,'at_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-frozen-only', action='store_true', help='read code hashes only; no market input')
    args = parser.parse_args()
    if args.verify_frozen_only:
        print(json.dumps({'frozen_key':VS.frozen_key(),'sha256':R.frozen_hashes()},indent=2))
    else:
        main()
