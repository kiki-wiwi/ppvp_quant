"""
strategy_state_machine.py — 1H 箱体底部“二次确认 + 看涨 K 线”显式状态机（只做多）。

状态定义（逐根 1H 收盘后更新）
------------------------------
0 NO_BOX     尚无已确认箱体
1 ARMED      State 1：箱体激活，等待第 1 次“回落”触及支撑带 [VAL-k1*ATR, VAL+k2*ATR]
2 TOUCH1     State 2a：第 1 次测试进行中（记录测试低点）
3 DEPARTED   State 2b：价格整根 K 线回到支撑带上方（low > 带上沿，已脱离），等待第 2 次回踩
4 TOUCH2     State 3：第 2 次回踩进行中；触及支撑带、收盘 ≥ 带下沿、且为看涨 Pinbar/吞没 → 入场信号
5 INVALID    出现“放量实体跌穿支撑带下沿” → 该支撑区失效（解除条件见下）
6 EXHAUSTED  （仅 roll_failed_second_touch=False）第 2 次测试失败后不再接受信号，直至新箱体

关键规则（含对规格模糊处的显式解释；各条规则的合成回归测试见 test_audit_regressions.py）
------------------------------------------------------------------------------------
* “回落触及”：第 1 次测试必须从上方到来。箱体激活时若上一根收盘已低于带下沿，或刚发出过信号，
  需先出现一根整根位于带上方的 K 线（low > 带上沿），之后的触及才算第 1 次测试（need_above 标志）。
  激活时上一根收盘 ≥ 带下沿（价格在带内或带上方），则激活后的首次触及即视为第 1 次测试。
  ARMED 期间若箱体更新到同一支撑区（新 VAL 与设置该要求时的 VAL 相距 ≤ (k1+k2)*ATR），该要求保留；
  只有整根位于带上方的 K 线能清除它，换到不同支撑区时按新箱体重新判定。
* 第 2 次测试必须与第 1 次测试“分离”：脱离后最高价较当根带上沿反弹 ≥ min_bounce_atr*ATR，
  或当根与第 1 次测试最后一根带内 K 线的索引差 ≥ min_bars_between_touches；否则并入第 1 次测试。
* 允许插针跌破 VAL / 第 1 次低点（SFP 流动性清扫），“站回有效区”解释为信号 K 线收盘 ≥ 带下沿。
* 确认形态可出现在第 2 次测试连续在带内的任意一根（max_pattern_lag_bars=None，基线）；
  设为 0 则只接受重新进入支撑带的那一根（规格“当根”的严格解释）。
* 止损 SL = min(第 1 次测试低点, 第 2 次测试锚点低点[, 吞没前一根低点]) - sl_buffer_atr*ATR；
  锚点 sl_anchor='test'（基线）= 第 2 次测试全部 K 线的最低点，'pattern' = 仅确认形态 K 线低点。
* 止盈 TP = 箱体 VAH（基线）；POC 一并输出供分批止盈变体使用。
* 一次二次确认只产生一个信号；信号发出后序列被消费（回到 ARMED 且 need_above=True），
  必须先脱离到带上方，再出现全新的两次测试才会再次发信号。
  过滤器（min_RR / 箱体宽度 / 宏观状态）只决定该信号是否交易，不改变状态机序列。
* 箱体更新：每个新确认枢轴都会生成新箱体。若新箱体 VAL 与“本轮第 1 次测试开始时箱体的 VAL”相距
  ≤ (k1+k2)*ATR（同一支撑区），测试进度保留（carry）；否则重新 ARMED。
* 失效（INVALID）：箱体失效位 = 支撑带下沿 VAL-k1*ATR；放量实体跌穿 = 阴线、收盘 < 带下沿、
  实体 ≥ break_body_atr*ATR、成交量 ≥ break_vol_mult*均量。
  放量跌穿记忆：每条记录被放量跌穿的支撑带 VAL 与跌穿 K 线的收盘时刻 t，最多同时保留 MEM_SLOTS 条、互不覆盖。
  在 ARMED/TOUCH/DEPARTED/EXHAUSTED 中出现放量跌穿时写入一条并进入 INVALID；INVALID 期间若与某条记忆属同一
  支撑区的箱体再次放量跌穿，只有收盘也低于该条记忆的带下沿（VAL_mem - k1*ATR）才把其 t 后移；
  若属其他支撑区（如被否决的箱体），则写入新的一条。
  每次箱体切换（任何状态）时：先使 t ≤ 新箱体窗口起点的记忆失效（窗口起点随箱体单调不减，此后所有箱体都完全由
  该次跌穿后的数据构成，因此不会死锁）；若新箱体 VAL 与任一有效记忆相距 ≤ (k1+k2)*ATR，则状态变为或保持 INVALID；
  否则 INVALID 解除、重新 ARMED（或按 carry 保留进度）。这条规则在任何状态下检查：绕道经过其他支撑区再回到被跌穿
  的支撑位，或期间又有另一个支撑区被跌穿，同样会被判为失效。
* reject_broken_at_activation：箱体激活时上一根收盘已低于其带下沿 → 仅否决这一个箱体（INVALID 至下一个箱体），
  不写入放量跌穿记忆；下一个箱体按自身条件重新判定。
* 所有判定只使用第 i 根 1H 收盘时已知的信息：箱体取 available_time ≤ open_time[i]，
  ATR/均量取第 i-1 根收盘值（第 i 根开盘时已知），形态取第 i 根及之前的 OHLC。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import StrategyConfig
from numba_compat import njit
from ppvp_engine import build_4h, build_boxes, map_boxes_to_1h, shift_known_at_open, to_ns, wilder_atr

NO_BOX, ARMED, TOUCH1, DEPARTED, TOUCH2, INVALID, EXHAUSTED = range(7)
MEM_SLOTS = 8   # 放量跌穿记忆容量（同时有效的被跌穿支撑区个数上限；条目在“全新结构”箱体出现后自动失效）
STATE_NAMES = {0: "NO_BOX", 1: "ARMED", 2: "TOUCH1", 3: "DEPARTED", 4: "TOUCH2", 5: "INVALID", 6: "EXHAUSTED"}
PAT_PIN, PAT_ENGULF = 1, 2


# ----------------------------------------------------------------------------
# 看涨 K 线形态（向量化，只用当根与前一根）
# ----------------------------------------------------------------------------
def bullish_pinbar(o, h, l, c, cfg: StrategyConfig) -> np.ndarray:
    """下影线 ≥ pin_wick_frac*振幅、下影线 ≥ pinbar_ratio*实体、收盘位置 ≥ pin_close_pos。

    由于 (c-l) ≥ 下影线，pin_close_pos ≤ pin_wick_frac 时收盘位置条件被下影线条件蕴含、不起约束作用
    （此时不应把它列入参数平原/扰动表）。规格未要求实体颜色；pin_require_bull_close=True 时额外要求收阳。
    """
    rng = h - l
    body = np.abs(c - o)
    lower = np.minimum(o, c) - l
    with np.errstate(invalid="ignore"):
        ok = (rng > 0) & (lower >= cfg.pin_wick_frac * rng) & (lower >= cfg.pinbar_ratio * body) \
            & ((c - l) >= cfg.pin_close_pos * rng)
        if cfg.pin_require_bull_close:
            ok &= c > o
    return ok & cfg.use_pinbar


def bullish_engulfing(o, h, l, c, cfg: StrategyConfig) -> np.ndarray:
    po = np.concatenate([[np.nan], o[:-1]])
    pc = np.concatenate([[np.nan], c[:-1]])
    prev_body = po - pc                                  # 前一根阴线实体（>0）
    lo_cur = np.where(np.abs(o - pc) <= cfg.engulf_gap_tol * pc, np.minimum(o, pc), o)
    with np.errstate(invalid="ignore", divide="ignore"):
        cover = (np.minimum(c, po) - np.maximum(lo_cur, pc)) / prev_body
        ok = (prev_body > 0) & (c > o) & (cover >= cfg.engulf_ratio - 1e-12)
    ok[0] = False
    return ok & cfg.use_engulfing


# ----------------------------------------------------------------------------
# 状态机核心（Numba JIT；未安装 numba 时同一代码以纯 Python 运行）
# ----------------------------------------------------------------------------
@njit(cache=True)
def _mem_add(hv_vals, hv_ts, val, t):
    """写入一条放量跌穿记忆：优先使用空槽；槽满时覆盖最早失效（跌穿时刻最早）的条目。"""
    slot = -1
    for e in range(hv_ts.shape[0]):
        if hv_ts[e] == 0:
            slot = e
            break
    if slot < 0:
        slot = 0
        for e in range(1, hv_ts.shape[0]):
            if hv_ts[e] < hv_ts[slot]:
                slot = e
    hv_vals[slot] = val
    hv_ts[slot] = t


@njit(cache=True)
def _run_state_machine(o, h, l, c, v, atr, vsma, open_ns, box_idx, box_val, box_vah, box_wstart,
                       pin, eng, k1, k2, min_bars, min_bounce, carry, roll,
                       brk_body, brk_vol, sl_buf, reject_broken, max_lag, sl_pattern):
    n = o.shape[0]
    state = np.zeros(n, np.int8)
    zlo = np.full(n, np.nan)
    zhi = np.full(n, np.nan)
    brk = np.zeros(n, np.bool_)
    sig = np.zeros(n, np.bool_)
    pat = np.zeros(n, np.int8)
    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    s_t1_low = np.full(n, np.nan)
    s_t1_bar = np.full(n, -1, np.int64)
    s_t2_low = np.full(n, np.nan)
    s_t2_bar = np.full(n, -1, np.int64)
    s_t2_start = np.full(n, -1, np.int64)
    s_box = np.full(n, -1, np.int64)
    s_need = np.zeros(n, np.bool_)

    hour = np.int64(3600) * np.int64(1000000000)
    st = 0
    cur = -1
    t1_low = np.nan
    t1_bar = -1
    t1_last = -1
    max_high = np.nan
    t2_low = np.nan
    t2_bar = -1
    t2_start = -1
    seq_val = np.nan          # 本轮第 1 次测试开始时所在箱体的 VAL（carry 的比较基准）
    t2_val = np.nan           # 第 2 次测试开始时所在箱体的 VAL（滚动后成为新的 seq_val）
    hv_vals = np.full(MEM_SLOTS, np.nan)            # 放量跌穿记忆：被放量跌穿的支撑带 VAL（多条，互不覆盖）
    hv_ts = np.zeros(MEM_SLOTS, np.int64)          # 对应跌穿 K 线收盘时刻；0 = 空槽或已失效
    need_above = True         # True：须先出现整根位于带上方的 K 线，触及才算第 1 次测试
    need_val = np.nan         # 设置 need_above 时所在箱体的 VAL（同一支撑区的箱体更新不清除该要求）
    for i in range(n):
        b = box_idx[i]
        a = atr[i]
        if b != cur:
            if b < 0:
                st = 0
            else:
                ov = (k1 + k2) * a
                hv_block = False
                for e in range(MEM_SLOTS):
                    if hv_ts[e] > 0:
                        if box_wstart[b] >= hv_ts[e]:
                            hv_ts[e] = 0                   # 窗口起点单调不减：此后所有箱体都不含该次跌穿前的数据
                            hv_vals[e] = np.nan
                        elif (not np.isnan(a)) and abs(box_val[b] - hv_vals[e]) <= ov:
                            hv_block = True
                was_armed = st == 1
                keep = False
                if (not hv_block) and carry and cur >= 0 and (st == 2 or st == 3 or st == 4) and not np.isnan(a):
                    if abs(box_val[b] - seq_val) <= ov:
                        keep = True                        # 同一支撑区：保留测试进度
                if hv_block or not keep:
                    t1_low = np.nan
                    t1_bar = -1
                    t1_last = -1
                    max_high = np.nan
                    t2_low = np.nan
                    t2_bar = -1
                    t2_start = -1
                    seq_val = np.nan
                    t2_val = np.nan
                if hv_block:
                    st = 5                                 # 回到被放量跌穿、且仍含跌穿前数据的支撑区：失效
                elif not keep:
                    st = 1
                    zl_new = box_val[b] - k1 * a
                    below = not (i > 0 and c[i - 1] >= zl_new)          # a 为 NaN 时比较为 False → 需先到带上方
                    if was_armed and need_above and (not np.isnan(a)) and abs(box_val[b] - need_val) <= ov:
                        need_above = True                  # 同一支撑区的箱体更新不清除未完成的“先到带上方”要求
                    else:
                        need_above = below
                        need_val = box_val[b]
                    if reject_broken and i > 0 and not np.isnan(a) and c[i - 1] < zl_new:
                        st = 5                             # 仅否决这一个箱体，不写入放量跌穿记忆
            cur = b
        s_box[i] = cur
        if st == 0 or np.isnan(a):
            state[i] = st
            s_need[i] = need_above
            continue
        zl = box_val[cur] - k1 * a
        zh = box_val[cur] + k2 * a
        zlo[i] = zl
        zhi[i] = zh
        is_brk = (c[i] < o[i]) and (c[i] < zl) and (o[i] - c[i] >= brk_body * a) and (v[i] >= brk_vol * vsma[i])
        brk[i] = is_brk
        if st == 5 or st == 6:
            if is_brk:
                same = False
                if st == 5:
                    for e in range(MEM_SLOTS):
                        if hv_ts[e] > 0 and abs(box_val[cur] - hv_vals[e]) <= (k1 + k2) * a:
                            same = True
                            if c[i] < hv_vals[e] - k1 * a:
                                hv_ts[e] = open_ns[i] + hour   # 同一支撑区再次放量跌穿被跌穿的带下沿：记忆时刻后移
                if not same:
                    _mem_add(hv_vals, hv_ts, box_val[cur], open_ns[i] + hour)   # EXHAUSTED 或其他支撑区被放量跌穿
                    st = 5
            state[i] = st
            s_need[i] = need_above
            continue
        if is_brk:
            st = 5
            _mem_add(hv_vals, hv_ts, box_val[cur], open_ns[i] + hour)
            state[i] = st
            s_need[i] = need_above
            continue
        touched = l[i] <= zh
        if st == 1:
            if need_above:
                if l[i] > zh:
                    need_above = False                     # 已到带上方：之后的触及才是“回落”测试
            elif touched:
                st = 2
                t1_low = l[i]
                t1_bar = i
                t1_last = i
                seq_val = box_val[cur]
        elif st == 2:
            if touched:
                if l[i] < t1_low:
                    t1_low = l[i]
                    t1_bar = i
                t1_last = i
            else:
                st = 3
                max_high = h[i]
        elif st == 3:
            if touched:
                if (max_high - zh >= min_bounce * a) or (i - t1_last >= min_bars):
                    st = 4
                    t2_low = l[i]
                    t2_bar = i
                    t2_start = i
                    t2_val = box_val[cur]
                else:
                    st = 2
                    if l[i] < t1_low:
                        t1_low = l[i]
                        t1_bar = i
                    t1_last = i
            else:
                if h[i] > max_high:
                    max_high = h[i]
        elif st == 4:
            if touched:
                if l[i] < t2_low:
                    t2_low = l[i]
                    t2_bar = i
            else:
                if roll:
                    st = 3
                    t1_low = t2_low
                    t1_bar = t2_bar
                    t1_last = i - 1
                    seq_val = t2_val
                    max_high = h[i]
                    t2_low = np.nan
                    t2_bar = -1
                    t2_start = -1
                    t2_val = np.nan
                else:
                    st = 6
        if st == 4 and touched and c[i] >= zl and (max_lag < 0 or i - t2_start <= max_lag):
            p = 0
            if pin[i]:
                p |= 1
            if eng[i]:
                p |= 2
            if p > 0:
                if sl_pattern:
                    lowest = min(t1_low, l[i])
                else:
                    lowest = min(t1_low, t2_low)
                if (p & 2) and i > 0 and l[i - 1] < lowest:
                    lowest = l[i - 1]
                sig[i] = True
                pat[i] = p
                sl[i] = lowest - sl_buf * a
                tp[i] = box_vah[cur]
                s_t1_low[i] = t1_low
                s_t1_bar[i] = t1_bar
                s_t2_low[i] = t2_low
                s_t2_bar[i] = t2_bar
                s_t2_start[i] = t2_start
                st = 1                                   # 序列被消费：须先脱离到带上方，再出现全新的两次测试
                need_above = True
                need_val = box_val[cur]
                t1_low = np.nan
                t1_bar = -1
                t1_last = -1
                max_high = np.nan
                t2_low = np.nan
                t2_bar = -1
                t2_start = -1
                seq_val = np.nan
                t2_val = np.nan
        state[i] = st
        s_need[i] = need_above
    return (state, zlo, zhi, brk, sig, pat, sl, tp, s_t1_low, s_t1_bar, s_t2_low, s_t2_bar, s_t2_start,
            s_box, s_need)


# ----------------------------------------------------------------------------
# 宏观状态过滤（只用决策时刻已收盘的日线 / 4H 数据）
# ----------------------------------------------------------------------------
def daily_closes_with_warmup(df1h: pd.DataFrame, spot_daily: pd.Series | None) -> pd.Series:
    """日线收盘序列：永续 1H 聚合的完整日线；2020 年之前以现货日线收盘预热（仅价格，不含成交量）。

    索引为日线开盘日期 D，该日线在 D+1 00:00 收盘后才可用。
    """
    d = df1h["close"].resample("1D").last()
    last_close = df1h.index[-1] + pd.Timedelta(hours=1)
    d = d[(d.index + pd.Timedelta(days=1)) <= last_close].dropna()   # 只保留已收盘的日线
    if spot_daily is not None and len(d):
        warm = spot_daily[spot_daily.index < d.index[0]]
        d = pd.concat([warm, d])
    return d


def regime_block_mask(df1h: pd.DataFrame, df4h: pd.DataFrame, cfg: StrategyConfig,
                      spot_daily: pd.Series | None) -> np.ndarray:
    """True = 第 i 根 1H 收盘时处于需规避的极端空头状态（只在决策时刻 close_time[i] 已知的数据上计算）。"""
    n = len(df1h)
    if cfg.regime_filter is None:
        return np.zeros(n, dtype=bool)
    close_1h_ns = to_ns(df1h.index) + np.int64(3600 * 10**9)
    if cfg.regime_filter == "daily_sma":
        d = daily_closes_with_warmup(df1h, spot_daily)
        sma = d.rolling(cfg.regime_sma_days, min_periods=cfg.regime_sma_days).mean()
        slope = sma - sma.shift(cfg.regime_slope_days)
        bear = ((d < sma) & (slope < 0)).to_numpy()
        avail = to_ns(d.index) + np.int64(86400 * 10**9)              # 日线 D 于 D+1 00:00 可用
        j = np.searchsorted(avail, close_1h_ns, side="right") - 1
        out = np.zeros(n, dtype=bool)
        ok = j >= 0
        out[ok] = bear[j[ok]]
        return out
    if cfg.regime_filter == "crash_cooldown":
        o4, h4, l4, c4, v4 = (df4h[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close", "volume"))
        atr4 = shift_known_at_open(wilder_atr(h4, l4, c4, 14))
        vs4 = shift_known_at_open(pd.Series(v4).rolling(20, min_periods=20).mean().to_numpy())
        with np.errstate(invalid="ignore"):
            crash = ((o4 - c4) >= cfg.crash_body_atr4 * atr4) & (v4 >= cfg.crash_vol_mult4 * vs4)
        ev = to_ns(df4h["available_time"])[crash]                    # 崩跌 4H 收盘时刻
        j = np.searchsorted(ev, close_1h_ns, side="right") - 1
        out = np.zeros(n, dtype=bool)
        ok = j >= 0
        out[ok] = (close_1h_ns[ok] - ev[j[ok]]) < np.int64(cfg.crash_cooldown_hours * 3600 * 10**9)
        return out
    raise ValueError(f"unknown regime_filter {cfg.regime_filter}")


# ----------------------------------------------------------------------------
# 对外接口
# ----------------------------------------------------------------------------
def indicator_arrays(df1h: pd.DataFrame, cfg: StrategyConfig) -> dict:
    """状态机与回测共用的逐根指标（全部为第 i 根收盘决策时已知的量）。"""
    o, h, l, c, v = (df1h[k].to_numpy(dtype=np.float64) for k in ("open", "high", "low", "close", "volume"))
    atr_prev = shift_known_at_open(wilder_atr(h, l, c, cfg.atr_len))
    vsma_prev = shift_known_at_open(pd.Series(v).rolling(cfg.vol_sma_len, min_periods=cfg.vol_sma_len).mean().to_numpy())
    return {"o": o, "h": h, "l": l, "c": c, "v": v, "atr_prev": atr_prev, "vsma_prev": vsma_prev,
            "pin": bullish_pinbar(o, h, l, c, cfg), "eng": bullish_engulfing(o, h, l, c, cfg)}


def state_machine_params(cfg: StrategyConfig) -> tuple:
    """状态机核心的标量参数（顺序与 _run_state_machine 签名一致）。"""
    return (float(cfg.zone_k1), float(cfg.zone_k2), int(cfg.min_bars_between_touches), float(cfg.min_bounce_atr),
            bool(cfg.carry_touches_on_box_update), bool(cfg.roll_failed_second_touch),
            float(cfg.break_body_atr), float(cfg.break_vol_mult), float(cfg.sl_buffer_atr),
            bool(cfg.reject_broken_at_activation),
            int(-1 if cfg.max_pattern_lag_bars is None else cfg.max_pattern_lag_bars),
            bool(cfg.sl_anchor == "pattern"))


def state_machine_args(df1h: pd.DataFrame, cfg: StrategyConfig, boxes: pd.DataFrame, ind: dict | None = None) -> tuple:
    """组装状态机核心的全部输入（向量化预计算部分）。"""
    ind = ind if ind is not None else indicator_arrays(df1h, cfg)
    box_idx = map_boxes_to_1h(df1h, boxes)
    nb = len(boxes)
    bval = boxes["VAL"].to_numpy(dtype=np.float64) if nb else np.zeros(0)
    bvah = boxes["VAH"].to_numpy(dtype=np.float64) if nb else np.zeros(0)
    bws = to_ns(boxes["window_start"]) if nb else np.zeros(0, dtype=np.int64)
    return (ind["o"], ind["h"], ind["l"], ind["c"], ind["v"], ind["atr_prev"], ind["vsma_prev"], to_ns(df1h.index),
            box_idx, bval, bvah, bws, ind["pin"], ind["eng"]) + state_machine_params(cfg)


def generate_signals(df1h: pd.DataFrame, cfg: StrategyConfig, boxes: pd.DataFrame | None = None,
                     df4h: pd.DataFrame | None = None, spot_daily: pd.Series | None = None,
                     use_jit: bool = True) -> pd.DataFrame:
    """对 1H 数据逐根运行状态机，返回逐根诊断与信号 DataFrame（索引 = 1H open_time）。

    除信号与价位外，还输出全部中间量（pin/eng/vsma/regime_raw/brk_bodyvol/k1_atr/need_above），
    使防未来函数闸门在每个截断点都能逐位比较它们，而不只是在恰好落在信号 K 线上的截断点。
    """
    if boxes is None or df4h is None:
        boxes, df4h = build_boxes(df1h, cfg, df4h)
    ind = indicator_arrays(df1h, cfg)
    args = state_machine_args(df1h, cfg, boxes, ind)
    o, c, v, atr_prev, vsma_prev = ind["o"], ind["c"], ind["v"], ind["atr_prev"], ind["vsma_prev"]
    nb = len(boxes)
    fn = _run_state_machine if use_jit else _run_state_machine.py_func
    (state, zlo, zhi, brk, sig_raw, pat, sl, tp, t1_low, t1_bar, t2_low, t2_bar, t2_start, sbox, need) = fn(*args)

    def box_col(name):
        out = np.full(len(df1h), np.nan)
        ok = sbox >= 0
        if nb:
            out[ok] = boxes[name].to_numpy(dtype=np.float64)[sbox[ok]]
        return out

    val, poc, vah, plow, phigh = (box_col(k) for k in ("VAL", "POC", "VAH", "P_low", "P_high"))
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = np.where(sig_raw, (tp - c) / (c - sl), np.nan)
        block_rr = sig_raw & ~((tp > c) & (rr >= cfg.min_rr))
        width = vah - val
        block_width = np.zeros(len(df1h), dtype=bool)
        if cfg.max_box_width_atr is not None:
            block_width |= sig_raw & (width / atr_prev > cfg.max_box_width_atr)
        if cfg.max_box_width_pct is not None:
            block_width |= sig_raw & (width / val > cfg.max_box_width_pct)
        # 持仓期“跌穿支撑带”离场所需的、与箱体无关的部分（回测器用入场时箱体的 VAL 冻结参照位）
        brk_bodyvol = (c < o) & (o - c >= cfg.break_body_atr * atr_prev) & (v >= cfg.break_vol_mult * vsma_prev)
    regime_raw = regime_block_mask(df1h, df4h, cfg, spot_daily)
    block_regime = sig_raw & regime_raw
    signal = sig_raw & ~block_rr & ~block_width & ~block_regime

    out = pd.DataFrame({
        "box_id": sbox, "VAL": val, "POC": poc, "VAH": vah, "P_low": plow, "P_high": phigh,
        "atr": atr_prev, "vsma": vsma_prev, "zone_lo": zlo, "zone_hi": zhi, "state": state, "need_above": need,
        "breakdown": brk, "brk_bodyvol": brk_bodyvol, "k1_atr": cfg.zone_k1 * atr_prev,
        "pin": ind["pin"], "eng": ind["eng"], "regime_raw": regime_raw,
        "sig_raw": sig_raw, "signal": signal, "pattern": pat, "entry_ref": np.where(sig_raw, c, np.nan),
        "sl": sl, "tp": tp, "poc_tp": np.where(sig_raw, poc, np.nan), "rr": rr,
        "t1_low": t1_low, "t1_bar": t1_bar, "t2_low": t2_low, "t2_bar": t2_bar, "t2_start": t2_start,
        "block_rr": block_rr, "block_width": block_width, "block_regime": block_regime,
    }, index=df1h.index)
    return out
