import numpy as np


def run_sm_instr(o, h, l, c, v, atr, vsma, open_ns, box_idx, box_val, box_vah, box_wstart,
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
    s_seq_start = np.full(n, -1, np.int64)
    s_seq_box = np.full(n, -1, np.int64)
    s_seq_val = np.full(n, np.nan)
    s_t2_val = np.full(n, np.nan)
    s_carry = np.zeros(n, np.bool_)
    seq_start = -1
    seq_box = -1
    s_chain_start = np.full(n, -1, np.int64)
    s_nroll = np.zeros(n, np.int64)
    chain_start = -1
    nroll = 0
    s_cdist = np.full(n, np.nan)

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
    # 放量跌穿记忆（紧凑列表 [0, hcnt)）：被跌穿支撑带的 VAL 与跌穿 K 线收盘时刻。每根 K 线至多写入一条，
    # 容量 n + 1 永远不会溢出，有效记忆不会被覆盖。
    hv_vals = np.empty(n + 1)
    hv_ts = np.empty(n + 1, np.int64)
    hcnt = 0
    # “须先到带上方”要求（紧凑列表 [0, ncnt)）：锚点 VAL、写入时刻、是否被当前箱体认领。每根 K 线至多写入一条。
    nd_val = np.empty(n + 1)
    nd_t = np.empty(n + 1, np.int64)
    nd_cl = np.zeros(n + 1, np.bool_)
    ncnt = 0
    need_above = True         # = 是否存在被当前箱体认领的未完成要求（尚无箱体时无意义，输出为 True）
    for i in range(n):
        b = box_idx[i]
        a = atr[i]
        if b != cur:
            if b < 0:
                st = 0
            else:
                ov = (k1 + k2) * a
                # 放量跌穿记忆：先使窗口起点 ≥ 跌穿时刻的条目永久失效（窗口起点随箱体单调不减），再检查同区
                e = hcnt - 1
                while e >= 0:
                    if box_wstart[b] >= hv_ts[e]:
                        hcnt -= 1
                        hv_vals[e] = hv_vals[hcnt]
                        hv_ts[e] = hv_ts[hcnt]
                    e -= 1
                hv_block = False
                if not np.isnan(a):
                    for e in range(hcnt):
                        if abs(box_val[b] - hv_vals[e]) <= ov:
                            hv_block = True
                            break
                keep = False
                if (not hv_block) and carry and cur >= 0 and (st == 2 or st == 3 or st == 4) and not np.isnan(a):
                    if abs(box_val[b] - seq_val) <= ov:
                        keep = True                        # 同一支撑区：保留测试进度
                s_carry[i] = keep and not hv_block
                if cur >= 0 and (st == 2 or st == 3 or st == 4) and not np.isnan(a):
                    s_cdist[i] = abs(box_val[b] - seq_val) / a
                if hv_block or not keep:
                    seq_start = -1
                    seq_box = -1
                    chain_start = -1
                    nroll = 0
                    t1_low = np.nan
                    t1_bar = -1
                    t1_last = -1
                    max_high = np.nan
                    t2_low = np.nan
                    t2_bar = -1
                    t2_start = -1
                    seq_val = np.nan
                    t2_val = np.nan
                    # 新箱体认领同区的全部未完成“须先到带上方”要求（其余要求保留，按各自锚点的带上沿清除）
                    need_above = False
                    for e in range(ncnt):
                        nd_cl[e] = (not np.isnan(a)) and abs(box_val[b] - nd_val[e]) <= ov
                        if nd_cl[e]:
                            need_above = True
                # 未被认领（不在当前支撑区）的要求：出现完全由其写入之后的数据构成的箱体（全新结构）时失效
                e = ncnt - 1
                while e >= 0:
                    if (not nd_cl[e]) and box_wstart[b] >= nd_t[e]:
                        ncnt -= 1
                        nd_val[e] = nd_val[ncnt]
                        nd_t[e] = nd_t[ncnt]
                        nd_cl[e] = nd_cl[ncnt]
                    e -= 1
                if hv_block:
                    st = 5                                 # 回到被放量跌穿、且仍含跌穿前数据的支撑区：失效
                elif not keep:
                    st = 1
                    zl_new = box_val[b] - k1 * a
                    below = not (i > 0 and c[i - 1] >= zl_new)          # a 为 NaN 时比较为 False → 需先到带上方
                    if below and not need_above:
                        nd_val[ncnt] = box_val[b]
                        nd_t[ncnt] = open_ns[i]            # 依据第 i-1 根收盘（第 i 根开盘时已知）
                        nd_cl[ncnt] = True
                        ncnt += 1
                        need_above = True
                    if reject_broken and i > 0 and not np.isnan(a) and c[i - 1] < zl_new:
                        st = 5                             # 否决该箱体；否决本身不写入放量跌穿记忆
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
        if ncnt > 0:                                       # 整根位于某支撑区带上方的 K 线清除该区的要求（任何状态）
            e = ncnt - 1
            while e >= 0:
                if l[i] > (zh if nd_cl[e] else nd_val[e] + k2 * a):
                    ncnt -= 1
                    nd_val[e] = nd_val[ncnt]
                    nd_t[e] = nd_t[ncnt]
                    nd_cl[e] = nd_cl[ncnt]
                e -= 1
            need_above = False
            for e in range(ncnt):
                if nd_cl[e]:
                    need_above = True
                    break
        is_brk = (c[i] < o[i]) and (c[i] < zl) and (o[i] - c[i] >= brk_body * a) and (v[i] >= brk_vol * vsma[i])
        brk[i] = is_brk
        if st == 5 or st == 6:
            if is_brk:
                same = False
                if st == 5:
                    for e in range(hcnt):
                        if abs(box_val[cur] - hv_vals[e]) <= (k1 + k2) * a:
                            same = True
                            if c[i] < hv_vals[e] - k1 * a:
                                hv_ts[e] = open_ns[i] + hour   # 同一支撑区再次放量跌穿被跌穿的带下沿：记忆时刻后移
                if not same:
                    hv_vals[hcnt] = box_val[cur]               # EXHAUSTED 或其他支撑区被放量跌穿：写入新的一条
                    hv_ts[hcnt] = open_ns[i] + hour
                    hcnt += 1
                    st = 5
            state[i] = st
            s_need[i] = need_above
            continue
        if is_brk:
            st = 5
            hv_vals[hcnt] = box_val[cur]
            hv_ts[hcnt] = open_ns[i] + hour
            hcnt += 1
            state[i] = st
            s_need[i] = need_above
            continue
        touched = l[i] <= zh
        if st == 1:
            if need_above:
                pass                                       # 本根未到带上方（到了已在上方清除）：触及不算第 1 次测试
            elif touched:
                st = 2
                seq_start = i
                seq_box = cur
                chain_start = i
                nroll = 0
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
                    seq_start = t2_start
                    seq_box = s_box[t2_start]
                    nroll += 1
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
                s_seq_start[i] = seq_start
                s_seq_box[i] = seq_box
                s_seq_val[i] = seq_val
                s_t2_val[i] = t2_val
                s_chain_start[i] = chain_start
                s_nroll[i] = nroll
                seq_start = -1
                seq_box = -1
                chain_start = -1
                nroll = 0
                st = 1                                   # 序列被消费：须先脱离到带上方，再出现全新的两次测试
                nd_val[ncnt] = box_val[cur]
                nd_t[ncnt] = open_ns[i] + hour
                nd_cl[ncnt] = True
                ncnt += 1
                need_above = True
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
            s_box, s_need, s_seq_start, s_seq_box, s_seq_val, s_t2_val, s_carry, s_chain_start, s_nroll, s_cdist)
