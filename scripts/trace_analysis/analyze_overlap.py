#!/usr/bin/env python3
"""
Compare AMD MI350X vs NV B200 8-GPU traces for OVERLAP and IMBALANCE.

For each trace we compute, per iteration, per GPU:
  - critical-path (iter wall-time on the slowest stream)
  - per-stream (tid) busy time
  - per-category busy time on each stream
  - exposed-RCCL = total RCCL on that GPU minus RCCL time that is
    overlapped by compute on a *different* stream of the same GPU
  - count of distinct active streams (>=5% of iter wall-time)
  - inter-iter gap (time GPU is idle between iterations)

For cross-GPU imbalance:
  - per-iter min/median/max wall-time across GPUs
  - per-iter critical-path GPU id
  - per-iter "wait at sync" = median->max delta (this is the delay the
    fastest GPUs add to the slowest because of barrier-style RCCL)
"""
import json, sys, statistics, collections, os

def load(path):
    with open(path) as f:
        d = json.load(f)
    return d['traceEvents'] if isinstance(d, dict) else d

def per_iter_per_gpu(ev):
    """Returns {iter: {pid: [events]}} keeping only ph=='X' (complete events)
    that have an 'iter' arg."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in ev:
        if e.get('ph') != 'X': continue
        a = e.get('args') or {}
        it = a.get('iter')
        if it is None: continue
        out[it][e['pid']].append(e)
    return out

def stream_busy(events):
    """Returns {tid: total_dur_us}"""
    s = collections.defaultdict(float)
    for e in events:
        s[e['tid']] += e.get('dur', 0.0)
    return s

def category_busy(events):
    s = collections.defaultdict(float)
    for e in events:
        s[e.get('cat','?')] += e.get('dur', 0.0)
    return s

def category_busy_by_stream(events):
    s = collections.defaultdict(lambda: collections.defaultdict(float))
    for e in events:
        s[e['tid']][e.get('cat','?')] += e.get('dur', 0.0)
    return s

def merge_intervals(events, key=lambda e: True):
    """Returns list of (start, end) merged intervals for events matching key,
    on the union of all streams."""
    iv = sorted(((e['ts'], e['ts'] + e.get('dur', 0.0))
                 for e in events if key(e) and e.get('dur', 0)),
                key=lambda x: x[0])
    if not iv: return []
    out = [iv[0]]
    for s, en in iv[1:]:
        if s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], en))
        else:
            out.append((s, en))
    return out

def total_interval(intervals):
    return sum(b - a for a, b in intervals)

def overlap(a_intervals, b_intervals):
    """Total time covered by BOTH a and b intervals (intersection length)."""
    i = j = 0
    tot = 0.0
    while i < len(a_intervals) and j < len(b_intervals):
        a0, a1 = a_intervals[i]
        b0, b1 = b_intervals[j]
        lo = max(a0, b0); hi = min(a1, b1)
        if hi > lo:
            tot += hi - lo
        if a1 <= b1:
            i += 1
        else:
            j += 1
    return tot

def iter_wall(events):
    if not events: return 0.0, 0.0
    starts = [e['ts'] for e in events]
    ends = [e['ts'] + e.get('dur', 0.0) for e in events]
    return min(starts), max(ends)

def analyze(path, label):
    print('='*80)
    print(f'{label}: {path}')
    print('='*80)
    ev = load(path)
    print(f'  events={len(ev)}')
    pi = per_iter_per_gpu(ev)
    iters = sorted(pi.keys())
    print(f'  iters={iters[:5]}{"..." if len(iters)>5 else ""}  total={len(iters)}')
    pids = sorted({p for it in iters for p in pi[it]})
    print(f'  GPUs (pids)={pids}')

    # === per-iter per-GPU summary ===
    rows = []
    for it in iters:
        for pid in pids:
            evs = pi[it].get(pid, [])
            if not evs: continue
            iter_start, iter_end = iter_wall(evs)
            wall = iter_end - iter_start
            sb = stream_busy(evs)
            cb = category_busy(evs)
            cb_per_stream = category_busy_by_stream(evs)

            # Streams where >=5% of iter is busy
            active_streams = sum(1 for tid, t in sb.items() if t >= 0.05 * wall)

            # RCCL intervals (any stream)
            rccl_iv = merge_intervals(evs, key=lambda e: e.get('cat') == 'rccl' or e.get('cat') in ('allreduce', 'emb_a2a'))
            rccl_total = total_interval(rccl_iv)

            # Compute (non-RCCL, non-host) intervals
            comp_iv = merge_intervals(evs, key=lambda e: e.get('cat') not in ('rccl', 'allreduce', 'emb_a2a', 'host_api', 'launch', 'iter_marker', '?', 'memset', 'memcpy'))
            comp_total = total_interval(comp_iv)

            # Memcpy intervals
            mcp_iv = merge_intervals(evs, key=lambda e: e.get('cat') == 'memcpy')

            # Overlap
            ov_rccl_comp = overlap(rccl_iv, comp_iv)
            exposed_rccl = rccl_total - ov_rccl_comp
            overlap_rate = (ov_rccl_comp / rccl_total) if rccl_total else 0.0

            rows.append(dict(it=it, pid=pid,
                             wall=wall,
                             rccl=rccl_total,
                             comp=comp_total,
                             rccl_ov=ov_rccl_comp,
                             rccl_exposed=exposed_rccl,
                             overlap_rate=overlap_rate,
                             active_streams=active_streams,
                             cb=dict(cb)))

    # === aggregate per-GPU averages ===
    print('\n--- PER-GPU AVERAGES (across iters) ---')
    print(f'{"GPU":>4}  {"wall_us":>9} {"comp_us":>9} {"rccl_us":>9} {"rccl_ov":>8} {"exposed":>8} {"ov%":>5} {"streams":>7}')
    pid_avg = {}
    for pid in pids:
        rs = [r for r in rows if r['pid'] == pid]
        if not rs: continue
        wall = statistics.mean(r['wall'] for r in rs)
        rccl = statistics.mean(r['rccl'] for r in rs)
        comp = statistics.mean(r['comp'] for r in rs)
        rccl_ov = statistics.mean(r['rccl_ov'] for r in rs)
        exp = statistics.mean(r['rccl_exposed'] for r in rs)
        ovr = statistics.mean(r['overlap_rate'] for r in rs)
        streams = statistics.mean(r['active_streams'] for r in rs)
        pid_avg[pid] = dict(wall=wall, rccl=rccl, comp=comp, exposed=exp, overlap_rate=ovr, streams=streams)
        print(f'{pid:>4}  {wall:>9.0f} {comp:>9.0f} {rccl:>9.0f} {rccl_ov:>8.0f} {exp:>8.0f} {ovr*100:>5.1f} {streams:>7.1f}')

    # === cross-GPU imbalance per iter ===
    print('\n--- CROSS-GPU IMBALANCE (per iter) ---')
    print(f'{"iter":>5}  {"min_us":>8} {"med_us":>8} {"max_us":>8} {"max_pid":>7} {"wait_us":>8}  spread%')
    for it in iters:
        rs = [r for r in rows if r['it'] == it]
        if not rs: continue
        walls = sorted((r['wall'], r['pid']) for r in rs)
        mn = walls[0][0]; mx = walls[-1][0]; mx_pid = walls[-1][1]
        med = statistics.median(w for w,_ in walls)
        wait = mx - med
        spread_pct = ((mx - mn) / mn * 100) if mn else 0
        print(f'{it:>5}  {mn:>8.0f} {med:>8.0f} {mx:>8.0f} {mx_pid:>7} {wait:>8.0f}  {spread_pct:>5.1f}%')

    # === per-stream category breakdown for slowest GPU at first iter ===
    print('\n--- PER-STREAM CATEGORY BREAKDOWN (slowest GPU, first iter) ---')
    if rows:
        first_it = iters[0]
        rs = [r for r in rows if r['it'] == first_it]
        slow = max(rs, key=lambda r: r['wall'])
        slow_pid = slow['pid']
        evs = pi[first_it][slow_pid]
        cb_per_stream = category_busy_by_stream(evs)
        # rank streams by total busy
        ranked = sorted(((tid, sum(d.values())) for tid, d in cb_per_stream.items()),
                        key=lambda x: -x[1])
        print(f'  slowest pid={slow_pid}  wall={slow["wall"]:.0f}us')
        for tid, busy in ranked[:8]:
            cats = cb_per_stream[tid]
            top_cats = sorted(cats.items(), key=lambda x: -x[1])[:5]
            cat_str = ' '.join(f'{c}={d:.0f}' for c, d in top_cats)
            print(f'  tid={tid:>5}  busy={busy:>7.0f}us  {cat_str}')

    return rows, pid_avg

if __name__ == '__main__':
    amd, amd_avg = analyze('/home/chcai/traces/hctr/iter_steady_all8gpus.json', 'AMD MI350X (5-iter steady, iter 95-99)')
    nv, nv_avg = analyze('/home/chcai/traces/hctr/nsys_bs1x_5iter_8gpu.all8gpu.iter1000-1005.json', 'NV B200 (5-iter steady, iter 1000-1004)')

    # === SIDE-BY-SIDE OVERLAP COMPARISON ===
    print('\n' + '='*80)
    print('SIDE-BY-SIDE OVERLAP & IMBALANCE')
    print('='*80)
    def avg(d, k):
        return statistics.mean(v[k] for v in d.values())
    for k in ('wall', 'comp', 'rccl', 'exposed', 'overlap_rate', 'streams'):
        ad = avg(amd_avg, k); nd = avg(nv_avg, k)
        if k == 'overlap_rate':
            print(f'  {k:>14}:  AMD={ad*100:>6.1f}%   NV={nd*100:>6.1f}%   delta={(nd-ad)*100:+.1f}pp')
        elif k == 'streams':
            print(f'  {k:>14}:  AMD={ad:>6.1f}    NV={nd:>6.1f}    delta={nd-ad:+.1f}')
        else:
            print(f'  {k:>14}:  AMD={ad:>6.0f}us   NV={nd:>6.0f}us   delta={nd-ad:+.0f}us  ({(nd-ad)/ad*100 if ad else 0:+.1f}%)')

    # critical-path GPU consistency
    print('\n--- CRITICAL-PATH GPU (the GPU that consistently dictates iter wall-time) ---')
    for label, rows in [('AMD', amd), ('NV', nv)]:
        cnt = collections.Counter()
        for it in sorted({r['it'] for r in rows}):
            rs = [r for r in rows if r['it'] == it]
            slow = max(rs, key=lambda r: r['wall'])
            cnt[slow['pid']] += 1
        print(f'  {label}: {dict(cnt)}')
