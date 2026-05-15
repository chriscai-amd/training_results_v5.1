#!/usr/bin/env python3
"""
Classify every stream (tid) in both traces by its kernel-category mix to
determine which streams are HCTR-explicit (app code) vs. library-internal
(cuBLAS/NCCL/cuDNN auxiliary streams).

A stream is heuristically "library-internal" if:
  - it has very few distinct categories (1-2)
  - it carries one of: pure rccl/allreduce/emb_a2a, pure mlp_*, pure
    sparse_prep, or pure memcpy
  - and / or its first-event timestamp is well after iter start (lazy-create)

A stream is "app-explicit" if:
  - it carries a wide mix of HCTR categories (emb + mlp + sparse + memset),
    consistent with a named HCTR stream like "default"/"dp"/"mp"/"prefetch"
"""
import json, collections, statistics

def load(p):
    with open(p) as f:
        d = json.load(f)
    return d['traceEvents'] if isinstance(d, dict) else d

def per_iter_per_gpu(ev):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in ev:
        if e.get('ph') != 'X': continue
        a = e.get('args') or {}
        it = a.get('iter')
        if it is None: continue
        out[it][e['pid']].append(e)
    return out

def classify_streams(events, label, pid, it):
    """For a single (pid, iter), break down each tid by category and produce
    a classification."""
    by_tid = collections.defaultdict(list)
    for e in events:
        by_tid[e['tid']].append(e)

    rows = []
    iter_start = min(e['ts'] for e in events)
    iter_end = max(e['ts'] + e.get('dur', 0) for e in events)

    for tid, evs in by_tid.items():
        cats = collections.Counter()
        cat_dur = collections.defaultdict(float)
        for e in evs:
            c = e.get('cat', '?')
            cats[c] += 1
            cat_dur[c] += e.get('dur', 0)
        total_dur = sum(cat_dur.values())
        first_ts = min(e['ts'] for e in evs) - iter_start
        last_ts = max(e['ts'] + e.get('dur', 0) for e in evs) - iter_start
        n_distinct_cats = len([c for c, d in cat_dur.items() if d >= 0.05 * total_dur])
        top_cat, top_dur = max(cat_dur.items(), key=lambda x: x[1])
        top_share = top_dur / total_dur if total_dur else 0

        # Heuristic classification
        # - launch/host_api/iter_marker are host-side scheduler artifacts
        #   (we will skip them but mark them)
        host_only = all(c in ('launch', 'host_api', 'iter_marker', '?') for c in cat_dur if cat_dur[c] > 0)

        # Real GPU streams
        rccl_share = (cat_dur.get('rccl',0) + cat_dur.get('allreduce',0) + cat_dur.get('emb_a2a',0)) / total_dur if total_dur else 0
        mlp_share = sum(cat_dur.get(c,0) for c in ('mlp_fwd','mlp_bwd_dgrad','mlp_bwd_wgrad','fused_fma','interaction')) / total_dur if total_dur else 0
        emb_share = sum(cat_dur.get(c,0) for c in ('emb_fwd','emb_reduce','emb_scatter','opt_emb','emb_other')) / total_dur if total_dur else 0
        sparse_share = cat_dur.get('sparse_prep',0) / total_dur if total_dur else 0
        memcpy_share = cat_dur.get('memcpy',0) / total_dur if total_dur else 0
        memset_share = cat_dur.get('memset',0) / total_dur if total_dur else 0
        loss_share = cat_dur.get('loss', 0) / total_dur if total_dur else 0
        dtype_share = cat_dur.get('dtype_cast', 0) / total_dur if total_dur else 0

        # Class:
        if host_only:
            cls = 'HOST'
        elif rccl_share > 0.85:
            cls = 'COMM-only (NCCL/RCCL internal)'
        elif mlp_share > 0.85 and rccl_share < 0.05:
            cls = 'MLP-only (cuBLAS/hipBLASLt internal?)'
        elif emb_share > 0.85 and rccl_share < 0.05:
            cls = 'EMB-only'
        elif sparse_share > 0.85:
            cls = 'SPARSE-only'
        elif memcpy_share > 0.85:
            cls = 'MEMCPY-only'
        elif n_distinct_cats >= 4:
            cls = 'APP-MIXED (likely HCTR-named stream)'
        elif rccl_share > 0.30 and (mlp_share > 0.30 or emb_share > 0.30 or sparse_share > 0.30):
            cls = 'APP-COMM+COMPUTE (RCCL serialised w/ compute)'
        else:
            cls = 'OTHER'

        rows.append(dict(tid=tid, total_dur=total_dur, first_ts=first_ts, last_ts=last_ts,
                         class_=cls, top_cat=top_cat, top_share=top_share,
                         n_cats=n_distinct_cats,
                         rccl_share=rccl_share, mlp_share=mlp_share,
                         emb_share=emb_share, sparse_share=sparse_share,
                         memcpy_share=memcpy_share, memset_share=memset_share,
                         cat_dur=dict(cat_dur)))
    return rows

def show(label, rows):
    print(f'\n--- {label} ---')
    print(f'{"tid":>5} {"busy":>7} {"first":>5} {"last":>5} {"class":<45} {"top_cat":<20} top%  cats  rccl% mlp% emb% sps% cpy%')
    rows = sorted(rows, key=lambda r: -r['total_dur'])
    for r in rows:
        if r['total_dur'] < 5:  # skip tiny
            continue
        print(f'{r["tid"]:>5} {r["total_dur"]:>7.0f} {r["first_ts"]:>5.0f} {r["last_ts"]:>5.0f} {r["class_"]:<45} '
              f'{r["top_cat"]:<20} {r["top_share"]*100:>4.0f}  {r["n_cats"]:>3}  '
              f'{r["rccl_share"]*100:>4.0f} {r["mlp_share"]*100:>4.0f} {r["emb_share"]*100:>4.0f} '
              f'{r["sparse_share"]*100:>4.0f} {r["memcpy_share"]*100:>4.0f}')

def main():
    for label, path in [
        ('AMD MI350X (iter 95, pid=100)', '/home/chcai/traces/hctr/iter_steady_all8gpus.json'),
        ('NV B200 (iter 1000, pid=100)', '/home/chcai/traces/hctr/nsys_bs1x_5iter_8gpu.all8gpu.iter1000-1005.json'),
    ]:
        ev = load(path)
        pi = per_iter_per_gpu(ev)
        first_iter = sorted(pi.keys())[0]
        events = pi[first_iter][100]
        rows = classify_streams(events, label, 100, first_iter)

        # Aggregate by class for headline
        by_class = collections.Counter()
        dur_by_class = collections.defaultdict(float)
        for r in rows:
            if r['total_dur'] < 5: continue
            by_class[r['class_']] += 1
            dur_by_class[r['class_']] += r['total_dur']
        print(f'\n{"="*100}')
        print(f'{label}: iter {first_iter}, GPU 0 (pid=100)')
        print(f'{"="*100}')
        print('\nStream class summary:')
        for cls, n in sorted(by_class.items(), key=lambda x: -dur_by_class[x[0]]):
            print(f'  {n:>3} streams in class "{cls}"  (total busy {dur_by_class[cls]:.0f} us)')
        show('per-stream detail', rows)

if __name__ == '__main__':
    main()
