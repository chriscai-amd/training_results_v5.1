#!/usr/bin/env python3
"""
Side-by-side stream comparison: AMD MI350X vs NV B200 8-GPU traces.

For each trace, for GPU 0 (pid=100), iter 0, classify every stream (tid)
by its dominant kernel category mix. Then attempt to align AMD and NV
streams by their content signature: which streams exist on both, and
which exist only on one side.
"""
import json, collections

def load(p):
    with open(p) as f:
        d = json.load(f)
    return d['traceEvents'] if isinstance(d, dict) else d

def per_gpu_iter(ev, pid, it):
    out = []
    for e in ev:
        if e.get('ph') != 'X': continue
        if e.get('pid') != pid: continue
        a = e.get('args') or {}
        if a.get('iter') != it: continue
        out.append(e)
    return out

def classify(events):
    by_tid = collections.defaultdict(list)
    for e in events:
        by_tid[e['tid']].append(e)
    rows = []
    for tid, evs in by_tid.items():
        cats = collections.defaultdict(float)
        for e in evs:
            cats[e.get('cat', '?')] += e.get('dur', 0)
        total = sum(cats.values())
        if total < 5:
            continue  # skip near-empty
        # Compute a content signature
        rccl = (cats.get('rccl', 0) + cats.get('allreduce', 0) +
                cats.get('emb_a2a', 0))
        mlp = sum(cats.get(c, 0) for c in
                  ('mlp_fwd', 'mlp_bwd_dgrad', 'mlp_bwd_wgrad',
                   'fused_fma', 'interaction',
                   'fwd_bwd_mlp_fwd', 'fwd_bwd_interaction'))
        emb = sum(cats.get(c, 0) for c in
                  ('emb_fwd', 'emb_reduce', 'emb_scatter', 'opt_emb',
                   'emb_other', 'fwd_bwd_emb_fwd', 'fwd_bwd_emb_a2a'))
        sparse = cats.get('sparse_prep', 0)
        memcpy = cats.get('memcpy', 0)
        loss = cats.get('loss', 0)
        dtype = cats.get('dtype_cast', 0)
        memset = cats.get('memset', 0)

        rccl_s = rccl/total if total else 0
        mlp_s = mlp/total if total else 0
        emb_s = emb/total if total else 0
        sparse_s = sparse/total if total else 0
        memcpy_s = memcpy/total if total else 0

        # Coarse class
        n_cats = len([c for c, d in cats.items() if d >= 0.05*total])
        if rccl_s >= 0.85:
            cls = 'PURE-COMM (RCCL/AR/A2A only)'
        elif mlp_s >= 0.85 and rccl_s < 0.05:
            cls = 'PURE-MLP (cuBLASLt-internal worker?)'
        elif emb_s >= 0.85 and rccl_s < 0.05:
            cls = 'PURE-EMB'
        elif sparse_s >= 0.85:
            cls = 'PURE-SPARSE-PREP'
        elif memcpy_s >= 0.85:
            cls = 'PURE-MEMCPY'
        elif rccl_s >= 0.30 and (mlp_s + emb_s + sparse_s) >= 0.30:
            cls = 'MIXED-COMM-COMPUTE (HCTR app stream)'
        elif n_cats >= 4:
            cls = 'MIXED-APP (HCTR app stream)'
        elif mlp_s >= 0.50:
            cls = 'MLP-DOMINATED (HCTR default-like)'
        elif emb_s >= 0.50:
            cls = 'EMB-DOMINATED (HCTR mp/dp-like)'
        elif sparse_s >= 0.50:
            cls = 'SPARSE-DOMINATED'
        else:
            cls = 'OTHER'
        # Top categories sorted
        top_cats = sorted(cats.items(), key=lambda x: -x[1])[:4]
        sig = ' '.join(f'{c}={d:.0f}' for c, d in top_cats)
        rows.append(dict(tid=tid, busy=total, cls=cls,
                         rccl_pct=rccl_s*100, mlp_pct=mlp_s*100,
                         emb_pct=emb_s*100, sparse_pct=sparse_s*100,
                         sig=sig))
    rows.sort(key=lambda r: -r['busy'])
    return rows

def show(label, rows):
    print(f'\n=== {label} ===')
    print(f'  ({len(rows)} active streams)')
    print(f'  {"tid":>6} {"busy_us":>8} {"R%":>4} {"M%":>4} {"E%":>4} {"S%":>4}  class                                        signature')
    for r in rows:
        print(f'  {r["tid"]:>6} {r["busy"]:>8.0f} {r["rccl_pct"]:>4.0f} '
              f'{r["mlp_pct"]:>4.0f} {r["emb_pct"]:>4.0f} {r["sparse_pct"]:>4.0f}  '
              f'{r["cls"]:<44} {r["sig"]}')

# ---
nv = load('/home/chcai/traces/hctr/nsys_bs1x_5iter_8gpu.all8gpu.iter1000-1005.json')
amd = load('/home/chcai/traces/hctr/iter_steady_all8gpus.json')

# Pick GPU 0 (pid=100), first iter in each trace
nv_events = per_gpu_iter(nv, 100, 1000)
amd_events = per_gpu_iter(amd, 100, 95)

nv_rows = classify(nv_events)
amd_rows = classify(amd_events)

show('NV B200 — pid=100 iter 1000', nv_rows)
show('AMD MI350X — pid=100 iter 95', amd_rows)

# Align
print('\n' + '='*100)
print('SIDE-BY-SIDE ALIGNMENT (matched by class signature)')
print('='*100)

# Buckets we care about
buckets = [
    'MIXED-APP (HCTR app stream)',
    'MIXED-COMM-COMPUTE (HCTR app stream)',
    'EMB-DOMINATED (HCTR mp/dp-like)',
    'SPARSE-DOMINATED',
    'PURE-COMM (RCCL/AR/A2A only)',
    'PURE-MLP (cuBLASLt-internal worker?)',
    'PURE-EMB',
    'PURE-SPARSE-PREP',
    'PURE-MEMCPY',
    'MLP-DOMINATED (HCTR default-like)',
    'OTHER',
]

def by_class(rows, cls):
    return [r for r in rows if r['cls'] == cls]

print(f'  {"class":<46} {"NV count":>8} {"NV busy us":>10} {"AMD count":>8} {"AMD busy us":>10}')
print(f'  {"-"*46} {"-"*8} {"-"*10} {"-"*8} {"-"*10}')
for cls in buckets:
    nv_b = by_class(nv_rows, cls)
    amd_b = by_class(amd_rows, cls)
    nv_n = len(nv_b)
    nv_busy = sum(r['busy'] for r in nv_b)
    amd_n = len(amd_b)
    amd_busy = sum(r['busy'] for r in amd_b)
    if nv_n + amd_n == 0:
        continue
    print(f'  {cls:<46} {nv_n:>8} {nv_busy:>10.0f} {amd_n:>8} {amd_busy:>10.0f}')

# Common vs unique
nv_classes = set(r['cls'] for r in nv_rows)
amd_classes = set(r['cls'] for r in amd_rows)
common = nv_classes & amd_classes
nv_only = nv_classes - amd_classes
amd_only = amd_classes - nv_classes
print(f'\n  COMMON classes: {sorted(common)}')
print(f'  NV-ONLY classes: {sorted(nv_only)}')
print(f'  AMD-ONLY classes: {sorted(amd_only)}')

# Headline
print('\n' + '='*100)
print('HEADLINE')
print('='*100)
print(f'  NV B200    pid=100 iter 1000: {len(nv_rows)} active streams (≥5us busy)')
print(f'  AMD MI350X pid=100 iter 95:   {len(amd_rows)} active streams')
print(f'  Delta:                         {len(nv_rows) - len(amd_rows):+d} streams (NV has more)')
