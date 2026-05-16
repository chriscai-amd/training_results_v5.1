#!/usr/bin/env python3
"""Parse hipBLASLt / Tensile log output into a structured GEMM-shapes
sidecar JSON.

This is the "Method 2 (lite)" companion to rocprofv3_to_perfetto_annotated.py.
The converter (Method 1) already extracts macro_tile / mfma / dtype from each
hipBLASLt kernel name in the trace. This extractor parses the
hipBLASLt + Tensile debug output collected at capture time and emits the
list of unique GEMM problem shapes (M, N, K) that actually fired during
the run, plus a best-effort kernel-name -> probable-shape mapping based
on macro-tile divisibility.

Why not exact 1:1 mapping per kernel event?
   hipBLASLt's autotuner evaluates many candidate solutions per call;
   each candidate logs its own "Software match: Cijk_..." line, but
   only the chosen one actually dispatches a kernel. Without an
   LD_PRELOAD shim or HCTR-side roctx markers, we can't tell which
   "Software match" line corresponds to which kernel dispatch. So we
   emit the SET of distinct problem shapes seen + likely macro-tile
   matches; the user can cross-reference with the per-event
   macro_tile in the Perfetto JSON.

Input sources (require HIPBLASLT_LOG_LEVEL=4 + TENSILE_DB=0xff at run):
  * hipblaslt.log        -- HIPBLASLT_LOG_FILE output; has
                            "Software match: Cijk_..." kernel names
  * stdout.log           -- TENSILE_DB stdout/stderr capture; has
                            "[a/b/c/d]3-tensor<Half>( sizes(X,Y,Z),
                            strides(...) )" tensor descriptors

Usage:
  python3 extract_gemm_shapes.py \\
      --hipblaslt-log /path/to/hipblaslt.log \\
      --stdout-log    /path/to/stdout.log \\
      --output        /path/to/gemm_shapes.json
"""
import argparse, json, re, sys
from collections import Counter, defaultdict


_RE_TENSOR = re.compile(
    r"\[([abcd])\]3-tensor<([A-Za-z]+)>\( sizes\((\d+), (\d+), (\d+)\),"
    r" strides\(([^)]+)\)"
)
_RE_SOFTWARE_MATCH = re.compile(
    r"Software match: (Cijk_\w+)\["
)
_RE_MT = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")


def parse_tensor_blocks(stdout_path):
    """Walk the TENSILE_DB stdout and group consecutive [a],[b],[c],[d]
    tensors into contraction blocks. Skip the (1,1,1) placeholder blocks
    that fire on every internal hipBLASLt op.
    """
    blocks = []                # list of dicts {a:(X,Y,Z), b, c, d, dtype}
    current = {}
    with open(stdout_path, "rb") as f:
        for raw in f:
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            m = _RE_TENSOR.search(line)
            if not m:
                continue
            tag = m.group(1)        # a / b / c / d
            dtype = m.group(2)
            sx, sy, sz = int(m.group(3)), int(m.group(4)), int(m.group(5))
            current[tag] = (sx, sy, sz)
            current[f"{tag}_strides"] = m.group(6)
            current["dtype"] = dtype
            if tag == "d":          # last tag in a block; flush
                # skip placeholder (1,1,1) blocks
                if (current.get("a") not in (None, (1, 1, 1)) or
                        current.get("c") not in (None, (1, 1, 1))):
                    blocks.append(dict(current))
                current = {}
    return blocks


def parse_software_matches(hipblaslt_log_path):
    """Extract every 'Software match: Cijk_...' kernel name (= a
    candidate solution evaluated for some matmul call). Returns the
    list of unique kernel names + a Counter of how many times each
    was evaluated.
    """
    names = []
    with open(hipblaslt_log_path, "rb") as f:
        for raw in f:
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            m = _RE_SOFTWARE_MATCH.search(line)
            if m:
                names.append(m.group(1))
    return names


def infer_mnk(block):
    """Given a tensor block {a, b, c, d, dtype}, infer (M, N, K).
    For Tensile contractions with layout Cijk_Ailk_Bjlk (or similar):
      [a] = (K, M, batch)  i.e. A^T storage
      [b] = (K, N, batch)
      [c] = (M, N, batch)  output
    """
    a, b, c = block.get("a"), block.get("b"), block.get("c")
    if not (a and b and c):
        return None
    # c sizes give us M and N
    m, n = c[0], c[1]
    batch = c[2]
    # K is the shared inner dim: typically a[0] == b[0] == K when
    # both A and B are stored K-major.
    k = a[1] if (a[0] == m) else (a[0] if (a[1] == m) else a[1])
    # consistency check: b should match (k, n)
    return {"m": m, "n": n, "k": k, "batch": batch, "dtype": block.get("dtype")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hipblaslt-log", required=False,
                    help="path to HIPBLASLT_LOG_FILE output")
    ap.add_argument("--stdout-log", required=True,
                    help="path to captured stdout (TENSILE_DB output)")
    ap.add_argument("--output", required=True, help="output JSON path")
    args = ap.parse_args()

    print(f"Parsing TENSILE_DB blocks from {args.stdout_log} ...", file=sys.stderr)
    blocks = parse_tensor_blocks(args.stdout_log)
    print(f"  found {len(blocks)} non-trivial contraction blocks", file=sys.stderr)

    # Aggregate by inferred (M, N, K, dtype)
    mnk_counts = Counter()
    for b in blocks:
        mnk = infer_mnk(b)
        if mnk:
            key = (mnk["m"], mnk["n"], mnk["k"], mnk["dtype"])
            mnk_counts[key] += 1

    unique_shapes = []
    for (m, n, k, dtype), cnt in mnk_counts.most_common():
        unique_shapes.append({
            "m": m, "n": n, "k": k, "dtype": dtype,
            "occurrences": cnt,
            "flops_estimate": 2 * m * n * k,
        })

    # Group by macro-tile from hipblaslt log to give a probable mapping
    sw_matches = []
    if args.hipblaslt_log:
        print(f"Parsing Software match lines from {args.hipblaslt_log} ...",
              file=sys.stderr)
        names = parse_software_matches(args.hipblaslt_log)
        match_counts = Counter(names)
        print(f"  found {len(match_counts)} unique kernel names",
              file=sys.stderr)
        for name, cnt in match_counts.most_common():
            mt_m = _RE_MT.search(name)
            if mt_m:
                sw_matches.append({
                    "kernel_name_prefix": name[:80] + "...",
                    "full_name": name,
                    "macro_tile_m": int(mt_m.group(1)),
                    "macro_tile_n": int(mt_m.group(2)),
                    "macro_tile_k": int(mt_m.group(3)),
                    "macro_tile":
                        f"{mt_m.group(1)}x{mt_m.group(2)}x{mt_m.group(3)}",
                    "evaluated_times": cnt,
                })

    # Best-effort: for each unique problem shape, list which macro-tiles
    # could plausibly handle it (M % tile_m == 0, N % tile_n == 0, or close).
    for shape in unique_shapes:
        candidates = []
        for sw in sw_matches:
            tm, tn = sw["macro_tile_m"], sw["macro_tile_n"]
            # allow up to 25% slack on N (Tensile pads/splits)
            m_fits = (shape["m"] % tm == 0) or (shape["m"] <= tm)
            n_fits = (shape["n"] % tn == 0) or (shape["n"] <= tn)
            if m_fits and n_fits:
                candidates.append({
                    "macro_tile": sw["macro_tile"],
                    "evaluated_times": sw["evaluated_times"],
                })
        shape["plausible_macro_tiles"] = candidates[:5]  # top 5

    out = {
        "schema_version": 1,
        "source": {
            "hipblaslt_log": args.hipblaslt_log,
            "stdout_log": args.stdout_log,
            "n_tensor_blocks": len(blocks),
            "n_unique_software_matches": len(sw_matches),
        },
        "unique_gemm_problem_shapes": unique_shapes,
        "all_evaluated_kernel_macro_tiles": sw_matches,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.output}", file=sys.stderr)
    print(f"\n=== unique GEMM problem shapes (top 10 by occurrences) ===",
          file=sys.stderr)
    for s in unique_shapes[:10]:
        print(f"  M={s['m']:>5}  N={s['n']:>6}  K={s['k']:>5}  "
              f"{s['dtype']:<5}  occ={s['occurrences']:>4}  "
              f"flops={s['flops_estimate']:>12,}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
