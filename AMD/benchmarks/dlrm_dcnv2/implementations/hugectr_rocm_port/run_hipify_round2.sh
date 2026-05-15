#!/usr/bin/env bash
# Hipify gpu_cache, HierarchicalKV, dynamic_embedding_table, tools in parallel
set -u

PORT_ROOT=/apps/chcai/hugectr_rocm_port
SRC_ROOT=$PORT_ROOT/hipify_pass

# rsync targets we haven't copied yet
for d in gpu_cache tools; do
    rsync -a --exclude='.git' \
        $PORT_ROOT/hugectr/$d/ $SRC_ROOT/$d/ 2>/dev/null
done
mkdir -p $SRC_ROOT/third_party
for d in HierarchicalKV dynamic_embedding_table; do
    rsync -a --exclude='.git' \
        $PORT_ROOT/hugectr/third_party/$d/ $SRC_ROOT/third_party/$d/ 2>/dev/null
done

cd $SRC_ROOT
echo "=== file counts after rsync ==="
for d in gpu_cache tools third_party/HierarchicalKV third_party/dynamic_embedding_table; do
    n=$(find "$d" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' -o -name '*.inl' \) 2>/dev/null | wc -l)
    printf "  %-40s %5d files\n" "$d" "$n"
done

# Build TODO list (files without .prehip backup)
todo=$(mktemp)
> "$todo"
for d in gpu_cache tools third_party/HierarchicalKV third_party/dynamic_embedding_table; do
    while IFS= read -r f; do
        [[ ! -f "${f}.prehip" ]] && echo "$f" >> "$todo"
    done < <(find "$d" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' -o -name '*.inl' \))
done
echo ""
echo "files to hipify: $(wc -l < "$todo")"

errlog=hipify_warnings_round2.log
> "$errlog"
echo ""
echo "=== running hipify-perl with -P 64 ==="
time xargs -P 64 -I{} hipify-perl -inplace "{}" 2>>"$errlog" < "$todo"

echo ""
for d in gpu_cache tools third_party/HierarchicalKV third_party/dynamic_embedding_table; do
    total=$(find "$d" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' -o -name '*.inl' \) | wc -l)
    proc=$(find "$d" -name '*.prehip' | wc -l)
    mod=0
    while IFS= read -r f; do
        if ! diff -q "${f%.prehip}" "$f" >/dev/null 2>&1; then
            mod=$((mod + 1))
        fi
    done < <(find "$d" -name '*.prehip')
    printf "  %-40s  total=%-4d processed=%-4d modified=%-4d\n" "$d" "$total" "$proc" "$mod"
done
echo ""
echo "warning lines (round 2): $(wc -l < "$errlog" 2>/dev/null || echo 0)"
echo "=== top warning patterns ==="
grep "warning:" "$errlog" 2>/dev/null | sed -E 's/.*warning:\s*//' | sort | uniq -c | sort -rn | head -25

rm -f "$todo"
