#!/usr/bin/env bash
# Run hipify-perl in parallel on remaining HugeCTR source files
set -u
cd /apps/chcai/hugectr_rocm_port/hipify_pass

# Ensure no stale procs
pkill -f hipify-perl 2>/dev/null
sleep 1

# Build list of unprocessed files (no .prehip backup yet)
todo=$(mktemp)
> "$todo"
while IFS= read -r f; do
    [[ ! -f "${f}.prehip" ]] && echo "$f" >> "$todo"
done < <(find HugeCTR -type f \( -name "*.cu" -o -name "*.cuh" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \))

n_remaining=$(wc -l < "$todo")
echo "files remaining: $n_remaining"

errlog=hipify_warnings.log
xargs -P 64 -I{} hipify-perl -inplace "{}" 2>>"$errlog" < "$todo"

total=$(find HugeCTR -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) | wc -l)
processed=$(find HugeCTR -name '*.prehip' | wc -l)
modified=0
while IFS= read -r f; do
    if ! diff -q "${f%.prehip}" "$f" >/dev/null 2>&1; then
        modified=$((modified + 1))
    fi
done < <(find HugeCTR -name '*.prehip')

echo ""
echo "total source files : $total"
echo "files processed    : $processed"
echo "files modified     : $modified"
echo "warning lines      : $(wc -l < "$errlog" 2>/dev/null || echo 0)"
echo ""
echo "=== top hipify warning patterns ==="
grep "warning:" "$errlog" 2>/dev/null | sed -E 's/.*warning:\s*//' | sort | uniq -c | sort -rn | head -20
rm -f "$todo"
