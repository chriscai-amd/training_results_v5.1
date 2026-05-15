#!/usr/bin/env bash
# Broad post-hipify include-fix sweep across all hipified sources.
set -u
ROOT=/apps/chcai/hugectr_rocm_port/hugectr_hip
cd $ROOT

# Files we touch
FILES=$(find HugeCTR gpu_cache third_party/HierarchicalKV third_party/dynamic_embedding_table tools \
        -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cc' -o -name '*.inl' \) \
        2>/dev/null | grep -v '\.prehip$')

echo "files in scope: $(echo "$FILES" | wc -l)"

apply_sed() {
    local desc="$1" expr="$2"
    local n=0
    while IFS= read -r f; do
        if grep -lqE "${3:-$2}" "$f" 2>/dev/null; then
            sed -i -E "$expr" "$f"
            n=$((n + 1))
        fi
    done <<< "$FILES"
    printf "  %-55s patched in %4d files\n" "$desc" "$n"
}

echo ""
echo "=== include path fixes ==="

# nccl.h -> rccl/rccl.h
apply_sed "<nccl.h> -> <rccl/rccl.h>" \
    's|#include[[:space:]]*<nccl\.h>|#include <rccl/rccl.h>|g' \
    '#include[[:space:]]*<nccl\.h>'

# bare hipblas.h -> hipblas/hipblas.h
apply_sed "<hipblas.h> -> <hipblas/hipblas.h>" \
    's|#include[[:space:]]*<hipblas\.h>|#include <hipblas/hipblas.h>|g' \
    '#include[[:space:]]*<hipblas\.h>'

# bare hiprand.h -> hiprand/hiprand.h
apply_sed "<hiprand.h> -> <hiprand/hiprand.h>" \
    's|#include[[:space:]]*<hiprand\.h>|#include <hiprand/hiprand.h>|g' \
    '#include[[:space:]]*<hiprand\.h>'

# bare hipblaslt.h -> hipblaslt/hipblaslt.h
apply_sed "<hipblaslt.h> -> <hipblaslt/hipblaslt.h>" \
    's|#include[[:space:]]*<hipblaslt\.h>|#include <hipblaslt/hipblaslt.h>|g' \
    '#include[[:space:]]*<hipblaslt\.h>'

# nvml.h -> drop (comment out, no AMD equivalent)
apply_sed "<nvml.h>  -> /* removed (no AMD equiv) */" \
    's|#include[[:space:]]*<nvml\.h>|/* HugeCTR ROCm port: <nvml.h> removed */|g' \
    '#include[[:space:]]*<nvml\.h>'

# hipDNN.h -> drop (excluded layers don't need cuDNN/MIOpen)
apply_sed "<hipDNN.h> -> /* removed */" \
    's|#include[[:space:]]*<hipDNN\.h>|/* HugeCTR ROCm port: <hipDNN.h> removed */|g' \
    '#include[[:space:]]*<hipDNN\.h>'

# cooperative_groups/reduce.h -> just cooperative_groups (HIP collapses sub-headers)
apply_sed "cooperative_groups/reduce.h -> cooperative_groups" \
    's|#include[[:space:]]*<cooperative_groups/reduce\.h>|#include <hip/amd_detail/amd_hip_cooperative_groups.h>|g' \
    '#include[[:space:]]*<cooperative_groups/reduce\.h>'

echo ""
echo "=== verify nothing left ==="
for kw in '<nccl\.h>' '<hipDNN\.h>' '<nvml\.h>' '<cooperative_groups/reduce\.h>'; do
    n=$(grep -rEln "$kw" $ROOT --include='*.cpp' --include='*.hpp' --include='*.cu' --include='*.cuh' --include='*.h' --include='*.cc' --include='*.inl' 2>/dev/null | grep -v '\.prehip$' | wc -l)
    printf "  %-40s remaining in %d files\n" "$kw" "$n"
done
