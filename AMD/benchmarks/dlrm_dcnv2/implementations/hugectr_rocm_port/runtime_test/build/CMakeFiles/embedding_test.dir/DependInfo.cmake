
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/apps/chcai/hugectr_rocm_port/runtime_test/embedding_test.cpp" "/apps/chcai/hugectr_rocm_port/runtime_test/build/CMakeFiles/embedding_test.dir/embedding_test.cpp.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "FMT_HEADER_ONLY"
  "HUGECTR_ROCM_PORT"
  "SPDLOG_FMT_EXTERNAL"
  "SPDLOG_HEADER_ONLY"
  "__HIP_PLATFORM_AMD__"
  "__HIP_ROCclr__=1"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/apps/chcai/hugectr_rocm_port/runtime_test/../hugectr_hip"
  "/apps/chcai/hugectr_rocm_port/runtime_test/../hugectr_hip/HugeCTR"
  "/apps/chcai/hugectr_rocm_port/runtime_test/../hugectr_hip/HugeCTR/include"
  "/apps/chcai/hugectr_rocm_port/runtime_test/../hugectr_hip/third_party"
  "/apps/chcai/hugectr_rocm_port/runtime_test/../hugectr_hip/third_party/json/single_include"
  "/apps/chcai/hugectr_rocm_port/runtime_test/../hugectr_hip/third_party/parallel-hashmap"
  "/opt/rocm/include/hipblas"
  "/opt/rocm/include/hipblaslt"
  "/opt/rocm/include/hiprand"
  "/opt/rocm/include/rccl"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "" "embedding_test" "gcc" "CMakeFiles/embedding_test.dir/link.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
