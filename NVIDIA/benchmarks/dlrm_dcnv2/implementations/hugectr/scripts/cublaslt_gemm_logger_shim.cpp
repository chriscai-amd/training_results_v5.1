// LD_PRELOAD shim that logs every cublasLtMatmul call's input/output
// shape, dtypes, transposes, and epilogue to a JSONL log file.
//
// Purpose: HCTR runs under use_cuda_graph=True, which means matmuls are
// baked into a CUDA graph during model.compile() and replayed via
// cudaGraphLaunch. cuBLASLt's built-in NVTX (CUBLASLT_NVTX_LEVEL=N)
// only emits TIMING RANGES with no payload, and even those only fire
// during graph CAPTURE, not graph REPLAY. So to recover the per-matmul
// shape info for tooltips in a trace converter (analogous to what
// PyTorch profiler shows for aten::mm), we need to read the
// cublasLtMatmulDesc + cublasLtMatrixLayout descriptors ourselves
// from a shim and write them out for offline consumption.
//
// Build (inside the mlperf-nvidia:recommendation-hugectr container):
//   g++ -O2 -fPIC -shared -o cublaslt_gemm_logger_shim.so \
//       cublaslt_gemm_logger_shim.cpp -ldl
//
// Use:
//   docker run ... \
//     -e LD_PRELOAD=/path/cublaslt_gemm_logger_shim.so \
//     -e SHIM_GEMM_LOG=/results/gemm_init_log.jsonl  ...
//
// Output: one JSON line per cublasLtMatmul call:
//   {"seq":12, "tid":141, "ts_ns":1778881234567890, "M":512, "N":6912,
//    "K":13, "ld_a":13, "ld_b":13, "ld_c":512, "ld_d":512,
//    "dt_a":"R_16F","dt_b":"R_16F","dt_c":"R_16F","dt_d":"R_16F",
//    "compute":"COMPUTE_32F", "op_a":"N","op_b":"T",
//    "epilogue":"BIAS_RELU"}
//
// During HCTR's init phase (model.compile + first iter warmup), every
// MLP matmul call goes through this shim and gets logged in deterministic
// order. After graph capture is complete, replay no longer calls
// cublasLtMatmul on the host, so the log is finite (~50-100 lines for
// our DLRM model).

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <syscall.h>
#include <pthread.h>
#include <atomic>

// ---- Minimal cublasLt typedefs (no header dependency at build time) ----
typedef int cublasStatus_t;
typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct cublasLtMatmulDescOpaque_t* cublasLtMatmulDesc_t;
typedef struct cublasLtMatrixLayoutOpaque_t* cublasLtMatrixLayout_t;
typedef struct cublasLtMatmulAlgoOpaque_t {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;

// Attribute enum values (verified against
// /usr/local/cuda/include/cublasLt.h in the mlperf-nvidia container,
// CUDA 12.8):
//   CUBLASLT_MATRIX_LAYOUT_TYPE       = 0
//   CUBLASLT_MATRIX_LAYOUT_ORDER      = 1
//   CUBLASLT_MATRIX_LAYOUT_ROWS       = 2
//   CUBLASLT_MATRIX_LAYOUT_COLS       = 3
//   CUBLASLT_MATRIX_LAYOUT_LD         = 4
//   CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0
//   CUBLASLT_MATMUL_DESC_SCALE_TYPE   = 1
//   CUBLASLT_MATMUL_DESC_TRANSA       = 3
//   CUBLASLT_MATMUL_DESC_TRANSB       = 4
//   CUBLASLT_MATMUL_DESC_EPILOGUE     = 7   (NOT 23 -- that's a different attr)
#define ML_LAYOUT_TYPE  0
#define ML_LAYOUT_ROWS  2
#define ML_LAYOUT_COLS  3
#define ML_LAYOUT_LD    4
#define ML_DESC_COMPUTE 0
#define ML_DESC_TRANSA  3
#define ML_DESC_TRANSB  4
#define ML_DESC_EPILOGUE 7

typedef cublasStatus_t (*pfn_layout_get_attr_t)(
    cublasLtMatrixLayout_t layout, int attr, void* value,
    size_t size_in_bytes, size_t* size_written);
typedef cublasStatus_t (*pfn_desc_get_attr_t)(
    cublasLtMatmulDesc_t desc, int attr, void* value,
    size_t size_in_bytes, size_t* size_written);
typedef cublasStatus_t (*pfn_matmul_t)(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t op,
    const void* alpha,
    const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C, cublasLtMatrixLayout_t Cdesc,
    void* D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes,
    void* stream);

static pfn_matmul_t           real_matmul = nullptr;
static pfn_layout_get_attr_t  real_layout_get_attr = nullptr;
static pfn_desc_get_attr_t    real_desc_get_attr = nullptr;

static FILE* g_log = nullptr;
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<uint64_t> g_seq{0};

static const char* dtype_str(int dt) {
    // cudaDataType_t enum from library_types.h (CUDA 12.x):
    //   CUDA_R_32F=0  CUDA_R_64F=1  CUDA_R_16F=2  CUDA_R_8I=3
    //   CUDA_C_32F=4  CUDA_C_64F=5  CUDA_C_16F=6  CUDA_C_8I=7
    //   CUDA_R_8U=8   CUDA_C_8U=9   CUDA_R_32I=10 CUDA_C_32I=11
    //   CUDA_R_32U=12 CUDA_C_32U=13 CUDA_R_16BF=14 CUDA_C_16BF=15
    //   ... CUDA_R_8FE4M3=28 CUDA_R_8FE5M2=29
    switch (dt) {
        case 0:  return "R_32F";   case 1:  return "R_64F";   case 2:  return "R_16F";
        case 3:  return "R_8I";    case 4:  return "C_32F";   case 5:  return "C_64F";
        case 6:  return "C_16F";   case 7:  return "C_8I";    case 8:  return "R_8U";
        case 9:  return "C_8U";    case 10: return "R_32I";   case 11: return "C_32I";
        case 12: return "R_32U";   case 13: return "C_32U";   case 14: return "R_16BF";
        case 15: return "C_16BF";  case 28: return "R_8FE4M3"; case 29: return "R_8FE5M2";
        default: { static char b[32]; snprintf(b, sizeof(b), "dt_%d", dt); return b; }
    }
}
static const char* op_str(int op) {
    switch (op) { case 0: return "N"; case 1: return "T"; case 2: return "C";
                  default: return "?"; }
}
static const char* compute_str(int c) {
    // cublasComputeType_t enum
    switch (c) {
        case 64: return "COMPUTE_16F";
        case 65: return "COMPUTE_16F_PEDANTIC";
        case 68: return "COMPUTE_32F";
        case 69: return "COMPUTE_32F_PEDANTIC";
        case 74: return "COMPUTE_32F_FAST_16F";
        case 75: return "COMPUTE_32F_FAST_16BF";
        case 77: return "COMPUTE_32F_FAST_TF32";
        case 70: return "COMPUTE_64F";
        case 72: return "COMPUTE_32I";
        default: { static char b[32]; snprintf(b, sizeof(b), "comp_%d", c); return b; }
    }
}
static const char* epilogue_str(uint32_t e) {
    // cublasLtEpilogue_t. Compose names for the common ones.
    static char buf[128];
    char* p = buf; *p = 0;
    auto add = [&](const char* s) {
        if (p != buf) { *p++ = '+'; *p = 0; }
        size_t l = strlen(s); memcpy(p, s, l); p += l; *p = 0;
    };
    if (e == 1 /*DEFAULT*/) return "DEFAULT";
    if (e &  2) add("RELU");        // CUBLASLT_EPILOGUE_RELU = 2
    if (e &  4) add("BIAS");        // CUBLASLT_EPILOGUE_BIAS = 4
    if (e &  8) add("GELU");        // CUBLASLT_EPILOGUE_GELU = 8
    if (e & 16) add("AUX");         // CUBLASLT_EPILOGUE_RELU_AUX = 0x82, GELU_AUX = 0x88
    if (e & 0x100) add("DRELU");    // CUBLASLT_EPILOGUE_DRELU = 0x102
    if (e & 0x200) add("DGELU");    // CUBLASLT_EPILOGUE_DGELU = 0x208
    if (e & 0x400) add("BGRADA");   // CUBLASLT_EPILOGUE_BGRADA = 0x401
    if (e & 0x800) add("BGRADB");   // CUBLASLT_EPILOGUE_BGRADB = 0x801
    if (buf[0] == 0) snprintf(buf, sizeof(buf), "epi_0x%x", e);
    return buf;
}

static void shim_init(void) {
    static int inited = 0;
    if (inited) return; inited = 1;

    // Resolve real functions via dlopen fallback (RTLD_NEXT may not have
    // libcublasLt in its scope when LD_PRELOAD loads us early).
    const char* libs[] = {"libcublasLt.so.13","libcublasLt.so.12","libcublasLt.so",nullptr};
    void* h = nullptr;
    for (int i=0; libs[i]; ++i) { h = dlopen(libs[i], RTLD_NOW|RTLD_GLOBAL); if (h) break; }
    if (!h) { fprintf(stderr, "[gemm-shim] FATAL: cannot dlopen libcublasLt\n"); return; }
    real_matmul          = (pfn_matmul_t)         dlsym(h, "cublasLtMatmul");
    real_layout_get_attr = (pfn_layout_get_attr_t)dlsym(h, "cublasLtMatrixLayoutGetAttribute");
    real_desc_get_attr   = (pfn_desc_get_attr_t)  dlsym(h, "cublasLtMatmulDescGetAttribute");

    const char* log_path = getenv("SHIM_GEMM_LOG");
    if (!log_path || !*log_path) log_path = "/tmp/gemm_init_log.jsonl";
    g_log = fopen(log_path, "a");
    if (!g_log) {
        fprintf(stderr, "[gemm-shim] FATAL: cannot open log %s\n", log_path);
        return;
    }
    setvbuf(g_log, nullptr, _IOLBF, 0);   // line-buffered so we don't lose data on crash

    fprintf(stderr,
        "[gemm-shim] loaded.  real_matmul=%p real_layout_get=%p real_desc_get=%p  log=%s\n",
        (void*)real_matmul, (void*)real_layout_get_attr, (void*)real_desc_get_attr, log_path);
    fflush(stderr);
}

__attribute__((constructor))
static void shim_ctor() { shim_init(); }

extern "C" cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t op,
    const void* alpha,
    const void* A, cublasLtMatrixLayout_t Adesc,
    const void* B, cublasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C, cublasLtMatrixLayout_t Cdesc,
    void* D, cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo, void* workspace, size_t wsz, void* stream)
{
    if (!real_matmul) shim_init();

    // Read M/N/K, ld, dtypes, op, epilogue from the descriptors BEFORE calling.
    // (The descriptors are caller-owned and live for at least the call duration.)
    uint64_t rows_A=0, cols_A=0, rows_B=0, cols_B=0, rows_D=0, cols_D=0;
    int64_t  ld_A=-1, ld_B=-1, ld_C=-1, ld_D=-1;
    int dt_A=-1, dt_B=-1, dt_C=-1, dt_D=-1;
    int op_a=-1, op_b=-1, compute=-1;
    uint32_t epi = 0;
    if (real_layout_get_attr) {
        real_layout_get_attr(Adesc, ML_LAYOUT_ROWS, &rows_A, sizeof(rows_A), nullptr);
        real_layout_get_attr(Adesc, ML_LAYOUT_COLS, &cols_A, sizeof(cols_A), nullptr);
        real_layout_get_attr(Adesc, ML_LAYOUT_LD,   &ld_A,   sizeof(ld_A),   nullptr);
        real_layout_get_attr(Adesc, ML_LAYOUT_TYPE, &dt_A,   sizeof(dt_A),   nullptr);
        real_layout_get_attr(Bdesc, ML_LAYOUT_ROWS, &rows_B, sizeof(rows_B), nullptr);
        real_layout_get_attr(Bdesc, ML_LAYOUT_COLS, &cols_B, sizeof(cols_B), nullptr);
        real_layout_get_attr(Bdesc, ML_LAYOUT_LD,   &ld_B,   sizeof(ld_B),   nullptr);
        real_layout_get_attr(Bdesc, ML_LAYOUT_TYPE, &dt_B,   sizeof(dt_B),   nullptr);
        real_layout_get_attr(Cdesc, ML_LAYOUT_LD,   &ld_C,   sizeof(ld_C),   nullptr);
        real_layout_get_attr(Cdesc, ML_LAYOUT_TYPE, &dt_C,   sizeof(dt_C),   nullptr);
        real_layout_get_attr(Ddesc, ML_LAYOUT_ROWS, &rows_D, sizeof(rows_D), nullptr);
        real_layout_get_attr(Ddesc, ML_LAYOUT_COLS, &cols_D, sizeof(cols_D), nullptr);
        real_layout_get_attr(Ddesc, ML_LAYOUT_LD,   &ld_D,   sizeof(ld_D),   nullptr);
        real_layout_get_attr(Ddesc, ML_LAYOUT_TYPE, &dt_D,   sizeof(dt_D),   nullptr);
    }
    if (real_desc_get_attr) {
        real_desc_get_attr(op, ML_DESC_TRANSA,   &op_a,   sizeof(op_a),   nullptr);
        real_desc_get_attr(op, ML_DESC_TRANSB,   &op_b,   sizeof(op_b),   nullptr);
        real_desc_get_attr(op, ML_DESC_COMPUTE,  &compute,sizeof(compute),nullptr);
        real_desc_get_attr(op, ML_DESC_EPILOGUE, &epi,    sizeof(epi),    nullptr);
    }

    // Derive M, N, K. (Standard cublasLt convention: A is M x K (op=N) or K x M (op=T),
    // B is K x N (op=N) or N x K (op=T), D is M x N.)
    uint64_t M = rows_D, N = cols_D;
    uint64_t K = (op_a == 0 /*N*/) ? cols_A : rows_A;

    // Timestamp + thread id for ordering
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t ts_ns = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
    pid_t tid = (pid_t)syscall(SYS_gettid);

    uint64_t seq = g_seq.fetch_add(1);

    if (g_log) {
        pthread_mutex_lock(&g_log_mu);
        fprintf(g_log,
            "{\"seq\":%lu,\"tid\":%d,\"ts_ns\":%lu,"
            "\"M\":%lu,\"N\":%lu,\"K\":%lu,"
            "\"ld_a\":%ld,\"ld_b\":%ld,\"ld_c\":%ld,\"ld_d\":%ld,"
            "\"dt_a\":\"%s\",\"dt_b\":\"%s\",\"dt_c\":\"%s\",\"dt_d\":\"%s\","
            "\"compute\":\"%s\",\"op_a\":\"%s\",\"op_b\":\"%s\","
            "\"epilogue\":\"%s\","
            "\"rows_a\":%lu,\"cols_a\":%lu,\"rows_b\":%lu,\"cols_b\":%lu}\n",
            seq, tid, ts_ns,
            M, N, K,
            ld_A, ld_B, ld_C, ld_D,
            dtype_str(dt_A), dtype_str(dt_B), dtype_str(dt_C), dtype_str(dt_D),
            compute_str(compute), op_str(op_a), op_str(op_b),
            epilogue_str(epi),
            rows_A, cols_A, rows_B, cols_B);
        pthread_mutex_unlock(&g_log_mu);
    }

    return real_matmul(lightHandle, op, alpha, A, Adesc, B, Bdesc, beta,
                       C, Cdesc, D, Ddesc, algo, workspace, wsz, stream);
}

__attribute__((destructor))
static void shim_fini() {
    fprintf(stderr, "[gemm-shim] exit. logged %lu cublasLtMatmul calls.\n",
            (unsigned long)g_seq.load());
    if (g_log) { fflush(g_log); fclose(g_log); }
}
