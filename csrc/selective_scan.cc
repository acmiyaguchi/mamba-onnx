#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <stdexcept>
#include <omp.h>

#include "onnxruntime_c_api.h"

// Helper macro for checking status
#define ORT_THROW_ON_ERROR(api, expr) \
    do { \
        OrtStatus* onnx_status = (expr); \
        if (onnx_status != nullptr) { \
            std::string msg = (api).GetErrorMessage(onnx_status); \
            (api).ReleaseStatus(onnx_status); \
            throw std::runtime_error(msg); \
        } \
    } while (0)

// =============================================================================
// Utility Functions
// =============================================================================

inline float softplus(float x) {
    if (x > 20.0f) return x;
    return std::log(1.0f + std::exp(x));
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// =============================================================================
// Templated Selective Scan Kernel
// Template parameters:
//   - UseFused: if true, applies Softplus to delta and SiLU gate to output
//   - UseExact: if true, uses exp(delta*A); otherwise uses linear approximation
// =============================================================================

template<bool UseFused, bool UseExact>
struct SelectiveScanKernelImpl {
    const OrtApi& api_;

    SelectiveScanKernelImpl(const OrtApi& api, const OrtKernelInfo* info) : api_(api) {}

    void Compute(OrtKernelContext* context) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        // Inputs: u, delta, A, B, C, D, [z if fused]
        const OrtValue* u_val;
        ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetInput(context, 0, &u_val));
        const OrtValue* delta_val;
        ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetInput(context, 1, &delta_val));
        const OrtValue* A_val;
        ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetInput(context, 2, &A_val));
        const OrtValue* B_val;
        ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetInput(context, 3, &B_val));
        const OrtValue* C_val;
        ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetInput(context, 4, &C_val));
        const OrtValue* D_val;
        ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetInput(context, 5, &D_val));

        const OrtValue* z_val = nullptr;
        const float* z_data = nullptr;
        if constexpr (UseFused) {
            ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetInput(context, 6, &z_val));
            ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData((OrtValue*)z_val, (void**)&z_data));
        }

        // Get shapes
        OrtTensorTypeAndShapeInfo* u_info;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorTypeAndShape(u_val, &u_info));
        int64_t u_dims[3];
        size_t u_dim_count = 3;
        ORT_THROW_ON_ERROR(api_, api_.GetDimensions(u_info, u_dims, u_dim_count));
        int64_t batch = u_dims[0];
        int64_t dim = u_dims[1];
        int64_t seqlen = u_dims[2];
        api_.ReleaseTensorTypeAndShapeInfo(u_info);

        OrtTensorTypeAndShapeInfo* A_info;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorTypeAndShape(A_val, &A_info));
        int64_t A_dims[2];
        size_t A_dim_count = 2;
        ORT_THROW_ON_ERROR(api_, api_.GetDimensions(A_info, A_dims, A_dim_count));
        int64_t dstate = A_dims[1];
        api_.ReleaseTensorTypeAndShapeInfo(A_info);

        // Data pointers
        const float* u_data;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData((OrtValue*)u_val, (void**)&u_data));
        const float* delta_data;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData((OrtValue*)delta_val, (void**)&delta_data));
        const float* A_data;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData((OrtValue*)A_val, (void**)&A_data));
        const float* B_data;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData((OrtValue*)B_val, (void**)&B_data));
        const float* C_data;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData((OrtValue*)C_val, (void**)&C_data));
        const float* D_data;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData((OrtValue*)D_val, (void**)&D_data));

        // Output
        int64_t out_shape_arr[] = {batch, dim, seqlen};
        OrtValue* out_val;
        ORT_THROW_ON_ERROR(api_, api_.KernelContext_GetOutput(context, 0, out_shape_arr, 3, &out_val));
        float* out_data;
        ORT_THROW_ON_ERROR(api_, api_.GetTensorMutableData(out_val, (void**)&out_data));

        if (dstate != 16) {
            throw std::runtime_error("Only dstate=16 is supported in this kernel");
        }

        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t d = 0; d < dim; ++d) {
                const float* A_ptr = A_data + d * 16;
                float D_val_scalar = D_data[d];

                __m256 A_0 = _mm256_loadu_ps(A_ptr);
                __m256 A_1 = _mm256_loadu_ps(A_ptr + 8);

                __m256 h_0 = _mm256_setzero_ps();
                __m256 h_1 = _mm256_setzero_ps();

                for (int64_t l = 0; l < seqlen; ++l) {
                    int64_t u_idx = b * dim * seqlen + d * seqlen + l;
                    int64_t BC_idx = b * seqlen * 16 + l * 16;

                    float u_t = u_data[u_idx];
                    float delta_t = delta_data[u_idx];

                    // Apply Softplus if fused
                    if constexpr (UseFused) {
                        delta_t = softplus(delta_t);
                    }

                    __m256 u_vec = _mm256_set1_ps(u_t);
                    __m256 delta_vec = _mm256_set1_ps(delta_t);

                    __m256 B_t_0 = _mm256_loadu_ps(B_data + BC_idx);
                    __m256 B_t_1 = _mm256_loadu_ps(B_data + BC_idx + 8);

                    if constexpr (UseExact) {
                        // Exact: A_bar = exp(delta * A)
                        float A_bar_tmp[16], delta_A_tmp[16];
                        _mm256_storeu_ps(delta_A_tmp, _mm256_mul_ps(delta_vec, A_0));
                        _mm256_storeu_ps(delta_A_tmp + 8, _mm256_mul_ps(delta_vec, A_1));
                        for (int i = 0; i < 16; ++i) A_bar_tmp[i] = std::exp(delta_A_tmp[i]);
                        __m256 A_bar_0 = _mm256_loadu_ps(A_bar_tmp);
                        __m256 A_bar_1 = _mm256_loadu_ps(A_bar_tmp + 8);

                        __m256 B_bar_0 = _mm256_mul_ps(delta_vec, B_t_0);
                        __m256 B_bar_1 = _mm256_mul_ps(delta_vec, B_t_1);

                        h_0 = _mm256_fmadd_ps(A_bar_0, h_0, _mm256_mul_ps(B_bar_0, u_vec));
                        h_1 = _mm256_fmadd_ps(A_bar_1, h_1, _mm256_mul_ps(B_bar_1, u_vec));
                    } else {
                        // Fast: h = h + delta * (A*h + B*u)
                        __m256 tmp_0 = _mm256_mul_ps(B_t_0, u_vec);
                        __m256 tmp_1 = _mm256_mul_ps(B_t_1, u_vec);
                        tmp_0 = _mm256_fmadd_ps(A_0, h_0, tmp_0);
                        tmp_1 = _mm256_fmadd_ps(A_1, h_1, tmp_1);
                        h_0 = _mm256_fmadd_ps(delta_vec, tmp_0, h_0);
                        h_1 = _mm256_fmadd_ps(delta_vec, tmp_1, h_1);
                    }

                    __m256 C_t_0 = _mm256_loadu_ps(C_data + BC_idx);
                    __m256 C_t_1 = _mm256_loadu_ps(C_data + BC_idx + 8);

                    __m256 y_0 = _mm256_mul_ps(C_t_0, h_0);
                    __m256 y_1 = _mm256_mul_ps(C_t_1, h_1);

                    float y_temp[16];
                    _mm256_storeu_ps(y_temp, y_0);
                    _mm256_storeu_ps(y_temp + 8, y_1);

                    float y_scalar = 0;
                    for (int i = 0; i < 16; ++i) y_scalar += y_temp[i];
                    y_scalar += D_val_scalar * u_t;

                    // Apply Z-gate if fused
                    if constexpr (UseFused) {
                        y_scalar *= silu(z_data[u_idx]);
                    }

                    out_data[u_idx] = y_scalar;
                }
            }
        }
    }
};

// =============================================================================
// Type aliases for the 4 kernel variants
// =============================================================================

using SelectiveScanKernel = SelectiveScanKernelImpl<false, false>;           // 6-input, linear
using SelectiveScanExactKernel = SelectiveScanKernelImpl<false, true>;       // 6-input, exact
using SelectiveScanFusedKernel = SelectiveScanKernelImpl<true, false>;       // 7-input, linear
using SelectiveScanFusedExactKernel = SelectiveScanKernelImpl<true, true>;   // 7-input, exact

// =============================================================================
// ONNX Runtime Custom Op Registration
// =============================================================================

extern "C" {

// --- SelectiveScan (6-input, linear) ---
void* ORT_API_CALL CreateKernel_SS(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    return new SelectiveScanKernel(*api, info);
}
void ORT_API_CALL Compute_SS(void* k, OrtKernelContext* ctx) { ((SelectiveScanKernel*)k)->Compute(ctx); }
void ORT_API_CALL Destroy_SS(void* k) { delete (SelectiveScanKernel*)k; }
const char* ORT_API_CALL Name_SS(const OrtCustomOp*) { return "SelectiveScan"; }
size_t ORT_API_CALL InputCount_SS(const OrtCustomOp*) { return 6; }

// --- SelectiveScanExact (6-input, exact) ---
void* ORT_API_CALL CreateKernel_SSE(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    return new SelectiveScanExactKernel(*api, info);
}
void ORT_API_CALL Compute_SSE(void* k, OrtKernelContext* ctx) { ((SelectiveScanExactKernel*)k)->Compute(ctx); }
void ORT_API_CALL Destroy_SSE(void* k) { delete (SelectiveScanExactKernel*)k; }
const char* ORT_API_CALL Name_SSE(const OrtCustomOp*) { return "SelectiveScanExact"; }
size_t ORT_API_CALL InputCount_SSE(const OrtCustomOp*) { return 6; }

// --- SelectiveScanFused (7-input, linear) ---
void* ORT_API_CALL CreateKernel_SSF(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    return new SelectiveScanFusedKernel(*api, info);
}
void ORT_API_CALL Compute_SSF(void* k, OrtKernelContext* ctx) { ((SelectiveScanFusedKernel*)k)->Compute(ctx); }
void ORT_API_CALL Destroy_SSF(void* k) { delete (SelectiveScanFusedKernel*)k; }
const char* ORT_API_CALL Name_SSF(const OrtCustomOp*) { return "SelectiveScanFused"; }
size_t ORT_API_CALL InputCount_SSF(const OrtCustomOp*) { return 7; }

// --- SelectiveScanFusedExact (7-input, exact) ---
void* ORT_API_CALL CreateKernel_SSFE(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    return new SelectiveScanFusedExactKernel(*api, info);
}
void ORT_API_CALL Compute_SSFE(void* k, OrtKernelContext* ctx) { ((SelectiveScanFusedExactKernel*)k)->Compute(ctx); }
void ORT_API_CALL Destroy_SSFE(void* k) { delete (SelectiveScanFusedExactKernel*)k; }
const char* ORT_API_CALL Name_SSFE(const OrtCustomOp*) { return "SelectiveScanFusedExact"; }
size_t ORT_API_CALL InputCount_SSFE(const OrtCustomOp*) { return 7; }

// --- Shared functions ---
const char* ORT_API_CALL GetExecutionProviderType(const OrtCustomOp*) { return "CPUExecutionProvider"; }
ONNXTensorElementDataType ORT_API_CALL GetInputType(const OrtCustomOp*, size_t) { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
size_t ORT_API_CALL GetOutputTypeCount(const OrtCustomOp*) { return 1; }
ONNXTensorElementDataType ORT_API_CALL GetOutputType(const OrtCustomOp*, size_t) { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetInputCharacteristic(const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; }
OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetOutputCharacteristic(const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; }
OrtMemType ORT_API_CALL GetInputMemoryType(const OrtCustomOp*, size_t) { return OrtMemTypeDefault; }
int ORT_API_CALL GetStartVersion(const OrtCustomOp*) { return 1; }
int ORT_API_CALL GetEndVersion(const OrtCustomOp*) { return 2147483647; }

// Global op structs
OrtCustomOp op_ss, op_sse, op_ssf, op_ssfe;

// Helper to initialize an OrtCustomOp struct
void InitOp(OrtCustomOp& op,
            void* (*create)(const OrtCustomOp*, const OrtApi*, const OrtKernelInfo*),
            void (*compute)(void*, OrtKernelContext*),
            void (*destroy)(void*),
            const char* (*name)(const OrtCustomOp*),
            size_t (*input_count)(const OrtCustomOp*)) {
    memset(&op, 0, sizeof(OrtCustomOp));
    op.version = ORT_API_VERSION;
    op.CreateKernel = create;
    op.KernelCompute = compute;
    op.KernelDestroy = destroy;
    op.GetName = name;
    op.GetExecutionProviderType = GetExecutionProviderType;
    op.GetInputTypeCount = input_count;
    op.GetInputType = GetInputType;
    op.GetOutputTypeCount = GetOutputTypeCount;
    op.GetOutputType = GetOutputType;
    op.GetInputCharacteristic = GetInputCharacteristic;
    op.GetOutputCharacteristic = GetOutputCharacteristic;
    op.GetInputMemoryType = GetInputMemoryType;
    op.GetStartVersion = GetStartVersion;
    op.GetEndVersion = GetEndVersion;
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    if (!api) return nullptr;
    const OrtApi* ort_api = api->GetApi(ORT_API_VERSION);
    if (!ort_api) return nullptr;

    // Initialize all 4 ops
    InitOp(op_ss,   CreateKernel_SS,   Compute_SS,   Destroy_SS,   Name_SS,   InputCount_SS);
    InitOp(op_sse,  CreateKernel_SSE,  Compute_SSE,  Destroy_SSE,  Name_SSE,  InputCount_SSE);
    InitOp(op_ssf,  CreateKernel_SSF,  Compute_SSF,  Destroy_SSF,  Name_SSF,  InputCount_SSF);
    InitOp(op_ssfe, CreateKernel_SSFE, Compute_SSFE, Destroy_SSFE, Name_SSFE, InputCount_SSFE);

    // Create domain and add all ops
    OrtCustomOpDomain* domain = nullptr;
    if (ort_api->CreateCustomOpDomain("mamba", &domain)) return nullptr;
    if (ort_api->CustomOpDomain_Add(domain, &op_ss)) return nullptr;
    if (ort_api->CustomOpDomain_Add(domain, &op_sse)) return nullptr;
    if (ort_api->CustomOpDomain_Add(domain, &op_ssf)) return nullptr;
    if (ort_api->CustomOpDomain_Add(domain, &op_ssfe)) return nullptr;
    if (ort_api->AddCustomOpDomain(options, domain)) return nullptr;

    return nullptr;
}

}
