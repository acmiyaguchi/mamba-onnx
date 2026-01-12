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

// Helper macro for checking status (throws exception, caught by ORT or crashes if no handler)
#define ORT_THROW_ON_ERROR(api, expr) \
    do { \
        OrtStatus* onnx_status = (expr); \
        if (onnx_status != nullptr) { \
            std::string msg = (api).GetErrorMessage(onnx_status); \
            (api).ReleaseStatus(onnx_status); \
            throw std::runtime_error(msg); \
        } \
    } while (0)

struct MambaSelectiveScanKernel {
    const OrtApi& api_;

    MambaSelectiveScanKernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api) {}

    void Compute(OrtKernelContext* context) {
        // Enable Flush-to-Zero and Denormals-Are-Zero
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        // Inputs
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

        // Shapes
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
        int64_t dstate = A_dims[1]; // N
        api_.ReleaseTensorTypeAndShapeInfo(A_info);

        // Data Pointers
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
                
                __m256 ones = _mm256_set1_ps(1.0f);

                for (int64_t l = 0; l < seqlen; ++l) {
                    int64_t u_idx = b * dim * seqlen + d * seqlen + l;
                    int64_t BC_idx = b * seqlen * 16 + l * 16;

                    float u_t = u_data[u_idx];
                    float delta_t = delta_data[u_idx];

                    // Approximation: exp(x) ~ 1 + x
                    // A_bar = exp(delta * A) ~ 1 + delta * A
                    __m256 delta_vec = _mm256_set1_ps(delta_t);
                    
                    __m256 A_bar_0 = _mm256_fmadd_ps(delta_vec, A_0, ones);
                    __m256 A_bar_1 = _mm256_fmadd_ps(delta_vec, A_1, ones);

                    __m256 B_t_0 = _mm256_loadu_ps(B_data + BC_idx);
                    __m256 B_t_1 = _mm256_loadu_ps(B_data + BC_idx + 8);

                    __m256 B_bar_0 = _mm256_mul_ps(delta_vec, B_t_0);
                    __m256 B_bar_1 = _mm256_mul_ps(delta_vec, B_t_1);

                    __m256 u_vec = _mm256_set1_ps(u_t);
                    
                    __m256 B_u_0 = _mm256_mul_ps(B_bar_0, u_vec);
                    __m256 B_u_1 = _mm256_mul_ps(B_bar_1, u_vec);

                    h_0 = _mm256_fmadd_ps(A_bar_0, h_0, B_u_0);
                    h_1 = _mm256_fmadd_ps(A_bar_1, h_1, B_u_1);

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

                    out_data[u_idx] = y_scalar;
                }
            }
        }
    }
};

extern "C" {

void* ORT_API_CALL CreateKernel(const OrtCustomOp* op, const OrtApi* api, const OrtKernelInfo* info) {
    return new MambaSelectiveScanKernel(*api, info);
}

void ORT_API_CALL KernelCompute(void* op_kernel, OrtKernelContext* context) {
    ((MambaSelectiveScanKernel*)op_kernel)->Compute(context);
}

void ORT_API_CALL KernelDestroy(void* op_kernel) {
    delete (MambaSelectiveScanKernel*)op_kernel;
}

const char* ORT_API_CALL GetName(const OrtCustomOp* op) { return "SelectiveScan"; }
const char* ORT_API_CALL GetExecutionProviderType(const OrtCustomOp* op) { return "CPUExecutionProvider"; }

size_t ORT_API_CALL GetInputTypeCount(const OrtCustomOp* op) { return 6; }
ONNXTensorElementDataType ORT_API_CALL GetInputType(const OrtCustomOp* op, size_t index) { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

size_t ORT_API_CALL GetOutputTypeCount(const OrtCustomOp* op) { return 1; }
ONNXTensorElementDataType ORT_API_CALL GetOutputType(const OrtCustomOp* op, size_t index) { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetInputCharacteristic(const OrtCustomOp* op, size_t index) { return INPUT_OUTPUT_REQUIRED; }
OrtCustomOpInputOutputCharacteristic ORT_API_CALL GetOutputCharacteristic(const OrtCustomOp* op, size_t index) { return INPUT_OUTPUT_REQUIRED; }
OrtMemType ORT_API_CALL GetInputMemoryType(const OrtCustomOp* op, size_t index) { return OrtMemTypeDefault; }

int ORT_API_CALL GetStartVersion(const OrtCustomOp* op) { return 1; }
int ORT_API_CALL GetEndVersion(const OrtCustomOp* op) { return 2147483647; }

OrtCustomOp mamba_custom_op;

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
    if (!api) return nullptr;

    const OrtApi* ort_api = api->GetApi(ORT_API_VERSION);
    if (!ort_api) return nullptr;

    memset(&mamba_custom_op, 0, sizeof(OrtCustomOp));
    mamba_custom_op.version = ORT_API_VERSION;
    mamba_custom_op.CreateKernel = CreateKernel;
    mamba_custom_op.KernelCompute = KernelCompute;
    mamba_custom_op.KernelDestroy = KernelDestroy;
    mamba_custom_op.GetName = GetName;
    mamba_custom_op.GetExecutionProviderType = GetExecutionProviderType;
    mamba_custom_op.GetInputTypeCount = GetInputTypeCount;
    mamba_custom_op.GetInputType = GetInputType;
    mamba_custom_op.GetOutputTypeCount = GetOutputTypeCount;
    mamba_custom_op.GetOutputType = GetOutputType;
    mamba_custom_op.GetInputCharacteristic = GetInputCharacteristic;
    mamba_custom_op.GetOutputCharacteristic = GetOutputCharacteristic;
    mamba_custom_op.GetInputMemoryType = GetInputMemoryType;
    mamba_custom_op.GetStartVersion = GetStartVersion;
    mamba_custom_op.GetEndVersion = GetEndVersion;
    
    OrtCustomOpDomain* domain = nullptr;
    if (ort_api->CreateCustomOpDomain("mamba", &domain)) {
        return nullptr;
    }
    
    if (ort_api->CustomOpDomain_Add(domain, &mamba_custom_op)) {
        return nullptr;
    }
    
    if (ort_api->AddCustomOpDomain(options, domain)) {
        return nullptr;
    }
    
    return nullptr;
}

}