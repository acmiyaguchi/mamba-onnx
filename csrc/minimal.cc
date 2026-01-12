#include <cstdio>

// Define minimal structures to avoid including full headers if possible, 
// but using the header is better for correctness.
#include "onnxruntime_c_api.h"

extern "C" {
    OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
        printf("Minimal Register called\n");
        return nullptr;
    }
}

