# Native extension for mamba_onnx

# Optimized flags for AVX2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -O3 -Wall -Wno-unknown-pragmas")

# Include paths - vendored onnxruntime headers
# Path is relative to project root (3 levels up from csrc)
set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor")
include_directories(${VENDOR_DIR}/onnxruntime/include/onnxruntime/core/session)

# Define the shared library
add_library(mamba_ops SHARED selective_scan.cc)

# Allow undefined symbols (plugin loaded by ORT host)
if(UNIX AND NOT APPLE)
    target_link_options(mamba_ops PRIVATE "-Wl,--allow-shlib-undefined")
endif()
if(APPLE)
    target_link_options(mamba_ops PRIVATE "-undefined" "dynamic_lookup")
endif()

# OpenMP for parallel execution
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mamba_ops PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install the shared library into the package directory
# scikit-build-core puts this relative to the wheel root
# Since wheel.packages copies src/mamba_onnx to mamba_onnx/, we install there
install(TARGETS mamba_ops LIBRARY DESTINATION .)
