cmake_minimum_required(VERSION 3.18)
project(mamba_ops)

set(CMAKE_CXX_STANDARD 17)
# Optimized flags for AVX2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -Ofast -Wall -Wno-unknown-pragmas")

# Include paths
# We point to the vendored onnxruntime headers (relative to csrc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../vendor/onnxruntime/include/onnxruntime/core/session)

# Define the shared library
add_library(mamba_ops SHARED selective_scan.cc)
# add_library(minimal_ops SHARED minimal.cc) # Optional: enable if needed

# We don't link against libonnxruntime.so (it's loaded by the host)
# But we need to allow undefined symbols on Linux
if(UNIX AND NOT APPLE)
    target_link_options(mamba_ops PRIVATE "-Wl,--allow-shlib-undefined")
endif()
if(APPLE)
    target_link_options(mamba_ops PRIVATE "-undefined dynamic_lookup")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mamba_ops PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install the shared library into the python package directory
install(TARGETS mamba_ops DESTINATION mamba_onnx)
