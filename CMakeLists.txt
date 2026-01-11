cmake_minimum_required(VERSION 3.18)
project(mamba_ops)

set(CMAKE_CXX_STANDARD 17)
# Optimized flags for AVX2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -Ofast -Wall -Wno-unknown-pragmas")

# Include paths
# We point to the vendored onnxruntime headers
include_directories(${CMAKE_SOURCE_DIR}/vendor/onnxruntime/include/onnxruntime/core/session)

# Define the shared library
add_library(mamba_ops SHARED src/selective_scan.cc)
add_library(minimal_ops SHARED src/minimal.cc)

# Output directory for the library
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# Ensure output directory exists
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mamba_ops PUBLIC OpenMP::OpenMP_CXX)
endif()
