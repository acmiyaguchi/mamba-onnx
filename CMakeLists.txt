cmake_minimum_required(VERSION 3.18)
project(mamba_onnx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build the native extension
add_subdirectory(src/mamba_onnx/csrc)
