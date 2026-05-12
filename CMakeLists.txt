cmake_minimum_required(VERSION 3.16)
project(attn_extract CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

set(LLAMA_DIR "/Users/henneberger/llama.cpp" CACHE PATH "Path to llama.cpp checkout (must be built)")
set(LLAMA_BUILD_DIR "${LLAMA_DIR}/build")

if(NOT EXISTS "${LLAMA_BUILD_DIR}/bin/libllama.dylib")
    message(FATAL_ERROR "Expected built llama at ${LLAMA_BUILD_DIR}/bin/libllama.dylib. Build llama.cpp first.")
endif()

add_executable(attn_extract main.cpp)

# Homebrew libomp on macOS
set(OMP_PREFIX "/opt/homebrew/opt/libomp")
target_include_directories(attn_extract PRIVATE
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
    "${LLAMA_DIR}/common"
    "${LLAMA_DIR}/src"
    "${OMP_PREFIX}/include"
)
target_compile_options(attn_extract PRIVATE -Xpreprocessor -fopenmp)
target_link_libraries(attn_extract PRIVATE "${OMP_PREFIX}/lib/libomp.dylib")

target_link_directories(attn_extract PRIVATE "${LLAMA_BUILD_DIR}/bin")
target_link_libraries(attn_extract PRIVATE
    llama
    llama-common
    ggml
    ggml-base
    ggml-cpu
    ggml-metal
    ggml-blas
    z
)

# Make the runtime able to find the dylibs alongside the binary.
set_target_properties(attn_extract PROPERTIES
    BUILD_RPATH "${LLAMA_BUILD_DIR}/bin"
    INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin"
)
