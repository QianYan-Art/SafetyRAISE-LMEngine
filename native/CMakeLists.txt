cmake_minimum_required(VERSION 3.18)
project(safetyraise_native_kernels LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(safetyraise_cpu SHARED cpu/safetyraise_cpu.c)
target_include_directories(safetyraise_cpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
if(MSVC)
    target_compile_options(safetyraise_cpu PRIVATE /utf-8)
endif()
set_target_properties(safetyraise_cpu PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(SAFETYRAISE_BUILD_CUDA "在检测到 CUDA 编译器时构建 F16 线性插件" ON)
if(SAFETYRAISE_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_library(safetyraise_cuda SHARED cuda/safetyraise_cuda.cu)
        target_include_directories(safetyraise_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
        if(MSVC)
            target_compile_options(safetyraise_cuda PRIVATE /utf-8)
        endif()
        set_target_properties(safetyraise_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
        message(STATUS "SafetyRAISE CUDA 插件已启用；仅 F16 路径声明为可用")
    else()
        message(STATUS "未找到 CUDA 编译器，跳过可选 safetyraise_cuda 插件")
    endif()
endif()
