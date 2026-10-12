#include "safetyraise_kernel.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

struct safetyraise_kernel_context {
    char error[256];
};

struct safetyraise_kernel_weight {
    __half *device_bytes;
    uint64_t byte_len;
    uint64_t rows;
    uint64_t cols;
};

static void set_error(safetyraise_kernel_context *context, const char *message) {
    if (context == NULL) return;
    strncpy(context->error, message == NULL ? "" : message, sizeof(context->error) - 1u);
    context->error[sizeof(context->error) - 1u] = '\0';
}

static int check_cuda(safetyraise_kernel_context *context, cudaError_t error, const char *where) {
    if (error == cudaSuccess) return 1;
    char message[256];
    snprintf(message, sizeof(message), "%s: %s", where, cudaGetErrorString(error));
    set_error(context, message);
    return 0;
}

static int checked_product_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (out == NULL || (right != 0u && left > UINT64_MAX / right)) return 0;
    *out = left * right;
    return 1;
}

__global__ static void linear_f16_kernel(
    const __half *weights,
    const float *input,
    float *output,
    uint64_t rows,
    uint64_t cols,
    uint64_t output_cols) {
    const uint64_t output_count = rows * output_cols;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < output_count;
         index += stride) {
        const uint64_t row = index / output_cols;
        const uint64_t out = index % output_cols;
        float sum = 0.0f;
        for (uint64_t col = 0u; col < cols; ++col) {
            sum += input[row * cols + col] * __half2float(weights[out * cols + col]);
        }
        output[index] = sum;
    }
}

uint32_t safetyraise_kernel_abi_version(void) { return SAFETYRAISE_KERNEL_ABI_VERSION; }
const char *safetyraise_kernel_name(void) { return "safetyraise-cuda-f16"; }

int32_t safetyraise_kernel_create_context(
    const safetyraise_kernel_context_options *options,
    safetyraise_kernel_context **out_context) {
    if (options == NULL || out_context == NULL ||
        options->abi_version != SAFETYRAISE_KERNEL_ABI_VERSION ||
        options->struct_size < sizeof(safetyraise_kernel_context_options)) {
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }
    safetyraise_kernel_context *context =
        (safetyraise_kernel_context *)calloc(1u, sizeof(*context));
    if (context == NULL) return SAFETYRAISE_KERNEL_STATUS_OUT_OF_MEMORY;
    *out_context = context;
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

int32_t safetyraise_kernel_destroy_context(safetyraise_kernel_context *context) {
    free(context);
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

int32_t safetyraise_kernel_get_capabilities(
    safetyraise_kernel_context *context,
    safetyraise_kernel_capabilities *out_capabilities) {
    if (context == NULL || out_capabilities == NULL ||
        out_capabilities->struct_size < sizeof(safetyraise_kernel_capabilities)) {
        set_error(context, "capabilities 参数无效");
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }
    out_capabilities->abi_version = SAFETYRAISE_KERNEL_ABI_VERSION;
    out_capabilities->dtype_mask = UINT64_C(1) << SAFETYRAISE_KERNEL_DTYPE_F16;
    out_capabilities->device_kind = SAFETYRAISE_KERNEL_DEVICE_CUDA;
    out_capabilities->flags = 0u;
    out_capabilities->max_rows = 0u;
    out_capabilities->max_cols = 0u;
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

int32_t safetyraise_kernel_create_weight(
    safetyraise_kernel_context *context,
    const uint8_t *data,
    uint64_t byte_len,
    uint64_t rows,
    uint64_t cols,
    uint32_t dtype,
    safetyraise_kernel_weight **out_weight) {
    uint64_t elements = 0u;
    uint64_t required_bytes = 0u;
    if (context == NULL || out_weight == NULL || dtype != SAFETYRAISE_KERNEL_DTYPE_F16 ||
        !checked_product_u64(rows, cols, &elements) ||
        !checked_product_u64(elements, sizeof(__half), &required_bytes) ||
        byte_len != required_bytes || (byte_len != 0u && data == NULL) ||
        byte_len > (uint64_t)SIZE_MAX) {
        set_error(context, "CUDA 插件只支持合法的 F16 权重");
        return SAFETYRAISE_KERNEL_STATUS_UNSUPPORTED_DTYPE;
    }
    safetyraise_kernel_weight *weight =
        (safetyraise_kernel_weight *)calloc(1u, sizeof(*weight));
    if (weight == NULL) return SAFETYRAISE_KERNEL_STATUS_OUT_OF_MEMORY;
    if (!check_cuda(context, cudaMalloc((void **)&weight->device_bytes, (size_t)byte_len),
                    "cudaMalloc 权重")) {
        free(weight);
        return SAFETYRAISE_KERNEL_STATUS_DEVICE_ERROR;
    }
    if (!check_cuda(context,
                    cudaMemcpy(weight->device_bytes, data, (size_t)byte_len,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy 权重")) {
        cudaFree(weight->device_bytes);
        free(weight);
        return SAFETYRAISE_KERNEL_STATUS_DEVICE_ERROR;
    }
    weight->byte_len = byte_len;
    weight->rows = rows;
    weight->cols = cols;
    *out_weight = weight;
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

int32_t safetyraise_kernel_destroy_weight(
    safetyraise_kernel_context *context,
    safetyraise_kernel_weight *weight) {
    if (weight == NULL) return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    if (!check_cuda(context, cudaFree(weight->device_bytes), "cudaFree 权重")) {
        free(weight);
        return SAFETYRAISE_KERNEL_STATUS_DEVICE_ERROR;
    }
    free(weight);
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

int32_t safetyraise_kernel_linear(
    safetyraise_kernel_context *context,
    safetyraise_kernel_weight *weight,
    const float *input,
    uint64_t input_rows,
    uint64_t input_cols,
    float *output,
    uint64_t output_rows,
    uint64_t output_cols) {
    uint64_t input_count = 0u;
    uint64_t output_count = 0u;
    if (context == NULL || weight == NULL || input_rows != output_rows ||
        input_cols != weight->cols || output_cols != weight->rows ||
        !checked_product_u64(input_rows, input_cols, &input_count) ||
        !checked_product_u64(output_rows, output_cols, &output_count) ||
        input_count > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        output_count > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        (input_count != 0u && input == NULL) || (output_count != 0u && output == NULL)) {
        set_error(context, "CUDA 线性算子 shape 或指针无效");
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }
    if (output_count == 0u) return SAFETYRAISE_KERNEL_STATUS_OK;
    float *device_input = NULL;
    float *device_output = NULL;
    const size_t input_bytes = (size_t)(input_count * sizeof(float));
    const size_t output_bytes = (size_t)(output_count * sizeof(float));
    if (!check_cuda(context, cudaMalloc((void **)&device_input, input_bytes),
                    "cudaMalloc 输入") ||
        !check_cuda(context, cudaMalloc((void **)&device_output, output_bytes),
                    "cudaMalloc 输出") ||
        !check_cuda(context, cudaMemcpy(device_input, input, input_bytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy 输入")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return SAFETYRAISE_KERNEL_STATUS_DEVICE_ERROR;
    }
    const dim3 block(256u);
    uint64_t block_count = (output_count + (uint64_t)block.x - 1u) / block.x;
    if (block_count > 65535u) block_count = 65535u;
    const dim3 grid((unsigned int)block_count);
    linear_f16_kernel<<<grid, block>>>(weight->device_bytes, device_input, device_output,
                                        output_rows, input_cols, output_cols);
    if (!check_cuda(context, cudaGetLastError(), "CUDA kernel") ||
        !check_cuda(context, cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !check_cuda(context, cudaMemcpy(output, device_output, output_bytes,
                                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy 输出")) {
        cudaFree(device_input);
        cudaFree(device_output);
        return SAFETYRAISE_KERNEL_STATUS_DEVICE_ERROR;
    }
    cudaFree(device_input);
    cudaFree(device_output);
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

const char *safetyraise_kernel_last_error(safetyraise_kernel_context *context) {
    return context == NULL ? "没有上下文可供读取错误" : context->error;
}
