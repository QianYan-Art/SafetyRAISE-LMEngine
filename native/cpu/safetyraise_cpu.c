#include "safetyraise_kernel.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

struct safetyraise_kernel_context {
    char error[256];
};

struct safetyraise_kernel_weight {
    uint8_t *bytes;
    uint64_t byte_len;
    uint64_t rows;
    uint64_t cols;
    uint32_t dtype;
};

static void set_error(safetyraise_kernel_context *context, const char *message) {
    if (context == NULL) {
        return;
    }
    if (message == NULL) {
        context->error[0] = '\0';
        return;
    }
    strncpy(context->error, message, sizeof(context->error) - 1u);
    context->error[sizeof(context->error) - 1u] = '\0';
}

static int checked_size(uint64_t value, size_t *out) {
    if (out == NULL || value > (uint64_t)SIZE_MAX) {
        return 0;
    }
    *out = (size_t)value;
    return 1;
}

static int checked_product(uint64_t left, uint64_t right, uint64_t *out) {
    if (out == NULL || (right != 0u && left > UINT64_MAX / right)) {
        return 0;
    }
    *out = left * right;
    return 1;
}

static float half_to_float(uint16_t bits) {
    const uint32_t sign = ((uint32_t)bits & UINT32_C(0x8000)) << 16;
    const uint32_t exponent = ((uint32_t)bits >> 10) & UINT32_C(0x1f);
    const uint32_t fraction = (uint32_t)bits & UINT32_C(0x3ff);
    uint32_t result;

    if (exponent == 0u) {
        if (fraction == 0u) {
            result = sign;
        } else {
            uint32_t normalized = fraction;
            int32_t exponent_value = -14;
            while ((normalized & UINT32_C(0x400)) == 0u) {
                normalized <<= 1;
                --exponent_value;
            }
            normalized &= UINT32_C(0x3ff);
            result = sign | ((uint32_t)(exponent_value + 127) << 23) |
                     (normalized << 13);
        }
    } else if (exponent == 0x1fu) {
        result = sign | UINT32_C(0x7f800000) | (fraction << 13);
    } else {
        result = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }

    float value;
    memcpy(&value, &result, sizeof(value));
    return value;
}

static float weight_value(const safetyraise_kernel_weight *weight, uint64_t index) {
    if (weight->dtype == SAFETYRAISE_KERNEL_DTYPE_F32) {
        float value;
        memcpy(&value, weight->bytes + index * sizeof(float), sizeof(value));
        return value;
    }
    uint16_t bits;
    memcpy(&bits, weight->bytes + index * sizeof(uint16_t), sizeof(bits));
    return half_to_float(bits);
}

uint32_t safetyraise_kernel_abi_version(void) {
    return SAFETYRAISE_KERNEL_ABI_VERSION;
}

const char *safetyraise_kernel_name(void) {
    return "safetyraise-cpu-reference";
}

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
    if (context == NULL) {
        return SAFETYRAISE_KERNEL_STATUS_OUT_OF_MEMORY;
    }
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
        set_error(context, "capabilities 参数或 struct_size 无效");
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }
    out_capabilities->abi_version = SAFETYRAISE_KERNEL_ABI_VERSION;
    out_capabilities->dtype_mask = (UINT64_C(1) << SAFETYRAISE_KERNEL_DTYPE_F32) |
                                    (UINT64_C(1) << SAFETYRAISE_KERNEL_DTYPE_F16);
    out_capabilities->device_kind = SAFETYRAISE_KERNEL_DEVICE_HOST;
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
    uint64_t element_count;
    uint64_t element_size;
    uint64_t required_bytes;
    size_t allocation_size;

    if (context == NULL || out_weight == NULL ||
        (dtype != SAFETYRAISE_KERNEL_DTYPE_F32 &&
         dtype != SAFETYRAISE_KERNEL_DTYPE_F16) ||
        !checked_product(rows, cols, &element_count)) {
        set_error(context, "权重参数、dtype 或维度无效");
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }
    element_size = dtype == SAFETYRAISE_KERNEL_DTYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    if (!checked_product(element_count, element_size, &required_bytes) ||
        required_bytes != byte_len || !checked_size(byte_len, &allocation_size) ||
        (byte_len != 0u && data == NULL)) {
        set_error(context, "权重字节长度与 shape/dtype 不一致");
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }

    safetyraise_kernel_weight *weight =
        (safetyraise_kernel_weight *)calloc(1u, sizeof(*weight));
    if (weight == NULL) {
        set_error(context, "无法分配权重句柄");
        return SAFETYRAISE_KERNEL_STATUS_OUT_OF_MEMORY;
    }
    if (allocation_size != 0u) {
        weight->bytes = (uint8_t *)malloc(allocation_size);
        if (weight->bytes == NULL) {
            free(weight);
            set_error(context, "无法复制权重数据");
            return SAFETYRAISE_KERNEL_STATUS_OUT_OF_MEMORY;
        }
        memcpy(weight->bytes, data, allocation_size);
    }
    weight->byte_len = byte_len;
    weight->rows = rows;
    weight->cols = cols;
    weight->dtype = dtype;
    *out_weight = weight;
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

int32_t safetyraise_kernel_destroy_weight(
    safetyraise_kernel_context *context,
    safetyraise_kernel_weight *weight) {
    (void)context;
    if (weight == NULL) {
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }
    free(weight->bytes);
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
    uint64_t input_count;
    uint64_t output_count;
    if (context == NULL || weight == NULL ||
        input_rows != output_rows || input_cols != weight->cols ||
        output_cols != weight->rows || !checked_product(input_rows, input_cols, &input_count) ||
        !checked_product(output_rows, output_cols, &output_count) ||
        (input_count != 0u && input == NULL) || (output_count != 0u && output == NULL)) {
        set_error(context, "线性算子 shape 或指针无效");
        return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
    }

    for (uint64_t index = 0u; index < input_count; ++index) {
        if (!isfinite(input[index])) {
            set_error(context, "输入包含非有限值");
            return SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT;
        }
    }

    for (uint64_t row = 0u; row < input_rows; ++row) {
        for (uint64_t out = 0u; out < weight->rows; ++out) {
            float sum = 0.0f;
            for (uint64_t col = 0u; col < weight->cols; ++col) {
                const float input_value = input[row * input_cols + col];
                const float weight_value_at = weight_value(weight, out * weight->cols + col);
                sum += input_value * weight_value_at;
            }
            output[row * output_cols + out] = sum;
        }
    }
    return SAFETYRAISE_KERNEL_STATUS_OK;
}

const char *safetyraise_kernel_last_error(safetyraise_kernel_context *context) {
    return context == NULL ? "没有上下文可供读取错误" : context->error;
}
