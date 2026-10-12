#ifndef SAFETYRAISE_KERNEL_H
#define SAFETYRAISE_KERNEL_H

/*
 * SafetyRAISE-LMEngine 原生线性算子 ABI v1。
 *
 * 这个头文件只使用固定宽度整数、C 指针和显式长度。Rust 不会把自身布局
 * 直接跨过 ABI，插件也不能保存 input/output 指针；create_weight 成功后
 * 插件必须复制或上传 bytes，linear 返回后才允许释放临时输入输出。
 * Rust 后端在同一 context 上串行调用这些函数；插件若向其他宿主暴露并发
 * 使用，需要自行提供同步。
 */

#include <stdint.h>

#if defined(_WIN32)
#  define SAFETYRAISE_KERNEL_EXPORT __declspec(dllexport)
#else
#  define SAFETYRAISE_KERNEL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SAFETYRAISE_KERNEL_ABI_VERSION UINT32_C(1)

typedef struct safetyraise_kernel_context safetyraise_kernel_context;
typedef struct safetyraise_kernel_weight safetyraise_kernel_weight;

/* 返回值为零表示成功；非零值必须能由 last_error 转换为可诊断错误。 */
typedef enum safetyraise_kernel_status {
    SAFETYRAISE_KERNEL_STATUS_OK = 0,
    SAFETYRAISE_KERNEL_STATUS_INVALID_ARGUMENT = 1,
    SAFETYRAISE_KERNEL_STATUS_UNSUPPORTED_DTYPE = 2,
    SAFETYRAISE_KERNEL_STATUS_DIMENSION_OVERFLOW = 3,
    SAFETYRAISE_KERNEL_STATUS_OUT_OF_MEMORY = 4,
    SAFETYRAISE_KERNEL_STATUS_DEVICE_ERROR = 5,
    SAFETYRAISE_KERNEL_STATUS_INTERNAL_ERROR = 6,
    SAFETYRAISE_KERNEL_STATUS_ABI_MISMATCH = 7
} safetyraise_kernel_status;

/* dtype 数值固定，Rust 侧通过显式映射使用。dtype mask 的位号等于这些数值。 */
typedef enum safetyraise_kernel_dtype {
    SAFETYRAISE_KERNEL_DTYPE_F32 = 1,
    SAFETYRAISE_KERNEL_DTYPE_F16 = 2,
    SAFETYRAISE_KERNEL_DTYPE_BF16 = 3,
    SAFETYRAISE_KERNEL_DTYPE_Q4_0 = 4,
    SAFETYRAISE_KERNEL_DTYPE_Q8_0 = 5,
    SAFETYRAISE_KERNEL_DTYPE_Q4K = 6,
    SAFETYRAISE_KERNEL_DTYPE_Q6K = 7
} safetyraise_kernel_dtype;

typedef enum safetyraise_kernel_device_kind {
    SAFETYRAISE_KERNEL_DEVICE_HOST = 1,
    SAFETYRAISE_KERNEL_DEVICE_CUDA = 2,
    SAFETYRAISE_KERNEL_DEVICE_OTHER = 3
} safetyraise_kernel_device_kind;

/* create_context 的输入。struct_size 允许未来追加尾部字段。 */
typedef struct safetyraise_kernel_context_options {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t flags;
} safetyraise_kernel_context_options;

/* get_capabilities 的输出。max_rows/max_cols 为零表示不设该上限。 */
typedef struct safetyraise_kernel_capabilities {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t dtype_mask;
    uint32_t device_kind;
    uint32_t flags;
    uint64_t max_rows;
    uint64_t max_cols;
} safetyraise_kernel_capabilities;

SAFETYRAISE_KERNEL_EXPORT uint32_t safetyraise_kernel_abi_version(void);

SAFETYRAISE_KERNEL_EXPORT const char *safetyraise_kernel_name(void);

SAFETYRAISE_KERNEL_EXPORT int32_t safetyraise_kernel_create_context(
    const safetyraise_kernel_context_options *options,
    safetyraise_kernel_context **out_context);

/* 必须在释放动态库前调用；上下文拥有的设备同步和临时资源在此完成。 */
SAFETYRAISE_KERNEL_EXPORT int32_t safetyraise_kernel_destroy_context(
    safetyraise_kernel_context *context);

SAFETYRAISE_KERNEL_EXPORT int32_t safetyraise_kernel_get_capabilities(
    safetyraise_kernel_context *context,
    safetyraise_kernel_capabilities *out_capabilities);

/*
 * data 的布局是连续的 weight[rows, cols]，插件必须在返回前复制/上传它。
 * byte_len、rows、cols 都是 u64，插件应在转换为 size_t 前检查溢出。
 */
SAFETYRAISE_KERNEL_EXPORT int32_t safetyraise_kernel_create_weight(
    safetyraise_kernel_context *context,
    const uint8_t *data,
    uint64_t byte_len,
    uint64_t rows,
    uint64_t cols,
    uint32_t dtype,
    safetyraise_kernel_weight **out_weight);

SAFETYRAISE_KERNEL_EXPORT int32_t safetyraise_kernel_destroy_weight(
    safetyraise_kernel_context *context,
    safetyraise_kernel_weight *weight);

/*
 * input/output 均为主机端 f32 连续矩阵。CUDA 插件在返回前必须同步，或在
 * ABI flags 中明确异步契约；当前 Rust 后端只接受同步返回的结果。
 */
SAFETYRAISE_KERNEL_EXPORT int32_t safetyraise_kernel_linear(
    safetyraise_kernel_context *context,
    safetyraise_kernel_weight *weight,
    const float *input,
    uint64_t input_rows,
    uint64_t input_cols,
    float *output,
    uint64_t output_rows,
    uint64_t output_cols);

/* 返回 context 内部的 UTF-8/NUL 结尾错误文本；指针由 context 持有。 */
SAFETYRAISE_KERNEL_EXPORT const char *safetyraise_kernel_last_error(
    safetyraise_kernel_context *context);

#ifdef __cplusplus
}
#endif

#endif /* SAFETYRAISE_KERNEL_H */
