//! 可选 GPU 计算内核。
//!
//! 目前实现单行 matvec 和 decode MLP 的 SwiGLU+down projection GPU 路径。
//! 初始化或执行失败时上层会回退 CPU。
use std::borrow::Cow;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc,
};
#[cfg(test)]
use std::sync::{Mutex, MutexGuard, OnceLock};

use half::f16;
use wgpu::util::DeviceExt;

use crate::tensor::{Q8LinearWeight, Tensor};

const WORKGROUP_SIZE: u32 = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuSyncStats {
    pub submits: u64,
    pub poll_waits: u64,
    pub map_reads: u64,
    pub host_write_buffers: u64,
    pub position_write_buffers: u64,
}

impl GpuSyncStats {
    pub fn record(&mut self, other: Self) {
        self.submits += other.submits;
        self.poll_waits += other.poll_waits;
        self.map_reads += other.map_reads;
        self.host_write_buffers += other.host_write_buffers;
        self.position_write_buffers += other.position_write_buffers;
    }
}

static GPU_SUBMIT_COUNT: AtomicU64 = AtomicU64::new(0);
static GPU_POLL_WAIT_COUNT: AtomicU64 = AtomicU64::new(0);
static GPU_MAP_READ_COUNT: AtomicU64 = AtomicU64::new(0);
static GPU_HOST_WRITE_BUFFER_COUNT: AtomicU64 = AtomicU64::new(0);
static GPU_POSITION_WRITE_BUFFER_COUNT: AtomicU64 = AtomicU64::new(0);

fn record_submit() {
    GPU_SUBMIT_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn record_poll_wait() {
    GPU_POLL_WAIT_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn record_map_read() {
    GPU_MAP_READ_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn record_host_write_buffer() {
    GPU_HOST_WRITE_BUFFER_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn record_position_write_buffer() {
    GPU_POSITION_WRITE_BUFFER_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn write_buffer<T: bytemuck::NoUninit>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: wgpu::BufferAddress,
    data: &[T],
) {
    record_host_write_buffer();
    queue.write_buffer(buffer, offset, bytemuck::cast_slice(data));
}

fn write_position_buffer(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: wgpu::BufferAddress,
    data: &[u32],
) {
    record_host_write_buffer();
    record_position_write_buffer();
    queue.write_buffer(buffer, offset, bytemuck::cast_slice(data));
}

pub fn reset_sync_stats() {
    GPU_SUBMIT_COUNT.store(0, Ordering::Relaxed);
    GPU_POLL_WAIT_COUNT.store(0, Ordering::Relaxed);
    GPU_MAP_READ_COUNT.store(0, Ordering::Relaxed);
    GPU_HOST_WRITE_BUFFER_COUNT.store(0, Ordering::Relaxed);
    GPU_POSITION_WRITE_BUFFER_COUNT.store(0, Ordering::Relaxed);
}

pub fn sync_stats() -> GpuSyncStats {
    GpuSyncStats {
        submits: GPU_SUBMIT_COUNT.load(Ordering::Relaxed),
        poll_waits: GPU_POLL_WAIT_COUNT.load(Ordering::Relaxed),
        map_reads: GPU_MAP_READ_COUNT.load(Ordering::Relaxed),
        host_write_buffers: GPU_HOST_WRITE_BUFFER_COUNT.load(Ordering::Relaxed),
        position_write_buffers: GPU_POSITION_WRITE_BUFFER_COUNT.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
pub(crate) fn gpu_test_guard() -> MutexGuard<'static, ()> {
    static GPU_TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
    GPU_TEST_MUTEX
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap()
}

const MATVEC_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_idx = id.x;
    if (out_idx >= params.out_features) {
        return;
    }

    var sum = 0.0;
    let base = out_idx * params.in_features;
    for (var i = 0u; i < params.in_features; i = i + 1u) {
        sum = sum + input[i] * weight[base + i];
    }
    output[out_idx] = sum;
}
"#;

const RMS_NORM_SHADER: &str = r#"
struct Params {
    last_dim: u32,
    rows: u32,
    eps: f32,
    _pad0: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> partial_sum: array<f32, 64>;
var<workgroup> row_scale: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = workgroup_id.x;
    if (row >= params.rows) {
        return;
    }

    let lane = local_id.x;
    let row_base = row * params.last_dim;

    var sum = 0.0;
    for (var idx = lane; idx < params.last_dim; idx = idx + 64u) {
        let value = input[row_base + idx];
        sum = sum + value * value;
    }
    partial_sum[lane] = sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        let mean_sq = partial_sum[0] / f32(params.last_dim);
        row_scale = inverseSqrt(mean_sq + params.eps);
    }
    workgroupBarrier();

    for (var idx = lane; idx < params.last_dim; idx = idx + 64u) {
        let value = input[row_base + idx];
        output[row_base + idx] = value * row_scale * weight[idx];
    }
}
"#;

const RMS_NORM_ROPE_SHADER: &str = r#"
struct StaticParams {
    head_dim: u32,
    rows: u32,
    half_dim: u32,
    eps: f32,
};

struct DynamicParams {
    pos: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<storage, read> inv_freq: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> static_params: StaticParams;

@group(0) @binding(5)
var<uniform> dynamic_params: DynamicParams;

var<workgroup> partial_sum: array<f32, 64>;
var<workgroup> row_scale: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = workgroup_id.x;
    if (row >= static_params.rows) {
        return;
    }

    let lane = local_id.x;
    let row_base = row * static_params.head_dim;

    var sum = 0.0;
    for (var idx = lane; idx < static_params.head_dim; idx = idx + 64u) {
        let value = input[row_base + idx];
        sum = sum + value * value;
    }
    partial_sum[lane] = sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        let mean_sq = partial_sum[0] / f32(static_params.head_dim);
        row_scale = inverseSqrt(mean_sq + static_params.eps);
    }
    workgroupBarrier();

    for (var idx = lane; idx < static_params.half_dim; idx = idx + 64u) {
        let angle = f32(dynamic_params.pos) * inv_freq[idx];
        let cos_val = cos(angle);
        let sin_val = sin(angle);
        let x1 = input[row_base + idx] * row_scale * weight[idx];
        let x2 = input[row_base + idx + static_params.half_dim] * row_scale * weight[idx + static_params.half_dim];
        output[row_base + idx] = x1 * cos_val - x2 * sin_val;
        output[row_base + idx + static_params.half_dim] = x1 * sin_val + x2 * cos_val;
    }
}
"#;

const KV_APPEND_SHADER: &str = r#"
struct StaticParams {
    head_dim: u32,
    num_heads: u32,
    max_len: u32,
    _pad0: u32,
};

struct DynamicParams {
    position: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> cache: array<f32>;

@group(0) @binding(2)
var<uniform> static_params: StaticParams;

@group(0) @binding(3)
var<uniform> dynamic_params: DynamicParams;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let head = workgroup_id.x;
    if (head >= static_params.num_heads) {
        return;
    }

    let lane = local_id.x;
    let input_base = head * static_params.head_dim;
    let cache_base = head * static_params.max_len * static_params.head_dim + dynamic_params.position * static_params.head_dim;
    for (var idx = lane; idx < static_params.head_dim; idx = idx + 64u) {
        cache[cache_base + idx] = input[input_base + idx];
    }
}
"#;

const DECODE_GQA_ATTENTION_SHADER: &str = r#"
struct StaticParams {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_len: u32,
    kv_group_size: u32,
    scale: f32,
    _pad0: u32,
};

struct DynamicParams {
    position: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> q: array<f32>;

@group(0) @binding(1)
var<storage, read> key: array<f32>;

@group(0) @binding(2)
var<storage, read> value: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> static_params: StaticParams;

@group(0) @binding(5)
var<uniform> dynamic_params: DynamicParams;

var<workgroup> scores: array<f32, 2048>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let head = workgroup_id.x;
    if (head >= static_params.num_heads) {
        return;
    }

    let lane = local_id.x;
    let kv_head = head / static_params.kv_group_size;
    let q_base = head * static_params.head_dim;
    let cache_head_stride = static_params.max_len * static_params.head_dim;
    let k_base = kv_head * cache_head_stride;
    let v_base = kv_head * cache_head_stride;
    let out_base = head * static_params.head_dim;
    let seq_len_k = dynamic_params.position + 1u;

    var local_max = -3.4028234663852886e38;
    for (var pos = lane; pos < seq_len_k; pos = pos + 64u) {
        let kv_row_base = pos * static_params.head_dim;
        var score = 0.0;
        for (var d = 0u; d < static_params.head_dim; d = d + 1u) {
            score = score + q[q_base + d] * key[k_base + kv_row_base + d];
        }
        score = score * static_params.scale;
        scores[pos] = score;
        local_max = max(local_max, score);
    }

    partial[lane] = local_max;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            partial[lane] = max(partial[lane], partial[lane + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        shared_max = partial[0];
    }
    workgroupBarrier();

    var local_sum = 0.0;
    for (var pos = lane; pos < seq_len_k; pos = pos + 64u) {
        let weight = exp(scores[pos] - shared_max);
        scores[pos] = weight;
        local_sum = local_sum + weight;
    }
    partial[lane] = local_sum;
    workgroupBarrier();

    stride = 32u;
    loop {
        if (lane < stride) {
            partial[lane] = partial[lane] + partial[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        shared_sum = partial[0];
    }
    workgroupBarrier();

    let inv_sum = 1.0 / shared_sum;
    for (var d = lane; d < static_params.head_dim; d = d + 64u) {
        var acc = 0.0;
        for (var pos = 0u; pos < seq_len_k; pos = pos + 1u) {
            acc = acc + scores[pos] * value[v_base + pos * static_params.head_dim + d];
        }
        output[out_base + d] = acc * inv_sum;
    }
}
"#;

const DECODE_GQA_ATTENTION_SERIAL_SHADER: &str = r#"
struct StaticParams {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_len: u32,
    kv_group_size: u32,
    scale: f32,
    _pad0: u32,
};

struct DynamicParams {
    position: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> q: array<f32>;

@group(0) @binding(1)
var<storage, read> key: array<f32>;

@group(0) @binding(2)
var<storage, read> value: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> static_params: StaticParams;

@group(0) @binding(5)
var<uniform> dynamic_params: DynamicParams;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let head = workgroup_id.x;
    if (head >= static_params.num_heads) {
        return;
    }

    let kv_head = head / static_params.kv_group_size;
    let q_base = head * static_params.head_dim;
    let cache_head_stride = static_params.max_len * static_params.head_dim;
    let k_base = kv_head * cache_head_stride;
    let v_base = kv_head * cache_head_stride;
    let out_base = head * static_params.head_dim;
    let seq_len_k = dynamic_params.position + 1u;

    var max_score = -3.4028234663852886e38;
    var sum_exp = 0.0;
    for (var d = 0u; d < static_params.head_dim; d = d + 1u) {
        output[out_base + d] = 0.0;
    }

    for (var pos = 0u; pos < seq_len_k; pos = pos + 1u) {
        let kv_row_base = pos * static_params.head_dim;
        var score = 0.0;
        for (var d = 0u; d < static_params.head_dim; d = d + 1u) {
            score = score + q[q_base + d] * key[k_base + kv_row_base + d];
        }
        score = score * static_params.scale;

        if (score <= max_score) {
            let weight = exp(score - max_score);
            sum_exp = sum_exp + weight;
            for (var d = 0u; d < static_params.head_dim; d = d + 1u) {
                output[out_base + d] = output[out_base + d] + weight * value[v_base + kv_row_base + d];
            }
        } else {
            let rescale = exp(max_score - score);
            for (var d = 0u; d < static_params.head_dim; d = d + 1u) {
                output[out_base + d] = output[out_base + d] * rescale + value[v_base + kv_row_base + d];
            }
            sum_exp = sum_exp * rescale + 1.0;
            max_score = score;
        }
    }

    let inv_sum = 1.0 / sum_exp;
    for (var d = 0u; d < static_params.head_dim; d = d + 1u) {
        output[out_base + d] = output[out_base + d] * inv_sum;
    }
}
"#;

const DECODE_GQA_ATTENTION_PARALLEL_MAX_SEQ_LEN: usize = 2048;

const RESIDUAL_ADD_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.len) {
        return;
    }
    output[idx] = lhs[idx] + rhs[idx];
}
"#;

const Q8_MATVEC_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    words_per_row: u32,
    _pad0: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> qweight: array<u32>;

@group(0) @binding(2)
var<storage, read> scales: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

fn unpack_i8(word: u32, lane: u32) -> i32 {
    let byte = (word >> (lane * 8u)) & 0xffu;
    var signed = i32(byte);
    if (byte >= 128u) {
        signed = signed - 256;
    }
    return signed;
}

var<workgroup> partial_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let out_idx = workgroup_id.x;
    if (out_idx >= params.out_features) {
        return;
    }

    var sum = 0.0;
    let base = out_idx * params.words_per_row;
    let lane = local_id.x;
    let full_words = params.in_features / 4u;
    for (var word_idx = lane; word_idx < full_words; word_idx = word_idx + 64u) {
        let word = qweight[base + word_idx];
        let input_base = word_idx * 4u;
        sum = sum + input[input_base] * f32(unpack_i8(word, 0u));
        sum = sum + input[input_base + 1u] * f32(unpack_i8(word, 1u));
        sum = sum + input[input_base + 2u] * f32(unpack_i8(word, 2u));
        sum = sum + input[input_base + 3u] * f32(unpack_i8(word, 3u));
    }

    let tail = params.in_features & 3u;
    if (tail != 0u && lane == 0u) {
        let word = qweight[base + full_words];
        let input_base = full_words * 4u;
        if (tail >= 1u) {
            sum = sum + input[input_base] * f32(unpack_i8(word, 0u));
        }
        if (tail >= 2u) {
            sum = sum + input[input_base + 1u] * f32(unpack_i8(word, 1u));
        }
        if (tail >= 3u) {
            sum = sum + input[input_base + 2u] * f32(unpack_i8(word, 2u));
        }
    }
    partial_sum[lane] = sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[out_idx] = partial_sum[0] * scales[out_idx];
    }
}
"#;

const Q8_MATVEC_VEC4_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    words_per_row: u32,
    _pad0: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> qweight: array<u32>;

@group(0) @binding(2)
var<storage, read> scales: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

var<workgroup> partial_sum: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let out_idx = workgroup_id.x;
    if (out_idx >= params.out_features) {
        return;
    }

    var sum = 0.0;
    let base = out_idx * params.words_per_row;
    let lane = local_id.x;
    let full_words = params.in_features / 4u;
    for (var word_idx = lane; word_idx < full_words; word_idx = word_idx + 64u) {
        let weights = unpack4x8snorm(qweight[base + word_idx]) * 127.0;
        sum = sum + dot(input[word_idx], weights);
    }

    partial_sum[lane] = sum;
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            partial_sum[lane] = partial_sum[lane] + partial_sum[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[out_idx] = partial_sum[0] * scales[out_idx];
    }
}
"#;

const ARGMAX_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    words_per_row: u32,
    out_offset: u32,
};

struct ArgmaxResult {
    index: u32,
    value: f32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> qweight: array<u32>;

@group(0) @binding(2)
var<storage, read> scales: array<f32>;

@group(0) @binding(3)
var<storage, read_write> result: array<ArgmaxResult>;

@group(0) @binding(4)
var<uniform> params: Params;

fn unpack_i8(word: u32, lane: u32) -> i32 {
    let byte = (word >> (lane * 8u)) & 0xffu;
    var signed = i32(byte);
    if (byte >= 128u) {
        signed = signed - 256;
    }
    return signed;
}

var<workgroup> local_idx: array<u32, 64>;
var<workgroup> local_value: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let idx = global_id.x;
    let lane = local_id.x;
    if (idx < params.out_features) {
        var sum = 0.0;
        let base = idx * params.words_per_row;
        for (var i = 0u; i < params.in_features; i = i + 1u) {
            let word = qweight[base + (i / 4u)];
            let q = unpack_i8(word, i & 3u);
            sum = sum + input[i] * f32(q);
        }
        local_idx[lane] = params.out_offset + idx;
        local_value[lane] = sum * scales[idx];
    } else {
        local_idx[lane] = params.out_offset;
        local_value[lane] = -3.4028234663852886e38;
    }
    workgroupBarrier();

    var stride = 32u;
    loop {
        if (lane < stride) {
            let other_lane = lane + stride;
            if (local_value[other_lane] > local_value[lane]) {
                local_value[lane] = local_value[other_lane];
                local_idx[lane] = local_idx[other_lane];
            }
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        result[workgroup_id.x].index = local_idx[0];
        result[workgroup_id.x].value = local_value[0];
    }
}
"#;

const SWIGLU_SHADER: &str = r#"
struct Params {
    offset: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> gate: array<f32>;

@group(0) @binding(1)
var<storage, read> up: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) {
        return;
    }

    let g = gate[idx];
    output[params.offset + idx] = (g / (1.0 + exp(-g))) * up[idx];
}
"#;

#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

struct GpuContextInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matvec_pipeline: wgpu::ComputePipeline,
    matvec_bind_group_layout: wgpu::BindGroupLayout,
    rms_norm_pipeline: wgpu::ComputePipeline,
    rms_norm_bind_group_layout: wgpu::BindGroupLayout,
    rms_norm_rope_pipeline: wgpu::ComputePipeline,
    rms_norm_rope_bind_group_layout: wgpu::BindGroupLayout,
    kv_append_pipeline: wgpu::ComputePipeline,
    kv_append_bind_group_layout: wgpu::BindGroupLayout,
    decode_gqa_attention_pipeline: wgpu::ComputePipeline,
    decode_gqa_attention_serial_pipeline: wgpu::ComputePipeline,
    decode_gqa_attention_bind_group_layout: wgpu::BindGroupLayout,
    residual_add_pipeline: wgpu::ComputePipeline,
    residual_add_bind_group_layout: wgpu::BindGroupLayout,
    q8_matvec_pipeline: wgpu::ComputePipeline,
    q8_matvec_vec4_pipeline: wgpu::ComputePipeline,
    q8_matvec_bind_group_layout: wgpu::BindGroupLayout,
    argmax_pipeline: wgpu::ComputePipeline,
    argmax_bind_group_layout: wgpu::BindGroupLayout,
    swiglu_pipeline: wgpu::ComputePipeline,
    swiglu_bind_group_layout: wgpu::BindGroupLayout,
    max_chunk_bytes: wgpu::BufferAddress,
}

pub struct GpuMatVec {
    context: GpuContext,
    input_buffer: wgpu::Buffer,
    chunks: Vec<GpuMatVecChunk>,
    in_features: usize,
    out_features: usize,
}

pub struct GpuRmsNorm {
    context: GpuContext,
    input_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    _weight_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    shape: Vec<usize>,
    last_dim: usize,
    rows: usize,
}

pub struct GpuQkRmsNormRope {
    context: GpuContext,
    q_input_buffer: wgpu::Buffer,
    q_bind_group: wgpu::BindGroup,
    q_output_buffer: wgpu::Buffer,
    q_readback_buffer: wgpu::Buffer,
    _q_weight_buffer: wgpu::Buffer,
    q_static_params_buffer: wgpu::Buffer,
    position_buffer: wgpu::Buffer,
    q_shape: Vec<usize>,
    q_rows: usize,
    q_head_dim: usize,
    k_input_buffer: wgpu::Buffer,
    k_bind_group: wgpu::BindGroup,
    k_output_buffer: wgpu::Buffer,
    k_readback_buffer: wgpu::Buffer,
    _k_weight_buffer: wgpu::Buffer,
    k_static_params_buffer: wgpu::Buffer,
    _inv_freq_buffer: wgpu::Buffer,
    k_shape: Vec<usize>,
    k_rows: usize,
    k_head_dim: usize,
}

pub struct GpuQkRmsNormRopeConfig<'a> {
    pub q_weight: &'a Tensor,
    pub k_weight: &'a Tensor,
    pub q_shape: &'a [usize],
    pub k_shape: &'a [usize],
    pub pos: usize,
    pub inv_freq: &'a [f32],
    pub q_eps: f32,
    pub k_eps: f32,
}

pub struct GpuKvAppend {
    context: GpuContext,
    key_input_buffer: wgpu::Buffer,
    value_input_buffer: wgpu::Buffer,
    key_cache_buffer: Option<wgpu::Buffer>,
    value_cache_buffer: Option<wgpu::Buffer>,
    key_bind_group: Option<wgpu::BindGroup>,
    value_bind_group: Option<wgpu::BindGroup>,
    key_readback_buffer: Option<wgpu::Buffer>,
    value_readback_buffer: Option<wgpu::Buffer>,
    static_params_buffer: wgpu::Buffer,
    position_buffer: wgpu::Buffer,
    num_heads: usize,
    head_dim: usize,
    max_len: usize,
    current_len: usize,
}

pub struct GpuDecodeGqaAttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_len: usize,
    pub scale: f32,
}

pub struct GpuDecodeGqaAttention {
    context: GpuContext,
    q_buffer: wgpu::Buffer,
    key_buffer: wgpu::Buffer,
    value_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    static_params_buffer: wgpu::Buffer,
    position_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_len: usize,
}

pub struct GpuResidualAdd {
    context: GpuContext,
    lhs_buffer: wgpu::Buffer,
    rhs_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    shape: Vec<usize>,
    len: usize,
}

pub struct GpuResidentBuffer {
    context: GpuContext,
    buffer: wgpu::Buffer,
    shape: Vec<usize>,
    len: usize,
}

pub struct GpuPositionUniform {
    context: GpuContext,
    buffer: wgpu::Buffer,
}

pub struct GpuQ8MatVec {
    context: GpuContext,
    input_buffer: Option<wgpu::Buffer>,
    chunks: Vec<GpuQ8MatVecChunk>,
    in_features: usize,
    out_features: usize,
}

pub struct GpuQ8SameInputBatch {
    context: GpuContext,
    input_buffer: wgpu::Buffer,
    bind_groups: Vec<Vec<wgpu::BindGroup>>,
    in_features: usize,
    out_features: Vec<usize>,
}

pub struct GpuQ8SameInputBatchResidentCache {
    bind_groups: Vec<Vec<wgpu::BindGroup>>,
    _params_buffers: Vec<Vec<wgpu::Buffer>>,
}

pub struct GpuSwiGluDown {
    context: GpuContext,
    bind_groups: Vec<GpuSwiGluBindGroup>,
    _params_buffers: Vec<wgpu::Buffer>,
}

pub struct GpuQ8SwiGluDown {
    context: GpuContext,
    shared_input_buffer: wgpu::Buffer,
    gate_bind_groups: Vec<wgpu::BindGroup>,
    up_bind_groups: Vec<wgpu::BindGroup>,
    bind_groups: Vec<GpuSwiGluBindGroup>,
    _params_buffers: Vec<wgpu::Buffer>,
}

pub struct GpuQ8SwiGluDownResidentCache {
    gate_bind_groups: Vec<wgpu::BindGroup>,
    up_bind_groups: Vec<wgpu::BindGroup>,
    swiglu_bind_groups: Vec<GpuSwiGluBindGroup>,
    _swiglu_params_buffers: Vec<wgpu::Buffer>,
    down_bind_groups: Vec<wgpu::BindGroup>,
}

pub struct GpuRmsNormResidentCache {
    bind_group: wgpu::BindGroup,
}

pub struct GpuQkRmsNormRopeResidentCache {
    q_bind_group: wgpu::BindGroup,
    k_bind_group: wgpu::BindGroup,
    position_buffer: wgpu::Buffer,
}

pub struct GpuKvAppendResidentCache {
    key_bind_group: wgpu::BindGroup,
    value_bind_group: wgpu::BindGroup,
    position_buffer: wgpu::Buffer,
}

pub struct GpuDecodeGqaAttentionResidentCache {
    bind_group: wgpu::BindGroup,
    position_buffer: wgpu::Buffer,
}

pub struct GpuResidualAddResidentCache {
    bind_group: wgpu::BindGroup,
}

pub struct GpuQ8MatVecResidentCache {
    bind_groups: Vec<wgpu::BindGroup>,
}

struct GpuSwiGluBindGroup {
    bind_group: wgpu::BindGroup,
    out_features: usize,
}

struct GpuMatVecChunk {
    bind_group: wgpu::BindGroup,
    _weight_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    out_offset: usize,
    out_features: usize,
}

struct GpuQ8MatVecChunk {
    bind_group: Option<wgpu::BindGroup>,
    argmax_bind_group: Option<wgpu::BindGroup>,
    _qweight_buffer: wgpu::Buffer,
    _scales_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: Option<wgpu::Buffer>,
    argmax_buffer: Option<wgpu::Buffer>,
    argmax_readback_buffer: Option<wgpu::Buffer>,
    _params_buffer: wgpu::Buffer,
    _argmax_params_buffer: Option<wgpu::Buffer>,
    argmax_results: usize,
    words_per_row: usize,
    out_offset: usize,
    out_features: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuArgmaxResult {
    index: u32,
    value: f32,
}

unsafe impl bytemuck::Zeroable for GpuArgmaxResult {}
unsafe impl bytemuck::Pod for GpuArgmaxResult {}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("request_adapter failed: {e}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rsinfer-gpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| format!("request_device failed: {e}"))?;

        let matvec_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-lm-head-matvec"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MATVEC_SHADER)),
        });
        let matvec_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-lm-head-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let matvec_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-lm-head-pipeline-layout"),
                bind_group_layouts: &[Some(&matvec_bind_group_layout)],
                immediate_size: 0,
            });
        let matvec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-lm-head-pipeline"),
            layout: Some(&matvec_pipeline_layout),
            module: &matvec_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rms_norm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-rmsnorm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RMS_NORM_SHADER)),
        });
        let rms_norm_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-rmsnorm-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let rms_norm_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-rmsnorm-pipeline-layout"),
                bind_group_layouts: &[Some(&rms_norm_bind_group_layout)],
                immediate_size: 0,
            });
        let rms_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-rmsnorm-pipeline"),
            layout: Some(&rms_norm_pipeline_layout),
            module: &rms_norm_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let rms_norm_rope_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-rmsnorm-rope"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RMS_NORM_ROPE_SHADER)),
        });
        let rms_norm_rope_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-rmsnorm-rope-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let rms_norm_rope_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-rmsnorm-rope-pipeline-layout"),
                bind_group_layouts: &[Some(&rms_norm_rope_bind_group_layout)],
                immediate_size: 0,
            });
        let rms_norm_rope_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rsinfer-rmsnorm-rope-pipeline"),
                layout: Some(&rms_norm_rope_pipeline_layout),
                module: &rms_norm_rope_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let kv_append_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-kv-append"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(KV_APPEND_SHADER)),
        });
        let kv_append_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-kv-append-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let kv_append_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-kv-append-pipeline-layout"),
                bind_group_layouts: &[Some(&kv_append_bind_group_layout)],
                immediate_size: 0,
            });
        let kv_append_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-kv-append-pipeline"),
            layout: Some(&kv_append_pipeline_layout),
            module: &kv_append_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let decode_gqa_attention_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rsinfer-decode-gqa-attention"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DECODE_GQA_ATTENTION_SHADER)),
            });
        let decode_gqa_attention_serial_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rsinfer-decode-gqa-attention-serial"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DECODE_GQA_ATTENTION_SERIAL_SHADER)),
            });
        let decode_gqa_attention_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-decode-gqa-attention-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let decode_gqa_attention_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-decode-gqa-attention-pipeline-layout"),
                bind_group_layouts: &[Some(&decode_gqa_attention_bind_group_layout)],
                immediate_size: 0,
            });
        let decode_gqa_attention_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rsinfer-decode-gqa-attention-pipeline"),
                layout: Some(&decode_gqa_attention_pipeline_layout),
                module: &decode_gqa_attention_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let decode_gqa_attention_serial_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rsinfer-decode-gqa-attention-serial-pipeline"),
                layout: Some(&decode_gqa_attention_pipeline_layout),
                module: &decode_gqa_attention_serial_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let residual_add_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-residual-add"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RESIDUAL_ADD_SHADER)),
        });
        let residual_add_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-residual-add-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let residual_add_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-residual-add-pipeline-layout"),
                bind_group_layouts: &[Some(&residual_add_bind_group_layout)],
                immediate_size: 0,
            });
        let residual_add_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rsinfer-residual-add-pipeline"),
                layout: Some(&residual_add_pipeline_layout),
                module: &residual_add_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let q8_matvec_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-q8-matvec"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Q8_MATVEC_SHADER)),
        });
        let q8_matvec_vec4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-q8-matvec-vec4"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Q8_MATVEC_VEC4_SHADER)),
        });
        let q8_matvec_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-q8-matvec-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let q8_matvec_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-q8-matvec-pipeline-layout"),
                bind_group_layouts: &[Some(&q8_matvec_bind_group_layout)],
                immediate_size: 0,
            });
        let q8_matvec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-q8-matvec-pipeline"),
            layout: Some(&q8_matvec_pipeline_layout),
            module: &q8_matvec_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let q8_matvec_vec4_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rsinfer-q8-matvec-vec4-pipeline"),
                layout: Some(&q8_matvec_pipeline_layout),
                module: &q8_matvec_vec4_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let argmax_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-argmax"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ARGMAX_SHADER)),
        });
        let argmax_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-argmax-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let argmax_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-argmax-pipeline-layout"),
                bind_group_layouts: &[Some(&argmax_bind_group_layout)],
                immediate_size: 0,
            });
        let argmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-argmax-pipeline"),
            layout: Some(&argmax_pipeline_layout),
            module: &argmax_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let swiglu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-swiglu"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SWIGLU_SHADER)),
        });
        let swiglu_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-swiglu-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let swiglu_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-swiglu-pipeline-layout"),
                bind_group_layouts: &[Some(&swiglu_bind_group_layout)],
                immediate_size: 0,
            });
        let swiglu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-swiglu-pipeline"),
            layout: Some(&swiglu_pipeline_layout),
            module: &swiglu_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let limits = device.limits();
        let max_chunk_bytes = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);

        Ok(Self {
            inner: Arc::new(GpuContextInner {
                device,
                queue,
                matvec_pipeline,
                matvec_bind_group_layout,
                rms_norm_pipeline,
                rms_norm_bind_group_layout,
                rms_norm_rope_pipeline,
                rms_norm_rope_bind_group_layout,
                kv_append_pipeline,
                kv_append_bind_group_layout,
                decode_gqa_attention_pipeline,
                decode_gqa_attention_serial_pipeline,
                decode_gqa_attention_bind_group_layout,
                residual_add_pipeline,
                residual_add_bind_group_layout,
                q8_matvec_pipeline,
                q8_matvec_vec4_pipeline,
                q8_matvec_bind_group_layout,
                argmax_pipeline,
                argmax_bind_group_layout,
                swiglu_pipeline,
                swiglu_bind_group_layout,
                max_chunk_bytes,
            }),
        })
    }

    pub fn create_command_encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.inner.queue.submit(Some(encoder.finish()));
        record_submit();
    }

    pub fn read_back_many(&self, buffers: &[&GpuResidentBuffer]) -> Result<Vec<Tensor>, String> {
        if buffers.is_empty() {
            return Ok(Vec::new());
        }
        if buffers
            .iter()
            .any(|buffer| !Arc::ptr_eq(&self.inner, &buffer.context.inner))
        {
            return Err("GPU resident multi-readback requires shared GpuContext".to_string());
        }

        let readbacks: Vec<wgpu::Buffer> = buffers
            .iter()
            .map(|buffer| {
                self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rsinfer-resident-readback-many"),
                    size: bytes_len(buffer.len),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-resident-readback-many-encoder"),
                });
        for (buffer, readback) in buffers.iter().zip(&readbacks) {
            encoder.copy_buffer_to_buffer(&buffer.buffer, 0, readback, 0, bytes_len(buffer.len));
        }
        self.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let mut receivers = Vec::with_capacity(readbacks.len());
        for readback in &readbacks {
            let slice = readback.slice(..);
            let (tx, rx) = mpsc::channel();
            record_map_read();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result.map_err(|e| e.to_string()));
            });
            receivers.push(rx);
        }
        record_poll_wait();
        self.inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;
        for rx in receivers {
            rx.recv()
                .map_err(|e| format!("map callback failed: {e}"))??;
        }

        let mut tensors = Vec::with_capacity(buffers.len());
        for (buffer, readback) in buffers.iter().zip(&readbacks) {
            let slice = readback.slice(..);
            let mapped = slice.get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
            drop(mapped);
            readback.unmap();
            tensors.push(Tensor::from_f32_vec(&buffer.shape, values).map_err(|e| e.to_string())?);
        }
        Ok(tensors)
    }
}

impl GpuMatVec {
    pub fn from_f16_weight(
        weight: &[f16],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::from_f16_weight_with_context(&context, weight, out_features, in_features)
    }

    pub fn from_f16_weight_with_context(
        context: &GpuContext,
        weight: &[f16],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        if weight.len() != out_features * in_features {
            return Err(format!(
                "GPU matvec weight length mismatch: got {}, expected {}",
                weight.len(),
                out_features * in_features
            ));
        }

        let input_buffer = context.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-matvec-input"),
            size: bytes_len(in_features),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let row_bytes = bytes_len(in_features);
        let max_chunk_bytes = context.inner.max_chunk_bytes;
        if row_bytes == 0 || row_bytes > max_chunk_bytes {
            return Err(format!(
                "lm_head row is too large for one GPU storage buffer binding: row={row_bytes}, limit={max_chunk_bytes}"
            ));
        }
        let rows_per_chunk = (max_chunk_bytes / row_bytes).max(1) as usize;
        let mut chunks = Vec::new();
        let mut out_offset = 0usize;
        while out_offset < out_features {
            let rows = rows_per_chunk.min(out_features - out_offset);
            chunks.push(create_chunk(
                &context.inner.device,
                &context.inner.matvec_bind_group_layout,
                &input_buffer,
                weight,
                in_features,
                out_offset,
                rows,
            ));
            out_offset += rows;
        }

        Ok(Self {
            context: context.clone(),
            input_buffer,
            chunks,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let mut outputs = Self::forward_many_same_input(&[self], x)?;
        outputs
            .pop()
            .ok_or_else(|| "GPU matvec returned no output".to_string())
    }

    pub fn forward_many_same_input(
        matvecs: &[&GpuMatVec],
        x: &Tensor,
    ) -> Result<Vec<Tensor>, String> {
        if matvecs.is_empty() {
            return Ok(Vec::new());
        }
        let first = matvecs[0];
        if x.ndim() != 2 || x.shape() != [1, first.in_features] {
            return Err(format!(
                "GPU matvec expects [1, {}], got {:?}",
                first.in_features,
                x.shape()
            ));
        }
        if matvecs
            .iter()
            .any(|matvec| matvec.in_features != first.in_features)
        {
            return Err("GPU matvec batch requires the same input width".to_string());
        }
        if matvecs
            .iter()
            .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &first.context.inner))
        {
            return Err("GPU matvec batch requires a shared GpuContext".to_string());
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU matvec input is not contiguous".to_string())?;
        for matvec in matvecs {
            write_buffer(&matvec.context.inner.queue, &matvec.input_buffer, 0, input);
        }

        let mut encoder =
            first
                .context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-lm-head-encoder"),
                });
        for matvec in matvecs {
            for chunk in &matvec.chunks {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rsinfer-lm-head-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&first.context.inner.matvec_pipeline);
                    pass.set_bind_group(0, &chunk.bind_group, &[]);
                    let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                encoder.copy_buffer_to_buffer(
                    &chunk.output_buffer,
                    0,
                    &chunk.readback_buffer,
                    0,
                    bytes_len(chunk.out_features),
                );
            }
        }
        first.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let mut receivers = Vec::new();
        for (matvec_idx, matvec) in matvecs.iter().enumerate() {
            for (chunk_idx, chunk) in matvec.chunks.iter().enumerate() {
                let slice = chunk.readback_buffer.slice(..);
                let (tx, rx) = mpsc::channel();
                record_map_read();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send((matvec_idx, chunk_idx, result.map_err(|e| e.to_string())));
                });
                receivers.push(rx);
            }
        }
        record_poll_wait();
        first
            .context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, _, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut outputs = Vec::with_capacity(matvecs.len());
        for matvec in matvecs {
            let mut output = vec![0f32; matvec.out_features];
            for chunk in &matvec.chunks {
                let slice = chunk.readback_buffer.slice(..);
                let mapped = slice.get_mapped_range();
                let values = bytemuck::cast_slice::<u8, f32>(&mapped);
                output[chunk.out_offset..chunk.out_offset + chunk.out_features]
                    .copy_from_slice(values);
                drop(mapped);
                chunk.readback_buffer.unmap();
            }
            outputs.push(
                Tensor::from_f32_vec(&[1, matvec.out_features], output)
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(outputs)
    }

    pub fn forward_swiglu_down_same_input(
        gate: &GpuMatVec,
        up: &GpuMatVec,
        down: &GpuMatVec,
        x: &Tensor,
    ) -> Result<Tensor, String> {
        let fused = GpuSwiGluDown::new(gate, up, down)?;
        fused.forward(gate, up, down, x)
    }
}

impl GpuRmsNorm {
    pub fn new(weight: &Tensor, shape: &[usize], eps: f32) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::with_context(&context, weight, shape, eps)
    }

    pub fn with_context(
        context: &GpuContext,
        weight: &Tensor,
        shape: &[usize],
        eps: f32,
    ) -> Result<Self, String> {
        let (&last_dim, prefix) = shape
            .split_last()
            .ok_or_else(|| "GPU RMSNorm expects at least 1D shape".to_string())?;
        let rows = prefix.iter().product::<usize>().max(1);
        let weight_slice = weight.as_slice();
        if weight.shape() != [last_dim] || weight_slice.len() != last_dim {
            return Err(format!(
                "GPU RMSNorm weight shape mismatch: got {:?}, expected [{}]",
                weight.shape(),
                last_dim
            ));
        }
        let numel = shape.iter().product::<usize>();
        let device = &context.inner.device;
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-rmsnorm-input"),
            size: bytes_len(numel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-rmsnorm-weight"),
            contents: bytemuck::cast_slice(weight_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-rmsnorm-output"),
            size: bytes_len(numel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-rmsnorm-readback"),
            size: bytes_len(numel),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = [last_dim as u32, rows as u32, eps.to_bits(), 0_u32];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-rmsnorm-params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rsinfer-rmsnorm-bind-group"),
            layout: &context.inner.rms_norm_bind_group_layout,
            entries: &[
                buffer_entry(0, &input_buffer),
                buffer_entry(1, &weight_buffer),
                buffer_entry(2, &output_buffer),
                buffer_entry(3, &params_buffer),
            ],
        });

        Ok(Self {
            context: context.clone(),
            input_buffer,
            bind_group,
            output_buffer,
            readback_buffer,
            _weight_buffer: weight_buffer,
            _params_buffer: params_buffer,
            shape: shape.to_vec(),
            last_dim,
            rows,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        if x.shape() != self.shape.as_slice() {
            return Err(format!(
                "GPU RMSNorm input shape mismatch: got {:?}, expected {:?}",
                x.shape(),
                self.shape
            ));
        }
        let input = x.as_slice();
        if input.len() != self.rows * self.last_dim {
            return Err(format!(
                "GPU RMSNorm input length mismatch: got {}, expected {}",
                input.len(),
                self.rows * self.last_dim
            ));
        }
        self.context
            .inner
            .queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(input));

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-rmsnorm-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-rmsnorm-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.context.inner.rms_norm_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.rows as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback_buffer,
            0,
            bytes_len(self.rows * self.last_dim),
        );
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let slice = self.readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        record_map_read();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;
        rx.recv()
            .map_err(|e| format!("map callback failed: {e}"))??;

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        self.readback_buffer.unmap();
        Tensor::from_f32_vec(&self.shape, values).map_err(|e| e.to_string())
    }

    pub fn encode_resident(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<(), String> {
        let cache = self.prepare_resident_cache(input, output)?;
        self.encode_resident_with_cache(encoder, input, output, &cache)
    }

    pub fn prepare_resident_cache(
        &self,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<GpuRmsNormResidentCache, String> {
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU RMSNorm resident path requires shared GpuContext".to_string());
        }
        if input.shape != self.shape || output.shape != self.shape {
            return Err(format!(
                "GPU RMSNorm resident shape mismatch: input={:?}, output={:?}, expected {:?}",
                input.shape, output.shape, self.shape
            ));
        }
        let bind_group = self
            .context
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-rmsnorm-resident-bind-group"),
                layout: &self.context.inner.rms_norm_bind_group_layout,
                entries: &[
                    buffer_entry(0, &input.buffer),
                    buffer_entry(1, &self._weight_buffer),
                    buffer_entry(2, &output.buffer),
                    buffer_entry(3, &self._params_buffer),
                ],
            });
        Ok(GpuRmsNormResidentCache { bind_group })
    }

    pub fn encode_resident_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        cache: &GpuRmsNormResidentCache,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU RMSNorm resident path requires shared GpuContext".to_string());
        }
        if input.shape != self.shape || output.shape != self.shape {
            return Err(format!(
                "GPU RMSNorm resident shape mismatch: input={:?}, output={:?}, expected {:?}",
                input.shape, output.shape, self.shape
            ));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsinfer-rmsnorm-resident-pass"),
            timestamp_writes: None,
        });
        self.encode_resident_with_cache_in_pass(&mut pass, input, output, cache)?;
        Ok(())
    }

    pub fn encode_resident_with_cache_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        cache: &GpuRmsNormResidentCache,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU RMSNorm resident path requires shared GpuContext".to_string());
        }
        if input.shape != self.shape || output.shape != self.shape {
            return Err(format!(
                "GPU RMSNorm resident shape mismatch: input={:?}, output={:?}, expected {:?}",
                input.shape, output.shape, self.shape
            ));
        }
        pass.set_pipeline(&self.context.inner.rms_norm_pipeline);
        pass.set_bind_group(0, &cache.bind_group, &[]);
        pass.dispatch_workgroups(self.rows as u32, 1, 1);
        Ok(())
    }
}

impl GpuQkRmsNormRope {
    fn write_position_buffer(&self, buffer: &wgpu::Buffer, pos: usize) {
        let position_params = [pos as u32, 0_u32, 0_u32, 0_u32];
        write_position_buffer(&self.context.inner.queue, buffer, 0, &position_params);
    }

    pub fn new(config: GpuQkRmsNormRopeConfig<'_>) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::with_context(&context, config)
    }

    pub fn with_context(
        context: &GpuContext,
        config: GpuQkRmsNormRopeConfig<'_>,
    ) -> Result<Self, String> {
        let GpuQkRmsNormRopeConfig {
            q_weight,
            k_weight,
            q_shape,
            k_shape,
            pos,
            inv_freq,
            q_eps,
            k_eps,
        } = config;
        let (&q_head_dim, q_prefix) = q_shape
            .split_last()
            .ok_or_else(|| "GPU Q RMSNorm+RoPE expects at least 1D shape".to_string())?;
        let (&k_head_dim, k_prefix) = k_shape
            .split_last()
            .ok_or_else(|| "GPU K RMSNorm+RoPE expects at least 1D shape".to_string())?;
        let q_rows = q_prefix.iter().product::<usize>().max(1);
        let k_rows = k_prefix.iter().product::<usize>().max(1);
        if q_head_dim != k_head_dim {
            return Err(format!(
                "GPU Q/K RMSNorm+RoPE head_dim mismatch: q={}, k={}",
                q_head_dim, k_head_dim
            ));
        }
        if q_head_dim % 2 != 0 {
            return Err(format!(
                "GPU Q/K RMSNorm+RoPE requires even head_dim, got {}",
                q_head_dim
            ));
        }
        let half_dim = q_head_dim / 2;
        if inv_freq.len() != half_dim {
            return Err(format!(
                "GPU Q/K RMSNorm+RoPE inv_freq length mismatch: got {}, expected {}",
                inv_freq.len(),
                half_dim
            ));
        }
        if q_weight.shape() != [q_head_dim] || q_weight.as_slice().len() != q_head_dim {
            return Err(format!(
                "GPU Q RMSNorm+RoPE weight shape mismatch: got {:?}, expected [{}]",
                q_weight.shape(),
                q_head_dim
            ));
        }
        if k_weight.shape() != [k_head_dim] || k_weight.as_slice().len() != k_head_dim {
            return Err(format!(
                "GPU K RMSNorm+RoPE weight shape mismatch: got {:?}, expected [{}]",
                k_weight.shape(),
                k_head_dim
            ));
        }

        let device = &context.inner.device;
        let inv_freq_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-rmsnorm-rope-inv-freq"),
            contents: bytemuck::cast_slice(inv_freq),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let q_numel = q_shape.iter().product::<usize>();
        let q_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-q-rmsnorm-rope-input"),
            size: bytes_len(q_numel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-q-rmsnorm-rope-weight"),
            contents: bytemuck::cast_slice(q_weight.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let q_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-q-rmsnorm-rope-output"),
            size: bytes_len(q_numel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-q-rmsnorm-rope-readback"),
            size: bytes_len(q_numel),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let q_static_params = [
            q_head_dim as u32,
            q_rows as u32,
            half_dim as u32,
            q_eps.to_bits(),
        ];
        let q_static_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-q-rmsnorm-rope-static-params"),
            contents: bytemuck::cast_slice(&q_static_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let position_params = [pos as u32, 0_u32, 0_u32, 0_u32];
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-rmsnorm-rope-position"),
            contents: bytemuck::cast_slice(&position_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let q_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rsinfer-q-rmsnorm-rope-bind-group"),
            layout: &context.inner.rms_norm_rope_bind_group_layout,
            entries: &[
                buffer_entry(0, &q_input_buffer),
                buffer_entry(1, &q_weight_buffer),
                buffer_entry(2, &inv_freq_buffer),
                buffer_entry(3, &q_output_buffer),
                buffer_entry(4, &q_static_params_buffer),
                buffer_entry(5, &position_buffer),
            ],
        });

        let k_numel = k_shape.iter().product::<usize>();
        let k_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-k-rmsnorm-rope-input"),
            size: bytes_len(k_numel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k_weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-k-rmsnorm-rope-weight"),
            contents: bytemuck::cast_slice(k_weight.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-k-rmsnorm-rope-output"),
            size: bytes_len(k_numel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let k_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-k-rmsnorm-rope-readback"),
            size: bytes_len(k_numel),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k_static_params = [
            k_head_dim as u32,
            k_rows as u32,
            half_dim as u32,
            k_eps.to_bits(),
        ];
        let k_static_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-k-rmsnorm-rope-static-params"),
            contents: bytemuck::cast_slice(&k_static_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let k_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rsinfer-k-rmsnorm-rope-bind-group"),
            layout: &context.inner.rms_norm_rope_bind_group_layout,
            entries: &[
                buffer_entry(0, &k_input_buffer),
                buffer_entry(1, &k_weight_buffer),
                buffer_entry(2, &inv_freq_buffer),
                buffer_entry(3, &k_output_buffer),
                buffer_entry(4, &k_static_params_buffer),
                buffer_entry(5, &position_buffer),
            ],
        });

        Ok(Self {
            context: context.clone(),
            q_input_buffer,
            q_bind_group,
            q_output_buffer,
            q_readback_buffer,
            _q_weight_buffer: q_weight_buffer,
            q_static_params_buffer,
            position_buffer,
            q_shape: q_shape.to_vec(),
            q_rows,
            q_head_dim,
            k_input_buffer,
            k_bind_group,
            k_output_buffer,
            k_readback_buffer,
            _k_weight_buffer: k_weight_buffer,
            k_static_params_buffer,
            _inv_freq_buffer: inv_freq_buffer,
            k_shape: k_shape.to_vec(),
            k_rows,
            k_head_dim,
        })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor), String> {
        if q.shape() != self.q_shape.as_slice() {
            return Err(format!(
                "GPU Q RMSNorm+RoPE input shape mismatch: got {:?}, expected {:?}",
                q.shape(),
                self.q_shape
            ));
        }
        if k.shape() != self.k_shape.as_slice() {
            return Err(format!(
                "GPU K RMSNorm+RoPE input shape mismatch: got {:?}, expected {:?}",
                k.shape(),
                self.k_shape
            ));
        }
        let q_input = q.as_slice();
        let k_input = k.as_slice();
        if q_input.len() != self.q_rows * self.q_head_dim {
            return Err(format!(
                "GPU Q RMSNorm+RoPE input length mismatch: got {}, expected {}",
                q_input.len(),
                self.q_rows * self.q_head_dim
            ));
        }
        if k_input.len() != self.k_rows * self.k_head_dim {
            return Err(format!(
                "GPU K RMSNorm+RoPE input length mismatch: got {}, expected {}",
                k_input.len(),
                self.k_rows * self.k_head_dim
            ));
        }
        write_buffer(&self.context.inner.queue, &self.q_input_buffer, 0, q_input);
        write_buffer(&self.context.inner.queue, &self.k_input_buffer, 0, k_input);

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-qk-rmsnorm-rope-encoder"),
                });
        for (bind_group, rows, label) in [
            (
                &self.q_bind_group,
                self.q_rows as u32,
                "rsinfer-q-rmsnorm-rope-pass",
            ),
            (
                &self.k_bind_group,
                self.k_rows as u32,
                "rsinfer-k-rmsnorm-rope-pass",
            ),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.context.inner.rms_norm_rope_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(rows, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.q_output_buffer,
            0,
            &self.q_readback_buffer,
            0,
            bytes_len(self.q_rows * self.q_head_dim),
        );
        encoder.copy_buffer_to_buffer(
            &self.k_output_buffer,
            0,
            &self.k_readback_buffer,
            0,
            bytes_len(self.k_rows * self.k_head_dim),
        );
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let q_slice = self.q_readback_buffer.slice(..);
        let (q_tx, q_rx) = mpsc::channel();
        record_map_read();
        q_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = q_tx.send(result.map_err(|e| e.to_string()));
        });
        let k_slice = self.k_readback_buffer.slice(..);
        let (k_tx, k_rx) = mpsc::channel();
        record_map_read();
        k_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = k_tx.send(result.map_err(|e| e.to_string()));
        });
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;
        q_rx.recv()
            .map_err(|e| format!("q map callback failed: {e}"))??;
        k_rx.recv()
            .map_err(|e| format!("k map callback failed: {e}"))??;

        let q_mapped = q_slice.get_mapped_range();
        let q_values = bytemuck::cast_slice::<u8, f32>(&q_mapped).to_vec();
        drop(q_mapped);
        self.q_readback_buffer.unmap();

        let k_mapped = k_slice.get_mapped_range();
        let k_values = bytemuck::cast_slice::<u8, f32>(&k_mapped).to_vec();
        drop(k_mapped);
        self.k_readback_buffer.unmap();

        Ok((
            Tensor::from_f32_vec(&self.q_shape, q_values).map_err(|e| e.to_string())?,
            Tensor::from_f32_vec(&self.k_shape, k_values).map_err(|e| e.to_string())?,
        ))
    }

    pub fn encode_resident(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
    ) -> Result<(), String> {
        let cache = self.prepare_resident_cache(q_input, k_input, q_output, k_output)?;
        self.encode_resident_with_cache(encoder, q_input, k_input, q_output, k_output, &cache)
    }

    pub fn encode_resident_at_pos(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
        pos: usize,
    ) -> Result<(), String> {
        self.write_position_buffer(&self.position_buffer, pos);
        self.encode_resident_internal_cached(
            encoder,
            q_input,
            k_input,
            q_output,
            k_output,
            &self.prepare_resident_cache(q_input, k_input, q_output, k_output)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_at_pos_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
        pos: usize,
        cache: &GpuQkRmsNormRopeResidentCache,
    ) -> Result<(), String> {
        self.write_position_buffer(&cache.position_buffer, pos);
        self.encode_resident_internal_cached(encoder, q_input, k_input, q_output, k_output, cache)
    }

    pub fn write_resident_position_with_cache(
        &self,
        pos: usize,
        cache: &GpuQkRmsNormRopeResidentCache,
    ) {
        self.write_position_buffer(&cache.position_buffer, pos);
    }

    pub fn prepare_resident_cache(
        &self,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
    ) -> Result<GpuQkRmsNormRopeResidentCache, String> {
        self.prepare_resident_cache_with_position_buffer(
            q_input,
            k_input,
            q_output,
            k_output,
            &self.position_buffer,
        )
    }

    pub fn prepare_resident_cache_with_position_buffer(
        &self,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
        position_buffer: &wgpu::Buffer,
    ) -> Result<GpuQkRmsNormRopeResidentCache, String> {
        if !Arc::ptr_eq(&self.context.inner, &q_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &k_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &q_output.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &k_output.context.inner)
        {
            return Err(
                "GPU Q/K RMSNorm+RoPE resident path requires shared GpuContext".to_string(),
            );
        }
        if q_input.shape != self.q_shape || q_output.shape != self.q_shape {
            return Err(format!(
                "GPU Q RMSNorm+RoPE resident shape mismatch: input={:?}, output={:?}, expected {:?}",
                q_input.shape, q_output.shape, self.q_shape
            ));
        }
        if k_input.shape != self.k_shape || k_output.shape != self.k_shape {
            return Err(format!(
                "GPU K RMSNorm+RoPE resident shape mismatch: input={:?}, output={:?}, expected {:?}",
                k_input.shape, k_output.shape, self.k_shape
            ));
        }
        Ok(GpuQkRmsNormRopeResidentCache {
            q_bind_group: self
                .context
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-q-rmsnorm-rope-resident-bind-group"),
                    layout: &self.context.inner.rms_norm_rope_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &q_input.buffer),
                        buffer_entry(1, &self._q_weight_buffer),
                        buffer_entry(2, &self._inv_freq_buffer),
                        buffer_entry(3, &q_output.buffer),
                        buffer_entry(4, &self.q_static_params_buffer),
                        buffer_entry(5, position_buffer),
                    ],
                }),
            k_bind_group: self
                .context
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-k-rmsnorm-rope-resident-bind-group"),
                    layout: &self.context.inner.rms_norm_rope_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &k_input.buffer),
                        buffer_entry(1, &self._k_weight_buffer),
                        buffer_entry(2, &self._inv_freq_buffer),
                        buffer_entry(3, &k_output.buffer),
                        buffer_entry(4, &self.k_static_params_buffer),
                        buffer_entry(5, position_buffer),
                    ],
                }),
            position_buffer: position_buffer.clone(),
        })
    }

    pub fn encode_resident_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
        cache: &GpuQkRmsNormRopeResidentCache,
    ) -> Result<(), String> {
        self.encode_resident_internal_cached(encoder, q_input, k_input, q_output, k_output, cache)
    }

    fn encode_resident_internal_cached(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
        cache: &GpuQkRmsNormRopeResidentCache,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &q_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &k_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &q_output.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &k_output.context.inner)
        {
            return Err(
                "GPU Q/K RMSNorm+RoPE resident path requires shared GpuContext".to_string(),
            );
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsinfer-qk-rmsnorm-rope-resident-pass"),
            timestamp_writes: None,
        });
        self.encode_resident_in_pass_cached(
            &mut pass, q_input, k_input, q_output, k_output, cache,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_at_pos_with_cache_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
        pos: usize,
        cache: &GpuQkRmsNormRopeResidentCache,
    ) -> Result<(), String> {
        self.write_position_buffer(&cache.position_buffer, pos);
        self.encode_resident_in_pass_cached(pass, q_input, k_input, q_output, k_output, cache)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_in_pass_cached(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        q_input: &GpuResidentBuffer,
        k_input: &GpuResidentBuffer,
        q_output: &GpuResidentBuffer,
        k_output: &GpuResidentBuffer,
        cache: &GpuQkRmsNormRopeResidentCache,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &q_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &k_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &q_output.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &k_output.context.inner)
        {
            return Err(
                "GPU Q/K RMSNorm+RoPE resident path requires shared GpuContext".to_string(),
            );
        }
        pass.set_pipeline(&self.context.inner.rms_norm_rope_pipeline);
        pass.set_bind_group(0, &cache.q_bind_group, &[]);
        pass.dispatch_workgroups(self.q_rows as u32, 1, 1);
        pass.set_bind_group(0, &cache.k_bind_group, &[]);
        pass.dispatch_workgroups(self.k_rows as u32, 1, 1);
        Ok(())
    }
}

impl GpuKvAppend {
    fn write_position_buffer(&self, buffer: &wgpu::Buffer, position: usize) {
        let position_params = [position as u32, 0_u32, 0_u32, 0_u32];
        write_position_buffer(&self.context.inner.queue, buffer, 0, &position_params);
    }

    pub fn new(num_heads: usize, head_dim: usize, max_len: usize) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::with_context(&context, num_heads, head_dim, max_len)
    }

    pub fn with_context(
        context: &GpuContext,
        num_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> Result<Self, String> {
        Self::with_context_inner(context, num_heads, head_dim, max_len, false)
    }

    pub fn with_context_resident_only(
        context: &GpuContext,
        num_heads: usize,
        head_dim: usize,
        max_len: usize,
    ) -> Result<Self, String> {
        Self::with_context_inner(context, num_heads, head_dim, max_len, true)
    }

    fn with_context_inner(
        context: &GpuContext,
        num_heads: usize,
        head_dim: usize,
        max_len: usize,
        resident_only: bool,
    ) -> Result<Self, String> {
        if num_heads == 0 || head_dim == 0 || max_len == 0 {
            return Err("GPU KV append requires num_heads/head_dim/max_len > 0".to_string());
        }
        let device = &context.inner.device;
        let row_len = num_heads * head_dim;
        let cache_len = num_heads * max_len * head_dim;

        let key_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-kv-append-key-input"),
            size: bytes_len(row_len),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let value_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-kv-append-value-input"),
            size: bytes_len(row_len),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let static_params = [head_dim as u32, num_heads as u32, max_len as u32, 0_u32];
        let static_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-kv-append-static-params"),
            contents: bytemuck::cast_slice(&static_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let position_params = [0_u32, 0_u32, 0_u32, 0_u32];
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-kv-append-position"),
            contents: bytemuck::cast_slice(&position_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let (
            key_cache_buffer,
            value_cache_buffer,
            key_bind_group,
            value_bind_group,
            key_readback_buffer,
            value_readback_buffer,
        ) = if resident_only {
            (None, None, None, None, None, None)
        } else {
            let key_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-kv-append-key-cache"),
                size: bytes_len(cache_len),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let value_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-kv-append-value-cache"),
                size: bytes_len(cache_len),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let key_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-kv-append-key-readback"),
                size: bytes_len(cache_len),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let value_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-kv-append-value-readback"),
                size: bytes_len(cache_len),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let key_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-kv-append-key-bind-group"),
                layout: &context.inner.kv_append_bind_group_layout,
                entries: &[
                    buffer_entry(0, &key_input_buffer),
                    buffer_entry(1, &key_cache_buffer),
                    buffer_entry(2, &static_params_buffer),
                    buffer_entry(3, &position_buffer),
                ],
            });
            let value_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-kv-append-value-bind-group"),
                layout: &context.inner.kv_append_bind_group_layout,
                entries: &[
                    buffer_entry(0, &value_input_buffer),
                    buffer_entry(1, &value_cache_buffer),
                    buffer_entry(2, &static_params_buffer),
                    buffer_entry(3, &position_buffer),
                ],
            });
            (
                Some(key_cache_buffer),
                Some(value_cache_buffer),
                Some(key_bind_group),
                Some(value_bind_group),
                Some(key_readback_buffer),
                Some(value_readback_buffer),
            )
        };

        Ok(Self {
            context: context.clone(),
            key_input_buffer,
            value_input_buffer,
            key_cache_buffer,
            value_cache_buffer,
            key_bind_group,
            value_bind_group,
            key_readback_buffer,
            value_readback_buffer,
            static_params_buffer,
            position_buffer,
            num_heads,
            head_dim,
            max_len,
            current_len: 0,
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn snapshot_len(&self) -> usize {
        self.current_len
    }

    pub fn restore_len(&mut self, len: usize) -> Result<(), String> {
        if len > self.max_len {
            return Err(format!(
                "GPU KV restore length {} exceeds max_len {}",
                len, self.max_len
            ));
        }
        self.current_len = len;
        Ok(())
    }

    pub fn append_decode_one_raw(
        &mut self,
        position: usize,
        new_k: &[f32],
        new_v: &[f32],
    ) -> Result<(), String> {
        if position >= self.max_len {
            return Err(format!(
                "GPU KV append position {} exceeds max_len {}",
                position, self.max_len
            ));
        }
        let expected = self.num_heads * self.head_dim;
        if new_k.len() != expected || new_v.len() != expected {
            return Err(format!(
                "GPU KV append expects {} values per K/V row, got {}/{}",
                expected,
                new_k.len(),
                new_v.len()
            ));
        }
        write_buffer(&self.context.inner.queue, &self.key_input_buffer, 0, new_k);
        write_buffer(
            &self.context.inner.queue,
            &self.value_input_buffer,
            0,
            new_v,
        );
        self.write_position_buffer(&self.position_buffer, position);
        let key_bind_group = self.key_bind_group.as_ref().ok_or_else(|| {
            "GPU KV append standalone buffers are unavailable in resident-only mode".to_string()
        })?;
        let value_bind_group = self.value_bind_group.as_ref().ok_or_else(|| {
            "GPU KV append standalone buffers are unavailable in resident-only mode".to_string()
        })?;

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-kv-append-encoder"),
                });
        for (bind_group, label) in [
            (key_bind_group, "rsinfer-kv-append-key-pass"),
            (value_bind_group, "rsinfer-kv-append-value-pass"),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.context.inner.kv_append_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.num_heads as u32, 1, 1);
        }
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();
        self.current_len = self.current_len.max(position + 1);
        Ok(())
    }

    pub fn read_compact(&self) -> Result<(Tensor, Tensor), String> {
        let total_len = self.num_heads * self.max_len * self.head_dim;
        let key_cache_buffer = self.key_cache_buffer.as_ref().ok_or_else(|| {
            "GPU KV compact readback is unavailable in resident-only mode".to_string()
        })?;
        let value_cache_buffer = self.value_cache_buffer.as_ref().ok_or_else(|| {
            "GPU KV compact readback is unavailable in resident-only mode".to_string()
        })?;
        let key_readback_buffer = self.key_readback_buffer.as_ref().ok_or_else(|| {
            "GPU KV compact readback is unavailable in resident-only mode".to_string()
        })?;
        let value_readback_buffer = self.value_readback_buffer.as_ref().ok_or_else(|| {
            "GPU KV compact readback is unavailable in resident-only mode".to_string()
        })?;
        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-kv-readback-encoder"),
                });
        encoder.copy_buffer_to_buffer(
            key_cache_buffer,
            0,
            key_readback_buffer,
            0,
            bytes_len(total_len),
        );
        encoder.copy_buffer_to_buffer(
            value_cache_buffer,
            0,
            value_readback_buffer,
            0,
            bytes_len(total_len),
        );
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let key_slice = key_readback_buffer.slice(..);
        let (key_tx, key_rx) = mpsc::channel();
        record_map_read();
        key_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = key_tx.send(result.map_err(|e| e.to_string()));
        });
        let value_slice = value_readback_buffer.slice(..);
        let (value_tx, value_rx) = mpsc::channel();
        record_map_read();
        value_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = value_tx.send(result.map_err(|e| e.to_string()));
        });
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;
        key_rx
            .recv()
            .map_err(|e| format!("key map callback failed: {e}"))??;
        value_rx
            .recv()
            .map_err(|e| format!("value map callback failed: {e}"))??;

        let key_mapped = key_slice.get_mapped_range();
        let raw_key = bytemuck::cast_slice::<u8, f32>(&key_mapped);
        let mut compact_key = vec![0f32; self.num_heads * self.current_len * self.head_dim];
        let cache_head_len = self.max_len * self.head_dim;
        let compact_head_len = self.current_len * self.head_dim;
        for h in 0..self.num_heads {
            let src_start = h * cache_head_len;
            let dst_start = h * compact_head_len;
            compact_key[dst_start..dst_start + compact_head_len]
                .copy_from_slice(&raw_key[src_start..src_start + compact_head_len]);
        }
        drop(key_mapped);
        key_readback_buffer.unmap();

        let value_mapped = value_slice.get_mapped_range();
        let raw_value = bytemuck::cast_slice::<u8, f32>(&value_mapped);
        let mut compact_value = vec![0f32; self.num_heads * self.current_len * self.head_dim];
        for h in 0..self.num_heads {
            let src_start = h * cache_head_len;
            let dst_start = h * compact_head_len;
            compact_value[dst_start..dst_start + compact_head_len]
                .copy_from_slice(&raw_value[src_start..src_start + compact_head_len]);
        }
        drop(value_mapped);
        value_readback_buffer.unmap();

        Ok((
            Tensor::from_f32_vec(
                &[self.num_heads, self.current_len, self.head_dim],
                compact_key,
            )
            .map_err(|e| e.to_string())?,
            Tensor::from_f32_vec(
                &[self.num_heads, self.current_len, self.head_dim],
                compact_value,
            )
            .map_err(|e| e.to_string())?,
        ))
    }

    pub fn encode_resident(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        position: usize,
        key_input: &GpuResidentBuffer,
        value_input: &GpuResidentBuffer,
        key_cache: &GpuResidentBuffer,
        value_cache: &GpuResidentBuffer,
    ) -> Result<(), String> {
        let cache = self.prepare_resident_cache(key_input, value_input, key_cache, value_cache)?;
        self.encode_resident_with_cache(
            encoder,
            position,
            key_input,
            value_input,
            key_cache,
            value_cache,
            &cache,
        )
    }

    pub fn prepare_resident_cache(
        &self,
        key_input: &GpuResidentBuffer,
        value_input: &GpuResidentBuffer,
        key_cache: &GpuResidentBuffer,
        value_cache: &GpuResidentBuffer,
    ) -> Result<GpuKvAppendResidentCache, String> {
        self.prepare_resident_cache_with_position_buffer(
            key_input,
            value_input,
            key_cache,
            value_cache,
            &self.position_buffer,
        )
    }

    pub fn prepare_resident_cache_with_position_buffer(
        &self,
        key_input: &GpuResidentBuffer,
        value_input: &GpuResidentBuffer,
        key_cache: &GpuResidentBuffer,
        value_cache: &GpuResidentBuffer,
        position_buffer: &wgpu::Buffer,
    ) -> Result<GpuKvAppendResidentCache, String> {
        if !Arc::ptr_eq(&self.context.inner, &key_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key_cache.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_cache.context.inner)
        {
            return Err("GPU KV append resident path requires shared GpuContext".to_string());
        }

        let expected_cache_shape = vec![self.num_heads, self.max_len, self.head_dim];
        let expected_row_len = self.num_heads * self.head_dim;
        if key_input.len != expected_row_len || value_input.len != expected_row_len {
            return Err(format!(
                "GPU KV append resident row length mismatch: key={}, value={}, expected {}",
                key_input.len, value_input.len, expected_row_len
            ));
        }
        if key_cache.shape != expected_cache_shape || value_cache.shape != expected_cache_shape {
            return Err(format!(
                "GPU KV append resident cache shape mismatch: key={:?}, value={:?}, expected {:?}",
                key_cache.shape, value_cache.shape, expected_cache_shape
            ));
        }
        Ok(GpuKvAppendResidentCache {
            key_bind_group: self.context.inner.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-kv-append-key-resident-bind-group"),
                    layout: &self.context.inner.kv_append_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &key_input.buffer),
                        buffer_entry(1, &key_cache.buffer),
                        buffer_entry(2, &self.static_params_buffer),
                        buffer_entry(3, position_buffer),
                    ],
                },
            ),
            value_bind_group: self.context.inner.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-kv-append-value-resident-bind-group"),
                    layout: &self.context.inner.kv_append_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &value_input.buffer),
                        buffer_entry(1, &value_cache.buffer),
                        buffer_entry(2, &self.static_params_buffer),
                        buffer_entry(3, position_buffer),
                    ],
                },
            ),
            position_buffer: position_buffer.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_cache(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        position: usize,
        key_input: &GpuResidentBuffer,
        value_input: &GpuResidentBuffer,
        key_cache: &GpuResidentBuffer,
        value_cache: &GpuResidentBuffer,
        cache: &GpuKvAppendResidentCache,
    ) -> Result<(), String> {
        if position >= self.max_len {
            return Err(format!(
                "GPU KV append position {} exceeds max_len {}",
                position, self.max_len
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &key_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key_cache.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_cache.context.inner)
        {
            return Err("GPU KV append resident path requires shared GpuContext".to_string());
        }

        self.write_position_buffer(&cache.position_buffer, position);
        self.encode_resident_with_cache_current_position(
            encoder,
            position,
            key_input,
            value_input,
            key_cache,
            value_cache,
            cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_cache_current_position(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        position: usize,
        key_input: &GpuResidentBuffer,
        value_input: &GpuResidentBuffer,
        key_cache: &GpuResidentBuffer,
        value_cache: &GpuResidentBuffer,
        cache: &GpuKvAppendResidentCache,
    ) -> Result<(), String> {
        if position >= self.max_len {
            return Err(format!(
                "GPU KV append position {} exceeds max_len {}",
                position, self.max_len
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &key_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key_cache.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_cache.context.inner)
        {
            return Err("GPU KV append resident path requires shared GpuContext".to_string());
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsinfer-kv-append-resident-pass"),
            timestamp_writes: None,
        });
        self.encode_resident_in_pass_current_position(
            &mut pass,
            position,
            key_input,
            value_input,
            key_cache,
            value_cache,
            cache,
        )?;
        Ok(())
    }

    pub fn write_resident_position_with_cache(
        &self,
        position: usize,
        cache: &GpuKvAppendResidentCache,
    ) {
        self.write_position_buffer(&cache.position_buffer, position);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_cache_in_pass(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        position: usize,
        key_input: &GpuResidentBuffer,
        value_input: &GpuResidentBuffer,
        key_cache: &GpuResidentBuffer,
        value_cache: &GpuResidentBuffer,
        cache: &GpuKvAppendResidentCache,
    ) -> Result<(), String> {
        self.write_position_buffer(&cache.position_buffer, position);
        self.encode_resident_in_pass_current_position(
            pass,
            position,
            key_input,
            value_input,
            key_cache,
            value_cache,
            cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_in_pass_current_position(
        &mut self,
        pass: &mut wgpu::ComputePass<'_>,
        position: usize,
        key_input: &GpuResidentBuffer,
        value_input: &GpuResidentBuffer,
        key_cache: &GpuResidentBuffer,
        value_cache: &GpuResidentBuffer,
        cache: &GpuKvAppendResidentCache,
    ) -> Result<(), String> {
        if position >= self.max_len {
            return Err(format!(
                "GPU KV append position {} exceeds max_len {}",
                position, self.max_len
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &key_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key_cache.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value_cache.context.inner)
        {
            return Err("GPU KV append resident path requires shared GpuContext".to_string());
        }

        pass.set_pipeline(&self.context.inner.kv_append_pipeline);
        pass.set_bind_group(0, &cache.key_bind_group, &[]);
        pass.dispatch_workgroups(self.num_heads as u32, 1, 1);
        pass.set_bind_group(0, &cache.value_bind_group, &[]);
        pass.dispatch_workgroups(self.num_heads as u32, 1, 1);

        self.current_len = self.current_len.max(position + 1);
        Ok(())
    }
}

impl GpuDecodeGqaAttention {
    fn write_position_buffer(&self, buffer: &wgpu::Buffer, position: usize) {
        let position_params = [position as u32, 0_u32, 0_u32, 0_u32];
        write_position_buffer(&self.context.inner.queue, buffer, 0, &position_params);
    }

    pub fn new(config: GpuDecodeGqaAttentionConfig) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::with_context(&context, config)
    }

    pub fn with_context(
        context: &GpuContext,
        config: GpuDecodeGqaAttentionConfig,
    ) -> Result<Self, String> {
        let GpuDecodeGqaAttentionConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            max_len,
            scale,
        } = config;
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || max_len == 0 {
            return Err(
                "GPU decode GQA attention requires non-zero heads/head_dim/max_len".to_string(),
            );
        }
        if num_heads % num_kv_heads != 0 {
            return Err(format!(
                "GPU decode GQA attention requires num_heads {} divisible by num_kv_heads {}",
                num_heads, num_kv_heads
            ));
        }

        let device = &context.inner.device;
        let q_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-decode-gqa-q"),
            size: bytes_len(num_heads * head_dim),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let key_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-decode-gqa-key"),
            size: bytes_len(num_kv_heads * max_len * head_dim),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-decode-gqa-value"),
            size: bytes_len(num_kv_heads * max_len * head_dim),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-decode-gqa-output"),
            size: bytes_len(num_heads * head_dim),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-decode-gqa-readback"),
            size: bytes_len(num_heads * head_dim),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let static_params = [
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            max_len as u32,
            (num_heads / num_kv_heads) as u32,
            scale.to_bits(),
            0_u32,
        ];
        let static_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-decode-gqa-static-params"),
            contents: bytemuck::cast_slice(&static_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let position_params = [0_u32, 0_u32, 0_u32, 0_u32];
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-decode-gqa-position"),
            contents: bytemuck::cast_slice(&position_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rsinfer-decode-gqa-bind-group"),
            layout: &context.inner.decode_gqa_attention_bind_group_layout,
            entries: &[
                buffer_entry(0, &q_buffer),
                buffer_entry(1, &key_buffer),
                buffer_entry(2, &value_buffer),
                buffer_entry(3, &output_buffer),
                buffer_entry(4, &static_params_buffer),
                buffer_entry(5, &position_buffer),
            ],
        });

        Ok(Self {
            context: context.clone(),
            q_buffer,
            key_buffer,
            value_buffer,
            output_buffer,
            readback_buffer,
            static_params_buffer,
            position_buffer,
            bind_group,
            num_heads,
            num_kv_heads,
            head_dim,
            max_len,
        })
    }

    pub fn forward_raw(
        &self,
        q: &[f32],
        key: &[f32],
        value: &[f32],
        seq_len_k: usize,
    ) -> Result<Vec<f32>, String> {
        if seq_len_k == 0 || seq_len_k > self.max_len {
            return Err(format!(
                "GPU decode GQA attention seq_len_k {} exceeds max_len {}",
                seq_len_k, self.max_len
            ));
        }
        let q_expected = self.num_heads * self.head_dim;
        let cache_expected = self.num_kv_heads * self.max_len * self.head_dim;
        if q.len() != q_expected {
            return Err(format!(
                "GPU decode GQA attention q length mismatch: got {}, expected {}",
                q.len(),
                q_expected
            ));
        }
        if key.len() != cache_expected || value.len() != cache_expected {
            return Err(format!(
                "GPU decode GQA attention cache length mismatch: got key={}, value={}, expected {}",
                key.len(),
                value.len(),
                cache_expected
            ));
        }
        write_buffer(&self.context.inner.queue, &self.q_buffer, 0, q);
        write_buffer(&self.context.inner.queue, &self.key_buffer, 0, key);
        write_buffer(&self.context.inner.queue, &self.value_buffer, 0, value);
        self.write_position_buffer(&self.position_buffer, seq_len_k - 1);

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-decode-gqa-attention-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-decode-gqa-attention-pass"),
                timestamp_writes: None,
            });
            let pipeline = if seq_len_k <= DECODE_GQA_ATTENTION_PARALLEL_MAX_SEQ_LEN {
                &self.context.inner.decode_gqa_attention_pipeline
            } else {
                &self.context.inner.decode_gqa_attention_serial_pipeline
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.num_heads as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback_buffer,
            0,
            bytes_len(q_expected),
        );
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let slice = self.readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        record_map_read();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;
        rx.recv()
            .map_err(|e| format!("map callback failed: {e}"))??;

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        self.readback_buffer.unmap();
        Ok(values)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        position: usize,
    ) -> Result<(), String> {
        let cache = self.prepare_resident_cache(q, key, value, output)?;
        self.encode_resident_with_cache(encoder, q, key, value, output, position, &cache)
    }

    pub fn prepare_resident_cache(
        &self,
        q: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<GpuDecodeGqaAttentionResidentCache, String> {
        self.prepare_resident_cache_with_position_buffer(
            q,
            key,
            value,
            output,
            &self.position_buffer,
        )
    }

    pub fn prepare_resident_cache_with_position_buffer(
        &self,
        q: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        position_buffer: &wgpu::Buffer,
    ) -> Result<GpuDecodeGqaAttentionResidentCache, String> {
        if !Arc::ptr_eq(&self.context.inner, &q.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU decode GQA attention resident path requires shared GpuContext".to_string(),
            );
        }

        let q_expected = self.num_heads * self.head_dim;
        let cache_expected = self.num_kv_heads * self.max_len * self.head_dim;
        if q.len != q_expected || output.len != q_expected {
            return Err(format!(
                "GPU decode GQA attention resident q/output length mismatch: q={}, output={}, expected {}",
                q.len, output.len, q_expected
            ));
        }
        if key.len != cache_expected || value.len != cache_expected {
            return Err(format!(
                "GPU decode GQA attention resident cache length mismatch: key={}, value={}, expected {}",
                key.len, value.len, cache_expected
            ));
        }
        Ok(GpuDecodeGqaAttentionResidentCache {
            bind_group: self
                .context
                .inner
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-decode-gqa-resident-bind-group"),
                    layout: &self.context.inner.decode_gqa_attention_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &q.buffer),
                        buffer_entry(1, &key.buffer),
                        buffer_entry(2, &value.buffer),
                        buffer_entry(3, &output.buffer),
                        buffer_entry(4, &self.static_params_buffer),
                        buffer_entry(5, position_buffer),
                    ],
                }),
            position_buffer: position_buffer.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        position: usize,
        cache: &GpuDecodeGqaAttentionResidentCache,
    ) -> Result<(), String> {
        if position >= self.max_len {
            return Err(format!(
                "GPU decode GQA attention position {} exceeds max_len {}",
                position, self.max_len
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &q.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU decode GQA attention resident path requires shared GpuContext".to_string(),
            );
        }

        self.write_position_buffer(&cache.position_buffer, position);
        self.encode_resident_with_cache_current_position(
            encoder, q, key, value, output, position, cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_cache_current_position(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        position: usize,
        cache: &GpuDecodeGqaAttentionResidentCache,
    ) -> Result<(), String> {
        if position >= self.max_len {
            return Err(format!(
                "GPU decode GQA attention position {} exceeds max_len {}",
                position, self.max_len
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &q.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU decode GQA attention resident path requires shared GpuContext".to_string(),
            );
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsinfer-decode-gqa-resident-pass"),
            timestamp_writes: None,
        });
        self.encode_resident_in_pass_current_position(
            &mut pass, q, key, value, output, position, cache,
        )?;
        Ok(())
    }

    pub fn write_resident_position_with_cache(
        &self,
        position: usize,
        cache: &GpuDecodeGqaAttentionResidentCache,
    ) {
        self.write_position_buffer(&cache.position_buffer, position);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_cache_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        q: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        position: usize,
        cache: &GpuDecodeGqaAttentionResidentCache,
    ) -> Result<(), String> {
        self.write_position_buffer(&cache.position_buffer, position);
        self.encode_resident_in_pass_current_position(pass, q, key, value, output, position, cache)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_in_pass_current_position(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        q: &GpuResidentBuffer,
        key: &GpuResidentBuffer,
        value: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        position: usize,
        cache: &GpuDecodeGqaAttentionResidentCache,
    ) -> Result<(), String> {
        if position >= self.max_len {
            return Err(format!(
                "GPU decode GQA attention position {} exceeds max_len {}",
                position, self.max_len
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &q.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &key.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &value.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU decode GQA attention resident path requires shared GpuContext".to_string(),
            );
        }
        let pipeline = if position < DECODE_GQA_ATTENTION_PARALLEL_MAX_SEQ_LEN {
            &self.context.inner.decode_gqa_attention_pipeline
        } else {
            &self.context.inner.decode_gqa_attention_serial_pipeline
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &cache.bind_group, &[]);
        pass.dispatch_workgroups(self.num_heads as u32, 1, 1);
        Ok(())
    }
}

impl GpuResidualAdd {
    pub fn new(shape: &[usize]) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::with_context(&context, shape)
    }

    pub fn with_context(context: &GpuContext, shape: &[usize]) -> Result<Self, String> {
        let len = shape.iter().product::<usize>();
        if len == 0 {
            return Err("GPU residual add requires non-empty shape".to_string());
        }
        let device = &context.inner.device;
        let lhs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-residual-add-lhs"),
            size: bytes_len(len),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rhs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-residual-add-rhs"),
            size: bytes_len(len),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-residual-add-output"),
            size: bytes_len(len),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-residual-add-readback"),
            size: bytes_len(len),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = [len as u32, 0_u32, 0_u32, 0_u32];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-residual-add-params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rsinfer-residual-add-bind-group"),
            layout: &context.inner.residual_add_bind_group_layout,
            entries: &[
                buffer_entry(0, &lhs_buffer),
                buffer_entry(1, &rhs_buffer),
                buffer_entry(2, &output_buffer),
                buffer_entry(3, &params_buffer),
            ],
        });

        Ok(Self {
            context: context.clone(),
            lhs_buffer,
            rhs_buffer,
            output_buffer,
            readback_buffer,
            _params_buffer: params_buffer,
            bind_group,
            shape: shape.to_vec(),
            len,
        })
    }

    pub fn forward(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, String> {
        if lhs.shape() != self.shape.as_slice() || rhs.shape() != self.shape.as_slice() {
            return Err(format!(
                "GPU residual add input shape mismatch: lhs={:?}, rhs={:?}, expected {:?}",
                lhs.shape(),
                rhs.shape(),
                self.shape
            ));
        }
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        if lhs_slice.len() != self.len || rhs_slice.len() != self.len {
            return Err(format!(
                "GPU residual add input length mismatch: lhs={}, rhs={}, expected {}",
                lhs_slice.len(),
                rhs_slice.len(),
                self.len
            ));
        }
        self.context
            .inner
            .queue
            .write_buffer(&self.lhs_buffer, 0, bytemuck::cast_slice(lhs_slice));
        self.context
            .inner
            .queue
            .write_buffer(&self.rhs_buffer, 0, bytemuck::cast_slice(rhs_slice));

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-residual-add-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-residual-add-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.context.inner.residual_add_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = (self.len as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback_buffer,
            0,
            bytes_len(self.len),
        );
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let slice = self.readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        record_map_read();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;
        rx.recv()
            .map_err(|e| format!("map callback failed: {e}"))??;

        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        self.readback_buffer.unmap();
        Tensor::from_f32_vec(&self.shape, values).map_err(|e| e.to_string())
    }

    pub fn encode_resident(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        lhs: &GpuResidentBuffer,
        rhs: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<(), String> {
        let cache = self.prepare_resident_cache(lhs, rhs, output)?;
        self.encode_resident_with_cache(encoder, lhs, rhs, output, &cache)
    }

    pub fn prepare_resident_cache(
        &self,
        lhs: &GpuResidentBuffer,
        rhs: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<GpuResidualAddResidentCache, String> {
        if !Arc::ptr_eq(&self.context.inner, &lhs.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &rhs.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU residual add resident path requires shared GpuContext".to_string());
        }
        if lhs.shape != self.shape || rhs.shape != self.shape || output.shape != self.shape {
            return Err(format!(
                "GPU residual add resident shape mismatch: lhs={:?}, rhs={:?}, output={:?}, expected {:?}",
                lhs.shape, rhs.shape, output.shape, self.shape
            ));
        }
        let bind_group = self
            .context
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-residual-add-resident-bind-group"),
                layout: &self.context.inner.residual_add_bind_group_layout,
                entries: &[
                    buffer_entry(0, &lhs.buffer),
                    buffer_entry(1, &rhs.buffer),
                    buffer_entry(2, &output.buffer),
                    buffer_entry(3, &self._params_buffer),
                ],
            });
        Ok(GpuResidualAddResidentCache { bind_group })
    }

    pub fn encode_resident_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        lhs: &GpuResidentBuffer,
        rhs: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        cache: &GpuResidualAddResidentCache,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &lhs.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &rhs.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU residual add resident path requires shared GpuContext".to_string());
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsinfer-residual-add-resident-pass"),
            timestamp_writes: None,
        });
        self.encode_resident_with_cache_in_pass(&mut pass, lhs, rhs, output, cache)?;
        Ok(())
    }

    pub fn encode_resident_with_cache_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        lhs: &GpuResidentBuffer,
        rhs: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        cache: &GpuResidualAddResidentCache,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &lhs.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &rhs.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU residual add resident path requires shared GpuContext".to_string());
        }
        pass.set_pipeline(&self.context.inner.residual_add_pipeline);
        pass.set_bind_group(0, &cache.bind_group, &[]);
        let workgroups = (self.len as u32).div_ceil(WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}

impl GpuResidentBuffer {
    pub fn with_context(context: &GpuContext, shape: &[usize]) -> Result<Self, String> {
        let len = shape.iter().product::<usize>();
        if len == 0 {
            return Err("GPU resident buffer requires non-empty shape".to_string());
        }
        let buffer = context.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-resident-buffer"),
            size: bytes_len(len),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            context: context.clone(),
            buffer,
            shape: shape.to_vec(),
            len,
        })
    }

    pub fn from_tensor(context: &GpuContext, tensor: &Tensor) -> Result<Self, String> {
        let resident = Self::with_context(context, tensor.shape())?;
        resident.upload(tensor)?;
        Ok(resident)
    }

    pub fn shared_context(&self) -> GpuContext {
        self.context.clone()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn upload(&self, tensor: &Tensor) -> Result<(), String> {
        if tensor.shape() != self.shape.as_slice() {
            return Err(format!(
                "GPU resident buffer upload shape mismatch: got {:?}, expected {:?}",
                tensor.shape(),
                self.shape
            ));
        }
        let slice = tensor.as_slice();
        if slice.len() != self.len {
            return Err(format!(
                "GPU resident buffer upload length mismatch: got {}, expected {}",
                slice.len(),
                self.len
            ));
        }
        self.context
            .inner
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(slice));
        Ok(())
    }

    pub fn encode_copy_to(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dest: &GpuResidentBuffer,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &dest.context.inner) {
            return Err("GPU resident buffer copy requires shared GpuContext".to_string());
        }
        if self.shape != dest.shape || self.len != dest.len {
            return Err(format!(
                "GPU resident buffer copy shape mismatch: src={:?}, dst={:?}",
                self.shape, dest.shape
            ));
        }
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dest.buffer, 0, bytes_len(self.len));
        Ok(())
    }

    pub fn read_back(&self) -> Result<Tensor, String> {
        let readback_buffer = self
            .context
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-resident-readback"),
                size: bytes_len(self.len),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-resident-readback-encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &readback_buffer, 0, bytes_len(self.len));
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let slice = readback_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        record_map_read();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.map_err(|e| e.to_string()));
        });
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;
        rx.recv()
            .map_err(|e| format!("map callback failed: {e}"))??;
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        readback_buffer.unmap();
        Tensor::from_f32_vec(&self.shape, values).map_err(|e| e.to_string())
    }
}

impl GpuPositionUniform {
    pub fn with_context(context: &GpuContext, position: usize) -> Result<Self, String> {
        let params = [position as u32, 0_u32, 0_u32, 0_u32];
        let buffer = context
            .inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsinfer-shared-position-uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        Ok(Self {
            context: context.clone(),
            buffer,
        })
    }

    pub fn write_position(&self, position: usize) {
        let params = [position as u32, 0_u32, 0_u32, 0_u32];
        write_position_buffer(&self.context.inner.queue, &self.buffer, 0, &params);
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl GpuQ8MatVec {
    pub fn from_q8_weight(weight: &Q8LinearWeight) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::from_q8_weight_with_context(&context, weight)
    }

    pub fn from_q8_weight_with_context(
        context: &GpuContext,
        weight: &Q8LinearWeight,
    ) -> Result<Self, String> {
        Self::from_q8_weight_with_context_and_argmax(context, weight, true)
    }

    pub fn from_q8_weight_with_context_and_argmax(
        context: &GpuContext,
        weight: &Q8LinearWeight,
        enable_argmax: bool,
    ) -> Result<Self, String> {
        validate_q8_weight(weight)?;

        let input_buffer = context.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-q8-matvec-input"),
            size: bytes_len(weight.in_features),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let words_per_row = weight.in_features.div_ceil(4);
        let qrow_bytes = bytes_len_u32(words_per_row);
        let row_binding_bytes = qrow_bytes.max(bytes_len(1));
        let max_chunk_bytes = context.inner.max_chunk_bytes;
        if row_binding_bytes > max_chunk_bytes {
            return Err(format!(
                "Q8 matvec row is too large for one GPU storage buffer binding: row={row_binding_bytes}, limit={max_chunk_bytes}"
            ));
        }
        let rows_per_chunk = (max_chunk_bytes / row_binding_bytes).max(1) as usize;
        let mut chunks = Vec::new();
        let mut out_offset = 0usize;
        while out_offset < weight.out_features {
            let rows = rows_per_chunk.min(weight.out_features - out_offset);
            chunks.push(create_q8_chunk(
                &context.inner,
                &input_buffer,
                weight,
                words_per_row,
                out_offset,
                rows,
                enable_argmax,
            ));
            out_offset += rows;
        }

        Ok(Self {
            context: context.clone(),
            input_buffer: Some(input_buffer),
            chunks,
            in_features: weight.in_features,
            out_features: weight.out_features,
        })
    }

    pub fn supports_argmax(&self) -> bool {
        self.chunks
            .iter()
            .all(|chunk| chunk.argmax_bind_group.is_some())
    }

    fn matvec_pipeline(&self) -> &wgpu::ComputePipeline {
        if self.in_features.is_multiple_of(4) {
            &self.context.inner.q8_matvec_vec4_pipeline
        } else {
            &self.context.inner.q8_matvec_pipeline
        }
    }

    pub fn shared_context(&self) -> GpuContext {
        self.context.clone()
    }

    pub fn release_standalone_forward_resources(&mut self, keep_readback: bool) {
        self.input_buffer = None;
        for chunk in &mut self.chunks {
            chunk.bind_group = None;
            if !keep_readback {
                chunk.readback_buffer = None;
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let mut outputs = Self::forward_many_same_input(&[self], x)?;
        outputs
            .pop()
            .ok_or_else(|| "GPU Q8 matvec returned no output".to_string())
    }

    pub fn forward_raw(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let mut outputs = Self::forward_many_same_input_raw(&[self], input)?;
        outputs
            .pop()
            .ok_or_else(|| "GPU Q8 matvec returned no raw output".to_string())
    }

    pub fn forward_argmax(&self, x: &Tensor) -> Result<u32, String> {
        if x.ndim() != 2 || x.shape() != [1, self.in_features] {
            return Err(format!(
                "GPU Q8 argmax expects [1, {}], got {:?}",
                self.in_features,
                x.shape()
            ));
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU Q8 argmax input is not contiguous".to_string())?;
        let input_buffer = self
            .input_buffer
            .as_ref()
            .ok_or_else(|| "GPU Q8 argmax input buffer was released".to_string())?;
        self.context
            .inner
            .queue
            .write_buffer(input_buffer, 0, bytemuck::cast_slice(input));

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-q8-argmax-encoder"),
                });
        for chunk in &self.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-q8-argmax-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.context.inner.argmax_pipeline);
                pass.set_bind_group(
                    0,
                    chunk
                        .argmax_bind_group
                        .as_ref()
                        .ok_or_else(|| "GPU Q8 argmax bind group was not attached".to_string())?,
                    &[],
                );
                pass.dispatch_workgroups(chunk.argmax_results as u32, 1, 1);
            }
            let argmax_buffer = chunk
                .argmax_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 argmax output buffer was not attached".to_string())?;
            let argmax_readback = chunk
                .argmax_readback_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 argmax readback buffer was not attached".to_string())?;
            encoder.copy_buffer_to_buffer(
                argmax_buffer,
                0,
                argmax_readback,
                0,
                argmax_bytes_len(chunk.argmax_results),
            );
        }
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let mut receivers = Vec::new();
        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            let slice = chunk
                .argmax_readback_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 argmax readback buffer was not attached".to_string())?
                .slice(..);
            let (tx, rx) = mpsc::channel();
            record_map_read();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send((chunk_idx, result.map_err(|e| e.to_string())));
            });
            receivers.push(rx);
        }
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut best: Option<GpuArgmaxResult> = None;
        for chunk in &self.chunks {
            let argmax_readback = chunk
                .argmax_readback_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 argmax readback buffer was not attached".to_string())?;
            let slice = argmax_readback.slice(..);
            let mapped = slice.get_mapped_range();
            let results = bytemuck::cast_slice::<u8, GpuArgmaxResult>(&mapped);
            for &result in results {
                if best
                    .map(|current| result.value > current.value)
                    .unwrap_or(true)
                {
                    best = Some(result);
                }
            }
            drop(mapped);
            argmax_readback.unmap();
        }
        best.map(|result| result.index)
            .ok_or_else(|| "GPU Q8 argmax has no chunks".to_string())
    }

    pub fn encode_resident(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<(), String> {
        let cache = self.prepare_resident_cache(input, output)?;
        self.encode_resident_with_cache(encoder, input, output, &cache)
    }

    pub fn prepare_resident_cache(
        &self,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<GpuQ8MatVecResidentCache, String> {
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU Q8 matvec resident path requires shared GpuContext".to_string());
        }
        if input.len != self.in_features {
            return Err(format!(
                "GPU Q8 matvec resident input length mismatch: got {}, expected {}",
                input.len, self.in_features
            ));
        }
        if output.len != self.out_features {
            return Err(format!(
                "GPU Q8 matvec resident output length mismatch: got {}, expected {}",
                output.len, self.out_features
            ));
        }

        let mut bind_groups = Vec::with_capacity(self.chunks.len());
        for chunk in &self.chunks {
            bind_groups.push(self.context.inner.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-q8-matvec-resident-bind-group"),
                    layout: &self.context.inner.q8_matvec_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &input.buffer),
                        buffer_entry(1, &chunk._qweight_buffer),
                        buffer_entry(2, &chunk._scales_buffer),
                        buffer_entry(3, &chunk.output_buffer),
                        buffer_entry(4, &chunk._params_buffer),
                    ],
                },
            ));
        }
        Ok(GpuQ8MatVecResidentCache { bind_groups })
    }

    pub fn encode_resident_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        cache: &GpuQ8MatVecResidentCache,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err("GPU Q8 matvec resident path requires shared GpuContext".to_string());
        }
        if input.len != self.in_features {
            return Err(format!(
                "GPU Q8 matvec resident input length mismatch: got {}, expected {}",
                input.len, self.in_features
            ));
        }
        if output.len != self.out_features {
            return Err(format!(
                "GPU Q8 matvec resident output length mismatch: got {}, expected {}",
                output.len, self.out_features
            ));
        }
        if cache.bind_groups.len() != self.chunks.len() {
            return Err("GPU Q8 matvec resident cache chunk layout mismatch".to_string());
        }

        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-q8-matvec-resident-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.matvec_pipeline());
                pass.set_bind_group(0, &cache.bind_groups[chunk_idx], &[]);
                pass.dispatch_workgroups(chunk.out_features as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &chunk.output_buffer,
                0,
                &output.buffer,
                bytes_len(chunk.out_offset),
                bytes_len(chunk.out_features),
            );
        }
        Ok(())
    }

    pub fn forward_many_same_input(
        matvecs: &[&GpuQ8MatVec],
        x: &Tensor,
    ) -> Result<Vec<Tensor>, String> {
        if matvecs.is_empty() {
            return Ok(Vec::new());
        }
        let first = matvecs[0];
        if x.ndim() != 2 || x.shape() != [1, first.in_features] {
            return Err(format!(
                "GPU Q8 matvec expects [1, {}], got {:?}",
                first.in_features,
                x.shape()
            ));
        }
        if matvecs
            .iter()
            .any(|matvec| matvec.in_features != first.in_features)
        {
            return Err("GPU Q8 matvec batch requires the same input width".to_string());
        }
        if matvecs
            .iter()
            .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &first.context.inner))
        {
            return Err("GPU Q8 matvec batch requires a shared GpuContext".to_string());
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU Q8 matvec input is not contiguous".to_string())?;
        let outputs = Self::forward_many_same_input_raw(matvecs, input)?;
        let mut tensors = Vec::with_capacity(matvecs.len());
        for (matvec, output) in matvecs.iter().zip(outputs) {
            tensors.push(
                Tensor::from_f32_vec(&[1, matvec.out_features], output)
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(tensors)
    }

    pub fn forward_many_same_input_raw(
        matvecs: &[&GpuQ8MatVec],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if matvecs.is_empty() {
            return Ok(Vec::new());
        }
        let first = matvecs[0];
        if input.len() != first.in_features {
            return Err(format!(
                "GPU Q8 matvec raw input expects {} values, got {}",
                first.in_features,
                input.len()
            ));
        }
        if matvecs.iter().any(|matvec| matvec.input_buffer.is_none()) {
            return Err("GPU Q8 matvec batch requires standalone input buffers".to_string());
        }
        for matvec in matvecs {
            let input_buffer = matvec
                .input_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 matvec input buffer was released".to_string())?;
            matvec
                .context
                .inner
                .queue
                .write_buffer(input_buffer, 0, bytemuck::cast_slice(input));
        }

        let mut encoder =
            first
                .context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-q8-matvec-encoder"),
                });
        for matvec in matvecs {
            for chunk in &matvec.chunks {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rsinfer-q8-matvec-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(matvec.matvec_pipeline());
                    pass.set_bind_group(
                        0,
                        chunk
                            .bind_group
                            .as_ref()
                            .ok_or_else(|| "GPU Q8 matvec bind group was released".to_string())?,
                        &[],
                    );
                    pass.dispatch_workgroups(chunk.out_features as u32, 1, 1);
                }
                let readback_buffer = chunk
                    .readback_buffer
                    .as_ref()
                    .ok_or_else(|| "GPU Q8 matvec readback buffer was released".to_string())?;
                encoder.copy_buffer_to_buffer(
                    &chunk.output_buffer,
                    0,
                    readback_buffer,
                    0,
                    bytes_len(chunk.out_features),
                );
            }
        }
        first.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let mut receivers = Vec::new();
        for (matvec_idx, matvec) in matvecs.iter().enumerate() {
            for (chunk_idx, chunk) in matvec.chunks.iter().enumerate() {
                let slice = chunk
                    .readback_buffer
                    .as_ref()
                    .ok_or_else(|| "GPU Q8 matvec readback buffer was released".to_string())?
                    .slice(..);
                let (tx, rx) = mpsc::channel();
                record_map_read();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send((matvec_idx, chunk_idx, result.map_err(|e| e.to_string())));
                });
                receivers.push(rx);
            }
        }
        record_poll_wait();
        first
            .context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, _, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut outputs = Vec::with_capacity(matvecs.len());
        for matvec in matvecs {
            let mut output = vec![0f32; matvec.out_features];
            for chunk in &matvec.chunks {
                let readback_buffer = chunk
                    .readback_buffer
                    .as_ref()
                    .ok_or_else(|| "GPU Q8 matvec readback buffer was released".to_string())?;
                let slice = readback_buffer.slice(..);
                let mapped = slice.get_mapped_range();
                let values = bytemuck::cast_slice::<u8, f32>(&mapped);
                output[chunk.out_offset..chunk.out_offset + chunk.out_features]
                    .copy_from_slice(values);
                drop(mapped);
                readback_buffer.unmap();
            }
            outputs.push(output);
        }
        Ok(outputs)
    }
}

impl GpuQ8SameInputBatch {
    pub fn input_buffer(&self) -> &wgpu::Buffer {
        &self.input_buffer
    }

    fn matvec_pipeline(&self) -> &wgpu::ComputePipeline {
        if self.in_features.is_multiple_of(4) {
            &self.context.inner.q8_matvec_vec4_pipeline
        } else {
            &self.context.inner.q8_matvec_pipeline
        }
    }

    pub fn encode_resident_input_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner) {
            return Err(
                "GPU Q8 shared-input resident batch requires the original GpuContext".to_string(),
            );
        }
        if input.len != self.in_features {
            return Err(format!(
                "GPU Q8 shared-input resident batch input length mismatch: got {}, expected {}",
                input.len, self.in_features
            ));
        }
        encoder.copy_buffer_to_buffer(
            &input.buffer,
            0,
            &self.input_buffer,
            0,
            bytes_len(self.in_features),
        );
        Ok(())
    }

    pub fn new(matvecs: &[&GpuQ8MatVec]) -> Result<Self, String> {
        if matvecs.is_empty() {
            return Err("GPU Q8 shared-input batch requires at least one matvec".to_string());
        }
        let first = matvecs[0];
        if matvecs
            .iter()
            .any(|matvec| matvec.in_features != first.in_features)
        {
            return Err("GPU Q8 shared-input batch requires the same input width".to_string());
        }
        if matvecs
            .iter()
            .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &first.context.inner))
        {
            return Err("GPU Q8 shared-input batch requires a shared GpuContext".to_string());
        }

        let input_buffer = first
            .context
            .inner
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-q8-shared-input"),
                size: bytes_len(first.in_features),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let mut bind_groups = Vec::with_capacity(matvecs.len());
        for matvec in matvecs {
            let mut matvec_groups = Vec::with_capacity(matvec.chunks.len());
            for chunk in &matvec.chunks {
                matvec_groups.push(first.context.inner.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("rsinfer-q8-shared-input-bind-group"),
                        layout: &first.context.inner.q8_matvec_bind_group_layout,
                        entries: &[
                            buffer_entry(0, &input_buffer),
                            buffer_entry(1, &chunk._qweight_buffer),
                            buffer_entry(2, &chunk._scales_buffer),
                            buffer_entry(3, &chunk.output_buffer),
                            buffer_entry(4, &chunk._params_buffer),
                        ],
                    },
                ));
            }
            bind_groups.push(matvec_groups);
        }

        Ok(Self {
            context: first.context.clone(),
            input_buffer,
            bind_groups,
            in_features: first.in_features,
            out_features: matvecs.iter().map(|matvec| matvec.out_features).collect(),
        })
    }

    pub fn forward(&self, matvecs: &[&GpuQ8MatVec], x: &Tensor) -> Result<Vec<Tensor>, String> {
        if matvecs.len() != self.bind_groups.len() {
            return Err(format!(
                "GPU Q8 shared-input batch expected {} matvecs, got {}",
                self.bind_groups.len(),
                matvecs.len()
            ));
        }
        if x.ndim() != 2 || x.shape() != [1, self.in_features] {
            return Err(format!(
                "GPU Q8 shared-input batch expects [1, {}], got {:?}",
                self.in_features,
                x.shape()
            ));
        }
        for (idx, matvec) in matvecs.iter().enumerate() {
            if matvec.in_features != self.in_features
                || matvec.out_features != self.out_features[idx]
            {
                return Err("GPU Q8 shared-input batch matvec shape mismatch".to_string());
            }
            if !Arc::ptr_eq(&matvec.context.inner, &self.context.inner) {
                return Err(
                    "GPU Q8 shared-input batch requires the original GpuContext".to_string()
                );
            }
            if matvec.chunks.len() != self.bind_groups[idx].len() {
                return Err("GPU Q8 shared-input batch chunk layout mismatch".to_string());
            }
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU Q8 shared-input batch input is not contiguous".to_string())?;
        let outputs = self.forward_raw(matvecs, input)?;
        let mut tensors = Vec::with_capacity(matvecs.len());
        for (matvec, output) in matvecs.iter().zip(outputs) {
            tensors.push(
                Tensor::from_f32_vec(&[1, matvec.out_features], output)
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(tensors)
    }

    pub fn forward_raw(
        &self,
        matvecs: &[&GpuQ8MatVec],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if matvecs.len() != self.bind_groups.len() {
            return Err(format!(
                "GPU Q8 shared-input batch expected {} matvecs, got {}",
                self.bind_groups.len(),
                matvecs.len()
            ));
        }
        if input.len() != self.in_features {
            return Err(format!(
                "GPU Q8 shared-input batch raw input expects {} values, got {}",
                self.in_features,
                input.len()
            ));
        }
        self.context
            .inner
            .queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(input));

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-q8-shared-input-encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-q8-shared-input-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.matvec_pipeline());
            for (matvec_idx, matvec) in matvecs.iter().enumerate() {
                for (chunk_idx, chunk) in matvec.chunks.iter().enumerate() {
                    pass.set_bind_group(0, &self.bind_groups[matvec_idx][chunk_idx], &[]);
                    pass.dispatch_workgroups(chunk.out_features as u32, 1, 1);
                }
            }
        }
        for matvec in matvecs {
            for chunk in &matvec.chunks {
                let readback_buffer = chunk.readback_buffer.as_ref().ok_or_else(|| {
                    "GPU Q8 shared-input batch readback buffer was released".to_string()
                })?;
                encoder.copy_buffer_to_buffer(
                    &chunk.output_buffer,
                    0,
                    readback_buffer,
                    0,
                    bytes_len(chunk.out_features),
                );
            }
        }
        self.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let mut receivers = Vec::new();
        for (matvec_idx, matvec) in matvecs.iter().enumerate() {
            for (chunk_idx, chunk) in matvec.chunks.iter().enumerate() {
                let slice = chunk
                    .readback_buffer
                    .as_ref()
                    .ok_or_else(|| {
                        "GPU Q8 shared-input batch readback buffer was released".to_string()
                    })?
                    .slice(..);
                let (tx, rx) = mpsc::channel();
                record_map_read();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send((matvec_idx, chunk_idx, result.map_err(|e| e.to_string())));
                });
                receivers.push(rx);
            }
        }
        record_poll_wait();
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, _, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut outputs = Vec::with_capacity(matvecs.len());
        for matvec in matvecs {
            let mut output = vec![0f32; matvec.out_features];
            for chunk in &matvec.chunks {
                let readback_buffer = chunk.readback_buffer.as_ref().ok_or_else(|| {
                    "GPU Q8 shared-input batch readback buffer was released".to_string()
                })?;
                let slice = readback_buffer.slice(..);
                let mapped = slice.get_mapped_range();
                let values = bytemuck::cast_slice::<u8, f32>(&mapped);
                output[chunk.out_offset..chunk.out_offset + chunk.out_features]
                    .copy_from_slice(values);
                drop(mapped);
                readback_buffer.unmap();
            }
            outputs.push(output);
        }
        Ok(outputs)
    }

    pub fn encode_resident(
        &self,
        matvecs: &[&GpuQ8MatVec],
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        outputs: &[&GpuResidentBuffer],
    ) -> Result<(), String> {
        let cache = self.prepare_resident_cache(matvecs, input, outputs)?;
        self.encode_resident_with_cache(matvecs, encoder, input, outputs, &cache)
    }

    pub fn prepare_resident_cache(
        &self,
        matvecs: &[&GpuQ8MatVec],
        input: &GpuResidentBuffer,
        outputs: &[&GpuResidentBuffer],
    ) -> Result<GpuQ8SameInputBatchResidentCache, String> {
        if matvecs.len() != self.bind_groups.len() || outputs.len() != self.bind_groups.len() {
            return Err(format!(
                "GPU Q8 shared-input resident batch expected {} matvecs/outputs, got {}/{}",
                self.bind_groups.len(),
                matvecs.len(),
                outputs.len()
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner) {
            return Err(
                "GPU Q8 shared-input resident batch requires the original GpuContext".to_string(),
            );
        }
        if input.len != self.in_features {
            return Err(format!(
                "GPU Q8 shared-input resident batch input length mismatch: got {}, expected {}",
                input.len, self.in_features
            ));
        }
        for ((idx, matvec), output) in matvecs.iter().enumerate().zip(outputs.iter()) {
            if matvec.in_features != self.in_features
                || matvec.out_features != self.out_features[idx]
            {
                return Err("GPU Q8 shared-input resident batch matvec shape mismatch".to_string());
            }
            if !Arc::ptr_eq(&matvec.context.inner, &self.context.inner)
                || !Arc::ptr_eq(&output.context.inner, &self.context.inner)
            {
                return Err(
                    "GPU Q8 shared-input resident batch requires shared GpuContext".to_string(),
                );
            }
            if matvec.chunks.len() != self.bind_groups[idx].len() {
                return Err("GPU Q8 shared-input resident batch chunk layout mismatch".to_string());
            }
            if output.len != matvec.out_features {
                return Err(format!(
                    "GPU Q8 shared-input resident batch output length mismatch at index {}: got {}, expected {}",
                    idx, output.len, matvec.out_features
                ));
            }
        }

        let mut bind_groups = Vec::with_capacity(matvecs.len());
        let mut params_buffers = Vec::with_capacity(matvecs.len());
        for (matvec, output) in matvecs.iter().zip(outputs.iter()) {
            let mut matvec_groups = Vec::with_capacity(matvec.chunks.len());
            let mut matvec_params = Vec::with_capacity(matvec.chunks.len());
            for chunk in &matvec.chunks {
                let params = [
                    self.in_features as u32,
                    chunk.out_features as u32,
                    chunk.words_per_row as u32,
                    chunk.out_offset as u32,
                ];
                let params_buffer = self.context.inner.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("rsinfer-q8-shared-input-resident-params-chunk"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    },
                );
                matvec_groups.push(self.context.inner.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("rsinfer-q8-shared-input-resident-bind-group"),
                        layout: &self.context.inner.q8_matvec_bind_group_layout,
                        entries: &[
                            buffer_entry(0, &self.input_buffer),
                            buffer_entry(1, &chunk._qweight_buffer),
                            buffer_entry(2, &chunk._scales_buffer),
                            buffer_entry(3, &output.buffer),
                            buffer_entry(4, &params_buffer),
                        ],
                    },
                ));
                matvec_params.push(params_buffer);
            }
            bind_groups.push(matvec_groups);
            params_buffers.push(matvec_params);
        }

        Ok(GpuQ8SameInputBatchResidentCache {
            bind_groups,
            _params_buffers: params_buffers,
        })
    }

    pub fn encode_resident_with_cache(
        &self,
        matvecs: &[&GpuQ8MatVec],
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        outputs: &[&GpuResidentBuffer],
        cache: &GpuQ8SameInputBatchResidentCache,
    ) -> Result<(), String> {
        if matvecs.len() != cache.bind_groups.len() || outputs.len() != cache.bind_groups.len() {
            return Err("GPU Q8 shared-input resident batch cache size mismatch".to_string());
        }
        if matvecs.len() != self.bind_groups.len() || outputs.len() != self.bind_groups.len() {
            return Err(format!(
                "GPU Q8 shared-input resident batch expected {} matvecs/outputs, got {}/{}",
                self.bind_groups.len(),
                matvecs.len(),
                outputs.len()
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner) {
            return Err(
                "GPU Q8 shared-input resident batch requires the original GpuContext".to_string(),
            );
        }
        if input.len != self.in_features {
            return Err(format!(
                "GPU Q8 shared-input resident batch input length mismatch: got {}, expected {}",
                input.len, self.in_features
            ));
        }
        for ((idx, matvec), output) in matvecs.iter().enumerate().zip(outputs.iter()) {
            if matvec.in_features != self.in_features
                || matvec.out_features != self.out_features[idx]
            {
                return Err("GPU Q8 shared-input resident batch matvec shape mismatch".to_string());
            }
            if !Arc::ptr_eq(&matvec.context.inner, &self.context.inner)
                || !Arc::ptr_eq(&output.context.inner, &self.context.inner)
            {
                return Err(
                    "GPU Q8 shared-input resident batch requires shared GpuContext".to_string(),
                );
            }
            if matvec.chunks.len() != cache.bind_groups[idx].len() {
                return Err(
                    "GPU Q8 shared-input resident batch cache chunk layout mismatch".to_string(),
                );
            }
            if output.len != matvec.out_features {
                return Err(format!(
                    "GPU Q8 shared-input resident batch output length mismatch at index {}: got {}, expected {}",
                    idx, output.len, matvec.out_features
                ));
            }
        }

        encoder.copy_buffer_to_buffer(
            &input.buffer,
            0,
            &self.input_buffer,
            0,
            bytes_len(self.in_features),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsinfer-q8-shared-input-resident-pass"),
            timestamp_writes: None,
        });
        self.encode_resident_with_cache_in_pass(matvecs, input, outputs, cache, &mut pass)?;
        Ok(())
    }

    pub fn encode_resident_with_cache_in_pass(
        &self,
        matvecs: &[&GpuQ8MatVec],
        input: &GpuResidentBuffer,
        outputs: &[&GpuResidentBuffer],
        cache: &GpuQ8SameInputBatchResidentCache,
        pass: &mut wgpu::ComputePass<'_>,
    ) -> Result<(), String> {
        if matvecs.len() != cache.bind_groups.len() || outputs.len() != cache.bind_groups.len() {
            return Err("GPU Q8 shared-input resident batch cache size mismatch".to_string());
        }
        if matvecs.len() != self.bind_groups.len() || outputs.len() != self.bind_groups.len() {
            return Err(format!(
                "GPU Q8 shared-input resident batch expected {} matvecs/outputs, got {}/{}",
                self.bind_groups.len(),
                matvecs.len(),
                outputs.len()
            ));
        }
        if !Arc::ptr_eq(&self.context.inner, &input.context.inner) {
            return Err(
                "GPU Q8 shared-input resident batch requires the original GpuContext".to_string(),
            );
        }
        if input.len != self.in_features {
            return Err(format!(
                "GPU Q8 shared-input resident batch input length mismatch: got {}, expected {}",
                input.len, self.in_features
            ));
        }
        for ((idx, matvec), output) in matvecs.iter().enumerate().zip(outputs.iter()) {
            if matvec.in_features != self.in_features
                || matvec.out_features != self.out_features[idx]
            {
                return Err("GPU Q8 shared-input resident batch matvec shape mismatch".to_string());
            }
            if !Arc::ptr_eq(&matvec.context.inner, &self.context.inner)
                || !Arc::ptr_eq(&output.context.inner, &self.context.inner)
            {
                return Err(
                    "GPU Q8 shared-input resident batch requires shared GpuContext".to_string(),
                );
            }
            if matvec.chunks.len() != cache.bind_groups[idx].len() {
                return Err(
                    "GPU Q8 shared-input resident batch cache chunk layout mismatch".to_string(),
                );
            }
            if output.len != matvec.out_features {
                return Err(format!(
                    "GPU Q8 shared-input resident batch output length mismatch at index {}: got {}, expected {}",
                    idx, output.len, matvec.out_features
                ));
            }
        }

        pass.set_pipeline(self.matvec_pipeline());
        for (matvec_idx, matvec) in matvecs.iter().enumerate() {
            for (chunk_idx, chunk) in matvec.chunks.iter().enumerate() {
                pass.set_bind_group(0, &cache.bind_groups[matvec_idx][chunk_idx], &[]);
                pass.dispatch_workgroups(chunk.out_features as u32, 1, 1);
            }
        }
        Ok(())
    }
}

impl GpuSwiGluDown {
    pub fn new(gate: &GpuMatVec, up: &GpuMatVec, down: &GpuMatVec) -> Result<Self, String> {
        validate_swiglu_down_matvecs(gate, up, down)?;

        let device = &gate.context.inner.device;
        let mut bind_groups = Vec::with_capacity(gate.chunks.len());
        let mut params_buffers = Vec::with_capacity(gate.chunks.len());
        for (gate_chunk, up_chunk) in gate.chunks.iter().zip(&up.chunks) {
            let params = [
                gate_chunk.out_offset as u32,
                gate_chunk.out_features as u32,
                0_u32,
                0_u32,
            ];
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsinfer-swiglu-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-swiglu-bind-group"),
                layout: &gate.context.inner.swiglu_bind_group_layout,
                entries: &[
                    buffer_entry(0, &gate_chunk.output_buffer),
                    buffer_entry(1, &up_chunk.output_buffer),
                    buffer_entry(2, &down.input_buffer),
                    buffer_entry(3, &params_buffer),
                ],
            });
            params_buffers.push(params_buffer);
            bind_groups.push(GpuSwiGluBindGroup {
                bind_group,
                out_features: gate_chunk.out_features,
            });
        }

        Ok(Self {
            context: gate.context.clone(),
            bind_groups,
            _params_buffers: params_buffers,
        })
    }

    pub fn forward(
        &self,
        gate: &GpuMatVec,
        up: &GpuMatVec,
        down: &GpuMatVec,
        x: &Tensor,
    ) -> Result<Tensor, String> {
        validate_swiglu_down_inputs(gate, up, down, x)?;
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner) {
            return Err(
                "GPU fused MLP cached resources require the original GpuContext".to_string(),
            );
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU fused MLP input is not contiguous".to_string())?;
        for matvec in [gate, up] {
            write_buffer(&matvec.context.inner.queue, &matvec.input_buffer, 0, input);
        }

        let device = &self.context.inner.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rsinfer-fused-mlp-encoder"),
        });
        for matvec in [gate, up] {
            for chunk in &matvec.chunks {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-fused-mlp-gate-up-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gate.context.inner.matvec_pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        for cached in &self.bind_groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-swiglu-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gate.context.inner.swiglu_pipeline);
            pass.set_bind_group(0, &cached.bind_group, &[]);
            let workgroups = (cached.out_features as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        for chunk in &down.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-fused-mlp-down-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gate.context.inner.matvec_pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &chunk.output_buffer,
                0,
                &chunk.readback_buffer,
                0,
                bytes_len(chunk.out_features),
            );
        }
        gate.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let mut receivers = Vec::new();
        for (chunk_idx, chunk) in down.chunks.iter().enumerate() {
            let slice = chunk.readback_buffer.slice(..);
            let (tx, rx) = mpsc::channel();
            record_map_read();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send((chunk_idx, result.map_err(|e| e.to_string())));
            });
            receivers.push(rx);
        }
        record_poll_wait();
        gate.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut output = vec![0f32; down.out_features];
        for chunk in &down.chunks {
            let slice = chunk.readback_buffer.slice(..);
            let mapped = slice.get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&mapped);
            output[chunk.out_offset..chunk.out_offset + chunk.out_features].copy_from_slice(values);
            drop(mapped);
            chunk.readback_buffer.unmap();
        }

        Tensor::from_f32_vec(&[1, down.out_features], output).map_err(|e| e.to_string())
    }
}

impl GpuQ8SwiGluDown {
    pub fn new(gate: &GpuQ8MatVec, up: &GpuQ8MatVec, down: &GpuQ8MatVec) -> Result<Self, String> {
        validate_q8_swiglu_down_matvecs(gate, up, down)?;

        let device = &gate.context.inner.device;
        let shared_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-q8-swiglu-shared-input"),
            size: bytes_len(gate.in_features),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let down_input_buffer = down
            .input_buffer
            .as_ref()
            .ok_or_else(|| "GPU Q8 fused MLP down input buffer was released".to_string())?;
        let mut bind_groups = Vec::with_capacity(gate.chunks.len());
        let mut gate_bind_groups = Vec::with_capacity(gate.chunks.len());
        let mut up_bind_groups = Vec::with_capacity(up.chunks.len());
        let mut params_buffers = Vec::with_capacity(gate.chunks.len());
        for ((gate_chunk, up_chunk), down_chunk) in
            gate.chunks.iter().zip(&up.chunks).zip(&down.chunks)
        {
            gate_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-q8-swiglu-gate-bind-group"),
                layout: &gate.context.inner.q8_matvec_bind_group_layout,
                entries: &[
                    buffer_entry(0, &shared_input_buffer),
                    buffer_entry(1, &gate_chunk._qweight_buffer),
                    buffer_entry(2, &gate_chunk._scales_buffer),
                    buffer_entry(3, &gate_chunk.output_buffer),
                    buffer_entry(4, &gate_chunk._params_buffer),
                ],
            }));
            up_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-q8-swiglu-up-bind-group"),
                layout: &gate.context.inner.q8_matvec_bind_group_layout,
                entries: &[
                    buffer_entry(0, &shared_input_buffer),
                    buffer_entry(1, &up_chunk._qweight_buffer),
                    buffer_entry(2, &up_chunk._scales_buffer),
                    buffer_entry(3, &up_chunk.output_buffer),
                    buffer_entry(4, &up_chunk._params_buffer),
                ],
            }));
            let params = [
                gate_chunk.out_offset as u32,
                gate_chunk.out_features as u32,
                0_u32,
                0_u32,
            ];
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsinfer-q8-swiglu-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-q8-swiglu-bind-group"),
                layout: &gate.context.inner.swiglu_bind_group_layout,
                entries: &[
                    buffer_entry(0, &gate_chunk.output_buffer),
                    buffer_entry(1, &up_chunk.output_buffer),
                    buffer_entry(2, down_input_buffer),
                    buffer_entry(3, &params_buffer),
                ],
            });
            params_buffers.push(params_buffer);
            bind_groups.push(GpuSwiGluBindGroup {
                bind_group,
                out_features: down_chunk.out_features,
            });
        }

        Ok(Self {
            context: gate.context.clone(),
            shared_input_buffer,
            gate_bind_groups,
            up_bind_groups,
            bind_groups,
            _params_buffers: params_buffers,
        })
    }

    pub fn forward(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        x: &Tensor,
    ) -> Result<Tensor, String> {
        validate_q8_swiglu_down_inputs(gate, up, down, x)?;
        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU Q8 fused MLP input is not contiguous".to_string())?;
        let output = self.forward_raw(gate, up, down, input)?;
        Tensor::from_f32_vec(&[1, down.out_features], output).map_err(|e| e.to_string())
    }

    pub fn forward_raw(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        validate_q8_swiglu_down_matvecs(gate, up, down)?;
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner) {
            return Err(
                "GPU Q8 fused MLP cached resources require the original GpuContext".to_string(),
            );
        }
        if input.len() != gate.in_features {
            return Err(format!(
                "GPU Q8 fused MLP raw input expects {} values, got {}",
                gate.in_features,
                input.len()
            ));
        }
        write_buffer(
            &self.context.inner.queue,
            &self.shared_input_buffer,
            0,
            input,
        );

        let device = &self.context.inner.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rsinfer-q8-fused-mlp-encoder"),
        });
        for bind_groups in [&self.gate_bind_groups, &self.up_bind_groups] {
            for (chunk_idx, bind_group) in bind_groups.iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-q8-fused-mlp-gate-up-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(gate.matvec_pipeline());
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(gate.chunks[chunk_idx].out_features as u32, 1, 1);
            }
        }
        for cached in &self.bind_groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-q8-swiglu-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gate.context.inner.swiglu_pipeline);
            pass.set_bind_group(0, &cached.bind_group, &[]);
            let workgroups = (cached.out_features as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        for chunk in &down.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-q8-fused-mlp-down-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(down.matvec_pipeline());
                pass.set_bind_group(
                    0,
                    chunk.bind_group.as_ref().ok_or_else(|| {
                        "GPU Q8 fused MLP down bind group was released".to_string()
                    })?,
                    &[],
                );
                pass.dispatch_workgroups(chunk.out_features as u32, 1, 1);
            }
            let readback_buffer = chunk
                .readback_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 fused MLP down readback buffer was released".to_string())?;
            encoder.copy_buffer_to_buffer(
                &chunk.output_buffer,
                0,
                readback_buffer,
                0,
                bytes_len(chunk.out_features),
            );
        }
        gate.context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let mut receivers = Vec::new();
        for (chunk_idx, chunk) in down.chunks.iter().enumerate() {
            let slice = chunk
                .readback_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 fused MLP down readback buffer was released".to_string())?
                .slice(..);
            let (tx, rx) = mpsc::channel();
            record_map_read();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send((chunk_idx, result.map_err(|e| e.to_string())));
            });
            receivers.push(rx);
        }
        record_poll_wait();
        gate.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut output = vec![0f32; down.out_features];
        for chunk in &down.chunks {
            let readback_buffer = chunk
                .readback_buffer
                .as_ref()
                .ok_or_else(|| "GPU Q8 fused MLP down readback buffer was released".to_string())?;
            let slice = readback_buffer.slice(..);
            let mapped = slice.get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&mapped);
            output[chunk.out_offset..chunk.out_offset + chunk.out_features].copy_from_slice(values);
            drop(mapped);
            readback_buffer.unmap();
        }

        Ok(output)
    }

    pub fn encode_resident(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<(), String> {
        let hidden = GpuResidentBuffer::with_context(&self.context, &[1, gate.out_features])?;
        self.encode_resident_with_hidden(gate, up, down, encoder, input, &hidden, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_hidden(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        hidden: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<(), String> {
        validate_q8_swiglu_down_matvecs(gate, up, down)?;
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &hidden.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU Q8 fused MLP resident path requires the original shared GpuContext"
                    .to_string(),
            );
        }
        if input.len != gate.in_features {
            return Err(format!(
                "GPU Q8 fused MLP resident input length mismatch: got {}, expected {}",
                input.len, gate.in_features
            ));
        }
        if output.len != down.out_features {
            return Err(format!(
                "GPU Q8 fused MLP resident output length mismatch: got {}, expected {}",
                output.len, down.out_features
            ));
        }
        if hidden.len != gate.out_features {
            return Err(format!(
                "GPU Q8 fused MLP resident hidden length mismatch: got {}, expected {}",
                hidden.len, gate.out_features
            ));
        }

        let cache = self.prepare_resident_cache(gate, up, down, input, hidden, output)?;
        self.encode_resident_with_hidden_cache(gate, up, down, encoder, input, output, &cache)
    }

    pub fn prepare_resident_cache(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        input: &GpuResidentBuffer,
        hidden: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
    ) -> Result<GpuQ8SwiGluDownResidentCache, String> {
        validate_q8_swiglu_down_matvecs(gate, up, down)?;
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &hidden.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU Q8 fused MLP resident cache requires the original shared GpuContext"
                    .to_string(),
            );
        }
        if hidden.len != gate.out_features {
            return Err(format!(
                "GPU Q8 fused MLP resident cache hidden length mismatch: got {}, expected {}",
                hidden.len, gate.out_features
            ));
        }
        if input.len != gate.in_features {
            return Err(format!(
                "GPU Q8 fused MLP resident cache input length mismatch: got {}, expected {}",
                input.len, gate.in_features
            ));
        }
        if output.len != down.out_features {
            return Err(format!(
                "GPU Q8 fused MLP resident cache output length mismatch: got {}, expected {}",
                output.len, down.out_features
            ));
        }

        let mut gate_bind_groups = Vec::with_capacity(gate.chunks.len());
        let mut up_bind_groups = Vec::with_capacity(up.chunks.len());
        let mut swiglu_bind_groups = Vec::with_capacity(gate.chunks.len());
        let mut swiglu_params = Vec::with_capacity(gate.chunks.len());
        let mut down_bind_groups = Vec::with_capacity(down.chunks.len());
        for ((gate_chunk, up_chunk), down_chunk) in
            gate.chunks.iter().zip(&up.chunks).zip(&down.chunks)
        {
            gate_bind_groups.push(self.context.inner.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-q8-resident-cache-gate-bind-group"),
                    layout: &self.context.inner.q8_matvec_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &input.buffer),
                        buffer_entry(1, &gate_chunk._qweight_buffer),
                        buffer_entry(2, &gate_chunk._scales_buffer),
                        buffer_entry(3, &gate_chunk.output_buffer),
                        buffer_entry(4, &gate_chunk._params_buffer),
                    ],
                },
            ));
            up_bind_groups.push(self.context.inner.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-q8-resident-cache-up-bind-group"),
                    layout: &self.context.inner.q8_matvec_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &input.buffer),
                        buffer_entry(1, &up_chunk._qweight_buffer),
                        buffer_entry(2, &up_chunk._scales_buffer),
                        buffer_entry(3, &up_chunk.output_buffer),
                        buffer_entry(4, &up_chunk._params_buffer),
                    ],
                },
            ));
            let params = [
                gate_chunk.out_offset as u32,
                gate_chunk.out_features as u32,
                0_u32,
                0_u32,
            ];
            let params_buffer =
                self.context
                    .inner
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("rsinfer-q8-swiglu-resident-params"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group =
                self.context
                    .inner
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("rsinfer-q8-swiglu-resident-bind-group"),
                        layout: &self.context.inner.swiglu_bind_group_layout,
                        entries: &[
                            buffer_entry(0, &gate_chunk.output_buffer),
                            buffer_entry(1, &up_chunk.output_buffer),
                            buffer_entry(2, &hidden.buffer),
                            buffer_entry(3, &params_buffer),
                        ],
                    });
            swiglu_params.push(params_buffer);
            swiglu_bind_groups.push(GpuSwiGluBindGroup {
                bind_group,
                out_features: down_chunk.out_features,
            });
            down_bind_groups.push(self.context.inner.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("rsinfer-q8-matvec-resident-output-bind-group"),
                    layout: &self.context.inner.q8_matvec_bind_group_layout,
                    entries: &[
                        buffer_entry(0, &hidden.buffer),
                        buffer_entry(1, &down_chunk._qweight_buffer),
                        buffer_entry(2, &down_chunk._scales_buffer),
                        buffer_entry(3, &output.buffer),
                        buffer_entry(4, &down_chunk._params_buffer),
                    ],
                },
            ));
        }

        Ok(GpuQ8SwiGluDownResidentCache {
            gate_bind_groups,
            up_bind_groups,
            swiglu_bind_groups,
            _swiglu_params_buffers: swiglu_params,
            down_bind_groups,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_hidden_cache(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        cache: &GpuQ8SwiGluDownResidentCache,
    ) -> Result<(), String> {
        validate_q8_swiglu_down_matvecs(gate, up, down)?;
        if cache.gate_bind_groups.len() != gate.chunks.len()
            || cache.up_bind_groups.len() != up.chunks.len()
            || cache.swiglu_bind_groups.len() != gate.chunks.len()
            || cache.down_bind_groups.len() != down.chunks.len()
        {
            return Err("GPU Q8 fused MLP resident cache chunk layout mismatch".to_string());
        }
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU Q8 fused MLP resident cache path requires the original shared GpuContext"
                    .to_string(),
            );
        }
        if input.len != gate.in_features {
            return Err(format!(
                "GPU Q8 fused MLP resident cache input length mismatch: got {}, expected {}",
                input.len, gate.in_features
            ));
        }
        if output.len != down.out_features {
            return Err(format!(
                "GPU Q8 fused MLP resident cache output length mismatch: got {}, expected {}",
                output.len, down.out_features
            ));
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsinfer-q8-fused-mlp-resident-pass"),
            timestamp_writes: None,
        });
        self.encode_resident_with_hidden_cache_in_pass(
            gate, up, down, &mut pass, input, output, cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_with_hidden_cache_in_pass(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        pass: &mut wgpu::ComputePass<'_>,
        input: &GpuResidentBuffer,
        output: &GpuResidentBuffer,
        cache: &GpuQ8SwiGluDownResidentCache,
    ) -> Result<(), String> {
        validate_q8_swiglu_down_matvecs(gate, up, down)?;
        if cache.gate_bind_groups.len() != gate.chunks.len()
            || cache.up_bind_groups.len() != up.chunks.len()
            || cache.swiglu_bind_groups.len() != gate.chunks.len()
            || cache.down_bind_groups.len() != down.chunks.len()
        {
            return Err("GPU Q8 fused MLP resident cache chunk layout mismatch".to_string());
        }
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &input.context.inner)
            || !Arc::ptr_eq(&self.context.inner, &output.context.inner)
        {
            return Err(
                "GPU Q8 fused MLP resident cache path requires the original shared GpuContext"
                    .to_string(),
            );
        }
        if input.len != gate.in_features {
            return Err(format!(
                "GPU Q8 fused MLP resident cache input length mismatch: got {}, expected {}",
                input.len, gate.in_features
            ));
        }
        if output.len != down.out_features {
            return Err(format!(
                "GPU Q8 fused MLP resident cache output length mismatch: got {}, expected {}",
                output.len, down.out_features
            ));
        }

        pass.set_pipeline(gate.matvec_pipeline());
        for bind_groups in [&cache.gate_bind_groups, &cache.up_bind_groups] {
            for (chunk_idx, bind_group) in bind_groups.iter().enumerate() {
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(gate.chunks[chunk_idx].out_features as u32, 1, 1);
            }
        }
        pass.set_pipeline(&gate.context.inner.swiglu_pipeline);
        for cached in &cache.swiglu_bind_groups {
            pass.set_bind_group(0, &cached.bind_group, &[]);
            let workgroups = (cached.out_features as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        pass.set_pipeline(down.matvec_pipeline());
        for (chunk_idx, bind_group) in cache.down_bind_groups.iter().enumerate() {
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(down.chunks[chunk_idx].out_features as u32, 1, 1);
        }
        Ok(())
    }
}

fn validate_swiglu_down_matvecs(
    gate: &GpuMatVec,
    up: &GpuMatVec,
    down: &GpuMatVec,
) -> Result<(), String> {
    if gate.in_features != up.in_features {
        return Err("GPU fused MLP requires gate/up input width to match".to_string());
    }
    if gate.out_features != up.out_features || gate.out_features != down.in_features {
        return Err(format!(
            "GPU fused MLP shape mismatch: gate {}, up {}, down input {}",
            gate.out_features, up.out_features, down.in_features
        ));
    }
    if gate.chunks.iter().zip(&up.chunks).count() != gate.chunks.len()
        || gate.chunks.len() != up.chunks.len()
    {
        return Err("GPU fused MLP requires gate/up chunk count to match".to_string());
    }
    for (gate_chunk, up_chunk) in gate.chunks.iter().zip(&up.chunks) {
        if gate_chunk.out_offset != up_chunk.out_offset
            || gate_chunk.out_features != up_chunk.out_features
        {
            return Err("GPU fused MLP requires gate/up chunk layout to match".to_string());
        }
        if gate_chunk.out_offset > u32::MAX as usize || gate_chunk.out_features > u32::MAX as usize
        {
            return Err("GPU fused MLP chunk is too large for u32 shader params".to_string());
        }
    }
    if [&up, &down]
        .iter()
        .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &gate.context.inner))
    {
        return Err("GPU fused MLP requires a shared GpuContext".to_string());
    }
    Ok(())
}

fn validate_swiglu_down_inputs(
    gate: &GpuMatVec,
    up: &GpuMatVec,
    down: &GpuMatVec,
    x: &Tensor,
) -> Result<(), String> {
    validate_swiglu_down_matvecs(gate, up, down)?;
    if x.ndim() != 2 || x.shape() != [1, gate.in_features] {
        return Err(format!(
            "GPU fused MLP expects [1, {}], got {:?}",
            gate.in_features,
            x.shape()
        ));
    }
    Ok(())
}

fn validate_q8_swiglu_down_matvecs(
    gate: &GpuQ8MatVec,
    up: &GpuQ8MatVec,
    down: &GpuQ8MatVec,
) -> Result<(), String> {
    if gate.in_features != up.in_features {
        return Err("GPU Q8 fused MLP requires gate/up input width to match".to_string());
    }
    if gate.out_features != up.out_features || gate.out_features != down.in_features {
        return Err(format!(
            "GPU Q8 fused MLP shape mismatch: gate {}, up {}, down input {}",
            gate.out_features, up.out_features, down.in_features
        ));
    }
    if gate.chunks.iter().zip(&up.chunks).count() != gate.chunks.len()
        || gate.chunks.len() != up.chunks.len()
    {
        return Err("GPU Q8 fused MLP requires gate/up chunk count to match".to_string());
    }
    for (gate_chunk, up_chunk) in gate.chunks.iter().zip(&up.chunks) {
        if gate_chunk.out_offset != up_chunk.out_offset
            || gate_chunk.out_features != up_chunk.out_features
        {
            return Err("GPU Q8 fused MLP requires gate/up chunk layout to match".to_string());
        }
        if gate_chunk.out_offset > u32::MAX as usize || gate_chunk.out_features > u32::MAX as usize
        {
            return Err("GPU Q8 fused MLP chunk is too large for u32 shader params".to_string());
        }
    }
    if [&up, &down]
        .iter()
        .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &gate.context.inner))
    {
        return Err("GPU Q8 fused MLP requires a shared GpuContext".to_string());
    }
    Ok(())
}

fn validate_q8_swiglu_down_inputs(
    gate: &GpuQ8MatVec,
    up: &GpuQ8MatVec,
    down: &GpuQ8MatVec,
    x: &Tensor,
) -> Result<(), String> {
    validate_q8_swiglu_down_matvecs(gate, up, down)?;
    if x.ndim() != 2 || x.shape() != [1, gate.in_features] {
        return Err(format!(
            "GPU Q8 fused MLP expects [1, {}], got {:?}",
            gate.in_features,
            x.shape()
        ));
    }
    Ok(())
}

fn bytes_len(items: usize) -> wgpu::BufferAddress {
    (items * std::mem::size_of::<f32>()) as wgpu::BufferAddress
}

fn bytes_len_u32(items: usize) -> wgpu::BufferAddress {
    (items * std::mem::size_of::<u32>()) as wgpu::BufferAddress
}

fn argmax_bytes_len(items: usize) -> wgpu::BufferAddress {
    (items * std::mem::size_of::<GpuArgmaxResult>()) as wgpu::BufferAddress
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_chunk(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    input_buffer: &wgpu::Buffer,
    weight: &[f16],
    in_features: usize,
    out_offset: usize,
    out_features: usize,
) -> GpuMatVecChunk {
    let start = out_offset * in_features;
    let end = start + out_features * in_features;
    let weight_f32: Vec<f32> = weight[start..end]
        .iter()
        .map(|value| value.to_f32())
        .collect();
    let weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-lm-head-weight-chunk"),
        contents: bytemuck::cast_slice(&weight_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-lm-head-output-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-lm-head-readback-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = [in_features as u32, out_features as u32, 0_u32, 0_u32];
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-lm-head-params-chunk"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rsinfer-lm-head-bind-group-chunk"),
        layout: bind_group_layout,
        entries: &[
            buffer_entry(0, input_buffer),
            buffer_entry(1, &weight_buffer),
            buffer_entry(2, &output_buffer),
            buffer_entry(3, &params_buffer),
        ],
    });
    GpuMatVecChunk {
        bind_group,
        _weight_buffer: weight_buffer,
        output_buffer,
        readback_buffer,
        _params_buffer: params_buffer,
        out_offset,
        out_features,
    }
}

fn create_q8_chunk(
    context: &GpuContextInner,
    input_buffer: &wgpu::Buffer,
    weight: &Q8LinearWeight,
    words_per_row: usize,
    out_offset: usize,
    out_features: usize,
    enable_argmax: bool,
) -> GpuQ8MatVecChunk {
    let device = &context.device;
    let qweight_buffer = if weight.in_features.is_multiple_of(4) {
        let src_start = out_offset * weight.in_features;
        let src_end = src_start + out_features * weight.in_features;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-q8-matvec-weight-chunk"),
            contents: bytemuck::cast_slice(&weight.qweight[src_start..src_end]),
            usage: wgpu::BufferUsages::STORAGE,
        })
    } else {
        let mut packed = vec![0u32; out_features * words_per_row];
        for row in 0..out_features {
            let src_row = out_offset + row;
            let src_start = src_row * weight.in_features;
            let dst_start = row * words_per_row;
            pack_i8_row_to_u32(
                &weight.qweight[src_start..src_start + weight.in_features],
                &mut packed[dst_start..dst_start + words_per_row],
            );
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rsinfer-q8-matvec-weight-chunk"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-q8-matvec-scales-chunk"),
        contents: bytemuck::cast_slice(&weight.scales[out_offset..out_offset + out_features]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-q8-matvec-output-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-q8-matvec-readback-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let argmax_results = if enable_argmax {
        (out_features as u32).div_ceil(WORKGROUP_SIZE) as usize
    } else {
        0
    };
    let params = [
        weight.in_features as u32,
        out_features as u32,
        words_per_row as u32,
        0_u32,
    ];
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-q8-matvec-params-chunk"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rsinfer-q8-matvec-bind-group-chunk"),
        layout: &context.q8_matvec_bind_group_layout,
        entries: &[
            buffer_entry(0, input_buffer),
            buffer_entry(1, &qweight_buffer),
            buffer_entry(2, &scales_buffer),
            buffer_entry(3, &output_buffer),
            buffer_entry(4, &params_buffer),
        ],
    });
    let (argmax_bind_group, argmax_buffer, argmax_readback_buffer, argmax_params_buffer) =
        if argmax_results > 0 {
            let argmax_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-q8-argmax-result-chunk"),
                size: argmax_bytes_len(argmax_results),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let argmax_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rsinfer-q8-argmax-readback-chunk"),
                size: argmax_bytes_len(argmax_results),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let argmax_params = [
                weight.in_features as u32,
                out_features as u32,
                words_per_row as u32,
                out_offset as u32,
            ];
            let argmax_params_buffer =
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rsinfer-q8-argmax-params-chunk"),
                    contents: bytemuck::cast_slice(&argmax_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let argmax_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-q8-argmax-bind-group-chunk"),
                layout: &context.argmax_bind_group_layout,
                entries: &[
                    buffer_entry(0, input_buffer),
                    buffer_entry(1, &qweight_buffer),
                    buffer_entry(2, &scales_buffer),
                    buffer_entry(3, &argmax_buffer),
                    buffer_entry(4, &argmax_params_buffer),
                ],
            });
            (
                Some(argmax_bind_group),
                Some(argmax_buffer),
                Some(argmax_readback_buffer),
                Some(argmax_params_buffer),
            )
        } else {
            (None, None, None, None)
        };

    GpuQ8MatVecChunk {
        bind_group: Some(bind_group),
        argmax_bind_group,
        _qweight_buffer: qweight_buffer,
        _scales_buffer: scales_buffer,
        output_buffer,
        readback_buffer: Some(readback_buffer),
        argmax_buffer,
        argmax_readback_buffer,
        _params_buffer: params_buffer,
        _argmax_params_buffer: argmax_params_buffer,
        argmax_results,
        words_per_row,
        out_offset,
        out_features,
    }
}

fn validate_q8_weight(weight: &Q8LinearWeight) -> Result<(), String> {
    let expected = weight.out_features * weight.in_features;
    if weight.qweight.len() != expected {
        return Err(format!(
            "GPU Q8 matvec qweight length mismatch: got {}, expected {}",
            weight.qweight.len(),
            expected
        ));
    }
    if weight.scales.len() != weight.out_features {
        return Err(format!(
            "GPU Q8 matvec scale length mismatch: got {}, expected {}",
            weight.scales.len(),
            weight.out_features
        ));
    }
    if weight.in_features == 0 || weight.out_features == 0 {
        return Err("GPU Q8 matvec requires non-empty dimensions".to_string());
    }
    Ok(())
}

fn pack_i8_row_to_u32(input: &[i8], output: &mut [u32]) {
    output.fill(0);
    for (idx, &value) in input.iter().enumerate() {
        let word_idx = idx / 4;
        let shift = (idx % 4) * 8;
        output[word_idx] |= ((value as u8) as u32) << shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tensor::{
        linear_forward_f16, linear_forward_q8, rms_norm, rms_norm_per_head_inplace,
        rope_single_token_inplace, scaled_dot_product_attention_gqa_cached_decode_one_raw, silu,
        CachedAttention, Q8LinearWeight,
    };

    #[test]
    fn gpu_rms_norm_matches_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU RMSNorm test skipped: no usable wgpu adapter");
            return;
        };
        let x = Tensor::from_f32_slice(&[2, 4], &[0.25, -0.5, 2.0, 1.5, -1.0, 0.75, 0.5, -0.25])
            .unwrap();
        let weight = Tensor::from_f32_slice(&[4], &[1.0, 0.5, -0.75, 1.25]).unwrap();
        let cpu = rms_norm(&x, &weight, 1e-5).unwrap();
        let gpu = GpuRmsNorm::with_context(&context, &weight, &[2, 4], 1e-5).unwrap();
        let got = gpu.forward(&x).unwrap();
        let max_abs = cpu
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU RMSNorm max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_qk_rms_norm_rope_matches_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU Q/K RMSNorm+RoPE test skipped: no usable wgpu adapter");
            return;
        };
        let q = Tensor::from_f32_slice(
            &[1, 3, 4],
            &[
                0.25, -0.5, 2.0, 1.5, -1.0, 0.75, 0.5, -0.25, 1.25, -1.5, 0.8, 0.1,
            ],
        )
        .unwrap();
        let k = Tensor::from_f32_slice(&[1, 2, 4], &[0.6, -1.4, 1.0, 0.25, -0.5, 1.1, 0.3, -0.7])
            .unwrap();
        let q_weight = Tensor::from_f32_slice(&[4], &[1.0, 0.5, -0.75, 1.25]).unwrap();
        let k_weight = Tensor::from_f32_slice(&[4], &[0.75, -1.0, 1.2, 0.4]).unwrap();
        let inv_freq = [1.0_f32, 0.1_f32];
        let pos = 5usize;

        let mut expected_q = q.as_slice().to_vec();
        let mut expected_k = k.as_slice().to_vec();
        rms_norm_per_head_inplace(&mut expected_q, 3, 4, q_weight.as_slice(), 1e-5).unwrap();
        rms_norm_per_head_inplace(&mut expected_k, 2, 4, k_weight.as_slice(), 1e-5).unwrap();
        rope_single_token_inplace(&mut expected_q, 3, 4, pos, &inv_freq).unwrap();
        rope_single_token_inplace(&mut expected_k, 2, 4, pos, &inv_freq).unwrap();
        let expected_q = Tensor::from_f32_vec(&[1, 3, 4], expected_q).unwrap();
        let expected_k = Tensor::from_f32_vec(&[1, 2, 4], expected_k).unwrap();

        let gpu = GpuQkRmsNormRope::with_context(
            &context,
            GpuQkRmsNormRopeConfig {
                q_weight: &q_weight,
                k_weight: &k_weight,
                q_shape: &[1, 3, 4],
                k_shape: &[1, 2, 4],
                pos,
                inv_freq: &inv_freq,
                q_eps: 1e-5,
                k_eps: 1e-5,
            },
        )
        .unwrap();
        let (got_q, got_k) = gpu.forward(&q, &k).unwrap();

        let q_max_abs = expected_q
            .as_slice()
            .iter()
            .zip(got_q.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let k_max_abs = expected_k
            .as_slice()
            .iter()
            .zip(got_k.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            q_max_abs <= 1e-4,
            "GPU Q RMSNorm+RoPE max abs diff {q_max_abs} exceeded tolerance"
        );
        assert!(
            k_max_abs <= 1e-4,
            "GPU K RMSNorm+RoPE max abs diff {k_max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_kv_append_matches_cpu_layout_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU KV append test skipped: no usable wgpu adapter");
            return;
        };
        let mut gpu = GpuKvAppend::with_context(&context, 2, 2, 8).unwrap();
        gpu.append_decode_one_raw(0, &[1.0, 2.0, 10.0, 20.0], &[3.0, 4.0, 30.0, 40.0])
            .unwrap();
        gpu.append_decode_one_raw(1, &[5.0, 6.0, 50.0, 60.0], &[7.0, 8.0, 70.0, 80.0])
            .unwrap();

        let (key, value) = gpu.read_compact().unwrap();
        assert_eq!(gpu.current_len(), 2);
        assert_eq!(key.shape(), &[2, 2, 2]);
        assert_eq!(value.shape(), &[2, 2, 2]);
        assert_eq!(
            key.as_slice(),
            &[1.0, 2.0, 5.0, 6.0, 10.0, 20.0, 50.0, 60.0]
        );
        assert_eq!(
            value.as_slice(),
            &[3.0, 4.0, 7.0, 8.0, 30.0, 40.0, 70.0, 80.0]
        );
    }

    #[test]
    fn gpu_kv_restore_len_truncates_without_losing_prior_cache_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU KV restore test skipped: no usable wgpu adapter");
            return;
        };
        let mut gpu = GpuKvAppend::with_context(&context, 1, 2, 8).unwrap();
        gpu.append_decode_one_raw(0, &[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        gpu.append_decode_one_raw(1, &[5.0, 6.0], &[7.0, 8.0])
            .unwrap();
        let snapshot = gpu.snapshot_len();
        gpu.append_decode_one_raw(2, &[9.0, 10.0], &[11.0, 12.0])
            .unwrap();
        assert_eq!(gpu.current_len(), 3);

        gpu.restore_len(snapshot).unwrap();
        assert_eq!(gpu.current_len(), 2);
        gpu.append_decode_one_raw(2, &[13.0, 14.0], &[15.0, 16.0])
            .unwrap();

        let (key, value) = gpu.read_compact().unwrap();
        assert_eq!(key.shape(), &[1, 3, 2]);
        assert_eq!(value.shape(), &[1, 3, 2]);
        assert_eq!(key.as_slice(), &[1.0, 2.0, 5.0, 6.0, 13.0, 14.0]);
        assert_eq!(value.as_slice(), &[3.0, 4.0, 7.0, 8.0, 15.0, 16.0]);
    }

    #[test]
    fn gpu_decode_gqa_attention_matches_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU decode GQA attention test skipped: no usable wgpu adapter");
            return;
        };
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 2usize;
        let max_len = 5usize;
        let seq_len_k = 3usize;
        let q = vec![0.2, -0.4, 1.0, 0.5, -0.3, 0.7, 0.9, -0.2];

        let mut key = vec![0.0f32; num_kv_heads * max_len * head_dim];
        let mut value = vec![0.0f32; num_kv_heads * max_len * head_dim];
        key[0..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        key[10..16].copy_from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        value[0..6].copy_from_slice(&[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
        value[10..16].copy_from_slice(&[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let expected = scaled_dot_product_attention_gqa_cached_decode_one_raw(
            &q,
            CachedAttention {
                key: &key,
                value: &value,
                num_kv_heads,
                seq_len_k,
                head_dim,
                max_len,
            },
            num_heads,
            num_heads / num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

        let gpu = GpuDecodeGqaAttention::with_context(
            &context,
            GpuDecodeGqaAttentionConfig {
                num_heads,
                num_kv_heads,
                head_dim,
                max_len,
                scale,
            },
        )
        .unwrap();
        let got = gpu.forward_raw(&q, &key, &value, seq_len_k).unwrap();

        let max_abs = expected
            .iter()
            .zip(&got)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU decode GQA attention max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_decode_gqa_attention_long_context_uses_serial_fallback_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU decode GQA attention long-context test skipped: no usable wgpu adapter");
            return;
        };
        let num_heads = 2usize;
        let num_kv_heads = 1usize;
        let head_dim = 2usize;
        let max_len = DECODE_GQA_ATTENTION_PARALLEL_MAX_SEQ_LEN + 2;
        let seq_len_k = DECODE_GQA_ATTENTION_PARALLEL_MAX_SEQ_LEN + 1;
        let q = vec![0.2, -0.4, 0.7, 0.1];

        let mut key = vec![0.0f32; num_kv_heads * max_len * head_dim];
        let mut value = vec![0.0f32; num_kv_heads * max_len * head_dim];
        for pos in 0..seq_len_k {
            let base = pos * head_dim;
            key[base] = ((pos % 17) as f32 - 8.0) * 0.01;
            key[base + 1] = ((pos % 23) as f32 - 11.0) * 0.01;
            value[base] = ((pos % 29) as f32 - 14.0) * 0.02;
            value[base + 1] = ((pos % 31) as f32 - 15.0) * 0.02;
        }
        let scale = 1.0 / (head_dim as f32).sqrt();

        let expected = scaled_dot_product_attention_gqa_cached_decode_one_raw(
            &q,
            CachedAttention {
                key: &key,
                value: &value,
                num_kv_heads,
                seq_len_k,
                head_dim,
                max_len,
            },
            num_heads,
            num_heads / num_kv_heads,
            head_dim,
            scale,
        )
        .unwrap();

        let gpu = GpuDecodeGqaAttention::with_context(
            &context,
            GpuDecodeGqaAttentionConfig {
                num_heads,
                num_kv_heads,
                head_dim,
                max_len,
                scale,
            },
        )
        .unwrap();
        let got = gpu.forward_raw(&q, &key, &value, seq_len_k).unwrap();

        let max_abs = expected
            .iter()
            .zip(&got)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU decode GQA attention long-context fallback max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_residual_add_matches_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU residual add test skipped: no usable wgpu adapter");
            return;
        };
        let lhs = Tensor::from_f32_slice(&[1, 4], &[0.25, -0.5, 2.0, 1.5]).unwrap();
        let rhs = Tensor::from_f32_slice(&[1, 4], &[1.0, 0.75, -0.5, 0.25]).unwrap();
        let expected = lhs.add(&rhs).unwrap();
        let gpu = GpuResidualAdd::with_context(&context, &[1, 4]).unwrap();
        let got = gpu.forward(&lhs, &rhs).unwrap();
        let max_abs = expected
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-6,
            "GPU residual add max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn resident_rmsnorm_then_residual_add_avoids_mid_readback_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU resident chain test skipped: no usable wgpu adapter");
            return;
        };
        let input = Tensor::from_f32_slice(&[1, 4], &[0.25, -0.5, 2.0, 1.5]).unwrap();
        let residual = Tensor::from_f32_slice(&[1, 4], &[1.0, 0.75, -0.5, 0.25]).unwrap();
        let weight = Tensor::from_f32_slice(&[4], &[1.0, 0.5, -0.75, 1.25]).unwrap();

        let cpu_norm = rms_norm(&input, &weight, 1e-5).unwrap();
        let expected = cpu_norm.add(&residual).unwrap();

        let resident_input = GpuResidentBuffer::from_tensor(&context, &input).unwrap();
        let resident_residual = GpuResidentBuffer::from_tensor(&context, &residual).unwrap();
        let resident_norm = GpuResidentBuffer::with_context(&context, &[1, 4]).unwrap();
        let resident_output = GpuResidentBuffer::with_context(&context, &[1, 4]).unwrap();
        let gpu_norm = GpuRmsNorm::with_context(&context, &weight, &[1, 4], 1e-5).unwrap();
        let gpu_add = GpuResidualAdd::with_context(&context, &[1, 4]).unwrap();

        reset_sync_stats();
        let mut encoder =
            context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-resident-chain-test-encoder"),
                });
        gpu_norm
            .encode_resident(&mut encoder, &resident_input, &resident_norm)
            .unwrap();
        gpu_add
            .encode_resident(
                &mut encoder,
                &resident_norm,
                &resident_residual,
                &resident_output,
            )
            .unwrap();
        context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let got = resident_output.read_back().unwrap();
        let stats = sync_stats();
        assert_eq!(
            stats.submits, 2,
            "expected one compute submit + one final readback submit"
        );
        assert_eq!(stats.poll_waits, 1, "expected only final readback poll");
        assert_eq!(stats.map_reads, 1, "expected only final readback map");

        let max_abs = expected
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU resident chain max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn resident_qk_kv_attention_chain_avoids_mid_readback_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU resident attention chain test skipped: no usable wgpu adapter");
            return;
        };

        let q = Tensor::from_f32_slice(&[1, 2, 2], &[0.25, -0.5, 1.0, 0.75]).unwrap();
        let k = Tensor::from_f32_slice(&[1, 1, 2], &[0.6, -1.4]).unwrap();
        let v = Tensor::from_f32_slice(&[1, 2], &[1.5, -0.25]).unwrap();
        let residual = Tensor::from_f32_slice(&[1, 2, 2], &[0.1, 0.2, -0.3, 0.4]).unwrap();
        let q_weight = Tensor::from_f32_slice(&[2], &[1.0, 0.5]).unwrap();
        let k_weight = Tensor::from_f32_slice(&[2], &[0.75, -1.0]).unwrap();
        let inv_freq = [1.0_f32];
        let pos = 1usize;
        let max_len = 4usize;

        let mut expected_q = q.as_slice().to_vec();
        let mut expected_k = k.as_slice().to_vec();
        rms_norm_per_head_inplace(&mut expected_q, 2, 2, q_weight.as_slice(), 1e-5).unwrap();
        rms_norm_per_head_inplace(&mut expected_k, 1, 2, k_weight.as_slice(), 1e-5).unwrap();
        rope_single_token_inplace(&mut expected_q, 2, 2, pos, &inv_freq).unwrap();
        rope_single_token_inplace(&mut expected_k, 1, 2, pos, &inv_freq).unwrap();

        let mut key_cache = vec![0.0f32; max_len * 2];
        let mut value_cache = vec![0.0f32; max_len * 2];
        key_cache[0..2].copy_from_slice(&[0.2, 0.4]);
        value_cache[0..2].copy_from_slice(&[0.3, 0.9]);
        key_cache[pos * 2..(pos + 1) * 2].copy_from_slice(&expected_k);
        value_cache[pos * 2..(pos + 1) * 2].copy_from_slice(v.as_slice());

        let scale = 1.0 / (2.0f32).sqrt();
        let expected_attention = scaled_dot_product_attention_gqa_cached_decode_one_raw(
            &expected_q,
            CachedAttention {
                key: &key_cache,
                value: &value_cache,
                num_kv_heads: 1,
                seq_len_k: pos + 1,
                head_dim: 2,
                max_len,
            },
            2,
            2,
            2,
            scale,
        )
        .unwrap();
        let expected = Tensor::from_f32_vec(
            &[1, 2, 2],
            expected_attention
                .iter()
                .zip(residual.as_slice())
                .map(|(&a, &b)| a + b)
                .collect(),
        )
        .unwrap();

        let resident_q = GpuResidentBuffer::from_tensor(&context, &q).unwrap();
        let resident_k = GpuResidentBuffer::from_tensor(&context, &k).unwrap();
        let resident_v = GpuResidentBuffer::from_tensor(&context, &v).unwrap();
        let resident_residual = GpuResidentBuffer::from_tensor(&context, &residual).unwrap();
        let resident_q_out = GpuResidentBuffer::with_context(&context, &[1, 2, 2]).unwrap();
        let resident_k_out = GpuResidentBuffer::with_context(&context, &[1, 1, 2]).unwrap();
        let resident_key_cache = GpuResidentBuffer::from_tensor(
            &context,
            &Tensor::from_f32_vec(&[1, max_len, 2], key_cache[..].to_vec()).unwrap(),
        )
        .unwrap();
        let resident_value_cache = GpuResidentBuffer::from_tensor(
            &context,
            &Tensor::from_f32_vec(&[1, max_len, 2], value_cache[..].to_vec()).unwrap(),
        )
        .unwrap();
        let resident_attention = GpuResidentBuffer::with_context(&context, &[1, 2, 2]).unwrap();
        let resident_output = GpuResidentBuffer::with_context(&context, &[1, 2, 2]).unwrap();

        let qk_gpu = GpuQkRmsNormRope::with_context(
            &context,
            GpuQkRmsNormRopeConfig {
                q_weight: &q_weight,
                k_weight: &k_weight,
                q_shape: &[1, 2, 2],
                k_shape: &[1, 1, 2],
                pos,
                inv_freq: &inv_freq,
                q_eps: 1e-5,
                k_eps: 1e-5,
            },
        )
        .unwrap();
        let mut kv_gpu = GpuKvAppend::with_context(&context, 1, 2, max_len).unwrap();
        kv_gpu.restore_len(pos).unwrap();
        let attn_gpu = GpuDecodeGqaAttention::with_context(
            &context,
            GpuDecodeGqaAttentionConfig {
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                max_len,
                scale,
            },
        )
        .unwrap();
        let add_gpu = GpuResidualAdd::with_context(&context, &[1, 2, 2]).unwrap();

        reset_sync_stats();
        let mut encoder =
            context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-resident-attention-chain-test-encoder"),
                });
        qk_gpu
            .encode_resident(
                &mut encoder,
                &resident_q,
                &resident_k,
                &resident_q_out,
                &resident_k_out,
            )
            .unwrap();
        kv_gpu
            .encode_resident(
                &mut encoder,
                pos,
                &resident_k_out,
                &resident_v,
                &resident_key_cache,
                &resident_value_cache,
            )
            .unwrap();
        attn_gpu
            .encode_resident(
                &mut encoder,
                &resident_q_out,
                &resident_key_cache,
                &resident_value_cache,
                &resident_attention,
                pos,
            )
            .unwrap();
        add_gpu
            .encode_resident(
                &mut encoder,
                &resident_attention,
                &resident_residual,
                &resident_output,
            )
            .unwrap();
        context.inner.queue.submit(Some(encoder.finish()));
        record_submit();

        let got = resident_output.read_back().unwrap();
        let stats = sync_stats();
        assert_eq!(
            stats.submits, 2,
            "expected one compute submit + one final readback submit"
        );
        assert_eq!(stats.poll_waits, 1, "expected only final readback poll");
        assert_eq!(stats.map_reads, 1, "expected only final readback map");

        let max_abs = expected
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU resident attention chain max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn resident_q8_matvec_then_residual_add_avoids_mid_readback_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("GPU resident Q8 matvec chain test skipped: no usable wgpu adapter");
            return;
        };

        let weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let q8 = Q8LinearWeight::from_f16(&weight, 3, 2).unwrap();
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let residual = Tensor::from_f32_slice(&[1, 3], &[0.1, 0.2, -0.3]).unwrap();
        let expected = linear_forward_q8(&x, &q8).unwrap().add(&residual).unwrap();

        let resident_input = GpuResidentBuffer::from_tensor(&context, &x).unwrap();
        let resident_residual = GpuResidentBuffer::from_tensor(&context, &residual).unwrap();
        let resident_matvec = GpuResidentBuffer::with_context(&context, &[1, 3]).unwrap();
        let resident_output = GpuResidentBuffer::with_context(&context, &[1, 3]).unwrap();
        let gpu_matvec = GpuQ8MatVec::from_q8_weight_with_context(&context, &q8).unwrap();
        let gpu_add = GpuResidualAdd::with_context(&context, &[1, 3]).unwrap();

        reset_sync_stats();
        let mut encoder = context.create_command_encoder("rsinfer-resident-q8-matvec-test-encoder");
        gpu_matvec
            .encode_resident(&mut encoder, &resident_input, &resident_matvec)
            .unwrap();
        gpu_add
            .encode_resident(
                &mut encoder,
                &resident_matvec,
                &resident_residual,
                &resident_output,
            )
            .unwrap();
        context.submit(encoder);

        let got = resident_output.read_back().unwrap();
        let stats = sync_stats();
        assert_eq!(
            stats.submits, 2,
            "expected one compute submit + one final readback submit"
        );
        assert_eq!(stats.poll_waits, 1, "expected only final readback poll");
        assert_eq!(stats.map_reads, 1, "expected only final readback map");

        let max_abs = expected
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU resident Q8 matvec chain max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_matvec_matches_cpu_for_small_tensor_when_available() {
        let _guard = gpu_test_guard();
        let weight = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
        ];
        let x = Tensor::from_f32_slice(&[1, 3], &[0.25, -0.5, 2.0]).unwrap();
        let cpu = linear_forward_f16(&x, &weight, 2, 3).unwrap();
        let Ok(gpu) = GpuMatVec::from_f16_weight(&weight, 2, 3) else {
            eprintln!("GPU matvec test skipped: no usable wgpu adapter");
            return;
        };
        let got = gpu.forward(&x).unwrap();
        let max_abs = cpu
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU matvec max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_q8_matvec_matches_cpu_q8_when_available() {
        let _guard = gpu_test_guard();
        let weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
            f16::from_f32(1.25),
            f16::from_f32(-0.125),
            f16::from_f32(0.875),
            f16::from_f32(-1.5),
        ];
        let q8 = Q8LinearWeight::from_f16(&weight, 2, 5).unwrap();
        let x = Tensor::from_f32_slice(&[1, 5], &[0.6, -1.4, 1.0, 0.25, -0.5]).unwrap();
        let cpu = linear_forward_q8(&x, &q8).unwrap();
        let Ok(gpu) = GpuQ8MatVec::from_q8_weight(&q8) else {
            eprintln!("GPU Q8 matvec test skipped: no usable wgpu adapter");
            return;
        };
        let got = gpu.forward(&x).unwrap();
        let max_abs = cpu
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU Q8 matvec max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_q8_argmax_matches_cpu_q8_when_available() {
        let _guard = gpu_test_guard();
        let weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
            f16::from_f32(1.25),
            f16::from_f32(-0.125),
            f16::from_f32(0.875),
            f16::from_f32(-1.5),
            f16::from_f32(2.0),
            f16::from_f32(0.25),
            f16::from_f32(-0.5),
            f16::from_f32(1.0),
            f16::from_f32(0.75),
        ];
        let q8 = Q8LinearWeight::from_f16(&weight, 3, 5).unwrap();
        let x = Tensor::from_f32_slice(&[1, 5], &[0.6, -1.4, 1.0, 0.25, -0.5]).unwrap();
        let cpu = linear_forward_q8(&x, &q8).unwrap();
        let expected = cpu
            .as_slice()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap();
        let Ok(gpu) = GpuQ8MatVec::from_q8_weight(&q8) else {
            eprintln!("GPU Q8 argmax test skipped: no usable wgpu adapter");
            return;
        };

        assert_eq!(gpu.forward_argmax(&x).unwrap(), expected);
    }

    #[test]
    fn q8_batched_same_input_matvecs_match_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("batched GPU Q8 matvec test skipped: no usable wgpu adapter");
            return;
        };
        let weight_a = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let weight_b = [
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(-0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.75, -1.25]).unwrap();
        let q8_a = Q8LinearWeight::from_f16(&weight_a, 2, 2).unwrap();
        let q8_b = Q8LinearWeight::from_f16(&weight_b, 2, 2).unwrap();
        let cpu_a = linear_forward_q8(&x, &q8_a).unwrap();
        let cpu_b = linear_forward_q8(&x, &q8_b).unwrap();
        let gpu_a = GpuQ8MatVec::from_q8_weight_with_context(&context, &q8_a).unwrap();
        let gpu_b = GpuQ8MatVec::from_q8_weight_with_context(&context, &q8_b).unwrap();
        let outputs = GpuQ8MatVec::forward_many_same_input(&[&gpu_a, &gpu_b], &x).unwrap();

        assert_eq!(outputs.len(), 2);
        for (expected, actual) in cpu_a.as_slice().iter().zip(outputs[0].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
        for (expected, actual) in cpu_b.as_slice().iter().zip(outputs[1].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn q8_shared_input_batch_matches_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("shared-input GPU Q8 batch test skipped: no usable wgpu adapter");
            return;
        };
        let weight_a = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let weight_b = [
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(-0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.75, -1.25]).unwrap();
        let q8_a = Q8LinearWeight::from_f16(&weight_a, 2, 2).unwrap();
        let q8_b = Q8LinearWeight::from_f16(&weight_b, 2, 2).unwrap();
        let cpu_a = linear_forward_q8(&x, &q8_a).unwrap();
        let cpu_b = linear_forward_q8(&x, &q8_b).unwrap();
        let gpu_a = GpuQ8MatVec::from_q8_weight_with_context(&context, &q8_a).unwrap();
        let gpu_b = GpuQ8MatVec::from_q8_weight_with_context(&context, &q8_b).unwrap();
        let batch = GpuQ8SameInputBatch::new(&[&gpu_a, &gpu_b]).unwrap();
        let outputs = batch.forward(&[&gpu_a, &gpu_b], &x).unwrap();

        assert_eq!(outputs.len(), 2);
        for (expected, actual) in cpu_a.as_slice().iter().zip(outputs[0].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
        for (expected, actual) in cpu_b.as_slice().iter().zip(outputs[1].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn q8_shared_input_batch_survives_released_standalone_resources_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("shared-input GPU Q8 release test skipped: no usable wgpu adapter");
            return;
        };
        let weight_a = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let weight_b = [
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(-0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.75, -1.25]).unwrap();
        let q8_a = Q8LinearWeight::from_f16(&weight_a, 2, 2).unwrap();
        let q8_b = Q8LinearWeight::from_f16(&weight_b, 2, 2).unwrap();
        let cpu_a = linear_forward_q8(&x, &q8_a).unwrap();
        let cpu_b = linear_forward_q8(&x, &q8_b).unwrap();
        let mut gpu_a =
            GpuQ8MatVec::from_q8_weight_with_context_and_argmax(&context, &q8_a, false).unwrap();
        let mut gpu_b =
            GpuQ8MatVec::from_q8_weight_with_context_and_argmax(&context, &q8_b, false).unwrap();
        let batch = GpuQ8SameInputBatch::new(&[&gpu_a, &gpu_b]).unwrap();
        gpu_a.release_standalone_forward_resources(true);
        gpu_b.release_standalone_forward_resources(true);

        assert!(GpuQ8MatVec::forward_many_same_input(&[&gpu_a], &x).is_err());
        let outputs = batch.forward(&[&gpu_a, &gpu_b], &x).unwrap();

        assert_eq!(outputs.len(), 2);
        for (expected, actual) in cpu_a.as_slice().iter().zip(outputs[0].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
        for (expected, actual) in cpu_b.as_slice().iter().zip(outputs[1].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }
    #[test]
    fn shared_context_can_drive_multiple_matvecs_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("shared GPU context test skipped: no usable wgpu adapter");
            return;
        };
        let weight_a = [
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
        ];
        let weight_b = [
            f16::from_f32(-0.25),
            f16::from_f32(2.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[2.0, -1.0]).unwrap();
        let cpu_a = linear_forward_f16(&x, &weight_a, 2, 2).unwrap();
        let cpu_b = linear_forward_f16(&x, &weight_b, 2, 2).unwrap();
        let gpu_a = GpuMatVec::from_f16_weight_with_context(&context, &weight_a, 2, 2)
            .unwrap()
            .forward(&x)
            .unwrap();
        let gpu_b = GpuMatVec::from_f16_weight_with_context(&context, &weight_b, 2, 2)
            .unwrap()
            .forward(&x)
            .unwrap();

        for (expected, actual) in cpu_a.as_slice().iter().zip(gpu_a.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
        for (expected, actual) in cpu_b.as_slice().iter().zip(gpu_b.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn batched_same_input_matvecs_match_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("batched GPU matvec test skipped: no usable wgpu adapter");
            return;
        };
        let weight_a = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let weight_b = [
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(-0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.75, -1.25]).unwrap();
        let cpu_a = linear_forward_f16(&x, &weight_a, 2, 2).unwrap();
        let cpu_b = linear_forward_f16(&x, &weight_b, 2, 2).unwrap();
        let gpu_a = GpuMatVec::from_f16_weight_with_context(&context, &weight_a, 2, 2).unwrap();
        let gpu_b = GpuMatVec::from_f16_weight_with_context(&context, &weight_b, 2, 2).unwrap();
        let outputs = GpuMatVec::forward_many_same_input(&[&gpu_a, &gpu_b], &x).unwrap();

        assert_eq!(outputs.len(), 2);
        for (expected, actual) in cpu_a.as_slice().iter().zip(outputs[0].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
        for (expected, actual) in cpu_b.as_slice().iter().zip(outputs[1].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn fused_swiglu_down_matches_cpu_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("fused GPU MLP test skipped: no usable wgpu adapter");
            return;
        };
        let gate_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let up_weight = [
            f16::from_f32(1.0),
            f16::from_f32(0.25),
            f16::from_f32(-0.5),
            f16::from_f32(0.75),
            f16::from_f32(1.25),
            f16::from_f32(-1.0),
        ];
        let down_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-0.25),
            f16::from_f32(1.0),
            f16::from_f32(-1.5),
            f16::from_f32(0.75),
            f16::from_f32(0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let gate = linear_forward_f16(&x, &gate_weight, 3, 2).unwrap();
        let up = linear_forward_f16(&x, &up_weight, 3, 2).unwrap();
        let hidden = silu(&gate).mul(&up).unwrap();
        let cpu = linear_forward_f16(&hidden, &down_weight, 2, 3).unwrap();

        let gate_gpu =
            GpuMatVec::from_f16_weight_with_context(&context, &gate_weight, 3, 2).unwrap();
        let up_gpu = GpuMatVec::from_f16_weight_with_context(&context, &up_weight, 3, 2).unwrap();
        let down_gpu =
            GpuMatVec::from_f16_weight_with_context(&context, &down_weight, 2, 3).unwrap();
        let fused = GpuSwiGluDown::new(&gate_gpu, &up_gpu, &down_gpu).unwrap();
        let got = fused.forward(&gate_gpu, &up_gpu, &down_gpu, &x).unwrap();

        for (expected, actual) in cpu.as_slice().iter().zip(got.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn q8_fused_swiglu_down_matches_cpu_q8_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("fused GPU Q8 MLP test skipped: no usable wgpu adapter");
            return;
        };
        let gate_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let up_weight = [
            f16::from_f32(1.0),
            f16::from_f32(0.25),
            f16::from_f32(-0.5),
            f16::from_f32(0.75),
            f16::from_f32(1.25),
            f16::from_f32(-1.0),
        ];
        let down_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-0.25),
            f16::from_f32(1.0),
            f16::from_f32(-1.5),
            f16::from_f32(0.75),
            f16::from_f32(0.25),
        ];
        let gate_q8 = Q8LinearWeight::from_f16(&gate_weight, 3, 2).unwrap();
        let up_q8 = Q8LinearWeight::from_f16(&up_weight, 3, 2).unwrap();
        let down_q8 = Q8LinearWeight::from_f16(&down_weight, 2, 3).unwrap();
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let gate = linear_forward_q8(&x, &gate_q8).unwrap();
        let up = linear_forward_q8(&x, &up_q8).unwrap();
        let hidden = silu(&gate).mul(&up).unwrap();
        let cpu = linear_forward_q8(&hidden, &down_q8).unwrap();

        let gate_gpu = GpuQ8MatVec::from_q8_weight_with_context(&context, &gate_q8).unwrap();
        let up_gpu = GpuQ8MatVec::from_q8_weight_with_context(&context, &up_q8).unwrap();
        let down_gpu = GpuQ8MatVec::from_q8_weight_with_context(&context, &down_q8).unwrap();
        let fused = GpuQ8SwiGluDown::new(&gate_gpu, &up_gpu, &down_gpu).unwrap();
        let got = fused.forward(&gate_gpu, &up_gpu, &down_gpu, &x).unwrap();

        for (expected, actual) in cpu.as_slice().iter().zip(got.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn q8_fused_swiglu_down_survives_released_gate_up_resources_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("fused GPU Q8 release test skipped: no usable wgpu adapter");
            return;
        };
        let gate_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let up_weight = [
            f16::from_f32(1.0),
            f16::from_f32(0.25),
            f16::from_f32(-0.5),
            f16::from_f32(0.75),
            f16::from_f32(1.25),
            f16::from_f32(-1.0),
        ];
        let down_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-0.25),
            f16::from_f32(1.0),
            f16::from_f32(-1.5),
            f16::from_f32(0.75),
            f16::from_f32(0.25),
        ];
        let gate_q8 = Q8LinearWeight::from_f16(&gate_weight, 3, 2).unwrap();
        let up_q8 = Q8LinearWeight::from_f16(&up_weight, 3, 2).unwrap();
        let down_q8 = Q8LinearWeight::from_f16(&down_weight, 2, 3).unwrap();
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let gate = linear_forward_q8(&x, &gate_q8).unwrap();
        let up = linear_forward_q8(&x, &up_q8).unwrap();
        let hidden = silu(&gate).mul(&up).unwrap();
        let cpu = linear_forward_q8(&hidden, &down_q8).unwrap();

        let mut gate_gpu =
            GpuQ8MatVec::from_q8_weight_with_context_and_argmax(&context, &gate_q8, false).unwrap();
        let mut up_gpu =
            GpuQ8MatVec::from_q8_weight_with_context_and_argmax(&context, &up_q8, false).unwrap();
        let down_gpu =
            GpuQ8MatVec::from_q8_weight_with_context_and_argmax(&context, &down_q8, false).unwrap();
        let fused = GpuQ8SwiGluDown::new(&gate_gpu, &up_gpu, &down_gpu).unwrap();
        gate_gpu.release_standalone_forward_resources(false);
        up_gpu.release_standalone_forward_resources(false);

        assert!(gate_gpu.forward(&x).is_err());
        let got = fused.forward(&gate_gpu, &up_gpu, &down_gpu, &x).unwrap();

        for (expected, actual) in cpu.as_slice().iter().zip(got.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }
}
