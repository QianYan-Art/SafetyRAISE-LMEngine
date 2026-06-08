//! 数学算子实现
//!
//! 包含 softmax、rms_norm、silu、rope 等 LLM 推理所需的核心算子。

use half::f16;
use half::slice::HalfFloatSliceExt;
use ndarray::{ArrayD, Axis, IxDyn};
use rayon::prelude::*;

use super::Tensor;
use crate::error::{Result, RsinferError};

/// 行级对称 Q8 线性权重。
///
/// 每个输出行独立 scale：`w ~= q * scale`，q 存 i8 行主序 `[out_features, in_features]`。
#[derive(Clone, Debug)]
pub struct Q8LinearWeight {
    pub qweight: Vec<i8>,
    pub scales: Vec<f32>,
    pub out_features: usize,
    pub in_features: usize,
}

impl Q8LinearWeight {
    pub fn from_f16(weight: &[f16], out_features: usize, in_features: usize) -> Result<Self> {
        let expected = out_features * in_features;
        if weight.len() != expected {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![weight.len()],
            });
        }

        let mut qweight = vec![0i8; expected];
        let mut scales = vec![0f32; out_features];
        for (row, scale_slot) in scales.iter_mut().enumerate() {
            let start = row * in_features;
            let end = start + in_features;
            let max_abs = weight[start..end]
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            *scale_slot = scale;
            for (dst, value) in qweight[start..end].iter_mut().zip(&weight[start..end]) {
                let quantized = (value.to_f32() / scale).round().clamp(-127.0, 127.0);
                *dst = quantized as i8;
            }
        }

        Ok(Self {
            qweight,
            scales,
            out_features,
            in_features,
        })
    }
}

/// 线性层矩阵乘法：out = x @ W^T，权重以 **f16** 存储。
///
/// HuggingFace 的线性层权重存为 `[out_features, in_features]`，计算 `y = x @ W^T`。
/// 这里：
/// - 权重保持 f16，**省一半内存带宽**（解码阶段是带宽瓶颈）；
/// - 不显式转置 W，直接按行点积（W 每行连续，缓存友好）；
/// - 用 rayon 在输出特征维并行，每个任务复用一块转换缓冲，
///   借 `convert_to_f32_slice`（F16C 指令）把 f16 行批量转 f32 再算点积。
///
/// - `x`: `[m, k]`（m 个 token，每个 k 维，f32）
/// - `w`: f16 行主序 `[n, k]`
/// - 返回: `[m, n]`（f32）
pub fn linear_forward_f16(
    x: &Tensor,
    w: &[f16],
    out_features: usize,
    in_features: usize,
) -> Result<Tensor> {
    if x.ndim() != 2 {
        return Err(RsinferError::DimensionError(format!(
            "linear_forward 需要 2D 输入, got {}D",
            x.ndim()
        )));
    }
    let (m, k) = (x.shape()[0], x.shape()[1]);
    if k != in_features {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![m, in_features],
            actual: vec![m, k],
        });
    }
    let n = out_features;

    // 确保输入连续
    let x_std = x.data.as_standard_layout();
    let xs = x_std
        .as_slice()
        .ok_or_else(|| RsinferError::DimensionError("x 非连续内存".into()))?;

    // out_t 用 [n, m] 布局：每个输出特征 m 个连续值，便于按特征并行
    let mut out_t = vec![0f32; n * m];
    const JCHUNK: usize = 16; // 每个并行任务一次处理的输出特征数（复用同一转换缓冲）
    out_t
        .par_chunks_mut(m * JCHUNK)
        .enumerate()
        .for_each(|(c, block)| {
            let mut wf = vec![0f32; k]; // 复用缓冲，避免逐特征 alloc
            let jbase = c * JCHUNK;
            let feats = block.len() / m;
            for jj in 0..feats {
                let j = jbase + jj;
                w[j * k..(j + 1) * k].convert_to_f32_slice(&mut wf);
                let dst = &mut block[jj * m..(jj + 1) * m];
                for i in 0..m {
                    dst[i] = dot(&xs[i * k..(i + 1) * k], &wf);
                }
            }
        });

    // 转回 [m, n]
    let mut out = vec![0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            out[i * n + j] = out_t[j * m + i];
        }
    }
    Tensor::from_f32_slice(&[m, n], &out)
}

/// 线性层矩阵乘法：out = x @ W^T，权重以行级 Q8 存储。
pub fn linear_forward_q8(x: &Tensor, weight: &Q8LinearWeight) -> Result<Tensor> {
    if x.ndim() != 2 {
        return Err(RsinferError::DimensionError(format!(
            "linear_forward_q8 需要 2D 输入, got {}D",
            x.ndim()
        )));
    }
    let (m, k) = (x.shape()[0], x.shape()[1]);
    if k != weight.in_features {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![m, weight.in_features],
            actual: vec![m, k],
        });
    }

    let x_std = x.data.as_standard_layout();
    let xs = x_std
        .as_slice()
        .ok_or_else(|| RsinferError::DimensionError("x 非连续内存".into()))?;

    let n = weight.out_features;
    if m == 1 {
        let input = &xs[..k];
        let mut out = vec![0f32; n];
        const JCHUNK: usize = 16;
        out.par_chunks_mut(JCHUNK)
            .enumerate()
            .for_each(|(c, block)| {
                let jbase = c * JCHUNK;
                for (jj, dst) in block.iter_mut().enumerate() {
                    let j = jbase + jj;
                    let w_row = &weight.qweight[j * k..(j + 1) * k];
                    *dst = dot_q8(input, w_row, weight.scales[j]);
                }
            });
        return Tensor::from_f32_vec(&[1, n], out);
    }

    let mut out_t = vec![0f32; n * m];
    const JCHUNK: usize = 16;
    out_t
        .par_chunks_mut(m * JCHUNK)
        .enumerate()
        .for_each(|(c, block)| {
            let jbase = c * JCHUNK;
            let feats = block.len() / m;
            for jj in 0..feats {
                let j = jbase + jj;
                let w_row = &weight.qweight[j * k..(j + 1) * k];
                let scale = weight.scales[j];
                let dst = &mut block[jj * m..(jj + 1) * m];
                for i in 0..m {
                    dst[i] = dot_q8(&xs[i * k..(i + 1) * k], w_row, scale);
                }
            }
        });

    let mut out = vec![0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            out[i * n + j] = out_t[j * m + i];
        }
    }
    Tensor::from_f32_slice(&[m, n], &out)
}

/// 两个等长切片的点积。
///
/// 用 8 路独立累加器打断 f32 求和的串行依赖，配合 `target-cpu=native`
/// 让编译器自动生成 AVX/FMA 向量指令，比朴素 `.sum()` 快数倍。
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (xs, ys) in ca.by_ref().zip(cb.by_ref()) {
        for ((acc, &x), &y) in acc.iter_mut().zip(xs).zip(ys) {
            *acc += x * y;
        }
    }
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(&x, &y)| x * y)
        .sum();
    acc.iter().sum::<f32>() + tail
}

#[inline]
fn dot_q8(x: &[f32], q: &[i8], scale: f32) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0f32; LANES];
    let mut cx = x.chunks_exact(LANES);
    let mut cq = q.chunks_exact(LANES);
    for (xs, qs) in cx.by_ref().zip(cq.by_ref()) {
        for ((acc, &x), &q) in acc.iter_mut().zip(xs).zip(qs) {
            *acc += x * q as f32;
        }
    }
    let tail: f32 = cx
        .remainder()
        .iter()
        .zip(cq.remainder())
        .map(|(&x, &q)| x * q as f32)
        .sum();
    (acc.iter().sum::<f32>() + tail) * scale
}

/// Softmax 操作
///
/// 对指定维度应用 softmax: exp(x) / sum(exp(x))
pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    if dim >= x.ndim() {
        return Err(RsinferError::DimensionError(format!(
            "softmax dim {} out of range for {}D tensor",
            dim,
            x.ndim()
        )));
    }

    let data = &x.data;

    // 数值稳定性：减去最大值
    let max_vals = data.map_axis(Axis(dim), |lane| {
        lane.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    });

    // 广播减法
    let shifted = data - &max_vals.insert_axis(Axis(dim));

    // 计算 exp
    let exp_vals = shifted.mapv(f32::exp);

    // 计算 sum
    let sum_vals = exp_vals.sum_axis(Axis(dim));

    // 归一化
    let result = &exp_vals / &sum_vals.insert_axis(Axis(dim));

    Ok(Tensor { data: result })
}

/// RMS Normalization
///
/// x_normalized = x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let data = &x.data;
    let last_dim = x.ndim() - 1;

    // 计算 x^2
    let x_sq = data.mapv(|v| v * v);

    // 计算 mean(x^2)
    let mean_sq = x_sq
        .mean_axis(Axis(last_dim))
        .ok_or_else(|| RsinferError::DimensionError("Failed to compute mean".into()))?;

    // 计算 rsqrt(mean + eps)
    let rsqrt = mean_sq.mapv(|v| 1.0 / (v + eps).sqrt());

    // 广播乘法归一化
    let normalized = data * &rsqrt.insert_axis(Axis(last_dim));

    // 应用权重
    let result = &normalized * &weight.data;

    Ok(Tensor { data: result })
}

/// SiLU 激活函数 (Swish)
///
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(x: &Tensor) -> Tensor {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let result = x.data.mapv(|v| v / (1.0 + (-v).exp()));
    Tensor { data: result }
}

/// 旋转位置编码 (Rotary Position Embedding)
///
/// 对 Q 和 K 应用旋转位置编码。
/// q, k: [seq_len, num_heads, head_dim]
/// pos: 起始位置
/// theta: 旋转频率基数 (通常为 10000.0)
pub fn rope(q: &Tensor, k: &Tensor, pos: usize, theta: f32) -> Result<(Tensor, Tensor)> {
    let q_shape = q.shape();

    if q.ndim() != 3 || k.ndim() != 3 {
        return Err(RsinferError::DimensionError(
            "RoPE requires 3D tensors [seq_len, num_heads, head_dim]".into(),
        ));
    }

    let head_dim = q_shape[2];
    let half_dim = head_dim / 2;

    // 计算频率
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    rope_with_inv_freq(q, k, pos, &inv_freq)
}

pub fn rope_with_inv_freq(
    q: &Tensor,
    k: &Tensor,
    pos: usize,
    inv_freq: &[f32],
) -> Result<(Tensor, Tensor)> {
    let q_shape = q.shape();
    if q.ndim() != 3 || k.ndim() != 3 {
        return Err(RsinferError::DimensionError(
            "RoPE requires 3D tensors [seq_len, num_heads, head_dim]".into(),
        ));
    }

    let seq_len = q_shape[0];
    let head_dim = q_shape[2];
    let half_dim = head_dim / 2;
    if inv_freq.len() != half_dim {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![half_dim],
            actual: vec![inv_freq.len()],
        });
    }

    // 对 Q 应用旋转
    let q_rotated = apply_rope_to_tensor(&q.data, inv_freq, pos, seq_len, half_dim)?;

    // 对 K 应用旋转
    let k_rotated = apply_rope_to_tensor(&k.data, inv_freq, pos, seq_len, half_dim)?;

    Ok((Tensor { data: q_rotated }, Tensor { data: k_rotated }))
}

/// 对单个张量应用 RoPE
fn apply_rope_to_tensor(
    data: &ArrayD<f32>,
    inv_freq: &[f32],
    pos: usize,
    seq_len: usize,
    half_dim: usize,
) -> Result<ArrayD<f32>> {
    let shape = data.shape().to_vec();
    let mut result = data.clone();

    for seq_idx in 0..seq_len {
        let cur_pos = pos + seq_idx;

        // 对于每个位置，计算旋转角度
        for (freq_idx, &inv_f) in inv_freq.iter().enumerate() {
            let angle = cur_pos as f32 * inv_f;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            // 遍历所有 head
            for head_idx in 0..shape[1] {
                // 获取 x1 和 x2
                let x1 = result[[seq_idx, head_idx, freq_idx]];
                let x2 = result[[seq_idx, head_idx, freq_idx + half_dim]];

                // 应用旋转
                result[[seq_idx, head_idx, freq_idx]] = x1 * cos_val - x2 * sin_val;
                result[[seq_idx, head_idx, freq_idx + half_dim]] = x1 * sin_val + x2 * cos_val;
            }
        }
    }

    Ok(result)
}

/// 注意力机制核心计算
///
/// scores: [num_heads, seq_len_q, seq_len_k]
/// 返回: [num_heads, seq_len_q, seq_len_k]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // q: [num_heads, seq_len_q, head_dim]
    // k: [num_heads, seq_len_k, head_dim]
    // v: [num_heads, seq_len_k, head_dim]

    let num_heads = q.shape()[0];
    let seq_len_q = q.shape()[1];
    let seq_len_k = k.shape()[1];
    let head_dim = q.shape()[2];

    // 保证连续内存，便于按行切片
    let q_std = q.data.as_standard_layout();
    let k_std = k.data.as_standard_layout();
    let v_std = v.data.as_standard_layout();
    let qs = q_std.as_slice().unwrap();
    let ks = k_std.as_slice().unwrap();
    let vs = v_std.as_slice().unwrap();

    // 输出 [num_heads, seq_len_q, head_dim]，按 head 并行
    let mut output = vec![0f32; num_heads * seq_len_q * head_dim];
    let head_stride_q = seq_len_q * head_dim;
    let head_stride_k = seq_len_k * head_dim;

    // 因果掩码：本次调用里第 i 个 query 的绝对位置 = key_offset + i，
    // 它只能注意到 key 的 0..=(key_offset + i)。
    // prefill 时 seq_len_q==seq_len_k（offset=0）；decode 时 seq_len_q=1（offset=已缓存长度）。
    let key_offset = seq_len_k - seq_len_q;

    output
        .par_chunks_mut(head_stride_q)
        .enumerate()
        .for_each(|(h, out_head)| {
            let q_head = &qs[h * head_stride_q..(h + 1) * head_stride_q];
            let k_head = &ks[h * head_stride_k..(h + 1) * head_stride_k];
            let v_head = &vs[h * head_stride_k..(h + 1) * head_stride_k];

            // 每个 query 位置独立计算
            let mut scores = vec![0f32; seq_len_k];
            for i in 0..seq_len_q {
                let q_row = &q_head[i * head_dim..(i + 1) * head_dim];
                let causal_limit = key_offset + i; // 可注意到的最大 key 下标

                // 1) scores = q·k * scale，并做数值稳定 softmax（带因果掩码）
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=causal_limit {
                    let k_row = &k_head[j * head_dim..(j + 1) * head_dim];
                    let s = dot(q_row, k_row) * scale;
                    scores[j] = s;
                    if s > max_score {
                        max_score = s;
                    }
                }
                let mut sum_exp = 0f32;
                for s in scores[..=causal_limit].iter_mut() {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp;

                // 2) output = softmax(scores) @ v（仅累加未被掩码的 key）
                let out_row = &mut out_head[i * head_dim..(i + 1) * head_dim];
                for j in 0..=causal_limit {
                    let w = scores[j] * inv_sum;
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    for d in 0..head_dim {
                        out_row[d] += w * v_row[d];
                    }
                }
            }
        });

    Ok(Tensor {
        data: ArrayD::from_shape_vec(IxDyn(&[num_heads, seq_len_q, head_dim]), output)
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?,
    })
}

pub fn scaled_dot_product_attention_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    kv_group_size: usize,
    scale: f32,
) -> Result<Tensor> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();
    if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
        return Err(RsinferError::DimensionError(
            "GQA attention expects q/k/v to be 3D tensors".into(),
        ));
    }

    let num_heads = q_shape[0];
    let seq_len_q = q_shape[1];
    let head_dim = q_shape[2];
    let num_kv_heads = k_shape[0];
    let seq_len_k = k_shape[1];
    if kv_group_size == 0
        || num_kv_heads * kv_group_size != num_heads
        || v_shape[0] != num_kv_heads
        || v_shape[1] != seq_len_k
        || k_shape[2] != head_dim
        || v_shape[2] != head_dim
    {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![num_heads, seq_len_q, head_dim],
            actual: vec![num_kv_heads, seq_len_k, k_shape[2]],
        });
    }
    if seq_len_q > seq_len_k {
        return Err(RsinferError::DimensionError(format!(
            "GQA attention seq_len_q {seq_len_q} exceeds seq_len_k {seq_len_k}"
        )));
    }

    let q_std = q.data.as_standard_layout();
    let k_std = k.data.as_standard_layout();
    let v_std = v.data.as_standard_layout();
    let qs = q_std.as_slice().unwrap();
    let ks = k_std.as_slice().unwrap();
    let vs = v_std.as_slice().unwrap();

    let mut output = vec![0f32; num_heads * seq_len_q * head_dim];
    let head_stride_q = seq_len_q * head_dim;
    let head_stride_k = seq_len_k * head_dim;
    let key_offset = seq_len_k - seq_len_q;

    output
        .par_chunks_mut(head_stride_q)
        .enumerate()
        .for_each(|(h, out_head)| {
            let kv_head_idx = h / kv_group_size;
            let q_head = &qs[h * head_stride_q..(h + 1) * head_stride_q];
            let k_head = &ks[kv_head_idx * head_stride_k..(kv_head_idx + 1) * head_stride_k];
            let v_head = &vs[kv_head_idx * head_stride_k..(kv_head_idx + 1) * head_stride_k];

            let mut scores = vec![0f32; seq_len_k];
            for i in 0..seq_len_q {
                let q_row = &q_head[i * head_dim..(i + 1) * head_dim];
                let causal_limit = key_offset + i;

                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=causal_limit {
                    let k_row = &k_head[j * head_dim..(j + 1) * head_dim];
                    let s = dot(q_row, k_row) * scale;
                    scores[j] = s;
                    if s > max_score {
                        max_score = s;
                    }
                }
                let mut sum_exp = 0f32;
                for s in scores[..=causal_limit].iter_mut() {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp;

                let out_row = &mut out_head[i * head_dim..(i + 1) * head_dim];
                for j in 0..=causal_limit {
                    let w = scores[j] * inv_sum;
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    for d in 0..head_dim {
                        out_row[d] += w * v_row[d];
                    }
                }
            }
        });

    Ok(Tensor {
        data: ArrayD::from_shape_vec(IxDyn(&[num_heads, seq_len_q, head_dim]), output)
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?,
    })
}

pub struct CachedAttention<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub num_kv_heads: usize,
    pub seq_len_k: usize,
    pub head_dim: usize,
    pub max_len: usize,
}

pub fn scaled_dot_product_attention_gqa_cached(
    q: &Tensor,
    cache: CachedAttention<'_>,
    kv_group_size: usize,
    scale: f32,
) -> Result<Tensor> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(RsinferError::DimensionError(
            "cached GQA attention expects q to be 3D".into(),
        ));
    }

    let num_heads = q_shape[0];
    let seq_len_q = q_shape[1];
    let head_dim = q_shape[2];
    if kv_group_size == 0
        || cache.num_kv_heads * kv_group_size != num_heads
        || cache.head_dim != head_dim
        || seq_len_q > cache.seq_len_k
        || cache.key.len() < cache.num_kv_heads * cache.max_len * cache.head_dim
        || cache.value.len() < cache.num_kv_heads * cache.max_len * cache.head_dim
    {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![num_heads, seq_len_q, head_dim],
            actual: vec![cache.num_kv_heads, cache.seq_len_k, cache.head_dim],
        });
    }

    let q_std = q.data.as_standard_layout();
    let qs = q_std
        .as_slice()
        .ok_or_else(|| RsinferError::DimensionError("q tensor is not contiguous".into()))?;

    if seq_len_q == 1 {
        return scaled_dot_product_attention_gqa_cached_decode_one(
            qs,
            cache,
            num_heads,
            kv_group_size,
            head_dim,
            scale,
        );
    }

    let mut output = vec![0f32; num_heads * seq_len_q * head_dim];
    let head_stride_q = seq_len_q * head_dim;
    let cache_head_stride = cache.max_len * head_dim;
    let key_offset = cache.seq_len_k - seq_len_q;

    output
        .par_chunks_mut(head_stride_q)
        .enumerate()
        .for_each(|(h, out_head)| {
            let kv_head_idx = h / kv_group_size;
            let q_head = &qs[h * head_stride_q..(h + 1) * head_stride_q];
            let k_head =
                &cache.key[kv_head_idx * cache_head_stride..(kv_head_idx + 1) * cache_head_stride];
            let v_head = &cache.value
                [kv_head_idx * cache_head_stride..(kv_head_idx + 1) * cache_head_stride];

            let mut scores = vec![0f32; cache.seq_len_k];
            for i in 0..seq_len_q {
                let q_row = &q_head[i * head_dim..(i + 1) * head_dim];
                let causal_limit = key_offset + i;

                let mut max_score = f32::NEG_INFINITY;
                for j in 0..=causal_limit {
                    let k_row = &k_head[j * head_dim..(j + 1) * head_dim];
                    let s = dot(q_row, k_row) * scale;
                    scores[j] = s;
                    if s > max_score {
                        max_score = s;
                    }
                }
                let mut sum_exp = 0f32;
                for s in scores[..=causal_limit].iter_mut() {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                let inv_sum = 1.0 / sum_exp;

                let out_row = &mut out_head[i * head_dim..(i + 1) * head_dim];
                for j in 0..=causal_limit {
                    let w = scores[j] * inv_sum;
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    for d in 0..head_dim {
                        out_row[d] += w * v_row[d];
                    }
                }
            }
        });

    Ok(Tensor {
        data: ArrayD::from_shape_vec(IxDyn(&[num_heads, seq_len_q, head_dim]), output)
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?,
    })
}

fn scaled_dot_product_attention_gqa_cached_decode_one(
    qs: &[f32],
    cache: CachedAttention<'_>,
    num_heads: usize,
    kv_group_size: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Tensor> {
    let mut output = vec![0f32; num_heads * head_dim];
    let cache_head_stride = cache.max_len * head_dim;

    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(h, out_row)| {
            let kv_head_idx = h / kv_group_size;
            let q_row = &qs[h * head_dim..(h + 1) * head_dim];
            let k_head =
                &cache.key[kv_head_idx * cache_head_stride..(kv_head_idx + 1) * cache_head_stride];
            let v_head = &cache.value
                [kv_head_idx * cache_head_stride..(kv_head_idx + 1) * cache_head_stride];

            let mut max_score = f32::NEG_INFINITY;
            let mut sum_exp = 0f32;
            for j in 0..cache.seq_len_k {
                let k_row = &k_head[j * head_dim..(j + 1) * head_dim];
                let score = dot(q_row, k_row) * scale;
                if score <= max_score {
                    let weight = (score - max_score).exp();
                    sum_exp += weight;
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    for d in 0..head_dim {
                        out_row[d] += weight * v_row[d];
                    }
                } else {
                    let rescale = (max_score - score).exp();
                    for value in out_row.iter_mut() {
                        *value *= rescale;
                    }
                    let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                    for d in 0..head_dim {
                        out_row[d] += v_row[d];
                    }
                    sum_exp = sum_exp * rescale + 1.0;
                    max_score = score;
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for value in out_row.iter_mut() {
                *value *= inv_sum;
            }
        });

    Ok(Tensor {
        data: ArrayD::from_shape_vec(IxDyn(&[num_heads, 1, head_dim]), output)
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?,
    })
}

/// GQA (Grouped Query Attention) 的 KV 头扩展
///
/// 将 [num_kv_heads, seq_len, head_dim] 扩展到 [num_heads, seq_len, head_dim]
pub fn repeat_kv(x: &Tensor, num_repeats: usize) -> Result<Tensor> {
    if num_repeats == 1 {
        return Ok(x.clone());
    }

    let shape = x.shape();
    if shape.len() != 3 {
        return Err(RsinferError::DimensionError(
            "repeat_kv expects 3D tensor [num_kv_heads, seq_len, head_dim]".into(),
        ));
    }

    let num_kv_heads = shape[0];
    let seq_len = shape[1];
    let head_dim = shape[2];
    let num_heads = num_kv_heads * num_repeats;

    let mut result = ArrayD::zeros(IxDyn(&[num_heads, seq_len, head_dim]));

    for kv_head in 0..num_kv_heads {
        for rep in 0..num_repeats {
            let head_idx = kv_head * num_repeats + rep;
            for seq in 0..seq_len {
                for d in 0..head_dim {
                    result[[head_idx, seq, d]] = x.data[[kv_head, seq, d]];
                }
            }
        }
    }

    Ok(Tensor { data: result })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let x = Tensor::from_f32_slice(&[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        let result = softmax(&x, 1).unwrap();
        let sum: f32 = result.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_silu() {
        let x = Tensor::from_f32_slice(&[4], &[0.0, 1.0, -1.0, 2.0]).unwrap();
        let result = silu(&x);
        // silu(0) = 0
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm() {
        let x = Tensor::from_f32_slice(&[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = Tensor::from_f32_slice(&[4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let result = rms_norm(&x, &weight, 1e-5).unwrap();
        assert_eq!(result.shape(), &[1, 4]);
    }

    #[test]
    fn rope_with_inv_freq_matches_theta_path() {
        let q = Tensor::from_f32_slice(
            &[2, 2, 4],
            &[
                0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.1, 0.6, 0.2, -0.4, 0.3, 0.9, -0.5, 0.1, 0.8,
            ],
        )
        .unwrap();
        let k = Tensor::from_f32_slice(&[2, 1, 4], &[0.2, -0.1, 0.4, 0.3, -0.2, 0.7, 0.5, 0.6])
            .unwrap();
        let theta: f32 = 10_000.0;
        let inv_freq: Vec<f32> = (0..2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / 4.0))
            .collect();

        let (expected_q, expected_k) = rope(&q, &k, 5, theta).unwrap();
        let (actual_q, actual_k) = rope_with_inv_freq(&q, &k, 5, &inv_freq).unwrap();

        for (expected, actual) in expected_q.as_slice().iter().zip(actual_q.as_slice()) {
            assert!((expected - actual).abs() <= 1e-6);
        }
        for (expected, actual) in expected_k.as_slice().iter().zip(actual_k.as_slice()) {
            assert!((expected - actual).abs() <= 1e-6);
        }
    }

    #[test]
    fn gqa_attention_matches_repeated_kv_path() {
        let q = Tensor::from_f32_slice(
            &[4, 2, 2],
            &[
                0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.1, 0.6, 0.2, -0.4, 0.3, 0.9, -0.5, 0.1, 0.8,
            ],
        )
        .unwrap();
        let k = Tensor::from_f32_slice(
            &[2, 3, 2],
            &[
                0.2, -0.1, 0.4, 0.3, -0.2, 0.7, 0.5, 0.6, -0.3, 0.2, 0.8, -0.4,
            ],
        )
        .unwrap();
        let v = Tensor::from_f32_slice(
            &[2, 3, 2],
            &[
                0.3, 0.1, -0.2, 0.4, 0.7, -0.5, -0.1, 0.8, 0.6, -0.3, 0.2, 0.5,
            ],
        )
        .unwrap();
        let repeated_k = repeat_kv(&k, 2).unwrap();
        let repeated_v = repeat_kv(&v, 2).unwrap();
        let expected = scaled_dot_product_attention(&q, &repeated_k, &repeated_v, 0.5).unwrap();
        let actual = scaled_dot_product_attention_gqa(&q, &k, &v, 2, 0.5).unwrap();

        assert_eq!(expected.shape(), actual.shape());
        for (expected, actual) in expected.as_slice().iter().zip(actual.as_slice()) {
            assert!((expected - actual).abs() <= 1e-6);
        }
    }

    #[test]
    fn cached_gqa_attention_matches_compact_gqa_path() {
        let q = Tensor::from_f32_slice(
            &[4, 2, 2],
            &[
                0.1, 0.2, 0.3, 0.4, -0.2, 0.5, 0.7, -0.1, 0.6, 0.2, -0.4, 0.3, 0.9, -0.5, 0.1, 0.8,
            ],
        )
        .unwrap();
        let k = Tensor::from_f32_slice(
            &[2, 3, 2],
            &[
                0.2, -0.1, 0.4, 0.3, -0.2, 0.7, 0.5, 0.6, -0.3, 0.2, 0.8, -0.4,
            ],
        )
        .unwrap();
        let v = Tensor::from_f32_slice(
            &[2, 3, 2],
            &[
                0.3, 0.1, -0.2, 0.4, 0.7, -0.5, -0.1, 0.8, 0.6, -0.3, 0.2, 0.5,
            ],
        )
        .unwrap();
        let max_len = 5;
        let head_dim = 2;
        let mut key = vec![0.0; 2 * max_len * head_dim];
        let mut value = vec![0.0; key.len()];
        for h in 0..2 {
            let compact_start = h * 3 * head_dim;
            let cache_start = h * max_len * head_dim;
            key[cache_start..cache_start + 3 * head_dim]
                .copy_from_slice(&k.as_slice()[compact_start..compact_start + 3 * head_dim]);
            value[cache_start..cache_start + 3 * head_dim]
                .copy_from_slice(&v.as_slice()[compact_start..compact_start + 3 * head_dim]);
        }

        let expected = scaled_dot_product_attention_gqa(&q, &k, &v, 2, 0.5).unwrap();
        let actual = scaled_dot_product_attention_gqa_cached(
            &q,
            CachedAttention {
                key: &key,
                value: &value,
                num_kv_heads: 2,
                seq_len_k: 3,
                head_dim,
                max_len,
            },
            2,
            0.5,
        )
        .unwrap();

        assert_eq!(expected.shape(), actual.shape());
        for (expected, actual) in expected.as_slice().iter().zip(actual.as_slice()) {
            assert!((expected - actual).abs() <= 1e-6);
        }
    }

    #[test]
    fn cached_gqa_decode_one_matches_compact_gqa_path() {
        let q = Tensor::from_f32_slice(&[4, 1, 2], &[0.1, 0.2, -0.2, 0.5, 0.6, 0.2, 0.9, -0.5])
            .unwrap();
        let k = Tensor::from_f32_slice(
            &[2, 3, 2],
            &[
                0.2, -0.1, 0.4, 0.3, -0.2, 0.7, 0.5, 0.6, -0.3, 0.2, 0.8, -0.4,
            ],
        )
        .unwrap();
        let v = Tensor::from_f32_slice(
            &[2, 3, 2],
            &[
                0.3, 0.1, -0.2, 0.4, 0.7, -0.5, -0.1, 0.8, 0.6, -0.3, 0.2, 0.5,
            ],
        )
        .unwrap();
        let max_len = 5;
        let head_dim = 2;
        let mut key = vec![0.0; 2 * max_len * head_dim];
        let mut value = vec![0.0; key.len()];
        for h in 0..2 {
            let compact_start = h * 3 * head_dim;
            let cache_start = h * max_len * head_dim;
            key[cache_start..cache_start + 3 * head_dim]
                .copy_from_slice(&k.as_slice()[compact_start..compact_start + 3 * head_dim]);
            value[cache_start..cache_start + 3 * head_dim]
                .copy_from_slice(&v.as_slice()[compact_start..compact_start + 3 * head_dim]);
        }

        let expected = scaled_dot_product_attention_gqa(&q, &k, &v, 2, 0.5).unwrap();
        let actual = scaled_dot_product_attention_gqa_cached(
            &q,
            CachedAttention {
                key: &key,
                value: &value,
                num_kv_heads: 2,
                seq_len_k: 3,
                head_dim,
                max_len,
            },
            2,
            0.5,
        )
        .unwrap();

        assert_eq!(expected.shape(), actual.shape());
        for (expected, actual) in expected.as_slice().iter().zip(actual.as_slice()) {
            assert!((expected - actual).abs() <= 1e-6);
        }
    }

    #[test]
    fn test_linear_forward_q8_close_to_f16() {
        let weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let x = Tensor::from_f32_slice(&[2, 2], &[0.6, -1.4, 1.0, 0.25]).unwrap();
        let f16_out = linear_forward_f16(&x, &weight, 3, 2).unwrap();
        let q8 = Q8LinearWeight::from_f16(&weight, 3, 2).unwrap();
        let q8_out = linear_forward_q8(&x, &q8).unwrap();

        for (expected, actual) in f16_out.as_slice().iter().zip(q8_out.as_slice()) {
            assert!(
                (expected - actual).abs() <= 0.02,
                "q8 output {actual} too far from f16 {expected}"
            );
        }
    }

    #[test]
    fn test_linear_forward_q8_single_row_matches_multi_row_first_row() {
        let weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let q8 = Q8LinearWeight::from_f16(&weight, 3, 2).unwrap();
        let single = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let multi = Tensor::from_f32_slice(&[2, 2], &[0.6, -1.4, 1.0, 0.25]).unwrap();

        let single_out = linear_forward_q8(&single, &q8).unwrap();
        let multi_out = linear_forward_q8(&multi, &q8).unwrap();

        assert_eq!(single_out.shape(), &[1, 3]);
        assert_eq!(multi_out.shape(), &[2, 3]);
        assert_eq!(single_out.as_slice(), &multi_out.as_slice()[..3]);
    }
}
