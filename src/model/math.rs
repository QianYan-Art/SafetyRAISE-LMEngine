//! 与设备无关的参考数学。用来核对优化算子，不根据设备改变模型公式。

use crate::engine::state::{PagedKvCache, RecurrentState};
use crate::model::spec::{GdnSpec, RopeSpec};
use crate::{Result, RsinferError};

fn invalid(message: &str) -> RsinferError {
    RsinferError::DimensionError(message.into())
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

pub(crate) fn rms_norm(input: &[f32], weight: &[f32], eps: f32, offset: bool) -> Result<Vec<f32>> {
    if weight.is_empty()
        || !input.len().is_multiple_of(weight.len())
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid("RMSNorm 的形状或 epsilon 非法"));
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(weight.len()) {
        let variance = row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32;
        if !variance.is_finite() {
            return Err(invalid("RMSNorm 激活平方和溢出"));
        }
        let inv = (variance + eps).sqrt().recip();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(x, w)| x * inv * (w + if offset { 1.0 } else { 0.0 })),
        );
    }
    Ok(output)
}

pub(crate) fn apply_rope(
    values: &mut [f32],
    head_dim: usize,
    position: [usize; 3],
    rope: &RopeSpec,
) -> Result<()> {
    if head_dim == 0
        || !values.len().is_multiple_of(head_dim)
        || rope.rotary_dim == 0
        || rope.rotary_dim > head_dim
        || !rope.rotary_dim.is_multiple_of(2)
    {
        return Err(invalid("RoPE 输入形状非法"));
    }
    let half = rope.rotary_dim / 2;
    for head in values.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let axis = match rope.mrope_section {
                None => 0,
                Some(s) if rope.interleaved => {
                    if i % 3 == 1 && i < s[1] * 3 {
                        1
                    } else if i % 3 == 2 && i < s[2] * 3 {
                        2
                    } else {
                        0
                    }
                }
                Some(s) => {
                    if i < s[0] {
                        0
                    } else if i < s[0] + s[1] {
                        1
                    } else {
                        2
                    }
                }
            };
            let angle =
                position[axis] as f32 / rope.theta.powf((2 * i) as f32 / rope.rotary_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let (left, right) = (head[i], head[i + half]);
            head[i] = left * cos - right * sin;
            head[i + half] = right * cos + left * sin;
        }
    }
    Ok(())
}

/// 在线 softmax，辅助内存与上下文长度无关；缓存中只包含当前 query 可见的前缀。
pub(crate) fn causal_attention(
    query: &[f32],
    cache: &PagedKvCache,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    if heads == 0
        || kv_heads == 0
        || dim == 0
        || !heads.is_multiple_of(kv_heads)
        || query.len()
            != heads
                .checked_mul(dim)
                .ok_or_else(|| invalid("注意力维度溢出"))?
        || cache.width()
            != kv_heads
                .checked_mul(dim)
                .ok_or_else(|| invalid("KV 维度溢出"))?
        || cache.is_empty()
    {
        return Err(invalid("注意力头映射或缓存几何非法"));
    }
    let mut output = vec![0.0; query.len()];
    let scale = (dim as f32).sqrt().recip();
    for h in 0..heads {
        let q = &query[h * dim..(h + 1) * dim];
        let kv_offset = h / (heads / kv_heads) * dim;
        let out = &mut output[h * dim..(h + 1) * dim];
        let mut maximum = f32::NEG_INFINITY;
        let mut normalizer = 0.0;
        for token in 0..cache.len() {
            let (key, value) = cache.get(token)?;
            let score = q
                .iter()
                .zip(&key[kv_offset..kv_offset + dim])
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * scale;
            let next_max = maximum.max(score);
            let prior = (maximum - next_max).exp();
            let current = (score - next_max).exp();
            normalizer = normalizer * prior + current;
            for (o, v) in out.iter_mut().zip(&value[kv_offset..kv_offset + dim]) {
                *o = *o * prior + *v * current;
            }
            maximum = next_max;
        }
        for o in out {
            *o /= normalizer;
        }
    }
    Ok(output)
}

/// Q/K/V 的逐通道因果卷积，状态仅保存前 kernel-1 列。
pub(crate) fn causal_conv_silu(
    input: &[f32],
    weight: &[f32],
    kernel: usize,
    history: &mut [f32],
) -> Result<Vec<f32>> {
    if kernel == 0
        || weight.len()
            != input
                .len()
                .checked_mul(kernel)
                .ok_or_else(|| invalid("卷积维度溢出"))?
        || history.len() != input.len() * (kernel - 1)
    {
        return Err(invalid("因果卷积形状非法"));
    }
    let mut output = Vec::with_capacity(input.len());
    for (channel, x) in input.iter().enumerate() {
        let w = &weight[channel * kernel..(channel + 1) * kernel];
        let start = channel * (kernel - 1);
        let prior = &mut history[start..start + kernel - 1];
        let value = prior.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() + x * w[kernel - 1];
        output.push(silu(value));
        if !prior.is_empty() {
            prior.rotate_left(1);
            prior[kernel - 2] = *x;
        }
    }
    Ok(output)
}

/// Qwen3.5 Gated DeltaNet 的单步数学；矩阵按 [head,Dv,Dk] 保存。
pub(crate) fn gated_delta_step(
    qkv: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    spec: &GdnSpec,
    state: &mut RecurrentState,
) -> Result<Vec<f32>> {
    let key_width = spec.key_heads * spec.key_dim;
    let value_width = spec.value_heads * spec.value_dim;
    if spec.key_heads == 0
        || !spec.value_heads.is_multiple_of(spec.key_heads)
        || qkv.len() != 2 * key_width + value_width
        || [a.len(), b.len(), a_log.len(), dt_bias.len()]
            .iter()
            .any(|n| *n != spec.value_heads)
        || state.matrix.len() != spec.value_heads * spec.key_dim * spec.value_dim
    {
        return Err(invalid("GDN 输入与递归状态不匹配"));
    }
    let mut output = vec![0.0; value_width];
    let key_groups = spec.value_heads / spec.key_heads;
    let query_scale = (spec.key_dim as f32).sqrt().recip();
    for h in 0..spec.value_heads {
        let start = h / key_groups * spec.key_dim;
        let q = &qkv[start..start + spec.key_dim];
        let k = &qkv[key_width + start..key_width + start + spec.key_dim];
        let qi = (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
        let ki = (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
        let alpha = (-a_log[h].exp() * softplus(a[h] + dt_bias[h])).exp();
        let beta = sigmoid(b[h]);
        for v in 0..spec.value_dim {
            let offset = (h * spec.value_dim + v) * spec.key_dim;
            let row = &mut state.matrix[offset..offset + spec.key_dim];
            // 必须先衰减，再用衰减后的矩阵计算 delta error。
            for s in row.iter_mut() {
                *s *= alpha;
            }
            let prediction = row.iter().zip(k).map(|(s, k)| s * k * ki).sum::<f32>();
            let correction = beta * (qkv[2 * key_width + h * spec.value_dim + v] - prediction);
            for (s, k) in row.iter_mut().zip(k) {
                *s += correction * k * ki;
            }
            output[h * spec.value_dim + v] =
                row.iter().zip(q).map(|(s, q)| s * q * qi).sum::<f32>() * query_scale;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_online_softmax_matches_known_weighted_average() {
        let mut kv = PagedKvCache::new(1, 2, 1).unwrap();
        kv.append(&[0.0], &[2.0]).unwrap();
        kv.append(&[2.0_f32.ln()], &[8.0]).unwrap();
        let output = causal_attention(&[1.0, 0.0], &kv, 2, 1, 1).unwrap();
        assert!((output[0] - 6.0).abs() < 1e-6);
        assert!((output[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn delta_rule_decays_before_prediction_and_snapshot_restores() {
        let spec = GdnSpec {
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 1,
            conv_kernel: 1,
        };
        let mut state = RecurrentState {
            matrix: vec![0.0; 2],
            convolution: vec![],
        };
        let output = gated_delta_step(
            &[1.0, 0.0, 1.0, 0.0, 2.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &spec,
            &mut state,
        )
        .unwrap();
        assert!((output[0] - 1.0 / 2.0_f32.sqrt()).abs() < 2e-6);
        let saved = state.clone();
        let output = gated_delta_step(
            &[1.0, 0.0, 1.0, 0.0, 0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &spec,
            &mut state,
        )
        .unwrap();
        assert!((output[0] - 0.25 / 2.0_f32.sqrt()).abs() < 2e-6);
        assert!((saved.matrix[0] - 1.0).abs() < 2e-6);
    }

    #[test]
    fn partial_rope_preserves_tail_and_uses_axis_positions() {
        let rope = RopeSpec {
            theta: 10000.0,
            rotary_dim: 2,
            mrope_section: None,
            interleaved: true,
        };
        let mut x = [1.0, 0.0, 7.0, 9.0];
        apply_rope(&mut x, 4, [1, 8, 9], &rope).unwrap();
        assert!((x[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0_f32.sin()).abs() < 1e-6);
        assert_eq!(&x[2..], &[7.0, 9.0]);
    }

    #[test]
    fn convolution_retains_only_prior_raw_inputs() {
        let mut history = [0.0; 2];
        assert!(
            (causal_conv_silu(&[2.0], &[1.0, 2.0, 3.0], 3, &mut history).unwrap()[0] - silu(6.0))
                .abs()
                < 1e-6
        );
        assert_eq!(history, [0.0, 2.0]);
        assert!(
            (causal_conv_silu(&[1.0], &[1.0, 2.0, 3.0], 3, &mut history).unwrap()[0] - silu(7.0))
                .abs()
                < 1e-6
        );
        assert_eq!(history, [2.0, 1.0]);
    }
}
