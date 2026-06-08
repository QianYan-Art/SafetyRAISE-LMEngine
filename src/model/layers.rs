//! Transformer 层实现
//!
//! 包含 RMSNorm、Attention、MLP 和 TransformerBlock。

use half::f16;

use crate::engine::KVCache;
use crate::error::{Result, RsinferError};
use crate::gpu::GpuMatVec;
use crate::model::config::Qwen3Config;
use crate::model::weights::{get_linear_weight_f16, get_weight, WeightMap};
use crate::tensor::{
    linear_forward_f16, repeat_kv, rms_norm, rope, scaled_dot_product_attention, silu, Tensor,
};

/// RMS 归一化层
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_norm(x, &self.weight, self.eps)
    }
}

/// 线性层 (矩阵乘法 + 可选偏置)
///
/// 权重以 f16 存储（权重本就是 f16，无精度损失），计算时即时转 f32。
pub struct Linear {
    /// f16 权重，行主序 [out_features, in_features]
    pub weight: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
    pub bias: Option<Tensor>,
    gpu_matvec: Option<GpuMatVec>,
}

impl Linear {
    /// 从 f32 张量构建：把权重压成 f16 存储（输入权重原本就来自 f16，往返无损）
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let shape = weight.shape();
        let (out_features, in_features) = (shape[0], shape[1]);
        let weight: Vec<f16> = weight
            .as_slice()
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        Self {
            weight,
            out_features,
            in_features,
            bias,
            gpu_matvec: None,
        }
    }

    pub fn from_f16_weight(
        shape: &[usize],
        weight: Vec<f16>,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        if shape.len() != 2 {
            return Err(RsinferError::DimensionError(format!(
                "Linear weight 需要 2D, got {}D",
                shape.len()
            )));
        }
        let (out_features, in_features) = (shape[0], shape[1]);
        let expected = out_features * in_features;
        if weight.len() != expected {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![expected],
                actual: vec![weight.len()],
            });
        }
        Ok(Self {
            weight,
            out_features,
            in_features,
            bias,
            gpu_matvec: None,
        })
    }

    pub fn from_weight_map(weights: &WeightMap, name: &str, bias: Option<Tensor>) -> Result<Self> {
        let (shape, weight) = get_linear_weight_f16(weights, name)?;
        Self::from_f16_weight(&shape, weight, bias)
    }

    pub fn try_enable_gpu_matvec(&mut self) -> std::result::Result<(), String> {
        let accelerator =
            GpuMatVec::from_f16_weight(&self.weight, self.out_features, self.in_features)?;
        self.gpu_matvec = Some(accelerator);
        Ok(())
    }

    pub fn has_gpu_matvec(&self) -> bool {
        self.gpu_matvec.is_some()
    }

    /// 前向传播: x @ W^T + b
    ///
    /// x: [batch, in_features] -> result: [batch, out_features]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.bias.is_none() {
            if let Some(gpu_matvec) = &self.gpu_matvec {
                if let Ok(result) = gpu_matvec.forward(x) {
                    return Ok(result);
                }
            }
        }

        let result = linear_forward_f16(x, &self.weight, self.out_features, self.in_features)?;

        if let Some(bias) = &self.bias {
            result.add(bias)
        } else {
            Ok(result)
        }
    }
}

/// 注意力层
///
/// 支持 MHA 和 GQA (Grouped Query Attention)
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    /// Qwen3 特有：对每个 head 的 Q 做 RMSNorm（权重形状 [head_dim]）
    pub q_norm: RmsNorm,
    /// Qwen3 特有：对每个 head 的 K 做 RMSNorm（权重形状 [head_dim]）
    pub k_norm: RmsNorm,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}

impl Attention {
    /// 前向传播
    ///
    /// hidden_states: [seq_len, hidden_size]
    /// 返回: [seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
    ) -> Result<Tensor> {
        let seq_len = hidden_states.shape()[0];

        let q = self.q_proj.forward(hidden_states)?.reshape(&[
            seq_len,
            self.num_heads,
            self.head_dim,
        ])?;
        let k = self.k_proj.forward(hidden_states)?.reshape(&[
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ])?;
        let v = self.v_proj.forward(hidden_states)?.reshape(&[
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ])?;

        // Qwen3 QK-Norm：每个 head 沿 head_dim 做 RMSNorm，必须在 RoPE 之前
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = rope(&q, &k, position_offset, self.rope_theta)?;

        // [seq, head, dim] -> [head, seq, dim]
        let q = transpose_for_attention(&q, self.num_heads)?;
        let k = transpose_for_attention(&k, self.num_kv_heads)?;
        let v = transpose_for_attention(&v, self.num_kv_heads)?;

        kv_cache.append(layer_idx, &k, &v)?;
        let (cached_k, cached_v) = kv_cache.get(layer_idx)?;

        // GQA：把 kv head 复制到对应的 query head 组
        let kv_group_size = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(cached_k, kv_group_size)?;
        let v = repeat_kv(cached_v, kv_group_size)?;

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_product_attention(&q, &k, &v, scale)?;

        let attn = transpose_back(&attn, seq_len, self.num_heads)?
            .reshape(&[seq_len, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&attn)
    }
}

/// 将 [seq_len, num_heads, head_dim] 转置为 [num_heads, seq_len, head_dim]
fn transpose_for_attention(x: &Tensor, num_heads: usize) -> Result<Tensor> {
    let shape = x.shape();
    let seq_len = shape[0];
    let head_dim = shape[2];

    let mut result = ndarray::ArrayD::zeros(ndarray::IxDyn(&[num_heads, seq_len, head_dim]));

    for s in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                result[[h, s, d]] = x.data[[s, h, d]];
            }
        }
    }

    Ok(Tensor { data: result })
}

/// 将 [num_heads, seq_len, head_dim] 转置回 [seq_len, num_heads, head_dim]
fn transpose_back(x: &Tensor, seq_len: usize, num_heads: usize) -> Result<Tensor> {
    let head_dim = x.shape()[2];

    let mut result = ndarray::ArrayD::zeros(ndarray::IxDyn(&[seq_len, num_heads, head_dim]));

    for h in 0..num_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                result[[s, h, d]] = x.data[[h, s, d]];
            }
        }
    }

    Ok(Tensor { data: result })
}

/// MLP 层 (SwiGLU)
///
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = silu(&gate);
        let up = self.up_proj.forward(x)?;
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Transformer Block
///
/// 包含 RMSNorm -> Attention -> Residual -> RMSNorm -> MLP -> Residual
pub struct TransformerBlock {
    pub input_layernorm: RmsNorm,
    pub attention: Attention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(
        input_layernorm: RmsNorm,
        attention: Attention,
        post_attention_layernorm: RmsNorm,
        mlp: Mlp,
    ) -> Self {
        Self {
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp,
        }
    }

    /// 前向传播
    ///
    /// hidden_states: [seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
    ) -> Result<Tensor> {
        // RMSNorm -> Attention
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self
            .attention
            .forward(&normed, kv_cache, layer_idx, position_offset)?;

        // Residual connection
        let hidden_states = hidden_states.add(&attn_output)?;

        // RMSNorm -> MLP
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;

        // Residual connection
        hidden_states.add(&mlp_output)
    }
}

pub fn build_transformer_block(
    weights: &WeightMap,
    config: &Qwen3Config,
    layer_idx: usize,
) -> Result<TransformerBlock> {
    let prefix = format!("model.layers.{layer_idx}");
    let get = |suffix: &str| get_weight(weights, &format!("{prefix}.{suffix}"));
    let eps = config.rms_norm_eps;
    let linear = |suffix: &str| -> Result<Linear> {
        Linear::from_weight_map(weights, &format!("{prefix}.{suffix}"), None)
    };

    let attention = Attention {
        q_proj: linear("self_attn.q_proj.weight")?,
        k_proj: linear("self_attn.k_proj.weight")?,
        v_proj: linear("self_attn.v_proj.weight")?,
        o_proj: linear("self_attn.o_proj.weight")?,
        q_norm: RmsNorm::new(get("self_attn.q_norm.weight")?, eps),
        k_norm: RmsNorm::new(get("self_attn.k_norm.weight")?, eps),
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim(),
        rope_theta: config.rope_theta,
    };

    let mlp = Mlp::new(
        linear("mlp.gate_proj.weight")?,
        linear("mlp.up_proj.weight")?,
        linear("mlp.down_proj.weight")?,
    );

    Ok(TransformerBlock::new(
        RmsNorm::new(get("input_layernorm.weight")?, eps),
        attention,
        RmsNorm::new(get("post_attention_layernorm.weight")?, eps),
        mlp,
    ))
}
