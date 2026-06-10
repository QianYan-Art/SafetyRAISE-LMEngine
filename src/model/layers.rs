//! Transformer 层实现
//!
//! 包含 RMSNorm、Attention、MLP 和 TransformerBlock。

use half::f16;
use rayon::join;
use std::cell::RefCell;
use std::time::{Duration, Instant};

use crate::engine::{CachedKV, KVCache};
use crate::error::{Result, RsinferError};
use crate::gpu::{
    GpuContext, GpuDecodeGqaAttention, GpuDecodeGqaAttentionConfig, GpuKvAppend, GpuMatVec,
    GpuQ8MatVec, GpuQ8SameInputBatch, GpuQ8SwiGluDown, GpuQkRmsNormRope, GpuQkRmsNormRopeConfig,
    GpuResidentBuffer, GpuSwiGluDown,
};
use crate::model::config::Qwen3Config;
use crate::model::q8_sidecar::Q8SidecarCache;
use crate::model::weights::{get_linear_weight_f16, get_weight, WeightMap};
use crate::tensor::{
    linear_forward_f16, linear_forward_q8, linear_forward_q8_profiled, rms_norm,
    rms_norm_per_head_inplace, rope_single_token_inplace, rope_with_inv_freq,
    scaled_dot_product_attention_gqa_cached,
    scaled_dot_product_attention_gqa_cached_decode_one_raw, silu, CachedAttention, Q8LinearProfile,
    Q8LinearWeight, Tensor,
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
/// 权重以 f16 存储；在 warm Q8 sidecar fast path 下，已接入的 Q8-only 线性层可不再保留原始 f16 副本。
pub struct Linear {
    /// f16 权重，行主序 [out_features, in_features]；Q8-only fast path 下允许为空。
    pub weight: Vec<f16>,
    pub out_features: usize,
    pub in_features: usize,
    pub bias: Option<Tensor>,
    gpu_matvec: Option<GpuMatVec>,
    q8_gpu_matvec: Option<GpuQ8MatVec>,
    q8_weight: Option<Q8LinearWeight>,
    source_name: Option<String>,
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
            q8_gpu_matvec: None,
            q8_weight: None,
            source_name: None,
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
            q8_gpu_matvec: None,
            q8_weight: None,
            source_name: None,
        })
    }

    pub fn from_q8_weight(name: &str, q8_weight: Q8LinearWeight, bias: Option<Tensor>) -> Self {
        Self {
            weight: Vec::new(),
            out_features: q8_weight.out_features,
            in_features: q8_weight.in_features,
            bias,
            gpu_matvec: None,
            q8_gpu_matvec: None,
            q8_weight: Some(q8_weight),
            source_name: Some(name.to_string()),
        }
    }

    pub fn from_weight_map(weights: &WeightMap, name: &str, bias: Option<Tensor>) -> Result<Self> {
        let (shape, weight) = get_linear_weight_f16(weights, name)?;
        let mut linear = Self::from_f16_weight(&shape, weight, bias)?;
        linear.source_name = Some(name.to_string());
        Ok(linear)
    }

    pub fn from_weight_map_or_q8_sidecar(
        weights: &WeightMap,
        name: &str,
        bias: Option<Tensor>,
        sidecar: Option<&mut Q8SidecarCache>,
    ) -> Result<Self> {
        if let Some(cache) = sidecar {
            if let Some(q8_weight) = cache.load_existing(name)? {
                return Ok(Self::from_q8_weight(name, q8_weight, bias));
            }
        }
        Self::from_weight_map(weights, name, bias)
    }

    fn has_f16_weight(&self) -> bool {
        self.weight.len() == self.out_features * self.in_features
    }

    pub fn try_enable_gpu_matvec(&mut self) -> std::result::Result<(), String> {
        if !self.has_f16_weight() {
            return Err(format!(
                "Linear '{}' has no f16 weight for GPU matvec fallback",
                self.source_name.as_deref().unwrap_or("<unnamed>")
            ));
        }
        let accelerator =
            GpuMatVec::from_f16_weight(&self.weight, self.out_features, self.in_features)?;
        self.gpu_matvec = Some(accelerator);
        Ok(())
    }

    pub fn try_enable_gpu_matvec_with_context(
        &mut self,
        context: &GpuContext,
    ) -> std::result::Result<(), String> {
        if !self.has_f16_weight() {
            return Err(format!(
                "Linear '{}' has no f16 weight for GPU matvec fallback",
                self.source_name.as_deref().unwrap_or("<unnamed>")
            ));
        }
        let accelerator = GpuMatVec::from_f16_weight_with_context(
            context,
            &self.weight,
            self.out_features,
            self.in_features,
        )?;
        self.gpu_matvec = Some(accelerator);
        Ok(())
    }

    pub fn has_gpu_matvec(&self) -> bool {
        self.gpu_matvec.is_some()
    }

    pub fn gpu_matvec(&self) -> Option<&GpuMatVec> {
        self.gpu_matvec.as_ref()
    }

    pub fn q8_gpu_matvec(&self) -> Option<&GpuQ8MatVec> {
        self.q8_gpu_matvec.as_ref()
    }

    pub fn q8_gpu_matvec_mut(&mut self) -> Option<&mut GpuQ8MatVec> {
        self.q8_gpu_matvec.as_mut()
    }

    fn has_cpu_q8_only(&self) -> bool {
        self.bias.is_none()
            && self.q8_weight.is_some()
            && self.q8_gpu_matvec.is_none()
            && self.gpu_matvec.is_none()
    }

    pub fn has_q8_gpu_argmax(&self) -> bool {
        self.bias.is_none()
            && self
                .q8_gpu_matvec
                .as_ref()
                .map(GpuQ8MatVec::supports_argmax)
                .unwrap_or(false)
    }

    pub fn try_enable_q8_gpu_matvec_with_context(
        &mut self,
        context: &GpuContext,
    ) -> std::result::Result<(), String> {
        self.try_enable_q8_gpu_matvec_with_context_and_argmax(context, true)
    }

    pub fn try_enable_q8_gpu_matvec_with_context_and_argmax(
        &mut self,
        context: &GpuContext,
        enable_argmax: bool,
    ) -> std::result::Result<(), String> {
        if self.q8_weight.is_none() {
            self.try_enable_q8_weight().map_err(|err| err.to_string())?;
        }
        let q8_weight = self
            .q8_weight
            .as_ref()
            .ok_or_else(|| "Q8 weight was not attached".to_string())?;
        let accelerator =
            GpuQ8MatVec::from_q8_weight_with_context_and_argmax(context, q8_weight, enable_argmax)?;
        self.q8_gpu_matvec = Some(accelerator);
        Ok(())
    }

    pub fn release_q8_gpu_standalone_forward_resources(&mut self, keep_readback: bool) {
        if let Some(matvec) = self.q8_gpu_matvec.as_mut() {
            matvec.release_standalone_forward_resources(keep_readback);
        }
    }

    pub fn try_enable_q8_weight(&mut self) -> Result<()> {
        if self.q8_weight.is_some() {
            return Ok(());
        }
        if !self.has_f16_weight() {
            return Err(RsinferError::WeightError(format!(
                "Linear '{}' has no f16 weight to derive missing Q8 fallback",
                self.source_name.as_deref().unwrap_or("<unnamed>")
            )));
        }
        let q8 = Q8LinearWeight::from_f16(&self.weight, self.out_features, self.in_features)?;
        self.q8_weight = Some(q8);
        Ok(())
    }

    pub fn try_enable_q8_weight_with_sidecar(
        &mut self,
        sidecar: Option<&mut Q8SidecarCache>,
    ) -> Result<()> {
        if self.q8_weight.is_some() {
            return Ok(());
        }
        if let (Some(cache), Some(name)) = (sidecar, self.source_name.as_deref()) {
            match cache.load_or_create(name, self.out_features, self.in_features, || {
                if !self.has_f16_weight() {
                    return Err(RsinferError::WeightError(format!(
                        "Linear '{name}' has no f16 weight to rebuild missing Q8 sidecar entry"
                    )));
                }
                Q8LinearWeight::from_f16(&self.weight, self.out_features, self.in_features)
            }) {
                Ok(q8) => {
                    self.q8_weight = Some(q8);
                    return Ok(());
                }
                Err(err) => cache.record_fallback(name, err.to_string()),
            }
        }
        self.try_enable_q8_weight()
    }

    /// 前向传播: x @ W^T + b
    ///
    /// x: [batch, in_features] -> result: [batch, out_features]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.bias.is_none() {
            if let Some(q8_gpu_matvec) = &self.q8_gpu_matvec {
                if let Ok(result) = q8_gpu_matvec.forward(x) {
                    return Ok(result);
                }
            }
            if let Some(gpu_matvec) = &self.gpu_matvec {
                if let Ok(result) = gpu_matvec.forward(x) {
                    return Ok(result);
                }
            }
            if let Some(q8_weight) = &self.q8_weight {
                if let Ok(result) = linear_forward_q8(x, q8_weight) {
                    return Ok(result);
                }
            }
        }

        if !self.has_f16_weight() {
            return Err(RsinferError::WeightError(format!(
                "Linear '{}' has no f16 fallback weight attached",
                self.source_name.as_deref().unwrap_or("<unnamed>")
            )));
        }
        let result = linear_forward_f16(x, &self.weight, self.out_features, self.in_features)?;

        if let Some(bias) = &self.bias {
            result.add(bias)
        } else {
            Ok(result)
        }
    }

    pub fn try_forward_q8_gpu_argmax(&self, x: &Tensor) -> std::result::Result<u32, String> {
        if !self.has_q8_gpu_argmax() {
            return Err("Q8 GPU argmax does not support bias".to_string());
        }
        let q8_gpu_matvec = self
            .q8_gpu_matvec
            .as_ref()
            .ok_or_else(|| "Q8 GPU matvec is not attached".to_string())?;
        q8_gpu_matvec.forward_argmax(x)
    }

    pub fn try_forward_q8_gpu_raw(&self, input: &[f32]) -> std::result::Result<Vec<f32>, String> {
        if self.bias.is_some() {
            return Err("Q8 GPU raw path does not support bias".to_string());
        }
        let q8_gpu_matvec = self
            .q8_gpu_matvec
            .as_ref()
            .ok_or_else(|| "Q8 GPU matvec is not attached".to_string())?;
        q8_gpu_matvec.forward_raw(input)
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
    rope_inv_freq: Vec<f32>,
    q8_qkv_batch: Option<GpuQ8SameInputBatch>,
    resident_runtime_state: RefCell<Option<ResidentAttentionRuntimeState>>,
}

struct ResidentAttentionRuntimeState {
    kv_gpu: GpuKvAppend,
    resident_key_cache: GpuResidentBuffer,
    resident_value_cache: GpuResidentBuffer,
    resident_q_pre: GpuResidentBuffer,
    resident_k_pre: GpuResidentBuffer,
    resident_q: GpuResidentBuffer,
    resident_k: GpuResidentBuffer,
    resident_v: GpuResidentBuffer,
    resident_attention: GpuResidentBuffer,
    qk_gpu: GpuQkRmsNormRope,
    attn_gpu: GpuDecodeGqaAttention,
    o_proj_gpu_cache: Option<crate::gpu::GpuQ8MatVecResidentCache>,
    qkv_batch_cache: Option<crate::gpu::GpuQ8SameInputBatchResidentCache>,
    qk_gpu_cache: crate::gpu::GpuQkRmsNormRopeResidentCache,
    kv_gpu_cache: crate::gpu::GpuKvAppendResidentCache,
    attn_gpu_cache: crate::gpu::GpuDecodeGqaAttentionResidentCache,
    synced_cache_id: u64,
    synced_cache_revision: u64,
    max_len: usize,
}

impl ResidentAttentionRuntimeState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        context: &GpuContext,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_len: usize,
        position_buffer: &wgpu::Buffer,
        q_weight: &Tensor,
        k_weight: &Tensor,
        rope_inv_freq: &[f32],
        q_eps: f32,
        k_eps: f32,
    ) -> std::result::Result<Self, String> {
        let kv_gpu =
            GpuKvAppend::with_context_resident_only(context, num_kv_heads, head_dim, max_len)?;
        let resident_key_cache =
            GpuResidentBuffer::with_context(context, &[num_kv_heads, max_len, head_dim])?;
        let resident_value_cache =
            GpuResidentBuffer::with_context(context, &[num_kv_heads, max_len, head_dim])?;
        let resident_q_pre = GpuResidentBuffer::with_context(context, &[1, num_heads, head_dim])?;
        let resident_k_pre =
            GpuResidentBuffer::with_context(context, &[1, num_kv_heads, head_dim])?;
        let resident_q = GpuResidentBuffer::with_context(context, &[1, num_heads, head_dim])?;
        let resident_k = GpuResidentBuffer::with_context(context, &[1, num_kv_heads, head_dim])?;
        let resident_v = GpuResidentBuffer::with_context(context, &[1, num_kv_heads, head_dim])?;
        let resident_attention =
            GpuResidentBuffer::with_context(context, &[1, num_heads, head_dim])?;
        let qk_gpu = GpuQkRmsNormRope::with_context(
            context,
            GpuQkRmsNormRopeConfig {
                q_weight,
                k_weight,
                q_shape: &[1, num_heads, head_dim],
                k_shape: &[1, num_kv_heads, head_dim],
                pos: 0,
                inv_freq: rope_inv_freq,
                q_eps,
                k_eps,
            },
        )?;
        let attn_gpu = GpuDecodeGqaAttention::with_context(
            context,
            GpuDecodeGqaAttentionConfig {
                num_heads,
                num_kv_heads,
                head_dim,
                max_len,
                scale: 1.0 / (head_dim as f32).sqrt(),
            },
        )?;
        let qk_gpu_cache = qk_gpu.prepare_resident_cache_with_position_buffer(
            &resident_q_pre,
            &resident_k_pre,
            &resident_q,
            &resident_k,
            position_buffer,
        )?;
        let kv_gpu_cache = kv_gpu.prepare_resident_cache_with_position_buffer(
            &resident_k,
            &resident_v,
            &resident_key_cache,
            &resident_value_cache,
            position_buffer,
        )?;
        let attn_gpu_cache = attn_gpu.prepare_resident_cache_with_position_buffer(
            &resident_q,
            &resident_key_cache,
            &resident_value_cache,
            &resident_attention,
            position_buffer,
        )?;
        Ok(Self {
            kv_gpu,
            resident_key_cache,
            resident_value_cache,
            resident_q_pre,
            resident_k_pre,
            resident_q,
            resident_k,
            resident_v,
            resident_attention,
            qk_gpu,
            attn_gpu,
            o_proj_gpu_cache: None,
            qkv_batch_cache: None,
            qk_gpu_cache,
            kv_gpu_cache,
            attn_gpu_cache,
            synced_cache_id: 0,
            synced_cache_revision: 0,
            max_len,
        })
    }
}

impl ResidentTransformerBlockRuntimeState {
    fn new(context: &GpuContext, block: &TransformerBlock, hidden_size: usize) -> Result<Self> {
        let gpu_input_norm = crate::gpu::GpuRmsNorm::with_context(
            context,
            &block.input_layernorm.weight,
            &[1, hidden_size],
            block.input_layernorm.eps,
        )
        .map_err(TransformerBlock::gpu_error)?;
        let gpu_post_norm = crate::gpu::GpuRmsNorm::with_context(
            context,
            &block.post_attention_layernorm.weight,
            &[1, hidden_size],
            block.post_attention_layernorm.eps,
        )
        .map_err(TransformerBlock::gpu_error)?;
        let gpu_add = crate::gpu::GpuResidualAdd::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_input = GpuResidentBuffer::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_norm1 = GpuResidentBuffer::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_attn = GpuResidentBuffer::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_hidden = GpuResidentBuffer::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_norm2 = GpuResidentBuffer::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_mlp_output = GpuResidentBuffer::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_mlp_hidden =
            GpuResidentBuffer::with_context(context, &[1, block.mlp.gate_proj.out_features])
                .map_err(TransformerBlock::gpu_error)?;
        let resident_output = GpuResidentBuffer::with_context(context, &[1, hidden_size])
            .map_err(TransformerBlock::gpu_error)?;
        let resident_position = crate::gpu::GpuPositionUniform::with_context(context, 0)
            .map_err(TransformerBlock::gpu_error)?;
        let resident_mlp_q8_cache = match (
            block.mlp.q8_gpu_swiglu_down.as_ref(),
            block.mlp.gate_proj.q8_gpu_matvec(),
            block.mlp.up_proj.q8_gpu_matvec(),
            block.mlp.down_proj.q8_gpu_matvec(),
        ) {
            (Some(fused), Some(gate), Some(up), Some(down)) => Some(
                fused
                    .prepare_resident_cache(
                        gate,
                        up,
                        down,
                        &resident_norm2,
                        &resident_mlp_hidden,
                        &resident_mlp_output,
                    )
                    .map_err(TransformerBlock::gpu_error)?,
            ),
            _ => None,
        };
        let gpu_input_norm_cache = gpu_input_norm
            .prepare_resident_cache(&resident_input, &resident_norm1)
            .map_err(TransformerBlock::gpu_error)?;
        let gpu_post_norm_cache = gpu_post_norm
            .prepare_resident_cache(&resident_hidden, &resident_norm2)
            .map_err(TransformerBlock::gpu_error)?;
        let gpu_input_residual_cache = gpu_add
            .prepare_resident_cache(&resident_input, &resident_attn, &resident_hidden)
            .map_err(TransformerBlock::gpu_error)?;
        let gpu_output_residual_cache = gpu_add
            .prepare_resident_cache(&resident_hidden, &resident_mlp_output, &resident_output)
            .map_err(TransformerBlock::gpu_error)?;
        Ok(Self {
            resident_input,
            resident_norm1,
            resident_attn,
            resident_hidden,
            resident_norm2,
            resident_mlp_output,
            resident_mlp_hidden,
            resident_output,
            resident_position,
            resident_mlp_q8_cache,
            gpu_input_norm,
            gpu_post_norm,
            gpu_add,
            gpu_input_norm_cache,
            gpu_post_norm_cache,
            gpu_input_residual_cache,
            gpu_output_residual_cache,
            slot_caches: None,
        })
    }

    fn ensure_slot_caches(
        &mut self,
        slot0: &GpuResidentBuffer,
        slot1: &GpuResidentBuffer,
    ) -> Result<()> {
        if self.slot_caches.is_some() {
            return Ok(());
        }
        let even = ResidentTransformerBlockIoCaches {
            input_norm_cache: self
                .gpu_input_norm
                .prepare_resident_cache(slot0, &self.resident_norm1)
                .map_err(TransformerBlock::gpu_error)?,
            input_residual_cache: self
                .gpu_add
                .prepare_resident_cache(slot0, &self.resident_attn, &self.resident_hidden)
                .map_err(TransformerBlock::gpu_error)?,
            output_residual_cache: self
                .gpu_add
                .prepare_resident_cache(&self.resident_hidden, &self.resident_mlp_output, slot1)
                .map_err(TransformerBlock::gpu_error)?,
        };
        let odd = ResidentTransformerBlockIoCaches {
            input_norm_cache: self
                .gpu_input_norm
                .prepare_resident_cache(slot1, &self.resident_norm1)
                .map_err(TransformerBlock::gpu_error)?,
            input_residual_cache: self
                .gpu_add
                .prepare_resident_cache(slot1, &self.resident_attn, &self.resident_hidden)
                .map_err(TransformerBlock::gpu_error)?,
            output_residual_cache: self
                .gpu_add
                .prepare_resident_cache(&self.resident_hidden, &self.resident_mlp_output, slot0)
                .map_err(TransformerBlock::gpu_error)?,
        };
        self.slot_caches = Some(ResidentTransformerBlockSlotCaches { even, odd });
        Ok(())
    }
}

struct ResidentTransformerBlockRuntimeState {
    #[allow(dead_code)]
    resident_input: GpuResidentBuffer,
    resident_norm1: GpuResidentBuffer,
    resident_attn: GpuResidentBuffer,
    resident_hidden: GpuResidentBuffer,
    resident_norm2: GpuResidentBuffer,
    resident_mlp_output: GpuResidentBuffer,
    resident_mlp_hidden: GpuResidentBuffer,
    #[allow(dead_code)]
    resident_output: GpuResidentBuffer,
    #[allow(dead_code)]
    resident_position: crate::gpu::GpuPositionUniform,
    resident_mlp_q8_cache: Option<crate::gpu::GpuQ8SwiGluDownResidentCache>,
    gpu_input_norm: crate::gpu::GpuRmsNorm,
    gpu_post_norm: crate::gpu::GpuRmsNorm,
    gpu_add: crate::gpu::GpuResidualAdd,
    #[allow(dead_code)]
    gpu_input_norm_cache: crate::gpu::GpuRmsNormResidentCache,
    gpu_post_norm_cache: crate::gpu::GpuRmsNormResidentCache,
    #[allow(dead_code)]
    gpu_input_residual_cache: crate::gpu::GpuResidualAddResidentCache,
    #[allow(dead_code)]
    gpu_output_residual_cache: crate::gpu::GpuResidualAddResidentCache,
    slot_caches: Option<ResidentTransformerBlockSlotCaches>,
}

struct ResidentTransformerBlockIoCaches {
    input_norm_cache: crate::gpu::GpuRmsNormResidentCache,
    input_residual_cache: crate::gpu::GpuResidualAddResidentCache,
    output_residual_cache: crate::gpu::GpuResidualAddResidentCache,
}

struct ResidentTransformerBlockSlotCaches {
    even: ResidentTransformerBlockIoCaches,
    odd: ResidentTransformerBlockIoCaches,
}

impl Attention {
    fn gpu_error(err: String) -> RsinferError {
        RsinferError::DimensionError(format!("GPU resident attention prototype failed: {err}"))
    }

    pub fn try_enable_q8_weights(
        &mut self,
        mut sidecar: Option<&mut Q8SidecarCache>,
    ) -> Result<usize> {
        let mut attached = 0usize;
        for linear in [
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.o_proj,
        ] {
            match sidecar.as_deref_mut() {
                Some(cache) => linear.try_enable_q8_weight_with_sidecar(Some(cache))?,
                None => linear.try_enable_q8_weight_with_sidecar(None)?,
            }
            attached += 1;
        }
        Ok(attached)
    }

    pub fn try_enable_gpu_matvecs(&mut self, context: &GpuContext) -> (usize, Vec<String>) {
        let mut attached = 0usize;
        let mut errors = Vec::new();
        for (name, linear) in [
            ("q_proj", &mut self.q_proj),
            ("k_proj", &mut self.k_proj),
            ("v_proj", &mut self.v_proj),
            ("o_proj", &mut self.o_proj),
        ] {
            match linear.try_enable_gpu_matvec_with_context(context) {
                Ok(()) => attached += 1,
                Err(err) => errors.push(format!("attention.{name}: {err}")),
            }
        }
        (attached, errors)
    }

    pub fn try_enable_q8_gpu_matvecs(&mut self, context: &GpuContext) -> (usize, Vec<String>) {
        let mut attached = 0usize;
        let mut errors = Vec::new();
        for (name, linear) in [
            ("q_proj", &mut self.q_proj),
            ("k_proj", &mut self.k_proj),
            ("v_proj", &mut self.v_proj),
            ("o_proj", &mut self.o_proj),
        ] {
            let enable_argmax = false;
            match linear.try_enable_q8_gpu_matvec_with_context_and_argmax(context, enable_argmax) {
                Ok(()) => attached += 1,
                Err(err) => errors.push(format!("attention.{name}.q8_gpu: {err}")),
            }
        }
        self.q8_qkv_batch = match (
            self.q_proj.q8_gpu_matvec(),
            self.k_proj.q8_gpu_matvec(),
            self.v_proj.q8_gpu_matvec(),
        ) {
            (Some(q_proj), Some(k_proj), Some(v_proj)) => {
                match GpuQ8SameInputBatch::new(&[q_proj, k_proj, v_proj]) {
                    Ok(batch) => {
                        self.q_proj
                            .release_q8_gpu_standalone_forward_resources(true);
                        self.k_proj
                            .release_q8_gpu_standalone_forward_resources(true);
                        self.v_proj
                            .release_q8_gpu_standalone_forward_resources(true);
                        Some(batch)
                    }
                    Err(err) => {
                        errors.push(format!("attention.qkv.q8_shared_input: {err}"));
                        None
                    }
                }
            }
            _ => None,
        };
        (attached, errors)
    }

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

        let (q, k, v) = if seq_len == 1 {
            match (
                self.q8_qkv_batch.as_ref(),
                self.q_proj.q8_gpu_matvec(),
                self.k_proj.q8_gpu_matvec(),
                self.v_proj.q8_gpu_matvec(),
            ) {
                (Some(batch), Some(q_proj), Some(k_proj), Some(v_proj)) => {
                    match batch.forward(&[q_proj, k_proj, v_proj], hidden_states) {
                        Ok(mut outputs) if outputs.len() == 3 => {
                            let v = outputs.pop().unwrap();
                            let k = outputs.pop().unwrap();
                            let q = outputs.pop().unwrap();
                            (q, k, v)
                        }
                        _ => self.forward_qkv_decode_fallback(hidden_states)?,
                    }
                }
                (_, Some(q_proj), Some(k_proj), Some(v_proj)) => {
                    match GpuQ8MatVec::forward_many_same_input(
                        &[q_proj, k_proj, v_proj],
                        hidden_states,
                    ) {
                        Ok(mut outputs) if outputs.len() == 3 => {
                            let v = outputs.pop().unwrap();
                            let k = outputs.pop().unwrap();
                            let q = outputs.pop().unwrap();
                            (q, k, v)
                        }
                        _ => self.forward_qkv_decode_fallback(hidden_states)?,
                    }
                }
                _ => self.forward_qkv_decode_fallback(hidden_states)?,
            }
        } else {
            (
                self.q_proj.forward(hidden_states)?,
                self.k_proj.forward(hidden_states)?,
                self.v_proj.forward(hidden_states)?,
            )
        };

        if seq_len == 1 {
            return self.forward_decode_one_fast(q, k, v, kv_cache, layer_idx, position_offset);
        }

        let q = q.reshape(&[seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[seq_len, self.num_kv_heads, self.head_dim])?;

        // Qwen3 QK-Norm：每个 head 沿 head_dim 做 RMSNorm，必须在 RoPE 之前
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = rope_with_inv_freq(&q, &k, position_offset, &self.rope_inv_freq)?;

        // [seq, head, dim] -> [head, seq, dim]
        let q = transpose_for_attention(&q, self.num_heads)?;
        let k = transpose_for_attention(&k, self.num_kv_heads)?;
        let v = transpose_for_attention(&v, self.num_kv_heads)?;

        kv_cache.append(layer_idx, &k, &v)?;
        let cached = kv_cache.get_cached(layer_idx)?;

        // GQA：直接按 query head 映射到对应的 KV head，避免实体复制 cached K/V。
        let kv_group_size = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_product_attention_gqa_cached(
            &q,
            CachedAttention {
                key: cached.key,
                value: cached.value,
                num_kv_heads: cached.num_heads,
                seq_len_k: cached.seq_len,
                head_dim: cached.head_dim,
                max_len: cached.capacity_len,
            },
            kv_group_size,
            scale,
        )?;

        let attn = transpose_back(&attn, seq_len, self.num_heads)?
            .reshape(&[seq_len, self.num_heads * self.head_dim])?;
        self.o_proj.forward(&attn)
    }

    fn forward_decode_one_fast(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
    ) -> Result<Tensor> {
        let mut q_data = q.as_slice().to_vec();
        let mut k_data = k.as_slice().to_vec();
        let v_data = v.as_slice();
        rms_norm_per_head_inplace(
            &mut q_data,
            self.num_heads,
            self.head_dim,
            self.q_norm.weight.as_slice(),
            self.q_norm.eps,
        )?;
        rms_norm_per_head_inplace(
            &mut k_data,
            self.num_kv_heads,
            self.head_dim,
            self.k_norm.weight.as_slice(),
            self.k_norm.eps,
        )?;
        rope_single_token_inplace(
            &mut q_data,
            self.num_heads,
            self.head_dim,
            position_offset,
            &self.rope_inv_freq,
        )?;
        rope_single_token_inplace(
            &mut k_data,
            self.num_kv_heads,
            self.head_dim,
            position_offset,
            &self.rope_inv_freq,
        )?;

        kv_cache.append_decode_one_raw(
            layer_idx,
            self.num_kv_heads,
            self.head_dim,
            &k_data,
            v_data,
        )?;
        let cached = kv_cache.get_cached(layer_idx)?;
        let kv_group_size = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let attn = scaled_dot_product_attention_gqa_cached_decode_one_raw(
            &q_data,
            CachedAttention {
                key: cached.key,
                value: cached.value,
                num_kv_heads: cached.num_heads,
                seq_len_k: cached.seq_len,
                head_dim: cached.head_dim,
                max_len: cached.capacity_len,
            },
            self.num_heads,
            kv_group_size,
            self.head_dim,
            scale,
        )?;
        let attn = Tensor::from_f32_vec(&[1, self.num_heads * self.head_dim], attn)?;
        self.o_proj.forward(&attn)
    }

    #[cfg(test)]
    fn forward_decode_one_resident_prototype(
        &self,
        hidden_states: &Tensor,
        cached_prefix: Option<CachedKV<'_>>,
        position_offset: usize,
    ) -> Result<Tensor> {
        if hidden_states.ndim() != 2 || hidden_states.shape() != [1, self.num_heads * self.head_dim]
        {
            return Err(RsinferError::DimensionError(format!(
                "resident decode-one attention prototype expects [1, {}], got {:?}",
                self.num_heads * self.head_dim,
                hidden_states.shape()
            )));
        }

        let q_proj = self
            .q_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing q_proj Q8 GPU matvec".to_string()))?;
        let context = q_proj.shared_context();

        if !hidden_states.shape()[1].is_multiple_of(self.head_dim) {
            return Err(RsinferError::DimensionError(
                "resident decode-one attention prototype got non-integral head width".into(),
            ));
        }

        let prefix_len = cached_prefix
            .as_ref()
            .map(|cached| cached.seq_len)
            .unwrap_or(0);
        if prefix_len != position_offset {
            return Err(RsinferError::DimensionError(format!(
                "resident decode-one attention prototype expects cached prefix len {} to equal position_offset {}",
                prefix_len, position_offset
            )));
        }

        if let Some(cached) = cached_prefix.as_ref() {
            if cached.num_heads != self.num_kv_heads || cached.head_dim != self.head_dim {
                return Err(RsinferError::DimensionError(format!(
                    "resident decode-one attention prototype cached prefix shape mismatch: got heads={}, head_dim={}, expected heads={}, head_dim={}",
                    cached.num_heads, cached.head_dim, self.num_kv_heads, self.head_dim
                )));
            }
        }

        let resident_output =
            GpuResidentBuffer::with_context(&context, &[1, self.num_heads * self.head_dim])
                .map_err(Self::gpu_error)?;
        let resident_hidden =
            GpuResidentBuffer::from_tensor(&context, hidden_states).map_err(Self::gpu_error)?;

        let mut encoder =
            context.create_command_encoder("rsinfer-resident-attention-prototype-encoder");
        self.encode_decode_one_resident_prototype(
            &mut encoder,
            &resident_hidden,
            &resident_output,
            cached_prefix,
            position_offset,
            None,
            None,
        )?;
        context.submit(encoder);

        resident_output.read_back().map_err(Self::gpu_error)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn encode_decode_one_resident_prototype(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resident_hidden: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
        cached_prefix: Option<CachedKV<'_>>,
        position_offset: usize,
        resident_k_capture: Option<&GpuResidentBuffer>,
        resident_v_capture: Option<&GpuResidentBuffer>,
    ) -> Result<()> {
        let q_proj = self
            .q_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing q_proj Q8 GPU matvec".to_string()))?;
        let k_proj = self
            .k_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing k_proj Q8 GPU matvec".to_string()))?;
        let v_proj = self
            .v_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing v_proj Q8 GPU matvec".to_string()))?;
        let o_proj = self
            .o_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing o_proj Q8 GPU matvec".to_string()))?;
        let context = q_proj.shared_context();
        let prefix_len = cached_prefix
            .as_ref()
            .map(|cached| cached.seq_len)
            .unwrap_or(0);
        let max_len = cached_prefix
            .as_ref()
            .map(|cached| cached.capacity_len)
            .unwrap_or(position_offset + 1)
            .max(position_offset + 1);
        let cache_shape = [self.num_kv_heads, max_len, self.head_dim];
        let cache_values_len = self.num_kv_heads * max_len * self.head_dim;
        let mut key_cache = vec![0.0f32; cache_values_len];
        let mut value_cache = vec![0.0f32; cache_values_len];
        if let Some(cached) = cached_prefix.as_ref() {
            let compact_head_len = cached.seq_len * cached.head_dim;
            let cache_head_len = max_len * cached.head_dim;
            let src_head_len = cached.capacity_len * cached.head_dim;
            for h in 0..cached.num_heads {
                let src_start = h * src_head_len;
                let dst_start = h * cache_head_len;
                key_cache[dst_start..dst_start + compact_head_len]
                    .copy_from_slice(&cached.key[src_start..src_start + compact_head_len]);
                value_cache[dst_start..dst_start + compact_head_len]
                    .copy_from_slice(&cached.value[src_start..src_start + compact_head_len]);
            }
        }
        let resident_q_pre =
            GpuResidentBuffer::with_context(&context, &[1, self.num_heads, self.head_dim])
                .map_err(Self::gpu_error)?;
        let resident_k_pre =
            GpuResidentBuffer::with_context(&context, &[1, self.num_kv_heads, self.head_dim])
                .map_err(Self::gpu_error)?;
        let resident_q =
            GpuResidentBuffer::with_context(&context, &[1, self.num_heads, self.head_dim])
                .map_err(Self::gpu_error)?;
        let owned_k = if resident_k_capture.is_none() {
            Some(
                GpuResidentBuffer::with_context(&context, &[1, self.num_kv_heads, self.head_dim])
                    .map_err(Self::gpu_error)?,
            )
        } else {
            None
        };
        let resident_k = resident_k_capture.unwrap_or_else(|| owned_k.as_ref().unwrap());
        let owned_v = if resident_v_capture.is_none() {
            Some(
                GpuResidentBuffer::with_context(&context, &[1, self.num_kv_heads, self.head_dim])
                    .map_err(Self::gpu_error)?,
            )
        } else {
            None
        };
        let resident_v = resident_v_capture.unwrap_or_else(|| owned_v.as_ref().unwrap());
        let resident_key_cache = GpuResidentBuffer::from_tensor(
            &context,
            &Tensor::from_f32_vec(&cache_shape, key_cache)?,
        )
        .map_err(Self::gpu_error)?;
        let resident_value_cache = GpuResidentBuffer::from_tensor(
            &context,
            &Tensor::from_f32_vec(&cache_shape, value_cache)?,
        )
        .map_err(Self::gpu_error)?;
        let resident_attention =
            GpuResidentBuffer::with_context(&context, &[1, self.num_heads, self.head_dim])
                .map_err(Self::gpu_error)?;
        let qk_gpu = GpuQkRmsNormRope::with_context(
            &context,
            GpuQkRmsNormRopeConfig {
                q_weight: &self.q_norm.weight,
                k_weight: &self.k_norm.weight,
                q_shape: &[1, self.num_heads, self.head_dim],
                k_shape: &[1, self.num_kv_heads, self.head_dim],
                pos: position_offset,
                inv_freq: &self.rope_inv_freq,
                q_eps: self.q_norm.eps,
                k_eps: self.k_norm.eps,
            },
        )
        .map_err(Self::gpu_error)?;
        let mut kv_gpu =
            GpuKvAppend::with_context(&context, self.num_kv_heads, self.head_dim, max_len)
                .map_err(Self::gpu_error)?;
        kv_gpu
            .restore_len(position_offset)
            .map_err(Self::gpu_error)?;
        let attn_gpu = GpuDecodeGqaAttention::with_context(
            &context,
            GpuDecodeGqaAttentionConfig {
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                max_len,
                scale: 1.0 / (self.head_dim as f32).sqrt(),
            },
        )
        .map_err(Self::gpu_error)?;

        if let Some(batch) = &self.q8_qkv_batch {
            batch
                .encode_resident(
                    &[q_proj, k_proj, v_proj],
                    encoder,
                    resident_hidden,
                    &[&resident_q_pre, &resident_k_pre, resident_v],
                )
                .map_err(Self::gpu_error)?;
        } else {
            q_proj
                .encode_resident(encoder, resident_hidden, &resident_q_pre)
                .map_err(Self::gpu_error)?;
            k_proj
                .encode_resident(encoder, resident_hidden, &resident_k_pre)
                .map_err(Self::gpu_error)?;
            v_proj
                .encode_resident(encoder, resident_hidden, resident_v)
                .map_err(Self::gpu_error)?;
        }
        qk_gpu
            .encode_resident(
                encoder,
                &resident_q_pre,
                &resident_k_pre,
                &resident_q,
                resident_k,
            )
            .map_err(Self::gpu_error)?;
        kv_gpu
            .encode_resident(
                encoder,
                prefix_len.max(position_offset),
                resident_k,
                resident_v,
                &resident_key_cache,
                &resident_value_cache,
            )
            .map_err(Self::gpu_error)?;
        attn_gpu
            .encode_resident(
                encoder,
                &resident_q,
                &resident_key_cache,
                &resident_value_cache,
                &resident_attention,
                position_offset,
            )
            .map_err(Self::gpu_error)?;
        o_proj
            .encode_resident(encoder, &resident_attention, resident_output)
            .map_err(Self::gpu_error)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_decode_one_resident_runtime(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resident_hidden: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
        position_offset: usize,
    ) -> Result<()> {
        let q_proj = self
            .q_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing q_proj Q8 GPU matvec".to_string()))?;
        let k_proj = self
            .k_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing k_proj Q8 GPU matvec".to_string()))?;
        let v_proj = self
            .v_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing v_proj Q8 GPU matvec".to_string()))?;
        let o_proj = self
            .o_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing o_proj Q8 GPU matvec".to_string()))?;
        let mut state_slot = self.resident_runtime_state.borrow_mut();
        let state = state_slot.as_mut().ok_or_else(|| {
            Self::gpu_error("resident runtime state should be initialized".to_string())
        })?;

        if let Some(batch) = &self.q8_qkv_batch {
            if state.qkv_batch_cache.is_none() {
                state.qkv_batch_cache = Some(
                    batch
                        .prepare_resident_cache(
                            &[q_proj, k_proj, v_proj],
                            resident_hidden,
                            &[
                                &state.resident_q_pre,
                                &state.resident_k_pre,
                                &state.resident_v,
                            ],
                        )
                        .map_err(Self::gpu_error)?,
                );
            }
            batch
                .encode_resident_input_copy(encoder, resident_hidden)
                .map_err(Self::gpu_error)?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-resident-attention-full-chain-pass"),
                timestamp_writes: None,
            });
            batch
                .encode_resident_with_cache_in_pass(
                    &[q_proj, k_proj, v_proj],
                    resident_hidden,
                    &[
                        &state.resident_q_pre,
                        &state.resident_k_pre,
                        &state.resident_v,
                    ],
                    state
                        .qkv_batch_cache
                        .as_ref()
                        .expect("resident qkv batch cache initialized"),
                    &mut pass,
                )
                .map_err(Self::gpu_error)?;
            state
                .qk_gpu
                .encode_resident_in_pass_cached(
                    &mut pass,
                    &state.resident_q_pre,
                    &state.resident_k_pre,
                    &state.resident_q,
                    &state.resident_k,
                    &state.qk_gpu_cache,
                )
                .map_err(Self::gpu_error)?;
            state
                .kv_gpu
                .encode_resident_in_pass_current_position(
                    &mut pass,
                    position_offset,
                    &state.resident_k,
                    &state.resident_v,
                    &state.resident_key_cache,
                    &state.resident_value_cache,
                    &state.kv_gpu_cache,
                )
                .map_err(Self::gpu_error)?;
            state
                .attn_gpu
                .encode_resident_in_pass_current_position(
                    &mut pass,
                    &state.resident_q,
                    &state.resident_key_cache,
                    &state.resident_value_cache,
                    &state.resident_attention,
                    position_offset,
                    &state.attn_gpu_cache,
                )
                .map_err(Self::gpu_error)?;
        } else {
            q_proj
                .encode_resident(encoder, resident_hidden, &state.resident_q_pre)
                .map_err(Self::gpu_error)?;
            k_proj
                .encode_resident(encoder, resident_hidden, &state.resident_k_pre)
                .map_err(Self::gpu_error)?;
            v_proj
                .encode_resident(encoder, resident_hidden, &state.resident_v)
                .map_err(Self::gpu_error)?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-resident-attention-chain-pass"),
                timestamp_writes: None,
            });
            state
                .qk_gpu
                .encode_resident_in_pass_cached(
                    &mut pass,
                    &state.resident_q_pre,
                    &state.resident_k_pre,
                    &state.resident_q,
                    &state.resident_k,
                    &state.qk_gpu_cache,
                )
                .map_err(Self::gpu_error)?;
            state
                .kv_gpu
                .encode_resident_in_pass_current_position(
                    &mut pass,
                    position_offset,
                    &state.resident_k,
                    &state.resident_v,
                    &state.resident_key_cache,
                    &state.resident_value_cache,
                    &state.kv_gpu_cache,
                )
                .map_err(Self::gpu_error)?;
            state
                .attn_gpu
                .encode_resident_in_pass_current_position(
                    &mut pass,
                    &state.resident_q,
                    &state.resident_key_cache,
                    &state.resident_value_cache,
                    &state.resident_attention,
                    position_offset,
                    &state.attn_gpu_cache,
                )
                .map_err(Self::gpu_error)?;
        }
        if state.o_proj_gpu_cache.is_none() {
            state.o_proj_gpu_cache = Some(
                o_proj
                    .prepare_resident_cache(&state.resident_attention, resident_output)
                    .map_err(Self::gpu_error)?,
            );
        }
        o_proj
            .encode_resident_with_cache(
                encoder,
                &state.resident_attention,
                resident_output,
                state
                    .o_proj_gpu_cache
                    .as_ref()
                    .expect("resident o_proj cache initialized"),
            )
            .map_err(Self::gpu_error)?;
        Ok(())
    }

    fn build_resident_cache_tensors(
        &self,
        cached_prefix: Option<CachedKV<'_>>,
        max_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cache_shape = [self.num_kv_heads, max_len, self.head_dim];
        let cache_values_len = self.num_kv_heads * max_len * self.head_dim;
        let mut key_cache = vec![0.0f32; cache_values_len];
        let mut value_cache = vec![0.0f32; cache_values_len];
        if let Some(cached) = cached_prefix {
            if cached.num_heads != self.num_kv_heads || cached.head_dim != self.head_dim {
                return Err(RsinferError::DimensionError(format!(
                    "resident attention cached prefix shape mismatch: got heads={}, head_dim={}, expected heads={}, head_dim={}",
                    cached.num_heads, cached.head_dim, self.num_kv_heads, self.head_dim
                )));
            }
            let compact_head_len = cached.seq_len * cached.head_dim;
            let cache_head_len = max_len * cached.head_dim;
            let src_head_len = cached.capacity_len * cached.head_dim;
            for h in 0..cached.num_heads {
                let src_start = h * src_head_len;
                let dst_start = h * cache_head_len;
                key_cache[dst_start..dst_start + compact_head_len]
                    .copy_from_slice(&cached.key[src_start..src_start + compact_head_len]);
                value_cache[dst_start..dst_start + compact_head_len]
                    .copy_from_slice(&cached.value[src_start..src_start + compact_head_len]);
            }
        }
        Ok((
            Tensor::from_f32_vec(&cache_shape, key_cache)?,
            Tensor::from_f32_vec(&cache_shape, value_cache)?,
        ))
    }

    fn ensure_resident_runtime_state(
        &self,
        context: &GpuContext,
        kv_cache: &KVCache,
        cached_prefix: Option<CachedKV<'_>>,
        position_offset: usize,
        position_buffer: &wgpu::Buffer,
    ) -> Result<()> {
        let desired_max_len = cached_prefix
            .as_ref()
            .map(|cached| cached.capacity_len.max(position_offset + 1))
            .unwrap_or_else(|| (position_offset + 1).max(64))
            .min(kv_cache.max_len());
        let mut state_slot = self.resident_runtime_state.borrow_mut();
        let needs_rebuild = state_slot
            .as_ref()
            .map(|state| state.max_len < desired_max_len)
            .unwrap_or(true);
        if needs_rebuild {
            *state_slot = Some(
                ResidentAttentionRuntimeState::new(
                    context,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    desired_max_len,
                    position_buffer,
                    &self.q_norm.weight,
                    &self.k_norm.weight,
                    &self.rope_inv_freq,
                    self.q_norm.eps,
                    self.k_norm.eps,
                )
                .map_err(Self::gpu_error)?,
            );
        }
        let state = state_slot
            .as_mut()
            .expect("resident runtime state should exist");
        let sync_from_cpu = state.synced_cache_id != kv_cache.cache_id()
            || state.synced_cache_revision != kv_cache.revision();
        if sync_from_cpu {
            let (key_cache, value_cache) =
                self.build_resident_cache_tensors(cached_prefix, state.max_len)?;
            state
                .resident_key_cache
                .upload(&key_cache)
                .map_err(Self::gpu_error)?;
            state
                .resident_value_cache
                .upload(&value_cache)
                .map_err(Self::gpu_error)?;
        }
        state
            .kv_gpu
            .restore_len(position_offset)
            .map_err(Self::gpu_error)?;
        state.synced_cache_id = kv_cache.cache_id();
        state.synced_cache_revision = kv_cache.revision();
        Ok(())
    }

    fn forward_qkv_decode_fallback(
        &self,
        hidden_states: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        match (
            self.q_proj.gpu_matvec(),
            self.k_proj.gpu_matvec(),
            self.v_proj.gpu_matvec(),
        ) {
            (Some(q_proj), Some(k_proj), Some(v_proj)) => {
                match GpuMatVec::forward_many_same_input(&[q_proj, k_proj, v_proj], hidden_states) {
                    Ok(mut outputs) if outputs.len() == 3 => {
                        let v = outputs.pop().unwrap();
                        let k = outputs.pop().unwrap();
                        let q = outputs.pop().unwrap();
                        Ok((q, k, v))
                    }
                    _ => Ok((
                        self.q_proj.forward(hidden_states)?,
                        self.k_proj.forward(hidden_states)?,
                        self.v_proj.forward(hidden_states)?,
                    )),
                }
            }
            _ => Ok((
                self.q_proj.forward(hidden_states)?,
                self.k_proj.forward(hidden_states)?,
                self.v_proj.forward(hidden_states)?,
            )),
        }
    }
}

/// 将 [seq_len, num_heads, head_dim] 转置为 [num_heads, seq_len, head_dim]
fn transpose_for_attention(x: &Tensor, num_heads: usize) -> Result<Tensor> {
    let shape = x.shape();
    let seq_len = shape[0];
    let head_dim = shape[2];
    if seq_len == 1 {
        return x.reshape(&[num_heads, 1, head_dim]);
    }

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
    if seq_len == 1 {
        return x.reshape(&[1, num_heads, head_dim]);
    }

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

fn rope_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

/// MLP 层 (SwiGLU)
///
/// output = down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
    gpu_swiglu_down: Option<GpuSwiGluDown>,
    q8_gpu_swiglu_down: Option<GpuQ8SwiGluDown>,
}

impl Mlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            gpu_swiglu_down: None,
            q8_gpu_swiglu_down: None,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, None)
    }

    pub fn forward_profiled(
        &self,
        x: &Tensor,
        profile: &mut TransformerBlockProfile,
    ) -> Result<Tensor> {
        self.forward_inner(x, Some(profile))
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        mut profile: Option<&mut TransformerBlockProfile>,
    ) -> Result<Tensor> {
        if x.shape()[0] == 1 {
            if let (Some(gate_proj), Some(up_proj), Some(down_proj)) = (
                self.gate_proj.q8_gpu_matvec(),
                self.up_proj.q8_gpu_matvec(),
                self.down_proj.q8_gpu_matvec(),
            ) {
                if let Some(fused) = &self.q8_gpu_swiglu_down {
                    if let Ok(output) = fused.forward(gate_proj, up_proj, down_proj, x) {
                        return Ok(output);
                    }
                }
            }
            if let (Some(gate_proj), Some(up_proj), Some(down_proj)) = (
                self.gate_proj.gpu_matvec(),
                self.up_proj.gpu_matvec(),
                self.down_proj.gpu_matvec(),
            ) {
                if let Some(fused) = &self.gpu_swiglu_down {
                    if let Ok(output) = fused.forward(gate_proj, up_proj, down_proj, x) {
                        return Ok(output);
                    }
                }
            }
        }

        let start = profile.as_ref().map(|_| Instant::now());
        let (gate, up) = if x.shape()[0] == 1 {
            match (self.gate_proj.gpu_matvec(), self.up_proj.gpu_matvec()) {
                (Some(gate_proj), Some(up_proj)) => {
                    match GpuMatVec::forward_many_same_input(&[gate_proj, up_proj], x) {
                        Ok(mut outputs) if outputs.len() == 2 => {
                            let up = outputs.pop().unwrap();
                            let gate = outputs.pop().unwrap();
                            (gate, up)
                        }
                        _ => (self.gate_proj.forward(x)?, self.up_proj.forward(x)?),
                    }
                }
                _ if self.gate_proj.has_cpu_q8_only() && self.up_proj.has_cpu_q8_only() => {
                    if let (Some(profile_ref), Some(gate_weight), Some(up_weight)) = (
                        profile.as_deref_mut(),
                        self.gate_proj.q8_weight.as_ref(),
                        self.up_proj.q8_weight.as_ref(),
                    ) {
                        let ((gate, gate_profile), (up, up_profile)) = join(
                            || {
                                let mut gate_profile = Q8LinearProfile::default();
                                let gate =
                                    linear_forward_q8_profiled(x, gate_weight, &mut gate_profile);
                                (gate, gate_profile)
                            },
                            || {
                                let mut up_profile = Q8LinearProfile::default();
                                let up = linear_forward_q8_profiled(x, up_weight, &mut up_profile);
                                (up, up_profile)
                            },
                        );
                        profile_ref.mlp_q8_gate_up_prep += gate_profile.prep + up_profile.prep;
                        profile_ref.mlp_q8_gate_up_dot += gate_profile.dot + up_profile.dot;
                        profile_ref.mlp_q8_gate_up_writeback +=
                            gate_profile.writeback + up_profile.writeback;
                        (gate?, up?)
                    } else {
                        let (gate, up) =
                            join(|| self.gate_proj.forward(x), || self.up_proj.forward(x));
                        (gate?, up?)
                    }
                }
                _ => (self.gate_proj.forward(x)?, self.up_proj.forward(x)?),
            }
        } else {
            (self.gate_proj.forward(x)?, self.up_proj.forward(x)?)
        };
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.mlp_gate_up += start.elapsed();
        }

        let start = profile.as_ref().map(|_| Instant::now());
        let gate = silu(&gate);
        let hidden = gate.mul(&up)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.mlp_silu_mul += start.elapsed();
        }

        let start = profile.as_ref().map(|_| Instant::now());
        let output = if let (Some(profile_ref), Some(down_weight)) =
            (profile.as_deref_mut(), self.down_proj.q8_weight.as_ref())
        {
            if self.down_proj.has_cpu_q8_only() {
                let mut down_profile = Q8LinearProfile::default();
                let output = linear_forward_q8_profiled(&hidden, down_weight, &mut down_profile)?;
                profile_ref.mlp_q8_down_proj_prep += down_profile.prep;
                profile_ref.mlp_q8_down_proj_dot += down_profile.dot;
                profile_ref.mlp_q8_down_proj_writeback += down_profile.writeback;
                output
            } else {
                self.down_proj.forward(&hidden)?
            }
        } else {
            self.down_proj.forward(&hidden)?
        };
        if let (Some(profile), Some(start)) = (profile, start) {
            profile.mlp_down_proj += start.elapsed();
        }
        Ok(output)
    }

    pub fn try_enable_q8_weights(
        &mut self,
        mut sidecar: Option<&mut Q8SidecarCache>,
    ) -> Result<usize> {
        let mut attached = 0usize;
        for linear in [&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj] {
            match sidecar.as_deref_mut() {
                Some(cache) => linear.try_enable_q8_weight_with_sidecar(Some(cache))?,
                None => linear.try_enable_q8_weight_with_sidecar(None)?,
            }
            attached += 1;
        }
        Ok(attached)
    }

    pub fn try_enable_gpu_matvecs(&mut self, context: &GpuContext) -> (usize, Vec<String>) {
        let mut attached = 0usize;
        let mut errors = Vec::new();
        for (name, linear) in [
            ("gate_proj", &mut self.gate_proj),
            ("up_proj", &mut self.up_proj),
            ("down_proj", &mut self.down_proj),
        ] {
            match linear.try_enable_gpu_matvec_with_context(context) {
                Ok(()) => attached += 1,
                Err(err) => errors.push(format!("mlp.{name}: {err}")),
            }
        }
        self.gpu_swiglu_down = match (
            self.gate_proj.gpu_matvec(),
            self.up_proj.gpu_matvec(),
            self.down_proj.gpu_matvec(),
        ) {
            (Some(gate_proj), Some(up_proj), Some(down_proj)) => {
                match GpuSwiGluDown::new(gate_proj, up_proj, down_proj) {
                    Ok(fused) => Some(fused),
                    Err(err) => {
                        errors.push(format!("mlp.swiglu_down: {err}"));
                        None
                    }
                }
            }
            _ => None,
        };
        (attached, errors)
    }

    pub fn try_enable_q8_gpu_matvecs(&mut self, context: &GpuContext) -> (usize, Vec<String>) {
        let mut attached = 0usize;
        let mut errors = Vec::new();
        for (name, linear) in [
            ("gate_proj", &mut self.gate_proj),
            ("up_proj", &mut self.up_proj),
            ("down_proj", &mut self.down_proj),
        ] {
            let enable_argmax = false;
            match linear.try_enable_q8_gpu_matvec_with_context_and_argmax(context, enable_argmax) {
                Ok(()) => attached += 1,
                Err(err) => errors.push(format!("mlp.{name}.q8_gpu: {err}")),
            }
        }
        self.q8_gpu_swiglu_down = match (
            self.gate_proj.q8_gpu_matvec(),
            self.up_proj.q8_gpu_matvec(),
            self.down_proj.q8_gpu_matvec(),
        ) {
            (Some(gate_proj), Some(up_proj), Some(down_proj)) => {
                match GpuQ8SwiGluDown::new(gate_proj, up_proj, down_proj) {
                    Ok(fused) => {
                        self.gate_proj
                            .release_q8_gpu_standalone_forward_resources(false);
                        self.up_proj
                            .release_q8_gpu_standalone_forward_resources(false);
                        Some(fused)
                    }
                    Err(err) => {
                        errors.push(format!("mlp.q8_swiglu_down: {err}"));
                        None
                    }
                }
            }
            _ => None,
        };
        (attached, errors)
    }

    fn gpu_error(err: String) -> RsinferError {
        RsinferError::DimensionError(format!("GPU resident MLP prototype failed: {err}"))
    }

    #[cfg(test)]
    fn encode_resident_prototype(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resident_input: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
    ) -> Result<()> {
        let gate = self
            .gate_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing gate_proj Q8 GPU matvec".to_string()))?;
        let hidden = GpuResidentBuffer::with_context(
            &gate.shared_context(),
            &[1, self.gate_proj.out_features],
        )
        .map_err(Self::gpu_error)?;
        self.encode_resident_prototype_with_hidden(
            encoder,
            resident_input,
            &hidden,
            resident_output,
        )
    }

    #[cfg(test)]
    fn encode_resident_prototype_with_hidden(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resident_input: &GpuResidentBuffer,
        resident_hidden: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
    ) -> Result<()> {
        self.encode_resident_prototype_with_hidden_cache(
            encoder,
            resident_input,
            resident_hidden,
            resident_output,
            None,
        )
    }

    #[cfg(test)]
    fn encode_resident_prototype_with_hidden_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resident_input: &GpuResidentBuffer,
        resident_hidden: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
        resident_cache: Option<&crate::gpu::GpuQ8SwiGluDownResidentCache>,
    ) -> Result<()> {
        let gate = self
            .gate_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing gate_proj Q8 GPU matvec".to_string()))?;
        let up = self
            .up_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing up_proj Q8 GPU matvec".to_string()))?;
        let down = self
            .down_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing down_proj Q8 GPU matvec".to_string()))?;
        let fused = self
            .q8_gpu_swiglu_down
            .as_ref()
            .ok_or_else(|| Self::gpu_error("missing fused Q8 SwiGLU helper".to_string()))?;
        match resident_cache {
            Some(cache) => fused
                .encode_resident_with_hidden_cache(
                    gate,
                    up,
                    down,
                    encoder,
                    resident_input,
                    resident_output,
                    cache,
                )
                .map_err(Self::gpu_error),
            None => fused
                .encode_resident_with_hidden(
                    gate,
                    up,
                    down,
                    encoder,
                    resident_input,
                    resident_hidden,
                    resident_output,
                )
                .map_err(Self::gpu_error),
        }
    }

    fn encode_resident_prototype_with_hidden_cache_in_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        resident_input: &GpuResidentBuffer,
        _resident_hidden: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
        resident_cache: Option<&crate::gpu::GpuQ8SwiGluDownResidentCache>,
    ) -> Result<()> {
        let gate = self
            .gate_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing gate_proj Q8 GPU matvec".to_string()))?;
        let up = self
            .up_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing up_proj Q8 GPU matvec".to_string()))?;
        let down = self
            .down_proj
            .q8_gpu_matvec()
            .ok_or_else(|| Self::gpu_error("missing down_proj Q8 GPU matvec".to_string()))?;
        let fused = self
            .q8_gpu_swiglu_down
            .as_ref()
            .ok_or_else(|| Self::gpu_error("missing fused Q8 SwiGLU helper".to_string()))?;
        let cache = resident_cache
            .ok_or_else(|| Self::gpu_error("missing resident fused Q8 SwiGLU cache".to_string()))?;
        fused
            .encode_resident_with_hidden_cache_in_pass(
                gate,
                up,
                down,
                pass,
                resident_input,
                resident_output,
                cache,
            )
            .map_err(Self::gpu_error)
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
    resident_runtime_state: RefCell<Option<ResidentTransformerBlockRuntimeState>>,
}

#[derive(Clone, Debug, Default)]
pub struct TransformerBlockProfile {
    pub input_norm: Duration,
    pub attention: Duration,
    pub post_norm: Duration,
    pub mlp: Duration,
    pub mlp_gate_up: Duration,
    pub mlp_silu_mul: Duration,
    pub mlp_down_proj: Duration,
    pub mlp_q8_gate_up_prep: Duration,
    pub mlp_q8_gate_up_dot: Duration,
    pub mlp_q8_gate_up_writeback: Duration,
    pub mlp_q8_down_proj_prep: Duration,
    pub mlp_q8_down_proj_dot: Duration,
    pub mlp_q8_down_proj_writeback: Duration,
    pub residual: Duration,
    pub resident_prefix_encode: Duration,
    pub resident_prefix_submit: Duration,
    pub resident_prefix_readback: Duration,
    pub total: Duration,
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
            resident_runtime_state: RefCell::new(None),
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
        self.forward_inner(hidden_states, kv_cache, layer_idx, position_offset, None)
    }

    pub fn forward_profiled(
        &self,
        hidden_states: &Tensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
        profile: &mut TransformerBlockProfile,
    ) -> Result<Tensor> {
        self.forward_inner(
            hidden_states,
            kv_cache,
            layer_idx,
            position_offset,
            Some(profile),
        )
    }

    fn forward_inner(
        &self,
        hidden_states: &Tensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
        mut profile: Option<&mut TransformerBlockProfile>,
    ) -> Result<Tensor> {
        let total_start = profile.as_ref().map(|_| Instant::now());

        // RMSNorm -> Attention
        let start = profile.as_ref().map(|_| Instant::now());
        let normed = self.input_layernorm.forward(hidden_states)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.input_norm += start.elapsed();
        }

        let start = profile.as_ref().map(|_| Instant::now());
        let attn_output = self
            .attention
            .forward(&normed, kv_cache, layer_idx, position_offset)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.attention += start.elapsed();
        }

        // Residual connection
        let start = profile.as_ref().map(|_| Instant::now());
        let hidden_states = hidden_states.add(&attn_output)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.residual += start.elapsed();
        }

        // RMSNorm -> MLP
        let start = profile.as_ref().map(|_| Instant::now());
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.post_norm += start.elapsed();
        }

        let start = profile.as_ref().map(|_| Instant::now());
        let mlp_output = if let Some(profile_ref) = profile.as_deref_mut() {
            self.mlp.forward_profiled(&normed, profile_ref)?
        } else {
            self.mlp.forward(&normed)?
        };
        if let (Some(profile_ref), Some(start)) = (profile.as_deref_mut(), start) {
            profile_ref.mlp += start.elapsed();
        }

        // Residual connection
        let start = profile.as_ref().map(|_| Instant::now());
        let output = hidden_states.add(&mlp_output)?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.residual += start.elapsed();
        }
        if let (Some(profile), Some(start)) = (profile, total_start) {
            profile.total += start.elapsed();
        }
        Ok(output)
    }

    pub fn try_enable_gpu_matvecs(&mut self, context: &GpuContext) -> (usize, Vec<String>) {
        let (attn_count, mut errors) = self.attention.try_enable_gpu_matvecs(context);
        let (mlp_count, mlp_errors) = self.mlp.try_enable_gpu_matvecs(context);
        errors.extend(mlp_errors);
        (attn_count + mlp_count, errors)
    }

    pub fn try_enable_q8_gpu_matvecs(&mut self, context: &GpuContext) -> (usize, Vec<String>) {
        let (attn_count, mut errors) = self.attention.try_enable_q8_gpu_matvecs(context);
        let (mlp_count, mlp_errors) = self.mlp.try_enable_q8_gpu_matvecs(context);
        errors.extend(mlp_errors);
        (attn_count + mlp_count, errors)
    }

    pub fn try_enable_q8_weights(&mut self, sidecar: Option<&mut Q8SidecarCache>) -> Result<usize> {
        let (attention, mlp) = if let Some(cache) = sidecar {
            (
                self.attention.try_enable_q8_weights(Some(&mut *cache))?,
                self.mlp.try_enable_q8_weights(Some(cache))?,
            )
        } else {
            (
                self.attention.try_enable_q8_weights(None)?,
                self.mlp.try_enable_q8_weights(None)?,
            )
        };
        Ok(attention + mlp)
    }

    fn gpu_error(err: String) -> RsinferError {
        RsinferError::DimensionError(format!(
            "GPU resident transformer block prototype failed: {err}"
        ))
    }

    fn ensure_resident_runtime_state(
        &self,
        context: &GpuContext,
        hidden_size: usize,
    ) -> Result<()> {
        let mut state_slot = self.resident_runtime_state.borrow_mut();
        if state_slot.is_none() {
            *state_slot = Some(ResidentTransformerBlockRuntimeState::new(
                context,
                self,
                hidden_size,
            )?);
        }
        Ok(())
    }

    fn ensure_resident_runtime_state_with_slots(
        &self,
        context: &GpuContext,
        hidden_size: usize,
        slot0: &GpuResidentBuffer,
        slot1: &GpuResidentBuffer,
    ) -> Result<()> {
        self.ensure_resident_runtime_state(context, hidden_size)?;
        let mut state_slot = self.resident_runtime_state.borrow_mut();
        let state = state_slot
            .as_mut()
            .expect("resident block runtime state should be initialized");
        state.ensure_slot_caches(slot0, slot1)?;
        Ok(())
    }

    #[cfg(test)]
    fn forward_decode_one_resident_prototype(
        &self,
        hidden_states: &Tensor,
        cached_prefix: Option<CachedKV<'_>>,
        position_offset: usize,
    ) -> Result<Tensor> {
        if hidden_states.ndim() != 2 || hidden_states.shape()[0] != 1 {
            return Err(Self::gpu_error(format!(
                "resident transformer block prototype expects [1, hidden], got {:?}",
                hidden_states.shape()
            )));
        }
        let hidden_size = hidden_states.shape()[1];
        let q_proj =
            self.attention.q_proj.q8_gpu_matvec().ok_or_else(|| {
                Self::gpu_error("missing attention q_proj Q8 GPU matvec".to_string())
            })?;
        let context = q_proj.shared_context();

        let resident_input =
            GpuResidentBuffer::from_tensor(&context, hidden_states).map_err(Self::gpu_error)?;
        let resident_norm1 = GpuResidentBuffer::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;
        let resident_attn = GpuResidentBuffer::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;
        let resident_hidden = GpuResidentBuffer::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;
        let resident_norm2 = GpuResidentBuffer::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;
        let resident_mlp = GpuResidentBuffer::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;
        let resident_output = GpuResidentBuffer::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;

        let gpu_input_norm = crate::gpu::GpuRmsNorm::with_context(
            &context,
            &self.input_layernorm.weight,
            &[1, hidden_size],
            self.input_layernorm.eps,
        )
        .map_err(Self::gpu_error)?;
        let gpu_post_norm = crate::gpu::GpuRmsNorm::with_context(
            &context,
            &self.post_attention_layernorm.weight,
            &[1, hidden_size],
            self.post_attention_layernorm.eps,
        )
        .map_err(Self::gpu_error)?;
        let gpu_add = crate::gpu::GpuResidualAdd::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;

        let mut encoder =
            context.create_command_encoder("rsinfer-resident-transformer-block-prototype-encoder");
        gpu_input_norm
            .encode_resident(&mut encoder, &resident_input, &resident_norm1)
            .map_err(Self::gpu_error)?;
        self.attention.encode_decode_one_resident_prototype(
            &mut encoder,
            &resident_norm1,
            &resident_attn,
            cached_prefix,
            position_offset,
            None,
            None,
        )?;
        gpu_add
            .encode_resident(
                &mut encoder,
                &resident_input,
                &resident_attn,
                &resident_hidden,
            )
            .map_err(Self::gpu_error)?;
        gpu_post_norm
            .encode_resident(&mut encoder, &resident_hidden, &resident_norm2)
            .map_err(Self::gpu_error)?;
        self.mlp
            .encode_resident_prototype(&mut encoder, &resident_norm2, &resident_mlp)?;
        gpu_add
            .encode_resident(
                &mut encoder,
                &resident_hidden,
                &resident_mlp,
                &resident_output,
            )
            .map_err(Self::gpu_error)?;
        context.submit(encoder);

        resident_output.read_back().map_err(Self::gpu_error)
    }

    #[cfg(test)]
    pub(crate) fn forward_decode_one_resident_runtime(
        &self,
        hidden_states: &Tensor,
        kv_cache: &mut KVCache,
        layer_idx: usize,
        position_offset: usize,
    ) -> Result<Tensor> {
        if hidden_states.ndim() != 2 || hidden_states.shape()[0] != 1 {
            return Err(Self::gpu_error(format!(
                "resident transformer block runtime expects [1, hidden], got {:?}",
                hidden_states.shape()
            )));
        }
        let q_proj =
            self.attention.q_proj.q8_gpu_matvec().ok_or_else(|| {
                Self::gpu_error("missing attention q_proj Q8 GPU matvec".to_string())
            })?;
        let context = q_proj.shared_context();

        let resident_input =
            GpuResidentBuffer::from_tensor(&context, hidden_states).map_err(Self::gpu_error)?;
        let hidden_size = hidden_states.shape()[1];
        let resident_output = GpuResidentBuffer::with_context(&context, &[1, hidden_size])
            .map_err(Self::gpu_error)?;

        let mut encoder =
            context.create_command_encoder("rsinfer-resident-transformer-block-runtime-encoder");
        self.encode_decode_one_resident_runtime(
            &mut encoder,
            &resident_input,
            &resident_output,
            kv_cache,
            layer_idx,
            position_offset,
        )?;
        context.submit(encoder);

        if layer_idx == 0 {
            kv_cache.set_current_len(position_offset + 1)?;
        }
        resident_output.read_back().map_err(Self::gpu_error)
    }

    #[cfg(test)]
    pub(crate) fn encode_decode_one_resident_runtime(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resident_input: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
        kv_cache: &KVCache,
        layer_idx: usize,
        position_offset: usize,
    ) -> Result<()> {
        let hidden_size = resident_input.len();
        let q_proj =
            self.attention.q_proj.q8_gpu_matvec().ok_or_else(|| {
                Self::gpu_error("missing attention q_proj Q8 GPU matvec".to_string())
            })?;
        let context = q_proj.shared_context();
        let cached_prefix = if position_offset == 0 {
            None
        } else {
            Some(kv_cache.get_cached(layer_idx)?)
        };
        self.ensure_resident_runtime_state(&context, hidden_size)?;
        let state_slot = self.resident_runtime_state.borrow();
        let state = state_slot
            .as_ref()
            .expect("resident block runtime state should be initialized");
        state.resident_position.write_position(position_offset);
        self.attention.ensure_resident_runtime_state(
            &context,
            kv_cache,
            cached_prefix,
            position_offset,
            state.resident_position.buffer(),
        )?;

        resident_input
            .encode_copy_to(encoder, &state.resident_input)
            .map_err(Self::gpu_error)?;
        state
            .gpu_input_norm
            .encode_resident_with_cache(
                encoder,
                &state.resident_input,
                &state.resident_norm1,
                &state.gpu_input_norm_cache,
            )
            .map_err(Self::gpu_error)?;
        self.attention.encode_decode_one_resident_runtime(
            encoder,
            &state.resident_norm1,
            &state.resident_attn,
            position_offset,
        )?;
        state
            .gpu_add
            .encode_resident_with_cache(
                encoder,
                &state.resident_input,
                &state.resident_attn,
                &state.resident_hidden,
                &state.gpu_input_residual_cache,
            )
            .map_err(Self::gpu_error)?;
        state
            .gpu_post_norm
            .encode_resident_with_cache(
                encoder,
                &state.resident_hidden,
                &state.resident_norm2,
                &state.gpu_post_norm_cache,
            )
            .map_err(Self::gpu_error)?;
        self.mlp.encode_resident_prototype_with_hidden_cache(
            encoder,
            &state.resident_norm2,
            &state.resident_mlp_hidden,
            &state.resident_mlp_output,
            state.resident_mlp_q8_cache.as_ref(),
        )?;
        state
            .gpu_add
            .encode_resident_with_cache(
                encoder,
                &state.resident_hidden,
                &state.resident_mlp_output,
                &state.resident_output,
                &state.gpu_output_residual_cache,
            )
            .map_err(Self::gpu_error)?;
        state
            .resident_output
            .encode_copy_to(encoder, resident_output)
            .map_err(Self::gpu_error)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_decode_one_resident_runtime_with_slot_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resident_input: &GpuResidentBuffer,
        resident_output: &GpuResidentBuffer,
        resident_slot0: &GpuResidentBuffer,
        resident_slot1: &GpuResidentBuffer,
        kv_cache: &KVCache,
        layer_idx: usize,
        position_offset: usize,
        position_buffer: &wgpu::Buffer,
    ) -> Result<()> {
        let hidden_size = resident_input.len();
        let q_proj =
            self.attention.q_proj.q8_gpu_matvec().ok_or_else(|| {
                Self::gpu_error("missing attention q_proj Q8 GPU matvec".to_string())
            })?;
        let context = q_proj.shared_context();
        let cached_prefix = if position_offset == 0 {
            None
        } else {
            Some(kv_cache.get_cached(layer_idx)?)
        };
        self.ensure_resident_runtime_state_with_slots(
            &context,
            hidden_size,
            resident_slot0,
            resident_slot1,
        )?;
        self.attention.ensure_resident_runtime_state(
            &context,
            kv_cache,
            cached_prefix,
            position_offset,
            position_buffer,
        )?;
        let state_slot = self.resident_runtime_state.borrow();
        let state = state_slot
            .as_ref()
            .expect("resident block runtime state should be initialized");
        let slot_caches = state
            .slot_caches
            .as_ref()
            .expect("resident block slot caches should be initialized");
        let io_caches = if layer_idx.is_multiple_of(2) {
            &slot_caches.even
        } else {
            &slot_caches.odd
        };

        state
            .gpu_input_norm
            .encode_resident_with_cache(
                encoder,
                resident_input,
                &state.resident_norm1,
                &io_caches.input_norm_cache,
            )
            .map_err(Self::gpu_error)?;
        self.attention.encode_decode_one_resident_runtime(
            encoder,
            &state.resident_norm1,
            &state.resident_attn,
            position_offset,
        )?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-resident-post-attention-chain-pass"),
                timestamp_writes: None,
            });
            state
                .gpu_add
                .encode_resident_with_cache_in_pass(
                    &mut pass,
                    resident_input,
                    &state.resident_attn,
                    &state.resident_hidden,
                    &io_caches.input_residual_cache,
                )
                .map_err(Self::gpu_error)?;
            state
                .gpu_post_norm
                .encode_resident_with_cache_in_pass(
                    &mut pass,
                    &state.resident_hidden,
                    &state.resident_norm2,
                    &state.gpu_post_norm_cache,
                )
                .map_err(Self::gpu_error)?;
            self.mlp
                .encode_resident_prototype_with_hidden_cache_in_pass(
                    &mut pass,
                    &state.resident_norm2,
                    &state.resident_mlp_hidden,
                    &state.resident_mlp_output,
                    state.resident_mlp_q8_cache.as_ref(),
                )?;
            state
                .gpu_add
                .encode_resident_with_cache_in_pass(
                    &mut pass,
                    &state.resident_hidden,
                    &state.resident_mlp_output,
                    resident_output,
                    &io_caches.output_residual_cache,
                )
                .map_err(Self::gpu_error)?;
        }
        Ok(())
    }
}

pub fn build_transformer_block(
    weights: &WeightMap,
    config: &Qwen3Config,
    layer_idx: usize,
) -> Result<TransformerBlock> {
    build_transformer_block_with_q8_sidecar(weights, config, layer_idx, None)
}

pub fn build_transformer_block_with_q8_sidecar(
    weights: &WeightMap,
    config: &Qwen3Config,
    layer_idx: usize,
    sidecar: Option<&mut Q8SidecarCache>,
) -> Result<TransformerBlock> {
    let prefix = format!("model.layers.{layer_idx}");
    let get = |suffix: &str| get_weight(weights, &format!("{prefix}.{suffix}"));
    let eps = config.rms_norm_eps;
    let (attention, mlp) = if let Some(cache) = sidecar {
        (
            Attention {
                q_proj: Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    None,
                    Some(&mut *cache),
                )?,
                k_proj: Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    None,
                    Some(&mut *cache),
                )?,
                v_proj: Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    None,
                    Some(&mut *cache),
                )?,
                o_proj: Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    None,
                    Some(&mut *cache),
                )?,
                q_norm: RmsNorm::new(get("self_attn.q_norm.weight")?, eps),
                k_norm: RmsNorm::new(get("self_attn.k_norm.weight")?, eps),
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim(),
                rope_theta: config.rope_theta,
                rope_inv_freq: rope_inv_freq(config.head_dim(), config.rope_theta),
                q8_qkv_batch: None,
                resident_runtime_state: RefCell::new(None),
            },
            Mlp::new(
                Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    None,
                    Some(&mut *cache),
                )?,
                Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    None,
                    Some(&mut *cache),
                )?,
                Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    None,
                    Some(&mut *cache),
                )?,
            ),
        )
    } else {
        let linear = |suffix: &str| -> Result<Linear> {
            Linear::from_weight_map(weights, &format!("{prefix}.{suffix}"), None)
        };
        (
            Attention {
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
                rope_inv_freq: rope_inv_freq(config.head_dim(), config.rope_theta),
                q8_qkv_batch: None,
                resident_runtime_state: RefCell::new(None),
            },
            Mlp::new(
                linear("mlp.gate_proj.weight")?,
                linear("mlp.up_proj.weight")?,
                linear("mlp.down_proj.weight")?,
            ),
        )
    };

    Ok(TransformerBlock::new(
        RmsNorm::new(get("input_layernorm.weight")?, eps),
        attention,
        RmsNorm::new(get("post_attention_layernorm.weight")?, eps),
        mlp,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{gpu_test_guard, reset_sync_stats, sync_stats};

    fn build_test_attention() -> Attention {
        let q_proj = Linear::from_f16_weight(
            &[4, 4],
            vec![
                f16::from_f32(0.5),
                f16::from_f32(-1.0),
                f16::from_f32(0.75),
                f16::from_f32(0.25),
                f16::from_f32(-0.5),
                f16::from_f32(0.4),
                f16::from_f32(1.1),
                f16::from_f32(-0.3),
                f16::from_f32(0.2),
                f16::from_f32(0.8),
                f16::from_f32(-0.6),
                f16::from_f32(1.0),
                f16::from_f32(-0.7),
                f16::from_f32(0.1),
                f16::from_f32(0.3),
                f16::from_f32(0.9),
            ],
            None,
        )
        .unwrap();
        let k_proj = Linear::from_f16_weight(
            &[2, 4],
            vec![
                f16::from_f32(0.6),
                f16::from_f32(-0.2),
                f16::from_f32(0.4),
                f16::from_f32(0.9),
                f16::from_f32(-0.8),
                f16::from_f32(0.5),
                f16::from_f32(0.7),
                f16::from_f32(-0.1),
            ],
            None,
        )
        .unwrap();
        let v_proj = Linear::from_f16_weight(
            &[2, 4],
            vec![
                f16::from_f32(0.3),
                f16::from_f32(0.7),
                f16::from_f32(-0.5),
                f16::from_f32(0.2),
                f16::from_f32(-0.4),
                f16::from_f32(1.0),
                f16::from_f32(0.6),
                f16::from_f32(-0.9),
            ],
            None,
        )
        .unwrap();
        let o_proj = Linear::from_f16_weight(
            &[4, 4],
            vec![
                f16::from_f32(0.25),
                f16::from_f32(-0.6),
                f16::from_f32(1.2),
                f16::from_f32(0.1),
                f16::from_f32(-0.3),
                f16::from_f32(0.5),
                f16::from_f32(0.8),
                f16::from_f32(-0.7),
                f16::from_f32(0.9),
                f16::from_f32(0.4),
                f16::from_f32(-0.2),
                f16::from_f32(0.6),
                f16::from_f32(-1.0),
                f16::from_f32(0.3),
                f16::from_f32(0.2),
                f16::from_f32(0.75),
            ],
            None,
        )
        .unwrap();
        Attention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RmsNorm::new(Tensor::from_f32_slice(&[2], &[1.0, 0.75]).unwrap(), 1e-5),
            k_norm: RmsNorm::new(Tensor::from_f32_slice(&[2], &[0.8, -1.1]).unwrap(), 1e-5),
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            rope_theta: 10000.0,
            rope_inv_freq: super::rope_inv_freq(2, 10000.0),
            q8_qkv_batch: None,
            resident_runtime_state: RefCell::new(None),
        }
    }

    fn build_test_mlp() -> Mlp {
        let gate_proj = Linear::from_f16_weight(
            &[6, 4],
            vec![
                f16::from_f32(0.5),
                f16::from_f32(-1.0),
                f16::from_f32(0.75),
                f16::from_f32(0.25),
                f16::from_f32(-0.5),
                f16::from_f32(0.4),
                f16::from_f32(1.1),
                f16::from_f32(-0.3),
                f16::from_f32(0.2),
                f16::from_f32(0.8),
                f16::from_f32(-0.6),
                f16::from_f32(1.0),
                f16::from_f32(-0.7),
                f16::from_f32(0.1),
                f16::from_f32(0.3),
                f16::from_f32(0.9),
                f16::from_f32(-0.4),
                f16::from_f32(0.6),
                f16::from_f32(-0.2),
                f16::from_f32(0.7),
                f16::from_f32(0.9),
                f16::from_f32(-0.5),
                f16::from_f32(0.4),
                f16::from_f32(0.2),
            ],
            None,
        )
        .unwrap();
        let up_proj = Linear::from_f16_weight(
            &[6, 4],
            vec![
                f16::from_f32(-0.25),
                f16::from_f32(0.75),
                f16::from_f32(0.4),
                f16::from_f32(-1.25),
                f16::from_f32(1.0),
                f16::from_f32(0.5),
                f16::from_f32(-0.4),
                f16::from_f32(0.2),
                f16::from_f32(0.6),
                f16::from_f32(-0.7),
                f16::from_f32(0.3),
                f16::from_f32(1.1),
                f16::from_f32(-0.8),
                f16::from_f32(0.9),
                f16::from_f32(0.2),
                f16::from_f32(0.1),
                f16::from_f32(0.45),
                f16::from_f32(-0.35),
                f16::from_f32(0.85),
                f16::from_f32(0.55),
                f16::from_f32(-0.15),
                f16::from_f32(0.95),
                f16::from_f32(-0.65),
                f16::from_f32(0.25),
            ],
            None,
        )
        .unwrap();
        let down_proj = Linear::from_f16_weight(
            &[4, 6],
            vec![
                f16::from_f32(0.3),
                f16::from_f32(-0.8),
                f16::from_f32(0.6),
                f16::from_f32(-0.4),
                f16::from_f32(0.2),
                f16::from_f32(1.2),
                f16::from_f32(-0.6),
                f16::from_f32(0.5),
                f16::from_f32(0.4),
                f16::from_f32(0.7),
                f16::from_f32(-0.9),
                f16::from_f32(0.1),
                f16::from_f32(0.8),
                f16::from_f32(-0.2),
                f16::from_f32(1.0),
                f16::from_f32(0.3),
                f16::from_f32(-0.5),
                f16::from_f32(0.6),
                f16::from_f32(-1.1),
                f16::from_f32(0.4),
                f16::from_f32(0.2),
                f16::from_f32(0.9),
                f16::from_f32(0.5),
                f16::from_f32(-0.3),
            ],
            None,
        )
        .unwrap();
        Mlp::new(gate_proj, up_proj, down_proj)
    }

    fn build_test_transformer_block() -> TransformerBlock {
        TransformerBlock::new(
            RmsNorm::new(
                Tensor::from_f32_slice(&[4], &[1.0, 0.8, -0.6, 1.2]).unwrap(),
                1e-5,
            ),
            build_test_attention(),
            RmsNorm::new(
                Tensor::from_f32_slice(&[4], &[0.7, -1.1, 0.9, 0.5]).unwrap(),
                1e-5,
            ),
            build_test_mlp(),
        )
    }

    #[test]
    fn linear_q8_optional_weight_matches_f16_path() {
        let weight = vec![
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let f16_linear = Linear::from_f16_weight(&[3, 2], weight.clone(), None).unwrap();
        let mut q8_linear = Linear::from_f16_weight(&[3, 2], weight, None).unwrap();
        q8_linear.try_enable_q8_weight().unwrap();

        let expected = f16_linear.forward(&x).unwrap();
        let actual = q8_linear.forward(&x).unwrap();

        for (expected, actual) in expected.as_slice().iter().zip(actual.as_slice()) {
            assert!(
                (expected - actual).abs() <= 0.02,
                "q8 Linear output {actual} too far from f16 {expected}"
            );
        }
    }

    #[test]
    fn linear_from_q8_weight_matches_f16_path_without_f16_fallback_copy() {
        let weight = vec![
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let f16_linear = Linear::from_f16_weight(&[3, 2], weight.clone(), None).unwrap();
        let q8_weight = Q8LinearWeight::from_f16(&weight, 3, 2).unwrap();
        let q8_only_linear = Linear::from_q8_weight("test.weight", q8_weight, None);

        let expected = f16_linear.forward(&x).unwrap();
        let actual = q8_only_linear.forward(&x).unwrap();

        for (expected, actual) in expected.as_slice().iter().zip(actual.as_slice()) {
            assert!(
                (expected - actual).abs() <= 0.02,
                "q8-only Linear output {actual} too far from f16 {expected}"
            );
        }
    }

    #[test]
    fn mlp_q8_single_token_parallel_gate_up_matches_multi_row_first_row() {
        let gate_weight = vec![
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let up_weight = vec![
            f16::from_f32(-0.25),
            f16::from_f32(0.75),
            f16::from_f32(0.4),
            f16::from_f32(-1.25),
            f16::from_f32(1.0),
            f16::from_f32(0.5),
        ];
        let down_weight = vec![
            f16::from_f32(0.3),
            f16::from_f32(-0.8),
            f16::from_f32(0.6),
            f16::from_f32(-0.4),
            f16::from_f32(0.2),
            f16::from_f32(1.2),
        ];
        let mut mlp = Mlp::new(
            Linear::from_f16_weight(&[3, 2], gate_weight, None).unwrap(),
            Linear::from_f16_weight(&[3, 2], up_weight, None).unwrap(),
            Linear::from_f16_weight(&[2, 3], down_weight, None).unwrap(),
        );
        mlp.try_enable_q8_weights(None).unwrap();
        let single = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let multi = Tensor::from_f32_slice(&[2, 2], &[0.6, -1.4, 1.0, 0.25]).unwrap();

        let single_out = mlp.forward(&single).unwrap();
        let multi_out = mlp.forward(&multi).unwrap();

        assert_eq!(single_out.shape(), &[1, 2]);
        assert_eq!(multi_out.shape(), &[2, 2]);
        for (single, multi) in single_out.as_slice().iter().zip(&multi_out.as_slice()[..2]) {
            assert!((single - multi).abs() <= 1e-6);
        }
    }

    #[test]
    fn transpose_for_attention_single_token_uses_same_element_order() {
        let x = Tensor::from_f32_slice(&[1, 3, 2], &[0.1, 0.2, 1.1, 1.2, 2.1, 2.2]).unwrap();

        let transposed = transpose_for_attention(&x, 3).unwrap();

        assert_eq!(transposed.shape(), &[3, 1, 2]);
        assert_eq!(transposed.as_slice(), x.as_slice());
    }

    #[test]
    fn transpose_back_single_token_uses_same_element_order() {
        let x = Tensor::from_f32_slice(&[3, 1, 2], &[0.1, 0.2, 1.1, 1.2, 2.1, 2.2]).unwrap();

        let transposed = transpose_back(&x, 1, 3).unwrap();

        assert_eq!(transposed.shape(), &[1, 3, 2]);
        assert_eq!(transposed.as_slice(), x.as_slice());
    }

    #[test]
    fn resident_decode_one_attention_prototype_matches_current_path_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("resident attention prototype test skipped: no usable wgpu adapter");
            return;
        };

        let mut attention = build_test_attention();
        attention.try_enable_q8_weights(None).unwrap();
        let (attached, errors) = attention.try_enable_q8_gpu_matvecs(&context);
        assert_eq!(attached, 4, "expected all four attention Q8 GPU matvecs");
        assert!(
            errors.is_empty(),
            "unexpected Q8 GPU matvec attach errors: {errors:?}"
        );

        let prefix_hidden = Tensor::from_f32_slice(&[1, 4], &[0.3, -0.7, 1.1, 0.5]).unwrap();
        let current_hidden = Tensor::from_f32_slice(&[1, 4], &[-0.2, 0.9, 0.4, -1.3]).unwrap();

        let mut prefix_cache = KVCache::new(1, 8);
        attention
            .forward(&prefix_hidden, &mut prefix_cache, 0, 0)
            .unwrap();
        let cached_prefix = prefix_cache.get_cached(0).unwrap();

        let mut expected_cache = prefix_cache.clone();
        let expected = attention
            .forward(&current_hidden, &mut expected_cache, 0, 1)
            .unwrap();

        reset_sync_stats();
        let got = attention
            .forward_decode_one_resident_prototype(&current_hidden, Some(cached_prefix), 1)
            .unwrap();
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
            max_abs <= 0.03,
            "resident attention prototype max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn resident_decode_one_transformer_block_prototype_matches_current_path_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("resident transformer block prototype test skipped: no usable wgpu adapter");
            return;
        };

        let mut block = build_test_transformer_block();
        block.try_enable_q8_weights(None).unwrap();
        let (attached, errors) = block.try_enable_q8_gpu_matvecs(&context);
        assert_eq!(attached, 7, "expected all attention+mlp Q8 GPU matvecs");
        assert!(
            errors.is_empty(),
            "unexpected transformer block Q8 GPU attach errors: {errors:?}"
        );

        let prefix_hidden = Tensor::from_f32_slice(&[1, 4], &[0.3, -0.7, 1.1, 0.5]).unwrap();
        let current_hidden = Tensor::from_f32_slice(&[1, 4], &[-0.2, 0.9, 0.4, -1.3]).unwrap();

        let mut prefix_cache = KVCache::new(1, 8);
        block
            .forward(&prefix_hidden, &mut prefix_cache, 0, 0)
            .unwrap();
        let cached_prefix = prefix_cache.get_cached(0).unwrap();

        let mut expected_cache = prefix_cache.clone();
        let expected = block
            .forward(&current_hidden, &mut expected_cache, 0, 1)
            .unwrap();

        reset_sync_stats();
        let got = block
            .forward_decode_one_resident_prototype(&current_hidden, Some(cached_prefix), 1)
            .unwrap();
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
            max_abs <= 0.05,
            "resident transformer block prototype max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn resident_decode_one_transformer_block_runtime_keeps_gpu_kv_across_tokens_and_restore_when_available(
    ) {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("resident transformer block runtime test skipped: no usable wgpu adapter");
            return;
        };

        let mut block = build_test_transformer_block();
        block.try_enable_q8_weights(None).unwrap();
        let (attached, errors) = block.try_enable_q8_gpu_matvecs(&context);
        assert_eq!(attached, 7, "expected all attention+mlp Q8 GPU matvecs");
        assert!(
            errors.is_empty(),
            "unexpected transformer block Q8 GPU attach errors: {errors:?}"
        );

        let prefix_hidden = Tensor::from_f32_slice(&[1, 4], &[0.3, -0.7, 1.1, 0.5]).unwrap();
        let token1_hidden = Tensor::from_f32_slice(&[1, 4], &[-0.2, 0.9, 0.4, -1.3]).unwrap();
        let token2_hidden = Tensor::from_f32_slice(&[1, 4], &[0.8, -0.1, -0.6, 1.4]).unwrap();

        let mut expected_cache = KVCache::new(1, 8);
        block
            .forward(&prefix_hidden, &mut expected_cache, 0, 0)
            .unwrap();
        let expected_token1 = block
            .forward(&token1_hidden, &mut expected_cache, 0, 1)
            .unwrap();
        let expected_snapshot = expected_cache.snapshot();
        let expected_token2 = block
            .forward(&token2_hidden, &mut expected_cache, 0, 2)
            .unwrap();

        let mut runtime_cache = KVCache::new(1, 8);
        block
            .forward(&prefix_hidden, &mut runtime_cache, 0, 0)
            .unwrap();

        reset_sync_stats();
        let got_token1 = block
            .forward_decode_one_resident_runtime(&token1_hidden, &mut runtime_cache, 0, 1)
            .unwrap();
        let stats1 = sync_stats();
        assert_eq!(
            stats1.submits, 2,
            "resident runtime token1 should keep exactly one compute submit and one final readback submit"
        );
        assert_eq!(
            stats1.poll_waits, 1,
            "resident runtime token1 should only poll for final output"
        );
        assert_eq!(
            stats1.map_reads, 1,
            "resident runtime token1 should only map final output"
        );
        assert_eq!(runtime_cache.current_len(), 2);

        let max_abs_token1 = expected_token1
            .as_slice()
            .iter()
            .zip(got_token1.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_token1 <= 0.05,
            "resident runtime token1 max abs diff {max_abs_token1} exceeded tolerance"
        );

        let runtime_snapshot = runtime_cache.snapshot();
        reset_sync_stats();
        let got_token2 = block
            .forward_decode_one_resident_runtime(&token2_hidden, &mut runtime_cache, 0, 2)
            .unwrap();
        let stats2 = sync_stats();
        assert_eq!(
            stats2.submits, 2,
            "resident runtime token2 should keep exactly one compute submit and one final readback submit"
        );
        assert_eq!(
            stats2.poll_waits, 1,
            "resident runtime token2 should only poll for final output"
        );
        assert_eq!(
            stats2.map_reads, 1,
            "resident runtime token2 should only map final output"
        );
        assert_eq!(runtime_cache.current_len(), 3);

        let max_abs_token2 = expected_token2
            .as_slice()
            .iter()
            .zip(got_token2.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_token2 <= 0.05,
            "resident runtime token2 max abs diff {max_abs_token2} exceeded tolerance"
        );

        runtime_cache.restore(runtime_snapshot).unwrap();
        expected_cache.restore(expected_snapshot).unwrap();

        reset_sync_stats();
        let replay_token2 = block
            .forward_decode_one_resident_runtime(&token2_hidden, &mut runtime_cache, 0, 2)
            .unwrap();
        let replay_stats = sync_stats();
        assert_eq!(
            replay_stats.submits, 2,
            "resident runtime replay should still keep exactly one compute submit and one final readback submit"
        );
        assert_eq!(
            replay_stats.poll_waits, 1,
            "resident runtime replay should only poll for final output"
        );
        assert_eq!(
            replay_stats.map_reads, 1,
            "resident runtime replay should only map final output"
        );

        let replay_expected = block
            .forward(&token2_hidden, &mut expected_cache, 0, 2)
            .unwrap();
        let replay_max_abs = replay_expected
            .as_slice()
            .iter()
            .zip(replay_token2.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            replay_max_abs <= 0.05,
            "resident runtime replay max abs diff {replay_max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn transpose_for_attention_multi_token_still_reorders_axes() {
        let x = Tensor::from_f32_slice(&[2, 2, 2], &[0.0, 0.1, 1.0, 1.1, 10.0, 10.1, 11.0, 11.1])
            .unwrap();

        let transposed = transpose_for_attention(&x, 2).unwrap();

        assert_eq!(transposed.shape(), &[2, 2, 2]);
        assert_eq!(
            transposed.as_slice(),
            &[0.0, 0.1, 10.0, 10.1, 1.0, 1.1, 11.0, 11.1]
        );
    }
}
