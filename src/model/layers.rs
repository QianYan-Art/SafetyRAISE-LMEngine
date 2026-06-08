//! Transformer 层实现
//!
//! 包含 RMSNorm、Attention、MLP 和 TransformerBlock。

use half::f16;

use crate::engine::KVCache;
use crate::error::{Result, RsinferError};
use crate::gpu::{
    GpuContext, GpuMatVec, GpuQ8MatVec, GpuQ8SameInputBatch, GpuQ8SwiGluDown, GpuSwiGluDown,
};
use crate::model::config::Qwen3Config;
use crate::model::q8_sidecar::Q8SidecarCache;
use crate::model::weights::{get_linear_weight_f16, get_weight, WeightMap};
use crate::tensor::{
    linear_forward_f16, linear_forward_q8, rms_norm, rope, scaled_dot_product_attention_gqa_cached,
    silu, CachedAttention, Q8LinearWeight, Tensor,
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

    pub fn from_weight_map(weights: &WeightMap, name: &str, bias: Option<Tensor>) -> Result<Self> {
        let (shape, weight) = get_linear_weight_f16(weights, name)?;
        let mut linear = Self::from_f16_weight(&shape, weight, bias)?;
        linear.source_name = Some(name.to_string());
        Ok(linear)
    }

    pub fn try_enable_gpu_matvec(&mut self) -> std::result::Result<(), String> {
        let accelerator =
            GpuMatVec::from_f16_weight(&self.weight, self.out_features, self.in_features)?;
        self.gpu_matvec = Some(accelerator);
        Ok(())
    }

    pub fn try_enable_gpu_matvec_with_context(
        &mut self,
        context: &GpuContext,
    ) -> std::result::Result<(), String> {
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

    pub fn has_q8_gpu_argmax(&self) -> bool {
        self.bias.is_none() && self.q8_gpu_matvec.is_some()
    }

    pub fn try_enable_q8_gpu_matvec_with_context(
        &mut self,
        context: &GpuContext,
    ) -> std::result::Result<(), String> {
        if self.q8_weight.is_none() {
            self.try_enable_q8_weight().map_err(|err| err.to_string())?;
        }
        let q8_weight = self
            .q8_weight
            .as_ref()
            .ok_or_else(|| "Q8 weight was not attached".to_string())?;
        let accelerator = GpuQ8MatVec::from_q8_weight_with_context(context, q8_weight)?;
        self.q8_gpu_matvec = Some(accelerator);
        Ok(())
    }

    pub fn try_enable_q8_weight(&mut self) -> Result<()> {
        let q8 = Q8LinearWeight::from_f16(&self.weight, self.out_features, self.in_features)?;
        self.q8_weight = Some(q8);
        Ok(())
    }

    pub fn try_enable_q8_weight_with_sidecar(
        &mut self,
        sidecar: Option<&mut Q8SidecarCache>,
    ) -> Result<()> {
        if let (Some(cache), Some(name)) = (sidecar, self.source_name.as_deref()) {
            match cache.load_or_create(name, self.out_features, self.in_features, || {
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
    q8_qkv_batch: Option<GpuQ8SameInputBatch>,
}

impl Attention {
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
            match linear.try_enable_q8_gpu_matvec_with_context(context) {
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
                    Ok(batch) => Some(batch),
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

        let q = q.reshape(&[seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[seq_len, self.num_kv_heads, self.head_dim])?;

        // Qwen3 QK-Norm：每个 head 沿 head_dim 做 RMSNorm，必须在 RoPE 之前
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = rope(&q, &k, position_offset, self.rope_theta)?;

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
                _ => (self.gate_proj.forward(x)?, self.up_proj.forward(x)?),
            }
        } else {
            (self.gate_proj.forward(x)?, self.up_proj.forward(x)?)
        };
        let gate = silu(&gate);
        let hidden = gate.mul(&up)?;
        self.down_proj.forward(&hidden)
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
            match linear.try_enable_q8_gpu_matvec_with_context(context) {
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
                    Ok(fused) => Some(fused),
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
        q8_qkv_batch: None,
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

#[cfg(test)]
mod tests {
    use super::*;

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
