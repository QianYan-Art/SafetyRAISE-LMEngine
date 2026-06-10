//! Qwen3 模型组装与前向传播。

use std::{cell::RefCell, path::Path};

use ndarray::{s, Array2};

use crate::engine::KVCache;
use crate::error::{Result, RsinferError};
use crate::gpu::{GpuContext, GpuPositionUniform, GpuResidentBuffer};
use crate::model::config::Qwen3Config;
use crate::model::layers::{
    build_transformer_block, build_transformer_block_with_q8_sidecar, Linear, RmsNorm,
    TransformerBlock, TransformerBlockProfile,
};
use crate::model::q8_sidecar::Q8SidecarCache;
use crate::model::weights::{get_weight, load_weights, load_weights_filtered, WeightMap};
use crate::runtime::{
    build_runtime_plan, QuantizationCacheMode, QuantizationMode, RuntimeOptions, RuntimePlan,
};
use crate::tensor::Tensor;

pub struct Qwen3Model {
    pub config: Qwen3Config,
    pub embed_tokens: Tensor, // [vocab_size, hidden_size]
    pub layers: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
    pub runtime_plan: RuntimePlan,
    resident_decode_prototype: bool,
    resident_runtime_state: RefCell<Option<ResidentQwen3RuntimeState>>,
}

struct ResidentQwen3RuntimeState {
    resident_hidden_a: GpuResidentBuffer,
    resident_hidden_b: GpuResidentBuffer,
    resident_position: GpuPositionUniform,
    hidden_size: usize,
}

impl Qwen3Model {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        Self::from_pretrained_with_options(model_dir, &RuntimeOptions::default())
    }

    pub fn from_pretrained_with_options<P: AsRef<Path>>(
        model_dir: P,
        runtime_options: &RuntimeOptions,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Qwen3Config::from_file(model_dir.join("config.json"))?;
        let mut q8_sidecar_error = None;
        let use_q8_sidecar = runtime_options.quantization == QuantizationMode::Q8
            && runtime_options.quantization_cache == QuantizationCacheMode::Auto;
        if use_q8_sidecar {
            match Q8SidecarCache::open(model_dir, runtime_options.q8_cache_dir.as_deref()) {
                Ok(cache) => {
                    if let Ok(weights) = load_weights_filtered(model_dir, |name| {
                        !should_skip_warm_q8_linear_weight(name)
                    }) {
                        if let Ok(mut model) = Self::from_weights_with_options_and_sidecar(
                            &config,
                            &weights,
                            runtime_options,
                            Some(cache),
                            None,
                        ) {
                            model.runtime_plan.notes.push(
                                "Warm Q8 sidecar fast-load skipped raw safetensors reads for transformer/lm_head linear weights."
                                    .to_string(),
                            );
                            return Ok(model);
                        }
                    }
                }
                Err(err) => q8_sidecar_error = Some(err.to_string()),
            }
        }

        let weights = load_weights(model_dir)?;
        let q8_sidecar = if use_q8_sidecar {
            match Q8SidecarCache::open(model_dir, runtime_options.q8_cache_dir.as_deref()) {
                Ok(cache) => Some(cache),
                Err(err) => {
                    if q8_sidecar_error.is_none() {
                        q8_sidecar_error = Some(err.to_string());
                    }
                    None
                }
            }
        } else {
            None
        };
        Self::from_weights_with_options_and_sidecar(
            &config,
            &weights,
            runtime_options,
            q8_sidecar,
            q8_sidecar_error,
        )
    }

    pub fn from_weights(config: &Qwen3Config, weights: &WeightMap) -> Result<Self> {
        Self::from_weights_with_options(config, weights, &RuntimeOptions::default())
    }

    pub fn from_weights_with_options(
        config: &Qwen3Config,
        weights: &WeightMap,
        runtime_options: &RuntimeOptions,
    ) -> Result<Self> {
        Self::from_weights_with_options_and_sidecar(config, weights, runtime_options, None, None)
    }

    fn from_weights_with_options_and_sidecar(
        config: &Qwen3Config,
        weights: &WeightMap,
        runtime_options: &RuntimeOptions,
        mut q8_sidecar: Option<Q8SidecarCache>,
        q8_sidecar_error: Option<String>,
    ) -> Result<Self> {
        let mut runtime_plan = build_runtime_plan(config, runtime_options);
        let gpu_context = if runtime_plan.should_try_gpu_backend() {
            match GpuContext::new() {
                Ok(context) => Some(context),
                Err(err) => {
                    runtime_plan.mark_transformer_gpu_fallback(err.clone());
                    runtime_plan.mark_lm_head_gpu_fallback(err);
                    None
                }
            }
        } else {
            None
        };
        let embed_tokens = get_weight(weights, "model.embed_tokens.weight")?;
        let use_q8 = runtime_options.quantization == QuantizationMode::Q8;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            if use_q8 {
                let layer = match q8_sidecar.as_mut() {
                    Some(cache) => build_transformer_block_with_q8_sidecar(
                        weights,
                        config,
                        layer_idx,
                        Some(cache),
                    )?,
                    None => build_transformer_block(weights, config, layer_idx)?,
                };
                layers.push(layer);
            } else {
                layers.push(build_transformer_block(weights, config, layer_idx)?);
            }
        }

        let norm = RmsNorm::new(
            get_weight(weights, "model.norm.weight")?,
            config.rms_norm_eps,
        );

        // tie_word_embeddings 时 lm_head 复用 embedding 权重
        let mut lm_head = if use_q8 {
            match q8_sidecar.as_mut() {
                Some(cache) => match Linear::from_weight_map_or_q8_sidecar(
                    weights,
                    "lm_head.weight",
                    None,
                    Some(cache),
                ) {
                    Ok(linear) => linear,
                    Err(_) if config.tie_word_embeddings => Linear::from_weight_map_or_q8_sidecar(
                        weights,
                        "model.embed_tokens.weight",
                        None,
                        Some(cache),
                    )?,
                    Err(_) => {
                        return Err(RsinferError::WeightError(
                            "缺少 lm_head.weight 且 tie_word_embeddings 为 false".into(),
                        ))
                    }
                },
                None => match Linear::from_weight_map(weights, "lm_head.weight", None) {
                    Ok(linear) => linear,
                    Err(_) if config.tie_word_embeddings => {
                        Linear::from_weight_map(weights, "model.embed_tokens.weight", None)?
                    }
                    Err(_) => {
                        return Err(RsinferError::WeightError(
                            "缺少 lm_head.weight 且 tie_word_embeddings 为 false".into(),
                        ))
                    }
                },
            }
        } else {
            match Linear::from_weight_map(weights, "lm_head.weight", None) {
                Ok(linear) => linear,
                Err(_) if config.tie_word_embeddings => {
                    Linear::from_weight_map(weights, "model.embed_tokens.weight", None)?
                }
                Err(_) => {
                    return Err(RsinferError::WeightError(
                        "缺少 lm_head.weight 且 tie_word_embeddings 为 false".into(),
                    ))
                }
            }
        };

        if use_q8 {
            match runtime_options.quantization_cache {
                QuantizationCacheMode::Auto => {
                    if let Some(err) = q8_sidecar_error {
                        runtime_plan.mark_q8_sidecar_unavailable(err);
                    }
                }
                QuantizationCacheMode::Off => runtime_plan.mark_q8_sidecar_disabled(),
            }

            let mut attached_linears = 0usize;
            for layer in &mut layers {
                attached_linears += match q8_sidecar.as_mut() {
                    Some(cache) => layer.try_enable_q8_weights(Some(cache))?,
                    None => layer.try_enable_q8_weights(None)?,
                };
            }
            match q8_sidecar.as_mut() {
                Some(cache) => lm_head.try_enable_q8_weight_with_sidecar(Some(cache))?,
                None => lm_head.try_enable_q8_weight_with_sidecar(None)?,
            }
            attached_linears += 1;
            runtime_plan.mark_q8_quantization(attached_linears);
            if let Some(cache) = &q8_sidecar {
                let report = cache.report();
                runtime_plan.mark_q8_sidecar_report(
                    report.dir.display(),
                    report.hits,
                    report.writes,
                    report.fallbacks.len(),
                );
            }
        }

        let planned_gpu_layers = runtime_plan.planned_transformer_gpu_layers();
        if let Some(context) = &gpu_context {
            let mut active_layers = 0usize;
            let mut attached_linears = 0usize;
            let mut fallback_errors = Vec::new();
            for (layer_idx, layer) in layers.iter_mut().enumerate().take(planned_gpu_layers) {
                let (attached, errors) = if use_q8 {
                    layer.try_enable_q8_gpu_matvecs(context)
                } else {
                    layer.try_enable_gpu_matvecs(context)
                };
                if attached > 0 {
                    active_layers += 1;
                    attached_linears += attached;
                }
                if !errors.is_empty() {
                    fallback_errors.push(format!("layer {layer_idx}: {}", errors.join("; ")));
                }
            }
            if use_q8 {
                runtime_plan.mark_transformer_decode_q8_gpu_layers(active_layers, attached_linears);
            } else {
                runtime_plan.mark_transformer_decode_gpu_layers(active_layers, attached_linears);
            }
            if !fallback_errors.is_empty() {
                runtime_plan.mark_transformer_gpu_fallback(fallback_errors.join(" | "));
            }

            if use_q8 {
                match lm_head.try_enable_q8_gpu_matvec_with_context_and_argmax(context, true) {
                    Ok(()) => runtime_plan.mark_lm_head_q8_gpu(),
                    Err(err) => runtime_plan.mark_lm_head_gpu_fallback(err),
                }
            } else {
                match lm_head.try_enable_gpu_matvec_with_context(context) {
                    Ok(()) if lm_head.has_gpu_matvec() => runtime_plan.mark_lm_head_gpu(),
                    Ok(()) => runtime_plan.mark_lm_head_gpu_fallback("backend was not attached"),
                    Err(err) => runtime_plan.mark_lm_head_gpu_fallback(err),
                }
            }
        }

        if runtime_options.internal_resident_decode_prototype {
            runtime_plan
                .notes
                .push("Internal resident decode prototype path is enabled for decode-one GPU layers; default behavior remains unchanged when this flag is off.".to_string());
        }

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            runtime_plan,
            resident_decode_prototype: runtime_options.internal_resident_decode_prototype,
            resident_runtime_state: RefCell::new(None),
        })
    }

    pub fn has_greedy_token_fast_path(&self) -> bool {
        self.lm_head.has_q8_gpu_argmax()
    }

    fn resident_gpu_prefix_len(&self, input_ids: &[u32]) -> usize {
        if !self.resident_decode_prototype || input_ids.len() != 1 {
            return 0;
        }
        self.runtime_plan
            .layer_devices
            .iter()
            .take_while(|&&device| device == crate::runtime::LayerDevice::Gpu)
            .count()
    }

    fn ensure_resident_runtime_state(
        &self,
        context: &GpuContext,
        hidden_size: usize,
    ) -> Result<()> {
        let mut state_slot = self.resident_runtime_state.borrow_mut();
        let needs_rebuild = state_slot
            .as_ref()
            .map(|state| state.hidden_size != hidden_size)
            .unwrap_or(true);
        if needs_rebuild {
            *state_slot = Some(ResidentQwen3RuntimeState {
                resident_hidden_a: GpuResidentBuffer::with_context(context, &[1, hidden_size])
                    .map_err(RsinferError::DimensionError)?,
                resident_hidden_b: GpuResidentBuffer::with_context(context, &[1, hidden_size])
                    .map_err(RsinferError::DimensionError)?,
                resident_position: GpuPositionUniform::with_context(context, 0)
                    .map_err(RsinferError::DimensionError)?,
                hidden_size,
            });
        }
        Ok(())
    }

    fn forward_hidden_decode_one_resident_prefix(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
    ) -> Result<Tensor> {
        let gpu_prefix_len = self.resident_gpu_prefix_len(input_ids);
        let mut hidden = self.embedding(input_ids)?;
        if gpu_prefix_len == 0 {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward(&hidden, kv_cache, layer_idx, position_offset)?;
            }
            return Ok(hidden);
        }

        let first_gpu_layer = &self.layers[0];
        let q_proj = first_gpu_layer
            .attention
            .q_proj
            .q8_gpu_matvec()
            .ok_or_else(|| {
                RsinferError::DimensionError(
                    "missing first resident q_proj Q8 GPU matvec".to_string(),
                )
            })?;
        let context = q_proj.shared_context();
        let hidden_size = hidden.shape()[1];
        self.ensure_resident_runtime_state(&context, hidden_size)?;
        let state_slot = self.resident_runtime_state.borrow();
        let state = state_slot
            .as_ref()
            .expect("resident qwen3 runtime state should be initialized");
        state
            .resident_hidden_a
            .upload(&hidden)
            .map_err(RsinferError::DimensionError)?;
        state.resident_position.write_position(position_offset);
        let mut encoder =
            context.create_command_encoder("rsinfer-resident-qwen3-gpu-prefix-encoder");
        for (layer_idx, layer) in self.layers.iter().take(gpu_prefix_len).enumerate() {
            let (resident_current, resident_next) = if layer_idx % 2 == 0 {
                (&state.resident_hidden_a, &state.resident_hidden_b)
            } else {
                (&state.resident_hidden_b, &state.resident_hidden_a)
            };
            layer.encode_decode_one_resident_runtime_with_slot_cache(
                &mut encoder,
                resident_current,
                resident_next,
                &state.resident_hidden_a,
                &state.resident_hidden_b,
                kv_cache,
                layer_idx,
                position_offset,
                state.resident_position.buffer(),
            )?;
        }
        context.submit(encoder);
        kv_cache.set_current_len(position_offset + 1)?;
        let resident_output = if gpu_prefix_len.is_multiple_of(2) {
            &state.resident_hidden_a
        } else {
            &state.resident_hidden_b
        };
        hidden = resident_output
            .read_back()
            .map_err(RsinferError::DimensionError)?;

        for (layer_idx, layer) in self.layers.iter().enumerate().skip(gpu_prefix_len) {
            hidden = layer.forward(&hidden, kv_cache, layer_idx, position_offset)?;
        }
        Ok(hidden)
    }

    fn forward_hidden_decode_one_resident_prefix_profiled(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
        layer_profiles: &mut [TransformerBlockProfile],
    ) -> Result<Tensor> {
        let gpu_prefix_len = self.resident_gpu_prefix_len(input_ids);
        let mut hidden = self.embedding(input_ids)?;
        if gpu_prefix_len == 0 {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_profiled(
                    &hidden,
                    kv_cache,
                    layer_idx,
                    position_offset,
                    &mut layer_profiles[layer_idx],
                )?;
            }
            return Ok(hidden);
        }

        let first_gpu_layer = &self.layers[0];
        let q_proj = first_gpu_layer
            .attention
            .q_proj
            .q8_gpu_matvec()
            .ok_or_else(|| {
                RsinferError::DimensionError(
                    "missing first resident q_proj Q8 GPU matvec".to_string(),
                )
            })?;
        let context = q_proj.shared_context();
        let hidden_size = hidden.shape()[1];
        self.ensure_resident_runtime_state(&context, hidden_size)?;
        let state_slot = self.resident_runtime_state.borrow();
        let state = state_slot
            .as_ref()
            .expect("resident qwen3 runtime state should be initialized");
        state
            .resident_hidden_a
            .upload(&hidden)
            .map_err(RsinferError::DimensionError)?;
        state.resident_position.write_position(position_offset);
        let prefix_start = std::time::Instant::now();
        let mut encoder =
            context.create_command_encoder("rsinfer-resident-qwen3-gpu-prefix-profiled-encoder");
        for (layer_idx, layer) in self.layers.iter().take(gpu_prefix_len).enumerate() {
            let (resident_current, resident_next) = if layer_idx % 2 == 0 {
                (&state.resident_hidden_a, &state.resident_hidden_b)
            } else {
                (&state.resident_hidden_b, &state.resident_hidden_a)
            };
            layer.encode_decode_one_resident_runtime_with_slot_cache(
                &mut encoder,
                resident_current,
                resident_next,
                &state.resident_hidden_a,
                &state.resident_hidden_b,
                kv_cache,
                layer_idx,
                position_offset,
                state.resident_position.buffer(),
            )?;
        }
        context.submit(encoder);
        kv_cache.set_current_len(position_offset + 1)?;
        let resident_output = if gpu_prefix_len.is_multiple_of(2) {
            &state.resident_hidden_a
        } else {
            &state.resident_hidden_b
        };
        hidden = resident_output
            .read_back()
            .map_err(RsinferError::DimensionError)?;
        layer_profiles[gpu_prefix_len - 1].total += prefix_start.elapsed();

        for (layer_idx, layer) in self.layers.iter().enumerate().skip(gpu_prefix_len) {
            hidden = layer.forward_profiled(
                &hidden,
                kv_cache,
                layer_idx,
                position_offset,
                &mut layer_profiles[layer_idx],
            )?;
        }
        Ok(hidden)
    }

    /// 返回最后一个位置的 logits: [1, vocab_size]。
    /// 自回归只需最后一步，prefill 时也借此跳过对整段序列的 lm_head。
    pub fn forward(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
    ) -> Result<Tensor> {
        let hidden =
            self.forward_hidden_decode_one_resident_prefix(input_ids, kv_cache, position_offset)?;
        let last = self.norm.forward(&hidden.last_row()?)?;
        self.lm_head.forward(&last)
    }

    pub fn forward_profiled(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
        layer_profiles: &mut Vec<TransformerBlockProfile>,
    ) -> Result<Tensor> {
        self.ensure_layer_profiles(layer_profiles);
        let hidden = self.forward_hidden_decode_one_resident_prefix_profiled(
            input_ids,
            kv_cache,
            position_offset,
            layer_profiles.as_mut_slice(),
        )?;
        let last = self.norm.forward(&hidden.last_row()?)?;
        self.lm_head.forward(&last)
    }

    /// Greedy-only fast path: return the lm_head argmax token without reading full logits when
    /// the active lm_head backend can do that directly. Callers must fall back to `forward` on error.
    pub fn forward_greedy_token(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
    ) -> Result<u32> {
        let hidden =
            self.forward_hidden_decode_one_resident_prefix(input_ids, kv_cache, position_offset)?;
        let last = self.norm.forward(&hidden.last_row()?)?;
        self.lm_head
            .try_forward_q8_gpu_argmax(&last)
            .map_err(RsinferError::DimensionError)
    }

    pub fn forward_greedy_token_profiled(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
        layer_profiles: &mut Vec<TransformerBlockProfile>,
    ) -> Result<u32> {
        self.ensure_layer_profiles(layer_profiles);
        let hidden = self.forward_hidden_decode_one_resident_prefix_profiled(
            input_ids,
            kv_cache,
            position_offset,
            layer_profiles.as_mut_slice(),
        )?;
        let last = self.norm.forward(&hidden.last_row()?)?;
        self.lm_head
            .try_forward_q8_gpu_argmax(&last)
            .map_err(RsinferError::DimensionError)
    }

    fn ensure_layer_profiles(&self, layer_profiles: &mut Vec<TransformerBlockProfile>) {
        if layer_profiles.len() < self.layers.len() {
            layer_profiles.resize_with(self.layers.len(), TransformerBlockProfile::default);
        }
    }

    fn embedding(&self, input_ids: &[u32]) -> Result<Tensor> {
        let hidden_size = self.config.hidden_size;
        let mut result = Array2::<f32>::zeros((input_ids.len(), hidden_size));
        for (i, &token_id) in input_ids.iter().enumerate() {
            let id = token_id as usize;
            if id >= self.config.vocab_size {
                return Err(RsinferError::DimensionError(format!(
                    "token id {id} 越界 (vocab_size {})",
                    self.config.vocab_size
                )));
            }
            result
                .row_mut(i)
                .assign(&self.embed_tokens.data.slice(s![id, ..]));
        }
        Ok(Tensor {
            data: result.into_dyn(),
        })
    }

    pub fn create_kv_cache(&self) -> KVCache {
        KVCache::new(
            self.config.num_hidden_layers,
            self.config.max_position_embeddings,
        )
    }
}

fn should_skip_warm_q8_linear_weight(name: &str) -> bool {
    name == "lm_head.weight"
        || name.ends_with("self_attn.q_proj.weight")
        || name.ends_with("self_attn.k_proj.weight")
        || name.ends_with("self_attn.v_proj.weight")
        || name.ends_with("self_attn.o_proj.weight")
        || name.ends_with("mlp.gate_proj.weight")
        || name.ends_with("mlp.up_proj.weight")
        || name.ends_with("mlp.down_proj.weight")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{gpu_test_guard, reset_sync_stats, sync_stats};
    use crate::model::weights::{LoadedTensor, WeightMap};
    use crate::runtime::{DevicePreference, LayerDevice};
    use half::f16;
    use std::collections::HashMap;

    fn test_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 4,
            intermediate_size: 6,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            explicit_head_dim: Some(2),
            vocab_size: 8,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            model_type: "qwen3".to_string(),
            torch_dtype: "float16".to_string(),
            tie_word_embeddings: false,
            eos_token_id: 7,
            bos_token_id: Some(0),
        }
    }

    fn tensor_1d(values: &[f32]) -> LoadedTensor {
        LoadedTensor::Tensor(Tensor::from_f32_slice(&[values.len()], values).unwrap())
    }

    fn f16_2d(shape: &[usize], values: &[f32]) -> LoadedTensor {
        LoadedTensor::F16 {
            shape: shape.to_vec(),
            data: values.iter().copied().map(f16::from_f32).collect(),
        }
    }

    fn layer_matrix_values(layer_idx: usize, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
        (0..rows * cols)
            .map(|idx| {
                let row = idx / cols;
                let col = idx % cols;
                let sign = if (row + col + layer_idx).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                sign * scale
                    * (1.0 + layer_idx as f32 * 0.17 + row as f32 * 0.11 + col as f32 * 0.07)
            })
            .collect()
    }

    fn layer_vector_values(layer_idx: usize, len: usize, base: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let sign = if (idx + layer_idx).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                sign * (base + idx as f32 * 0.09 + layer_idx as f32 * 0.05)
            })
            .collect()
    }

    fn test_weights() -> WeightMap {
        let mut weights: WeightMap = HashMap::new();
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            LoadedTensor::Tensor(
                Tensor::from_f32_slice(
                    &[8, 4],
                    &[
                        0.05, -0.10, 0.15, -0.20, 0.25, 0.30, -0.35, 0.40, -0.45, 0.50, 0.55,
                        -0.60, 0.65, -0.70, 0.75, 0.80, -0.85, 0.90, -0.95, 1.00, 1.05, -1.10,
                        1.15, -1.20, 1.25, 1.30, -1.35, 1.40, -1.45, 1.50, 1.55, -1.60,
                    ],
                )
                .unwrap(),
            ),
        );
        weights.insert(
            "model.norm.weight".to_string(),
            tensor_1d(&[1.0, -0.8, 0.9, 1.1]),
        );
        weights.insert(
            "lm_head.weight".to_string(),
            f16_2d(
                &[8, 4],
                &[
                    0.30, -0.25, 0.20, -0.15, -0.10, 0.12, -0.14, 0.16, 0.18, 0.22, -0.26, 0.30,
                    -0.34, 0.38, 0.42, -0.46, 0.50, -0.54, 0.58, 0.62, -0.66, 0.70, -0.74, 0.78,
                    0.82, -0.86, 0.90, -0.94, 0.98, 1.02, -1.06, 1.10,
                ],
            ),
        );

        for layer_idx in 0..3 {
            let prefix = format!("model.layers.{layer_idx}");
            weights.insert(
                format!("{prefix}.input_layernorm.weight"),
                tensor_1d(&layer_vector_values(layer_idx, 4, 0.8)),
            );
            weights.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                tensor_1d(&layer_vector_values(layer_idx + 1, 4, 0.7)),
            );
            weights.insert(
                format!("{prefix}.self_attn.q_norm.weight"),
                tensor_1d(&layer_vector_values(layer_idx, 2, 0.9)),
            );
            weights.insert(
                format!("{prefix}.self_attn.k_norm.weight"),
                tensor_1d(&layer_vector_values(layer_idx + 1, 2, 0.75)),
            );
            weights.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                f16_2d(&[4, 4], &layer_matrix_values(layer_idx, 4, 4, 0.12)),
            );
            weights.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                f16_2d(&[2, 4], &layer_matrix_values(layer_idx + 2, 2, 4, 0.10)),
            );
            weights.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                f16_2d(&[2, 4], &layer_matrix_values(layer_idx + 4, 2, 4, 0.11)),
            );
            weights.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                f16_2d(&[4, 4], &layer_matrix_values(layer_idx + 6, 4, 4, 0.09)),
            );
            weights.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                f16_2d(&[6, 4], &layer_matrix_values(layer_idx + 8, 6, 4, 0.08)),
            );
            weights.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                f16_2d(&[6, 4], &layer_matrix_values(layer_idx + 10, 6, 4, 0.07)),
            );
            weights.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                f16_2d(&[4, 6], &layer_matrix_values(layer_idx + 12, 4, 6, 0.06)),
            );
        }

        weights
    }

    fn attach_q8_gpu_layers(model: &mut Qwen3Model, context: &GpuContext, count: usize) {
        for layer in model.layers.iter_mut().take(count) {
            let (attached, errors) = layer.try_enable_q8_gpu_matvecs(context);
            assert_eq!(attached, 7, "expected all seven Q8 GPU matvecs to attach");
            assert!(
                errors.is_empty(),
                "unexpected layer GPU attach errors: {errors:?}"
            );
        }
    }

    #[test]
    fn resident_model_decode_matches_current_path_across_layers_and_restore_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("resident qwen3 model test skipped: no usable wgpu adapter");
            return;
        };

        let config = test_config();
        let weights = test_weights();
        let runtime_options = RuntimeOptions {
            device: DevicePreference::Cpu,
            gpu_layers: Some(2),
            quantization: QuantizationMode::Q8,
            quantization_cache: QuantizationCacheMode::Off,
            q8_cache_dir: None,
            internal_resident_decode_prototype: false,
        };

        let mut baseline =
            Qwen3Model::from_weights_with_options(&config, &weights, &runtime_options).unwrap();
        let mut resident =
            Qwen3Model::from_weights_with_options(&config, &weights, &runtime_options).unwrap();

        attach_q8_gpu_layers(&mut baseline, &context, 2);
        attach_q8_gpu_layers(&mut resident, &context, 2);
        resident.runtime_plan.layer_devices =
            vec![LayerDevice::Gpu, LayerDevice::Gpu, LayerDevice::Cpu];
        resident.resident_decode_prototype = true;

        let prompt = [1_u32, 3_u32];
        let token1 = [2_u32];
        let token2 = [4_u32];

        let mut baseline_cache = baseline.create_kv_cache();
        let mut resident_cache = resident.create_kv_cache();

        let baseline_prefill = baseline.forward(&prompt, &mut baseline_cache, 0).unwrap();
        let resident_prefill = resident.forward(&prompt, &mut resident_cache, 0).unwrap();
        let prefill_max_abs = baseline_prefill
            .as_slice()
            .iter()
            .zip(resident_prefill.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            prefill_max_abs <= 0.08,
            "prefill logits max abs diff {prefill_max_abs} exceeded tolerance"
        );

        let baseline_token1 = baseline
            .forward(&token1, &mut baseline_cache, prompt.len())
            .unwrap();
        reset_sync_stats();
        let resident_token1 = resident
            .forward(&token1, &mut resident_cache, prompt.len())
            .unwrap();
        let stats1 = sync_stats();
        assert_eq!(
            stats1.submits, 2,
            "resident GPU prefix should use one shared compute submit and one shared hidden readback submit"
        );
        assert_eq!(
            stats1.poll_waits, 1,
            "resident decode should only poll once for the shared hidden readback"
        );
        assert_eq!(
            stats1.map_reads, 1,
            "resident decode should only map once for the shared hidden readback"
        );
        let token1_max_abs = baseline_token1
            .as_slice()
            .iter()
            .zip(resident_token1.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            token1_max_abs <= 0.12,
            "decode token1 logits max abs diff {token1_max_abs} exceeded tolerance"
        );

        let baseline_snapshot = baseline_cache.snapshot();
        let resident_snapshot = resident_cache.snapshot();

        let baseline_token2 = baseline
            .forward(&token2, &mut baseline_cache, prompt.len() + 1)
            .unwrap();
        reset_sync_stats();
        let resident_token2 = resident
            .forward(&token2, &mut resident_cache, prompt.len() + 1)
            .unwrap();
        let stats2 = sync_stats();
        assert_eq!(
            stats2.submits, 2,
            "resident token2 should keep the shared GPU-prefix sync structure"
        );
        assert_eq!(
            stats2.poll_waits, 1,
            "resident token2 should only poll once for the shared hidden readback"
        );
        assert_eq!(
            stats2.map_reads, 1,
            "resident token2 should only map once for the shared hidden readback"
        );
        let token2_max_abs = baseline_token2
            .as_slice()
            .iter()
            .zip(resident_token2.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            token2_max_abs <= 0.12,
            "decode token2 logits max abs diff {token2_max_abs} exceeded tolerance"
        );

        baseline_cache.restore(baseline_snapshot).unwrap();
        resident_cache.restore(resident_snapshot).unwrap();

        let baseline_replay = baseline
            .forward(&token2, &mut baseline_cache, prompt.len() + 1)
            .unwrap();
        reset_sync_stats();
        let resident_replay = resident
            .forward(&token2, &mut resident_cache, prompt.len() + 1)
            .unwrap();
        let replay_stats = sync_stats();
        assert_eq!(
            replay_stats.submits, 2,
            "resident replay should keep the shared GPU-prefix sync structure"
        );
        assert_eq!(
            replay_stats.poll_waits, 1,
            "resident replay should only poll once for the shared hidden readback"
        );
        assert_eq!(
            replay_stats.map_reads, 1,
            "resident replay should only map once for the shared hidden readback"
        );
        let replay_max_abs = baseline_replay
            .as_slice()
            .iter()
            .zip(resident_replay.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            replay_max_abs <= 0.12,
            "decode replay logits max abs diff {replay_max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn resident_model_decode_profiled_uses_shared_gpu_prefix_sync_structure_when_available() {
        let _guard = gpu_test_guard();
        let Ok(context) = GpuContext::new() else {
            eprintln!("resident qwen3 profiled test skipped: no usable wgpu adapter");
            return;
        };

        let config = test_config();
        let weights = test_weights();
        let runtime_options = RuntimeOptions {
            device: DevicePreference::Cpu,
            gpu_layers: Some(2),
            quantization: QuantizationMode::Q8,
            quantization_cache: QuantizationCacheMode::Off,
            q8_cache_dir: None,
            internal_resident_decode_prototype: false,
        };

        let mut resident =
            Qwen3Model::from_weights_with_options(&config, &weights, &runtime_options).unwrap();
        attach_q8_gpu_layers(&mut resident, &context, 2);
        resident.runtime_plan.layer_devices =
            vec![LayerDevice::Gpu, LayerDevice::Gpu, LayerDevice::Cpu];
        resident.resident_decode_prototype = true;

        let prompt = [1_u32, 3_u32];
        let token1 = [2_u32];
        let mut kv_cache = resident.create_kv_cache();
        resident.forward(&prompt, &mut kv_cache, 0).unwrap();

        let mut layer_profiles = Vec::new();
        reset_sync_stats();
        let logits = resident
            .forward_profiled(&token1, &mut kv_cache, prompt.len(), &mut layer_profiles)
            .unwrap();
        let stats = sync_stats();

        assert_eq!(
            stats.submits, 2,
            "profiled resident GPU prefix should use one shared compute submit and one shared hidden readback submit"
        );
        assert_eq!(
            stats.poll_waits, 1,
            "profiled resident GPU prefix should only poll once for the shared hidden readback"
        );
        assert_eq!(
            stats.map_reads, 1,
            "profiled resident GPU prefix should only map once for the shared hidden readback"
        );
        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(layer_profiles.len(), config.num_hidden_layers);
        assert!(
            layer_profiles[1].total > std::time::Duration::ZERO,
            "the last GPU layer profile slot should accumulate the shared GPU prefix duration"
        );
        assert!(
            layer_profiles[2].total > std::time::Duration::ZERO,
            "the CPU tail layer should still record profiled work"
        );
    }
}
