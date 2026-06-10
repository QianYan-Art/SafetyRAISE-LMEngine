//! 自回归文本生成。

use std::path::Path;
use std::time::{Duration, Instant};

use crate::engine::sampler::{default_sampler, Sampler};
use crate::engine::KVCache;
use crate::error::{Result, RsinferError};
use crate::gpu::{reset_sync_stats, sync_stats, GpuSyncStats};
use crate::model::{Qwen3Model, TransformerBlockProfile};
use crate::runtime::RuntimeOptions;

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            stop_tokens: Vec::new(),
        }
    }
}

pub struct Generator {
    pub model: Qwen3Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub sampler: Box<dyn Sampler>,
    pub config: GenerationConfig,
}

#[derive(Clone, Debug, Default)]
pub struct FinalLayerTokenProfile {
    pub total_ms: f64,
    pub attention_ms: f64,
    pub mlp_ms: f64,
    pub mlp_gate_up_ms: f64,
    pub mlp_down_proj_ms: f64,
    pub mlp_q8_gate_up_dot_ms: f64,
    pub mlp_q8_down_proj_dot_ms: f64,
}

impl FinalLayerTokenProfile {
    fn from_layer_profile(profile: &TransformerBlockProfile) -> Self {
        Self {
            total_ms: profile.total.as_secs_f64() * 1000.0,
            attention_ms: profile.attention.as_secs_f64() * 1000.0,
            mlp_ms: profile.mlp.as_secs_f64() * 1000.0,
            mlp_gate_up_ms: profile.mlp_gate_up.as_secs_f64() * 1000.0,
            mlp_down_proj_ms: profile.mlp_down_proj.as_secs_f64() * 1000.0,
            mlp_q8_gate_up_dot_ms: profile.mlp_q8_gate_up_dot.as_secs_f64() * 1000.0,
            mlp_q8_down_proj_dot_ms: profile.mlp_q8_down_proj_dot.as_secs_f64() * 1000.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GenerationProfile {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_forward: Duration,
    pub prefill_sample: Duration,
    pub prefill_gpu_sync: GpuSyncStats,
    pub decode_forward: Duration,
    pub decode_forward_token_ms: Vec<f64>,
    pub decode_gpu_sync: GpuSyncStats,
    pub decode_gpu_sync_trace: Vec<GpuSyncStats>,
    pub decode_sample: Duration,
    pub text_decode: Duration,
    pub fast_path_tokens: usize,
    pub layer_profiles: Vec<TransformerBlockProfile>,
    pub final_layer_index: Option<usize>,
    pub final_layer_decode_trace: Vec<FinalLayerTokenProfile>,
}

impl GenerationProfile {
    pub fn avg_decode_forward_ms(&self) -> f64 {
        if self.generated_tokens <= 1 {
            0.0
        } else {
            self.decode_forward.as_secs_f64() * 1000.0 / (self.generated_tokens - 1) as f64
        }
    }

    fn record_final_layer_decode_trace(
        &mut self,
        layer_index: usize,
        layer_delta: &TransformerBlockProfile,
    ) {
        if let Some(existing) = self.final_layer_index {
            debug_assert_eq!(existing, layer_index);
        } else {
            self.final_layer_index = Some(layer_index);
        }
        self.final_layer_decode_trace
            .push(FinalLayerTokenProfile::from_layer_profile(layer_delta));
    }

    pub fn avg_decode_gpu_submits(&self) -> f64 {
        if self.generated_tokens <= 1 {
            0.0
        } else {
            self.decode_gpu_sync.submits as f64 / (self.generated_tokens - 1) as f64
        }
    }

    pub fn avg_decode_gpu_poll_waits(&self) -> f64 {
        if self.generated_tokens <= 1 {
            0.0
        } else {
            self.decode_gpu_sync.poll_waits as f64 / (self.generated_tokens - 1) as f64
        }
    }

    pub fn avg_decode_gpu_map_reads(&self) -> f64 {
        if self.generated_tokens <= 1 {
            0.0
        } else {
            self.decode_gpu_sync.map_reads as f64 / (self.generated_tokens - 1) as f64
        }
    }

    pub fn avg_decode_gpu_host_write_buffers(&self) -> f64 {
        if self.generated_tokens <= 1 {
            0.0
        } else {
            self.decode_gpu_sync.host_write_buffers as f64 / (self.generated_tokens - 1) as f64
        }
    }

    pub fn avg_decode_gpu_position_write_buffers(&self) -> f64 {
        if self.generated_tokens <= 1 {
            0.0
        } else {
            self.decode_gpu_sync.position_write_buffers as f64 / (self.generated_tokens - 1) as f64
        }
    }
}

#[derive(Clone, Debug)]
struct NextTokenProfileResult {
    token: u32,
    forward_time: Duration,
    sample_time: Duration,
    gpu_sync: GpuSyncStats,
    fast_path: bool,
    final_layer_delta: Option<TransformerBlockProfile>,
}

fn diff_transformer_block_profile(
    before: &TransformerBlockProfile,
    after: &TransformerBlockProfile,
) -> TransformerBlockProfile {
    TransformerBlockProfile {
        input_norm: after.input_norm.saturating_sub(before.input_norm),
        attention: after.attention.saturating_sub(before.attention),
        post_norm: after.post_norm.saturating_sub(before.post_norm),
        mlp: after.mlp.saturating_sub(before.mlp),
        mlp_gate_up: after.mlp_gate_up.saturating_sub(before.mlp_gate_up),
        mlp_silu_mul: after.mlp_silu_mul.saturating_sub(before.mlp_silu_mul),
        mlp_down_proj: after.mlp_down_proj.saturating_sub(before.mlp_down_proj),
        mlp_q8_gate_up_prep: after
            .mlp_q8_gate_up_prep
            .saturating_sub(before.mlp_q8_gate_up_prep),
        mlp_q8_gate_up_dot: after
            .mlp_q8_gate_up_dot
            .saturating_sub(before.mlp_q8_gate_up_dot),
        mlp_q8_gate_up_writeback: after
            .mlp_q8_gate_up_writeback
            .saturating_sub(before.mlp_q8_gate_up_writeback),
        mlp_q8_down_proj_prep: after
            .mlp_q8_down_proj_prep
            .saturating_sub(before.mlp_q8_down_proj_prep),
        mlp_q8_down_proj_dot: after
            .mlp_q8_down_proj_dot
            .saturating_sub(before.mlp_q8_down_proj_dot),
        mlp_q8_down_proj_writeback: after
            .mlp_q8_down_proj_writeback
            .saturating_sub(before.mlp_q8_down_proj_writeback),
        residual: after.residual.saturating_sub(before.residual),
        total: after.total.saturating_sub(before.total),
    }
}

fn final_layer_profile_delta(
    before: Option<&TransformerBlockProfile>,
    layer_profiles: &[TransformerBlockProfile],
) -> Option<TransformerBlockProfile> {
    let after = layer_profiles.last()?;
    Some(if let Some(before_profile) = before {
        diff_transformer_block_profile(before_profile, after)
    } else {
        after.clone()
    })
}

impl Generator {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        Self::from_pretrained_with_options(model_dir, &RuntimeOptions::default())
    }

    pub fn from_pretrained_with_options<P: AsRef<Path>>(
        model_dir: P,
        runtime_options: &RuntimeOptions,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model = Qwen3Model::from_pretrained_with_options(model_dir, runtime_options)?;

        let tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| RsinferError::TokenizerError(e.to_string()))?;

        let mut stop_tokens = vec![model.config.eos_token_id];
        for special in ["\x3c|im_end|>", "\x3c|endoftext|>"] {
            if let Some(id) = tokenizer.token_to_id(special) {
                if !stop_tokens.contains(&id) {
                    stop_tokens.push(id);
                }
            }
        }

        Ok(Self {
            model,
            tokenizer,
            sampler: default_sampler(),
            config: GenerationConfig {
                stop_tokens,
                ..Default::default()
            },
        })
    }

    pub fn with_sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = sampler;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = max_tokens;
        self
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| RsinferError::TokenizerError(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| RsinferError::TokenizerError(e.to_string()))
    }

    pub fn generate(&self, prompt: &str) -> Result<String> {
        self.generate_stream(prompt, |_| {})
    }

    /// 生成并通过 `on_text` 流式回调新产生的文本，返回完整结果。
    pub fn generate_stream<F: FnMut(&str)>(&self, prompt: &str, mut on_text: F) -> Result<String> {
        self.generate_stream_with_profile(prompt, |text| on_text(text))
            .map(|(text, _)| text)
    }

    pub fn generate_stream_with_profile<F: FnMut(&str)>(
        &self,
        prompt: &str,
        mut on_text: F,
    ) -> Result<(String, GenerationProfile)> {
        self.generate_stream_with_profile_options(prompt, false, false, |text| on_text(text))
    }

    pub fn generate_stream_with_profile_options<F: FnMut(&str)>(
        &self,
        prompt: &str,
        profile_layers: bool,
        profile_token_trace: bool,
        mut on_text: F,
    ) -> Result<(String, GenerationProfile)> {
        let input_ids = self.encode(prompt)?;
        let mut kv_cache = self.model.create_kv_cache();
        let mut profile = GenerationProfile {
            prompt_tokens: input_ids.len(),
            ..Default::default()
        };
        let track_final_layer_trace = profile_layers && profile_token_trace;

        let prefill = self.next_token_profile(
            &input_ids,
            &mut kv_cache,
            0,
            profile_layers,
            track_final_layer_trace,
            &mut profile,
        )?;
        profile.prefill_forward += prefill.forward_time;
        profile.prefill_sample += prefill.sample_time;
        profile.prefill_gpu_sync.record(prefill.gpu_sync);
        if prefill.fast_path {
            profile.fast_path_tokens += 1;
        }
        let prompt_len = input_ids.len();
        let mut next = prefill.token;

        // 逐 token 解码会截断多字节字符，故每步解码整段、只刷出已完整的新增后缀。
        let mut tokens: Vec<u32> = Vec::with_capacity(self.config.max_tokens);
        let mut printed = 0usize;

        for step in 0..self.config.max_tokens {
            if self.config.stop_tokens.contains(&next) {
                break;
            }
            tokens.push(next);

            let decode_start = Instant::now();
            let text = self.decode(&tokens)?;
            profile.text_decode += decode_start.elapsed();
            if !text.ends_with('\u{FFFD}') && text.len() > printed {
                on_text(&text[printed..]);
                printed = text.len();
            }

            if step + 1 < self.config.max_tokens {
                let next_token = self.next_token_profile(
                    &[next],
                    &mut kv_cache,
                    prompt_len + step,
                    profile_layers,
                    track_final_layer_trace,
                    &mut profile,
                )?;
                profile.decode_forward += next_token.forward_time;
                profile
                    .decode_forward_token_ms
                    .push(next_token.forward_time.as_secs_f64() * 1000.0);
                profile.decode_gpu_sync.record(next_token.gpu_sync);
                profile.decode_gpu_sync_trace.push(next_token.gpu_sync);
                if let (Some(layer_index), Some(layer_delta)) = (
                    profile.layer_profiles.len().checked_sub(1),
                    next_token.final_layer_delta.as_ref(),
                ) {
                    profile.record_final_layer_decode_trace(layer_index, layer_delta);
                }
                profile.decode_sample += next_token.sample_time;
                if next_token.fast_path {
                    profile.fast_path_tokens += 1;
                }
                next = next_token.token;
            }
        }

        profile.generated_tokens = tokens.len();
        let decode_start = Instant::now();
        let text = self.decode(&tokens)?;
        profile.text_decode += decode_start.elapsed();
        if text.len() > printed {
            on_text(&text[printed..]);
        }
        Ok((text, profile))
    }

    fn next_token_profile(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
        profile_layers: bool,
        track_final_layer_trace: bool,
        profile: &mut GenerationProfile,
    ) -> Result<NextTokenProfileResult> {
        reset_sync_stats();
        let forward_start = Instant::now();
        let final_layer_before = if track_final_layer_trace {
            profile.layer_profiles.last().cloned()
        } else {
            None
        };
        if self.sampler.is_greedy() && self.model.has_greedy_token_fast_path() {
            let snapshot = kv_cache.snapshot();
            let result = if profile_layers {
                self.model.forward_greedy_token_profiled(
                    input_ids,
                    kv_cache,
                    position_offset,
                    &mut profile.layer_profiles,
                )
            } else {
                self.model
                    .forward_greedy_token(input_ids, kv_cache, position_offset)
            };
            match result {
                Ok(token) => {
                    return Ok(NextTokenProfileResult {
                        token,
                        forward_time: forward_start.elapsed(),
                        sample_time: Duration::ZERO,
                        gpu_sync: sync_stats(),
                        fast_path: true,
                        final_layer_delta: final_layer_profile_delta(
                            final_layer_before.as_ref(),
                            &profile.layer_profiles,
                        ),
                    });
                }
                Err(_) => kv_cache.restore(snapshot)?,
            }
        }

        let logits = if profile_layers {
            self.model.forward_profiled(
                input_ids,
                kv_cache,
                position_offset,
                &mut profile.layer_profiles,
            )?
        } else {
            self.model.forward(input_ids, kv_cache, position_offset)?
        };
        let forward_time = forward_start.elapsed();
        let sample_start = Instant::now();
        let logits_slice = logits.as_slice();
        if logits_slice.len() != logits.numel() {
            return Err(crate::error::RsinferError::DimensionError(
                "logits tensor is not contiguous".into(),
            ));
        }
        let token = self.sampler.sample(logits_slice);
        Ok(NextTokenProfileResult {
            token,
            forward_time,
            sample_time: sample_start.elapsed(),
            gpu_sync: sync_stats(),
            fast_path: false,
            final_layer_delta: final_layer_profile_delta(
                final_layer_before.as_ref(),
                &profile.layer_profiles,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    #[test]
    fn transformer_block_profile_delta_is_fieldwise() {
        let before = TransformerBlockProfile {
            attention: ms(11),
            mlp: ms(13),
            mlp_gate_up: ms(17),
            mlp_down_proj: ms(19),
            mlp_q8_gate_up_dot: ms(23),
            mlp_q8_down_proj_dot: ms(29),
            total: ms(31),
            ..Default::default()
        };
        let after = TransformerBlockProfile {
            attention: ms(41),
            mlp: ms(53),
            mlp_gate_up: ms(67),
            mlp_down_proj: ms(79),
            mlp_q8_gate_up_dot: ms(83),
            mlp_q8_down_proj_dot: ms(97),
            total: ms(101),
            ..Default::default()
        };

        let delta = diff_transformer_block_profile(&before, &after);

        assert_eq!(delta.attention, ms(30));
        assert_eq!(delta.mlp, ms(40));
        assert_eq!(delta.mlp_gate_up, ms(50));
        assert_eq!(delta.mlp_down_proj, ms(60));
        assert_eq!(delta.mlp_q8_gate_up_dot, ms(60));
        assert_eq!(delta.mlp_q8_down_proj_dot, ms(68));
        assert_eq!(delta.total, ms(70));
    }

    #[test]
    fn generation_profile_records_final_layer_trace_in_ms() {
        let mut profile = GenerationProfile::default();
        let delta = TransformerBlockProfile {
            attention: Duration::from_micros(500),
            mlp: Duration::from_micros(1_500),
            mlp_gate_up: Duration::from_micros(250),
            mlp_down_proj: Duration::from_micros(750),
            mlp_q8_gate_up_dot: Duration::from_micros(1_000),
            mlp_q8_down_proj_dot: Duration::from_micros(1_250),
            total: Duration::from_micros(2_500),
            ..Default::default()
        };

        profile.record_final_layer_decode_trace(35, &delta);

        assert_eq!(profile.final_layer_index, Some(35));
        assert_eq!(profile.final_layer_decode_trace.len(), 1);
        let trace = &profile.final_layer_decode_trace[0];
        assert!((trace.total_ms - 2.5).abs() < 1e-9);
        assert!((trace.attention_ms - 0.5).abs() < 1e-9);
        assert!((trace.mlp_ms - 1.5).abs() < 1e-9);
        assert!((trace.mlp_gate_up_ms - 0.25).abs() < 1e-9);
        assert!((trace.mlp_down_proj_ms - 0.75).abs() < 1e-9);
        assert!((trace.mlp_q8_gate_up_dot_ms - 1.0).abs() < 1e-9);
        assert!((trace.mlp_q8_down_proj_dot_ms - 1.25).abs() < 1e-9);
    }
}
