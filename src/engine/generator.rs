//! 自回归文本生成。

use std::path::Path;
use std::time::{Duration, Instant};

use crate::engine::sampler::{default_sampler, Sampler};
use crate::engine::KVCache;
use crate::error::{Result, RsinferError};
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
pub struct GenerationProfile {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_forward: Duration,
    pub prefill_sample: Duration,
    pub decode_forward: Duration,
    pub decode_forward_token_ms: Vec<f64>,
    pub decode_sample: Duration,
    pub text_decode: Duration,
    pub fast_path_tokens: usize,
    pub layer_profiles: Vec<TransformerBlockProfile>,
}

impl GenerationProfile {
    pub fn avg_decode_forward_ms(&self) -> f64 {
        if self.generated_tokens <= 1 {
            0.0
        } else {
            self.decode_forward.as_secs_f64() * 1000.0 / (self.generated_tokens - 1) as f64
        }
    }
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
        self.generate_stream_with_profile_options(prompt, false, |text| on_text(text))
    }

    pub fn generate_stream_with_profile_options<F: FnMut(&str)>(
        &self,
        prompt: &str,
        profile_layers: bool,
        mut on_text: F,
    ) -> Result<(String, GenerationProfile)> {
        let input_ids = self.encode(prompt)?;
        let mut kv_cache = self.model.create_kv_cache();
        let mut profile = GenerationProfile {
            prompt_tokens: input_ids.len(),
            ..Default::default()
        };

        let (mut next, forward_time, sample_time, fast_path) =
            self.next_token_profile(&input_ids, &mut kv_cache, 0, profile_layers, &mut profile)?;
        profile.prefill_forward += forward_time;
        profile.prefill_sample += sample_time;
        if fast_path {
            profile.fast_path_tokens += 1;
        }
        let prompt_len = input_ids.len();

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
                let (token, forward_time, sample_time, fast_path) = self.next_token_profile(
                    &[next],
                    &mut kv_cache,
                    prompt_len + step,
                    profile_layers,
                    &mut profile,
                )?;
                profile.decode_forward += forward_time;
                profile
                    .decode_forward_token_ms
                    .push(forward_time.as_secs_f64() * 1000.0);
                profile.decode_sample += sample_time;
                if fast_path {
                    profile.fast_path_tokens += 1;
                }
                next = token;
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
        profile: &mut GenerationProfile,
    ) -> Result<(u32, Duration, Duration, bool)> {
        let forward_start = Instant::now();
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
                Ok(token) => return Ok((token, forward_start.elapsed(), Duration::ZERO, true)),
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
        Ok((token, forward_time, sample_start.elapsed(), false))
    }
}
