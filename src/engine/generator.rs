//! 自回归文本生成。

use std::path::Path;

use crate::error::{Result, RsinferError};
use crate::model::Qwen3Model;
use crate::engine::sampler::{Sampler, default_sampler};

pub struct GenerationConfig {
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_tokens: 256, stop_tokens: Vec::new() }
    }
}

pub struct Generator {
    pub model: Qwen3Model,
    pub tokenizer: tokenizers::Tokenizer,
    pub sampler: Box<dyn Sampler>,
    pub config: GenerationConfig,
}

impl Generator {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let model = Qwen3Model::from_pretrained(model_dir)?;

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
            config: GenerationConfig { stop_tokens, ..Default::default() },
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
        let input_ids = self.encode(prompt)?;
        let mut kv_cache = self.model.create_kv_cache();

        let logits = self.model.forward(&input_ids, &mut kv_cache, 0)?;
        let mut next = self.sampler.sample(&logits.to_1d()?);
        let prompt_len = input_ids.len();

        // 逐 token 解码会截断多字节字符，故每步解码整段、只刷出已完整的新增后缀。
        let mut tokens: Vec<u32> = Vec::with_capacity(self.config.max_tokens);
        let mut printed = 0usize;

        for step in 0..self.config.max_tokens {
            if self.config.stop_tokens.contains(&next) {
                break;
            }
            tokens.push(next);

            let text = self.decode(&tokens)?;
            if !text.ends_with('\u{FFFD}') && text.len() > printed {
                on_text(&text[printed..]);
                printed = text.len();
            }

            let logits = self.model.forward(&[next], &mut kv_cache, prompt_len + step)?;
            next = self.sampler.sample(&logits.to_1d()?);
        }

        let text = self.decode(&tokens)?;
        if text.len() > printed {
            on_text(&text[printed..]);
        }
        Ok(text)
    }
}
