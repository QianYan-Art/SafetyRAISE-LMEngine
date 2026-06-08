//! Qwen3 模型组装与前向传播。

use std::path::Path;

use ndarray::{s, Array2};

use crate::engine::KVCache;
use crate::error::{Result, RsinferError};
use crate::model::config::Qwen3Config;
use crate::model::layers::{build_transformer_block, Linear, RmsNorm, TransformerBlock};
use crate::model::weights::{get_weight, load_weights, WeightMap};
use crate::tensor::Tensor;

pub struct Qwen3Model {
    pub config: Qwen3Config,
    pub embed_tokens: Tensor, // [vocab_size, hidden_size]
    pub layers: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    pub lm_head: Linear,
}

impl Qwen3Model {
    pub fn from_pretrained<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Qwen3Config::from_file(model_dir.join("config.json"))?;
        let weights = load_weights(model_dir)?;
        Self::from_weights(&config, &weights)
    }

    pub fn from_weights(config: &Qwen3Config, weights: &WeightMap) -> Result<Self> {
        let embed_tokens = get_weight(weights, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(build_transformer_block(weights, config, layer_idx)?);
        }

        let norm = RmsNorm::new(
            get_weight(weights, "model.norm.weight")?,
            config.rms_norm_eps,
        );

        // tie_word_embeddings 时 lm_head 复用 embedding 权重
        let lm_head = match Linear::from_weight_map(weights, "lm_head.weight", None) {
            Ok(linear) => linear,
            Err(_) if config.tie_word_embeddings => {
                Linear::from_weight_map(weights, "model.embed_tokens.weight", None)?
            }
            Err(_) => {
                return Err(RsinferError::WeightError(
                    "缺少 lm_head.weight 且 tie_word_embeddings 为 false".into(),
                ))
            }
        };

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// 返回最后一个位置的 logits: [1, vocab_size]。
    /// 自回归只需最后一步，prefill 时也借此跳过对整段序列的 lm_head。
    pub fn forward(
        &self,
        input_ids: &[u32],
        kv_cache: &mut KVCache,
        position_offset: usize,
    ) -> Result<Tensor> {
        let mut hidden = self.embedding(input_ids)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, kv_cache, layer_idx, position_offset)?;
        }
        let last = self.norm.forward(&hidden.last_row()?)?;
        self.lm_head.forward(&last)
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
