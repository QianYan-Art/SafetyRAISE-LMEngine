//! LLaMA 模型配置解析
//!
//! 解析 HuggingFace 格式的 config.json 文件。

use serde::Deserialize;
use std::path::Path;
use std::fs;

use crate::error::Result;

/// LLaMA/Qwen 模型配置
///
/// 对应 HuggingFace 的 config.json 格式。
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    /// 隐藏层维度
    pub hidden_size: usize,
    
    /// MLP 中间层维度
    pub intermediate_size: usize,
    
    /// Transformer 层数
    pub num_hidden_layers: usize,
    
    /// 注意力头数
    pub num_attention_heads: usize,
    
    /// KV 头数 (用于 GQA)。若 config.json 未提供，则在 `from_file` 中
    /// 回退为 `num_attention_heads`（即标准 MHA）。
    #[serde(default)]
    pub num_key_value_heads: usize,
    
    /// 每个注意力头的维度 (可选，Qwen3 等模型显式指定)
    #[serde(default, rename = "head_dim")]
    pub explicit_head_dim: Option<usize>,
    
    /// 词汇表大小
    pub vocab_size: usize,
    
    /// 最大位置编码长度
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    
    /// RMSNorm 的 epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    
    /// RoPE 的 theta 值
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    
    /// 模型类型
    #[serde(default)]
    pub model_type: String,
    
    /// 数据类型
    #[serde(default)]
    pub torch_dtype: String,
    
    /// 是否绑定 embedding 和 lm_head 权重
    #[serde(default)]
    pub tie_word_embeddings: bool,
    
    /// EOS token ID
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    
    /// BOS token ID
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

fn default_max_position() -> usize {
    2048
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_eos_token_id() -> u32 {
    2 // 默认 LLaMA EOS token ID
}

impl LlamaConfig {
    /// 从 config.json 文件加载配置
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut config: LlamaConfig = serde_json::from_str(&content)?;

        // 未指定（或为 0）时回退为 num_attention_heads（标准 MHA）
        if config.num_key_value_heads == 0 {
            config.num_key_value_heads = config.num_attention_heads;
        }

        Ok(config)
    }
    
    /// 计算每个 head 的维度
    /// 
    /// 如果配置中显式指定了 head_dim，使用显式值；
    /// 否则使用 hidden_size / num_attention_heads
    pub fn head_dim(&self) -> usize {
        self.explicit_head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    
    /// 计算 GQA 的扩展倍数
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parse() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "model_type": "llama"
        }"#;
        
        let config: LlamaConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.head_dim(), 128);
    }
}
