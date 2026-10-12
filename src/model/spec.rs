//! 从实际模型配置建立数学合同；发行名、设备配置与量化格式不参与架构猜测。

use crate::{Result, RsinferError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    Qwen2,
    Qwen3,
    Qwen35,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSpec {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GdnSpec {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeSpec {
    pub theta: f32,
    pub rotary_dim: usize,
    pub mrope_section: Option<[usize; 3]>,
    pub interleaved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionSpec {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub family: ModelFamily,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub layers: Vec<LayerKind>,
    pub attention: AttentionSpec,
    pub gdn: Option<GdnSpec>,
    pub rope: RopeSpec,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub eos_token_ids: Vec<u32>,
    pub vision: Option<VisionSpec>,
    pub image_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
}

fn invalid(message: impl Into<String>) -> RsinferError {
    RsinferError::ConfigError(message.into())
}

fn positive(value: &Value, key: &str) -> Result<usize> {
    let n = value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|n| *n > 0);
    n.ok_or_else(|| invalid(format!("{key} 必须是可表示的正整数")))
}

fn optional_positive(value: &Value, key: &str, default: usize) -> Result<usize> {
    if value.get(key).is_some() {
        positive(value, key)
    } else {
        Ok(default)
    }
}

fn token_id(value: &Value, key: &str) -> Result<Option<u32>> {
    match value.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| invalid(format!("{key} 不是有效 token ID"))),
    }
}

fn finite_positive(value: &Value, key: &str, default: f32) -> Result<f32> {
    let n = match value.get(key) {
        None => default,
        Some(v) => v
            .as_f64()
            .ok_or_else(|| invalid(format!("{key} 必须是数值")))? as f32,
    };
    if n.is_finite() && n > 0.0 {
        Ok(n)
    } else {
        Err(invalid(format!("{key} 必须有限且大于零")))
    }
}

fn boolean(value: &Value, key: &str, default: bool) -> Result<bool> {
    match value.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| invalid(format!("{key} 必须是布尔值"))),
    }
}

impl ModelSpec {
    pub fn from_hf(root: &Value) -> Result<Self> {
        let text = root.get("text_config").unwrap_or(root);
        let kind = text
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("缺少文本 model_type"))?;
        let family = match kind {
            "qwen2" => ModelFamily::Qwen2,
            "qwen3" => ModelFamily::Qwen3,
            "qwen3_5_text" | "qwen3_5" => ModelFamily::Qwen35,
            other => return Err(RsinferError::Unsupported(format!("模型架构 {other}"))),
        };
        if let Some(architectures) = root.get("architectures") {
            let names = architectures
                .as_array()
                .ok_or_else(|| invalid("architectures 必须是数组"))?;
            if names.len() != 1 {
                return Err(invalid("architectures 必须唯一确定模型"));
            }
            let expected = match family {
                ModelFamily::Qwen2 => &["Qwen2ForCausalLM"][..],
                ModelFamily::Qwen3 => &["Qwen3ForCausalLM"][..],
                ModelFamily::Qwen35 => {
                    &["Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"][..]
                }
            };
            if !names[0]
                .as_str()
                .is_some_and(|name| expected.contains(&name))
            {
                return Err(invalid("architectures 与 model_type 不匹配"));
            }
        }
        if text
            .get("hidden_act")
            .and_then(Value::as_str)
            .is_some_and(|act| act != "silu")
        {
            return Err(RsinferError::Unsupported("文本激活函数不是 SiLU".into()));
        }
        if boolean(text, "use_sliding_window", false)? {
            return Err(RsinferError::Unsupported("当前未实现滑动窗口注意力".into()));
        }
        if boolean(text, "attention_bias", family == ModelFamily::Qwen2)?
            != (family == ModelFamily::Qwen2)
            || boolean(text, "attn_output_gate", family == ModelFamily::Qwen35)?
                != (family == ModelFamily::Qwen35)
        {
            return Err(RsinferError::Unsupported(
                "当前注意力 bias/gate 配置与已实现架构不一致".into(),
            ));
        }
        let hidden_size = positive(text, "hidden_size")?;
        let heads = positive(text, "num_attention_heads")?;
        let head_dim = if text.get("head_dim").is_some() {
            positive(text, "head_dim")?
        } else {
            if !hidden_size.is_multiple_of(heads) {
                return Err(invalid("hidden_size 不能整除注意力头数"));
            }
            hidden_size / heads
        };
        let attention = AttentionSpec {
            heads,
            kv_heads: optional_positive(text, "num_key_value_heads", heads)?,
            head_dim,
        };
        let count = positive(text, "num_hidden_layers")?;
        if count > 4096 {
            return Err(invalid("层数超过配置解析上限 4096"));
        }
        let layers = if let Some(kinds) = text.get("layer_types").filter(|v| !v.is_null()) {
            serde_json::from_value::<Vec<LayerKind>>(kinds.clone())?
        } else if family == ModelFamily::Qwen35 {
            let interval = optional_positive(text, "full_attention_interval", 4)?;
            (0..count)
                .map(|i| {
                    if (i + 1) % interval == 0 {
                        LayerKind::FullAttention
                    } else {
                        LayerKind::LinearAttention
                    }
                })
                .collect()
        } else {
            vec![LayerKind::FullAttention; count]
        };
        if layers.len() != count {
            return Err(invalid("layer_types 数量与 num_hidden_layers 不一致"));
        }
        let gdn = if layers.contains(&LayerKind::LinearAttention) {
            Some(GdnSpec {
                key_heads: positive(text, "linear_num_key_heads")?,
                value_heads: positive(text, "linear_num_value_heads")?,
                key_dim: positive(text, "linear_key_head_dim")?,
                value_dim: positive(text, "linear_value_head_dim")?,
                conv_kernel: positive(text, "linear_conv_kernel_dim")?,
            })
        } else {
            None
        };
        let rope = text
            .get("rope_parameters")
            .filter(|v| !v.is_null())
            .or_else(|| text.get("rope_scaling").filter(|v| !v.is_null()));
        if let (Some(a), Some(b)) = (
            text.get("rope_parameters").filter(|v| !v.is_null()),
            text.get("rope_scaling").filter(|v| !v.is_null()),
        ) {
            if a != b {
                return Err(invalid("rope_parameters 与 rope_scaling 冲突"));
            }
        }
        if let Some(r) = rope {
            if !r.is_object() {
                return Err(invalid("RoPE 参数必须是对象"));
            }
            let scaling = r
                .get("rope_type")
                .or_else(|| r.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("default");
            if scaling != "default" {
                return Err(RsinferError::Unsupported(format!("RoPE 缩放 {scaling}")));
            }
            if r.get("factor").is_some_and(|v| v.as_f64() != Some(1.0)) {
                return Err(RsinferError::Unsupported("当前只实现未缩放 RoPE".into()));
            }
        }
        let rope = rope.unwrap_or(text);
        let fraction = finite_positive(
            rope,
            "partial_rotary_factor",
            finite_positive(
                text,
                "partial_rotary_factor",
                if family == ModelFamily::Qwen35 {
                    0.25
                } else {
                    1.0
                },
            )?,
        )?;
        if fraction > 1.0 {
            return Err(invalid("partial_rotary_factor 不能超过 1"));
        }
        let rotary_dim = (head_dim as f64 * fraction as f64) as usize;
        let theta_default = finite_positive(
            text,
            "rope_theta",
            if family == ModelFamily::Qwen35 {
                10_000_000.0
            } else {
                1_000_000.0
            },
        )?;
        let rope = RopeSpec {
            theta: finite_positive(rope, "rope_theta", theta_default)?,
            rotary_dim,
            mrope_section: rope
                .get("mrope_section")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?,
            interleaved: boolean(rope, "mrope_interleaved", true)?,
        };
        let eos = text
            .get("eos_token_id")
            .or_else(|| root.get("eos_token_id"));
        let eos_token_ids = match eos.filter(|v| !v.is_null()) {
            None => vec![],
            Some(Value::Array(ids)) => ids
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or_else(|| invalid("eos_token_id 数组包含非法 ID"))
                })
                .collect::<Result<Vec<_>>>()?,
            Some(_) => token_id(
                if text.get("eos_token_id").is_some() {
                    text
                } else {
                    root
                },
                "eos_token_id",
            )?
            .into_iter()
            .collect(),
        };
        if let Some(v) = root.get("vision_config").filter(|v| !v.is_null()) {
            if v.get("model_type")
                .is_some_and(|t| !matches!(t.as_str(), Some("qwen3_5_vision" | "qwen3_5")))
                || v.get("hidden_act")
                    .is_some_and(|a| a.as_str() != Some("gelu_pytorch_tanh"))
                || boolean(v, "use_postshuffle_norm", false)?
            {
                return Err(RsinferError::Unsupported(
                    "视觉架构、激活或 merger norm 配置未实现".into(),
                ));
            }
            if let Some(r) = v.get("rope_parameters").filter(|v| !v.is_null()) {
                if !r.is_object()
                    || r.get("rope_type")
                        .is_some_and(|t| t.as_str() != Some("axial"))
                    || finite_positive(r, "rope_theta", 10_000.0)? != 10_000.0
                {
                    return Err(RsinferError::Unsupported(
                        "视觉仅支持 theta=10000 的 axial RoPE".into(),
                    ));
                }
            }
            if finite_positive(v, "rope_theta", 10_000.0)? != 10_000.0 {
                return Err(RsinferError::Unsupported("视觉 rope_theta 未实现".into()));
            }
        }
        let vision = root
            .get("vision_config")
            .filter(|v| !v.is_null())
            .map(|v| serde_json::from_value::<VisionSpec>(v.clone()))
            .transpose()?;
        let spec = Self {
            family,
            hidden_size,
            intermediate_size: positive(text, "intermediate_size")?,
            vocab_size: positive(text, "vocab_size")?,
            layers,
            attention,
            gdn,
            rope,
            rms_norm_eps: finite_positive(text, "rms_norm_eps", 1e-6)?,
            max_position_embeddings: positive(text, "max_position_embeddings")?,
            tie_word_embeddings: boolean(
                text,
                "tie_word_embeddings",
                boolean(root, "tie_word_embeddings", false)?,
            )?,
            eos_token_ids,
            vision,
            image_token_id: token_id(root, "image_token_id")?,
            vision_start_token_id: token_id(root, "vision_start_token_id")?,
            vision_end_token_id: token_id(root, "vision_end_token_id")?,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        let a = &self.attention;
        if [
            self.hidden_size,
            self.intermediate_size,
            self.vocab_size,
            self.layers.len(),
            self.max_position_embeddings,
            a.heads,
            a.kv_heads,
            a.head_dim,
        ]
        .contains(&0)
        {
            return Err(invalid("模型维度不能为零"));
        }
        if !a.heads.is_multiple_of(a.kv_heads) {
            return Err(invalid("Q 头数必须是 KV 头数的整数倍"));
        }
        if self.rope.rotary_dim == 0
            || self.rope.rotary_dim > a.head_dim
            || !self.rope.rotary_dim.is_multiple_of(2)
        {
            return Err(invalid("RoPE 维度须为不超过 head_dim 的正偶数"));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope.theta.is_finite()
            || self.rope.theta <= 0.0
        {
            return Err(invalid("归一化与 RoPE 参数须有限且大于零"));
        }
        if self.vocab_size > u32::MAX as usize
            || self
                .eos_token_ids
                .iter()
                .any(|id| *id as usize >= self.vocab_size)
        {
            return Err(invalid("词表或 EOS ID 超出有效范围"));
        }
        if self.layers.contains(&LayerKind::LinearAttention)
            && (self.family != ModelFamily::Qwen35 || self.gdn.is_none())
        {
            return Err(invalid("线性注意力必须属于 Qwen3.5 系并包含 GDN 配置"));
        }
        if let Some(g) = &self.gdn {
            if [
                g.key_heads,
                g.value_heads,
                g.key_dim,
                g.value_dim,
                g.conv_kernel,
            ]
            .contains(&0)
                || !g.value_heads.is_multiple_of(g.key_heads)
            {
                return Err(invalid("GDN 维度或 key/value 头映射非法"));
            }
        }
        if let Some(sections) = self.rope.mrope_section {
            let sum = sections
                .into_iter()
                .try_fold(0usize, |a, b| a.checked_add(b))
                .ok_or_else(|| invalid("MRoPE section 溢出"))?;
            if sum != self.rope.rotary_dim / 2 {
                return Err(invalid("MRoPE section 总和必须等于旋转维度的一半"));
            }
        }
        if let Some(v) = &self.vision {
            if self.family != ModelFamily::Qwen35
                || [
                    v.depth,
                    v.hidden_size,
                    v.intermediate_size,
                    v.num_heads,
                    v.patch_size,
                    v.spatial_merge_size,
                    v.temporal_patch_size,
                    v.num_position_embeddings,
                ]
                .contains(&0)
                || v.in_channels != 3
                || !v.hidden_size.is_multiple_of(v.num_heads)
                || v.out_hidden_size != self.hidden_size
                || v.num_position_embeddings.isqrt().checked_pow(2)
                    != Some(v.num_position_embeddings)
            {
                return Err(invalid("视觉配置的维度、通道或文本交接宽度非法"));
            }
            if self
                .image_token_id
                .is_none_or(|id| (id as usize) >= self.vocab_size)
            {
                return Err(invalid("视觉模型必须提供词表内的 image_token_id"));
            }
        }
        // 在任何权重或状态分配之前检查乘法边界。
        for dims in [
            vec![self.vocab_size, self.hidden_size, 4],
            vec![self.intermediate_size, self.hidden_size, 4],
            vec![a.heads, a.head_dim, self.hidden_size, 4],
        ] {
            checked_elements(&dims)?;
        }
        self.state_bytes(1)?;
        Ok(())
    }

    /// CPU 参考状态的预算，不包含权重、激活工作区及 allocator 开销。
    pub fn state_bytes(&self, context: usize) -> Result<usize> {
        let full = self
            .layers
            .iter()
            .filter(|k| **k == LayerKind::FullAttention)
            .count();
        let mut bytes = checked_elements(&[
            full,
            context,
            self.attention.kv_heads,
            self.attention.head_dim,
            2,
            4,
        ])?;
        if let Some(g) = &self.gdn {
            let recurrent = checked_elements(&[g.value_heads, g.key_dim, g.value_dim])?;
            let key_width = checked_elements(&[g.key_heads, g.key_dim, 2])?;
            let value_width = checked_elements(&[g.value_heads, g.value_dim])?;
            let width = key_width
                .checked_add(value_width)
                .ok_or_else(|| invalid("GDN 宽度溢出"))?;
            let convolution = checked_elements(&[width, g.conv_kernel.saturating_sub(1)])?;
            let per_layer = recurrent
                .checked_add(convolution)
                .ok_or_else(|| invalid("GDN 状态溢出"))?;
            let linear = self.layers.len() - full;
            bytes = bytes
                .checked_add(checked_elements(&[linear, per_layer, 4])?)
                .ok_or_else(|| invalid("总状态大小溢出"))?;
        }
        Ok(bytes)
    }
}

pub(crate) fn checked_elements(dims: &[usize]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| invalid("维度乘积溢出"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dense() -> Value {
        json!({"model_type":"qwen3","architectures":["Qwen3ForCausalLM"],"hidden_size":16,
            "intermediate_size":32,"vocab_size":64,"num_hidden_layers":2,"num_attention_heads":4,
            "num_key_value_heads":2,"head_dim":4,"max_position_embeddings":128,"eos_token_id":[1,2]})
    }

    #[test]
    fn rejects_architecture_and_dimension_mismatches() {
        let base = dense();
        let spec = ModelSpec::from_hf(&base).unwrap();
        assert_eq!(spec.state_bytes(8).unwrap(), 2 * 8 * 2 * 4 * 2 * 4);
        for (key, value) in [
            ("num_key_value_heads", json!(0)),
            ("num_key_value_heads", json!(3)),
            ("head_dim", json!(3)),
            ("eos_token_id", json!(64)),
            ("architectures", json!(["Qwen2ForCausalLM"])),
            ("rope_scaling", json!({"type":"yarn"})),
        ] {
            let mut bad = base.clone();
            bad[key] = value;
            assert!(ModelSpec::from_hf(&bad).is_err(), "{key}");
        }
    }

    #[test]
    fn hybrid_layer_pattern_has_distinct_recurrent_budget() {
        let mut text = dense();
        text["model_type"] = json!("qwen3_5_text");
        text.as_object_mut().unwrap().remove("architectures");
        text["num_hidden_layers"] = json!(4);
        text["head_dim"] = json!(8);
        for (key, value) in [
            ("linear_num_key_heads", 1),
            ("linear_num_value_heads", 2),
            ("linear_key_head_dim", 2),
            ("linear_value_head_dim", 3),
            ("linear_conv_kernel_dim", 4),
        ] {
            text[key] = json!(value);
        }
        let spec = ModelSpec::from_hf(&json!({"model_type":"qwen3_5","text_config":text})).unwrap();
        assert_eq!(
            spec.layers,
            [
                LayerKind::LinearAttention,
                LayerKind::LinearAttention,
                LayerKind::LinearAttention,
                LayerKind::FullAttention
            ]
        );
        assert_eq!(
            spec.state_bytes(8).unwrap(),
            8 * 2 * 8 * 2 * 4 + 3 * (2 * 2 * 3 + (2 * 2 + 2 * 3) * 3) * 4
        );
        assert!(spec.state_bytes(usize::MAX).is_err());
    }
}
