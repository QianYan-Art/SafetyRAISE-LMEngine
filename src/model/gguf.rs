//! GGUF Qwen 权重桥接。
//!
//! GGUF 是 llama.cpp/ggml 的执行布局，模型执行器使用 Hugging Face 风格的
//! 命名和参数语义。本模块只做元数据严格校验、名称映射及少数小型参数的
//! 语义恢复；投影矩阵和量化权重始终共享原只读映射，不整模反量化复制。
//!
//! 依据的上游固定快照为 llama.cpp `8bb3cf0d216a5ae48360179f2931daf284be4d01`：
//! `conversion/qwen.py`、`gguf-py/gguf/tensor_mapping.py`、`src/llama-arch.cpp`。

use std::collections::BTreeMap;

use serde_json::{Map, Number, Value};

use crate::artifact::{WeightStore, WeightTensor};
use crate::error::{Result, RsinferError};
use crate::model::spec::ModelSpec;

fn invalid(message: impl Into<String>) -> RsinferError {
    RsinferError::WeightError(message.into())
}

fn metadata<'a>(store: &'a WeightStore, key: &str) -> Result<&'a Value> {
    store
        .metadata()
        .get(key)
        .ok_or_else(|| invalid(format!("GGUF metadata 缺少 {key}")))
}

fn number(value: &Value, key: &str) -> Result<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
        .ok_or_else(|| invalid(format!("GGUF metadata {key} 不是非负整数")))
}

fn optional_number(store: &WeightStore, key: &str) -> Result<Option<u64>> {
    store
        .metadata()
        .get(key)
        .map(|value| number(value, key))
        .transpose()
}

fn required_usize(store: &WeightStore, key: &str) -> Result<usize> {
    usize::try_from(number(metadata(store, key)?, key)?)
        .map_err(|_| invalid(format!("GGUF metadata {key} 超过 usize")))
}

fn optional_usize(store: &WeightStore, key: &str) -> Result<Option<usize>> {
    optional_number(store, key)?
        .map(|value| {
            usize::try_from(value).map_err(|_| invalid(format!("GGUF metadata {key} 超过 usize")))
        })
        .transpose()
}

fn finite_number(value: &Value, key: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(format!("GGUF metadata {key} 不是有限浮点数")))
}

fn make_number(value: f64, key: &str) -> Result<Value> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| invalid(format!("无法编码 metadata {key} 的浮点数")))
}

fn mrope_sections(value: &Value, key: &str) -> Result<Value> {
    let values = value
        .as_array()
        .ok_or_else(|| invalid(format!("GGUF metadata {key} 必须是整数数组")))?;
    let mut sections = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let section = value
            .as_u64()
            .ok_or_else(|| invalid(format!("GGUF metadata {key}[{index}] 不是非负整数")))?;
        sections.push(section);
    }
    let sections = match sections.as_slice() {
        [first, second, third] => vec![*first, *second, *third],
        [first, second, third, 0] => vec![*first, *second, *third],
        _ => {
            return Err(invalid(format!(
                "GGUF metadata {key} 必须包含 3 项，或包含尾部为 0 的 4 项"
            )))
        }
    };
    Ok(Value::Array(
        sections
            .into_iter()
            .map(|section| Value::Number(Number::from(section)))
            .collect(),
    ))
}

fn string_value(store: &WeightStore, key: &str) -> Result<String> {
    metadata(store, key)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| invalid(format!("GGUF metadata {key} 不是字符串")))
}

fn set(map: &mut Map<String, Value>, key: &str, value: impl Into<Value>) {
    map.insert(key.to_owned(), value.into());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorTransform {
    None,
    Norm,
    ALog,
    VRows,
    VRowsQkv,
    VRowsConv,
    VVector,
    VColumns,
}

/// 将 Qwen2/Qwen3/Qwen3.5 GGUF 规范化为 native 执行器使用的 HF 命名。
///
/// 官方 GGUF 转换器把 GDN 的 V-head 写成 tiled 顺序，而 native 执行器按
/// grouped 顺序消费；本函数为受影响的 qkv、z、a、b、卷积、状态向量及
/// out 投影建立惰性 tiled → grouped 视图。无法确认的架构、分片 MTP 或
/// 不完整的 GDN 元数据会显式返回错误。
pub fn normalize(store: &WeightStore) -> Result<(WeightStore, ModelSpec)> {
    if store.metadata().is_empty() {
        return Err(invalid("GGUF 桥接需要带 metadata 的 GGUF WeightStore"));
    }
    if optional_number(store, "split.count")?.is_some_and(|count| count > 1) {
        return Err(RsinferError::Unsupported(
            "暂不支持跨多个 GGUF 文件的 split.count 分片".into(),
        ));
    }

    let architecture = string_value(store, "general.architecture")?;
    let prefix = match architecture.as_str() {
        "qwen2" | "qwen3" | "qwen35" => architecture.as_str(),
        "qwen35moe" => {
            return Err(RsinferError::Unsupported(
                "GGUF Qwen3.5 MoE 尚未接入稠密 native 执行器".into(),
            ))
        }
        other => {
            return Err(RsinferError::Unsupported(format!(
                "不支持的 GGUF 架构 {other}"
            )))
        }
    };
    if optional_number(store, &format!("{prefix}.nextn_predict_layers"))?.is_some_and(|n| n > 0) {
        return Err(RsinferError::Unsupported(
            "GGUF 包含 MTP/nextn 层，当前 native ModelSpec 未定义 MTP 状态".into(),
        ));
    }

    let config = gguf_config(store, prefix, &architecture)?;
    let spec = ModelSpec::from_hf(&config)?;
    let tensors = map_tensors(store, &architecture, &spec)?;
    let normalized = WeightStore::from_tensors(config, tensors)?;
    Ok((normalized, spec))
}

fn gguf_config(store: &WeightStore, prefix: &str, architecture: &str) -> Result<Value> {
    let hidden_size = required_usize(store, &format!("{prefix}.embedding_length"))?;
    let intermediate_size = required_usize(store, &format!("{prefix}.feed_forward_length"))?;
    let layers = required_usize(store, &format!("{prefix}.block_count"))?;
    let heads = required_usize(store, &format!("{prefix}.attention.head_count"))?;
    let kv_heads =
        optional_usize(store, &format!("{prefix}.attention.head_count_kv"))?.unwrap_or(heads);
    let context = required_usize(store, &format!("{prefix}.context_length"))?;
    let rms_eps = store
        .metadata()
        .get(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
        .map(|value| finite_number(value, &format!("{prefix}.attention.layer_norm_rms_epsilon")))
        .transpose()?
        .unwrap_or(1e-6);
    let head_dim = optional_usize(store, &format!("{prefix}.attention.key_length"))?
        .unwrap_or_else(|| hidden_size / heads.max(1));
    if heads == 0 || kv_heads == 0 || head_dim == 0 {
        return Err(invalid("GGUF attention 头参数必须为正数"));
    }
    if !hidden_size.is_multiple_of(heads)
        && !store
            .metadata()
            .contains_key(&format!("{prefix}.attention.key_length"))
    {
        return Err(invalid(
            "GGUF hidden_size 不能整除 attention.head_count，且缺少 key_length",
        ));
    }
    let vocab_size = optional_usize(store, &format!("{prefix}.vocab_size"))?
        .or_else(|| {
            store
                .get("token_embd.weight")
                .ok()
                .and_then(|tensor| tensor.shape().first().copied())
        })
        .ok_or_else(|| invalid("GGUF 缺少 vocab_size 且无法从 token_embd.weight 推断"))?;

    let model_type = match architecture {
        "qwen2" => "qwen2",
        "qwen3" => "qwen3",
        "qwen35" => "qwen3_5_text",
        _ => unreachable!(),
    };
    let architecture_name = match architecture {
        "qwen2" => "Qwen2ForCausalLM",
        "qwen3" => "Qwen3ForCausalLM",
        "qwen35" => "Qwen3_5ForCausalLM",
        _ => unreachable!(),
    };

    let mut config = Map::new();
    set(&mut config, "model_type", model_type);
    set(
        &mut config,
        "architectures",
        Value::Array(vec![Value::String(architecture_name.into())]),
    );
    set(&mut config, "hidden_size", hidden_size as u64);
    set(&mut config, "intermediate_size", intermediate_size as u64);
    set(&mut config, "vocab_size", vocab_size as u64);
    set(&mut config, "num_hidden_layers", layers as u64);
    set(&mut config, "num_attention_heads", heads as u64);
    set(&mut config, "num_key_value_heads", kv_heads as u64);
    set(&mut config, "head_dim", head_dim as u64);
    set(&mut config, "max_position_embeddings", context as u64);
    set(&mut config, "rms_norm_eps", rms_eps);
    set(&mut config, "hidden_act", "silu");
    set(
        &mut config,
        "tie_word_embeddings",
        !has_tensor(store, "output"),
    );

    if let Some(eos) = store.metadata().get("tokenizer.ggml.eos_token_id") {
        set(&mut config, "eos_token_id", eos.clone());
    }

    let rope_key = format!("{prefix}.rope.dimension_count");
    let rope_theta_key = format!("{prefix}.rope.freq_base");
    let mut rope = Map::new();
    if let Some(rotary_dim) = optional_usize(store, &rope_key)? {
        if rotary_dim > head_dim {
            return Err(invalid(format!("{rope_key} 大于 head_dim")));
        }
        set(
            &mut rope,
            "partial_rotary_factor",
            rotary_dim as f64 / head_dim as f64,
        );
    }
    if let Some(theta) = store.metadata().get(&rope_theta_key) {
        set(
            &mut rope,
            "rope_theta",
            make_number(finite_number(theta, &rope_theta_key)?, &rope_theta_key)?,
        );
    }
    set(&mut rope, "rope_type", "default");
    if let Some(sections) = store
        .metadata()
        .get(&format!("{prefix}.rope.dimension_sections"))
    {
        set(
            &mut rope,
            "mrope_section",
            mrope_sections(sections, &format!("{prefix}.rope.dimension_sections"))?,
        );
    }
    set(&mut config, "rope_parameters", Value::Object(rope));

    if architecture == "qwen35" {
        let gdn = [
            "ssm.group_count",
            "ssm.time_step_rank",
            "ssm.state_size",
            "ssm.inner_size",
            "ssm.conv_kernel",
        ];
        let present = gdn
            .iter()
            .filter(|suffix| store.metadata().contains_key(&format!("{prefix}.{suffix}")))
            .count();
        if present != 0 && present != gdn.len() {
            return Err(invalid("Qwen3.5 GGUF 的 GDN metadata 不完整"));
        }
        if present == gdn.len() {
            let key_heads = required_usize(store, &format!("{prefix}.ssm.group_count"))?;
            let value_heads = required_usize(store, &format!("{prefix}.ssm.time_step_rank"))?;
            let key_dim = required_usize(store, &format!("{prefix}.ssm.state_size"))?;
            let inner_size = required_usize(store, &format!("{prefix}.ssm.inner_size"))?;
            let conv = required_usize(store, &format!("{prefix}.ssm.conv_kernel"))?;
            if value_heads == 0 || inner_size == 0 || !inner_size.is_multiple_of(value_heads) {
                return Err(invalid("Qwen3.5 GDN inner_size 不能整除 value_heads"));
            }
            set(&mut config, "linear_num_key_heads", key_heads as u64);
            set(&mut config, "linear_num_value_heads", value_heads as u64);
            set(&mut config, "linear_key_head_dim", key_dim as u64);
            set(
                &mut config,
                "linear_value_head_dim",
                (inner_size / value_heads) as u64,
            );
            set(&mut config, "linear_conv_kernel_dim", conv as u64);
        }

        if let Some(recurrent) = store
            .metadata()
            .get(&format!("{prefix}.attention.recurrent_layers"))
        {
            let values = recurrent
                .as_array()
                .ok_or_else(|| invalid("GGUF recurrent_layers 必须是 bool 数组"))?;
            if values.len() != layers {
                return Err(invalid("GGUF recurrent_layers 数量与 block_count 不一致"));
            }
            let layer_types = values
                .iter()
                .map(|value| match value.as_bool() {
                    Some(true) => Ok(Value::String("linear_attention".into())),
                    Some(false) => Ok(Value::String("full_attention".into())),
                    None => Err(invalid("GGUF recurrent_layers 包含非 bool 值")),
                })
                .collect::<Result<Vec<_>>>()?;
            set(&mut config, "layer_types", Value::Array(layer_types));
        } else if let Some(interval) =
            optional_usize(store, &format!("{prefix}.full_attention_interval"))?
        {
            if interval == 0 {
                return Err(invalid("GGUF full_attention_interval 必须大于 0"));
            }
            set(&mut config, "full_attention_interval", interval as u64);
        }
    }
    Ok(Value::Object(config))
}

fn has_tensor(store: &WeightStore, base: &str) -> bool {
    store.names().any(|name| {
        name == base || name == format!("{base}.weight") || name == format!("{base}.bias")
    })
}

fn canonical_name(name: &str) -> (&str, &str) {
    if let Some(base) = name.strip_suffix(".weight") {
        (base, "weight")
    } else if let Some(base) = name.strip_suffix(".bias") {
        (base, "bias")
    } else {
        (name, "weight")
    }
}

fn parse_block_name(name: &str) -> Option<(usize, &str, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let index = rest[..dot].parse().ok()?;
    let (base, suffix) = canonical_name(&rest[dot + 1..]);
    Some((index, base, suffix))
}

fn mapped_name(
    name: &str,
    architecture: &str,
    spec: &ModelSpec,
) -> Result<Option<(String, TensorTransform)>> {
    let (base, suffix) = canonical_name(name);
    let global = match base {
        "token_embd" => Some(("model.embed_tokens".to_owned(), TensorTransform::None)),
        "output" => Some(("lm_head".to_owned(), TensorTransform::None)),
        "output_norm" if architecture == "qwen35" => {
            Some(("model.norm".to_owned(), TensorTransform::Norm))
        }
        "output_norm" => Some(("model.norm".to_owned(), TensorTransform::None)),
        _ => None,
    };
    if let Some((target, transform)) = global {
        let target = format!("{target}.{suffix}");
        return Ok(Some((target, transform)));
    }
    let Some((index, base, suffix)) = parse_block_name(name) else {
        return Ok(None);
    };
    if index >= spec.layers.len() {
        return Err(invalid(format!("GGUF 张量 {name} 的层号超出 block_count")));
    }
    let prefix = format!("model.layers.{index}");
    let target_base = match base {
        "attn_norm" => format!("{prefix}.input_layernorm"),
        "post_attention_norm" => format!("{prefix}.post_attention_layernorm"),
        "attn_q" => format!("{prefix}.self_attn.q_proj"),
        "attn_k" => format!("{prefix}.self_attn.k_proj"),
        "attn_v" => format!("{prefix}.self_attn.v_proj"),
        "attn_output" => format!("{prefix}.self_attn.o_proj"),
        "attn_q_norm" => format!("{prefix}.self_attn.q_norm"),
        "attn_k_norm" => format!("{prefix}.self_attn.k_norm"),
        "attn_gate" if architecture == "qwen35" => {
            format!("{prefix}.linear_attn.in_proj_z")
        }
        "ffn_gate" => format!("{prefix}.mlp.gate_proj"),
        "ffn_up" => format!("{prefix}.mlp.up_proj"),
        "ffn_down" => format!("{prefix}.mlp.down_proj"),
        "attn_qkv" if architecture == "qwen35" => format!("{prefix}.linear_attn.in_proj_qkv"),
        "ssm_alpha" if architecture == "qwen35" => format!("{prefix}.linear_attn.in_proj_a"),
        "ssm_beta" if architecture == "qwen35" => format!("{prefix}.linear_attn.in_proj_b"),
        "ssm_out" if architecture == "qwen35" => format!("{prefix}.linear_attn.out_proj"),
        "ssm_conv1d" if architecture == "qwen35" => format!("{prefix}.linear_attn.conv1d"),
        "ssm_a" if architecture == "qwen35" => format!("{prefix}.linear_attn.A_log"),
        "ssm_dt" if architecture == "qwen35" => format!("{prefix}.linear_attn.dt_bias"),
        "ssm_norm" if architecture == "qwen35" => format!("{prefix}.linear_attn.norm"),
        other if other.starts_with("ssm_") => {
            return Err(RsinferError::Unsupported(format!(
                "不支持的 {architecture} GDN 张量 {name}"
            )))
        }
        _ => return Ok(None),
    };
    let is_norm = architecture == "qwen35"
        && (base.ends_with("norm") || base == "attn_norm" || base == "post_attention_norm")
        && base != "ssm_norm";
    let transform = if is_norm {
        TensorTransform::Norm
    } else if architecture == "qwen35" && base == "ssm_a" {
        TensorTransform::ALog
    } else if architecture == "qwen35" && base == "ssm_conv1d" {
        TensorTransform::VRowsConv
    } else if architecture == "qwen35" && base == "ssm_dt" {
        TensorTransform::VVector
    } else if architecture == "qwen35" && base == "attn_qkv" {
        TensorTransform::VRowsQkv
    } else if architecture == "qwen35" && matches!(base, "attn_gate" | "ssm_alpha" | "ssm_beta") {
        TensorTransform::VRows
    } else if architecture == "qwen35" && base == "ssm_out" {
        TensorTransform::VColumns
    } else {
        TensorTransform::None
    };
    let target = if matches!(base, "ssm_a" | "ssm_dt") {
        target_base
    } else {
        format!("{target_base}.{suffix}")
    };
    Ok(Some((target, transform)))
}

fn tensor_rows_width(tensor: &WeightTensor) -> Result<(usize, usize)> {
    let shape = tensor.shape();
    let width = *shape.last().ok_or_else(|| invalid("张量形状不能为空"))?;
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, &dim| {
            rows.checked_mul(dim).ok_or_else(|| invalid("张量行数溢出"))
        })?;
    Ok((rows, width))
}

/// 返回 grouped 逻辑头到 GGUF tiled 物理头的索引映射。
fn v_head_permutation(key_heads: usize, value_heads: usize, width: usize) -> Result<Vec<usize>> {
    if key_heads == 0 || value_heads == 0 || width == 0 {
        return Err(invalid("GDN V-head permutation 参数必须为正数"));
    }
    if !value_heads.is_multiple_of(key_heads) {
        return Err(invalid("GDN value_heads 必须是 key_heads 的整数倍"));
    }
    let groups = value_heads / key_heads;
    let length = value_heads
        .checked_mul(width)
        .ok_or_else(|| invalid("GDN V-head permutation 长度溢出"))?;
    let mut mapping = Vec::with_capacity(length);
    for logical_head in 0..value_heads {
        let key_head = logical_head / groups;
        let tile = logical_head % groups;
        let physical_head = tile
            .checked_mul(key_heads)
            .and_then(|head| head.checked_add(key_head))
            .ok_or_else(|| invalid("GDN V-head permutation 索引溢出"))?;
        let physical_start = physical_head
            .checked_mul(width)
            .ok_or_else(|| invalid("GDN V-head permutation 偏移溢出"))?;
        for offset in 0..width {
            mapping.push(
                physical_start
                    .checked_add(offset)
                    .ok_or_else(|| invalid("GDN V-head permutation 索引溢出"))?,
            );
        }
    }
    Ok(mapping)
}

fn v_rows(
    tensor: &WeightTensor,
    key_heads: usize,
    value_heads: usize,
    head_width: usize,
) -> Result<WeightTensor> {
    let (rows, _) = tensor_rows_width(tensor)?;
    let expected = value_heads
        .checked_mul(head_width)
        .ok_or_else(|| invalid("GDN V-head 行数溢出"))?;
    if rows != expected {
        return Err(invalid(format!(
            "GDN V-head 行数不匹配：实际 {rows}，期望 {expected}"
        )));
    }
    tensor.permute_rows(v_head_permutation(key_heads, value_heads, head_width)?)
}

fn v_rows_qkv(
    tensor: &WeightTensor,
    key_heads: usize,
    key_dim: usize,
    value_heads: usize,
    value_dim: usize,
) -> Result<WeightTensor> {
    let (rows, _) = tensor_rows_width(tensor)?;
    let key_width = key_heads
        .checked_mul(key_dim)
        .ok_or_else(|| invalid("GDN QKV key 宽度溢出"))?;
    let value_width = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| invalid("GDN QKV value 宽度溢出"))?;
    let prefix = key_width
        .checked_mul(2)
        .ok_or_else(|| invalid("GDN QKV 前缀宽度溢出"))?;
    let expected = prefix
        .checked_add(value_width)
        .ok_or_else(|| invalid("GDN QKV 总行数溢出"))?;
    if rows != expected {
        return Err(invalid(format!(
            "GDN QKV 行数不匹配：实际 {rows}，期望 {expected}"
        )));
    }
    let mut mapping = Vec::with_capacity(expected);
    mapping.extend(0..prefix);
    mapping.extend(
        v_head_permutation(key_heads, value_heads, value_dim)?
            .into_iter()
            .map(|index| prefix + index),
    );
    tensor.permute_rows(mapping)
}

fn v_rows_conv(
    tensor: &WeightTensor,
    key_heads: usize,
    key_dim: usize,
    value_heads: usize,
    value_dim: usize,
) -> Result<WeightTensor> {
    v_rows_qkv(tensor, key_heads, key_dim, value_heads, value_dim)
}

fn v_vector(tensor: &WeightTensor, key_heads: usize, value_heads: usize) -> Result<WeightTensor> {
    let (rows, width) = tensor_rows_width(tensor)?;
    if rows != 1 || width != value_heads {
        return Err(invalid(format!(
            "GDN V-head 向量形状不匹配：实际 {:?}，期望 [{value_heads}]",
            tensor.shape()
        )));
    }
    tensor.permute_columns(v_head_permutation(key_heads, value_heads, 1)?)
}

fn v_columns(
    tensor: &WeightTensor,
    key_heads: usize,
    value_heads: usize,
    value_dim: usize,
) -> Result<WeightTensor> {
    let (_, width) = tensor_rows_width(tensor)?;
    let expected = value_heads
        .checked_mul(value_dim)
        .ok_or_else(|| invalid("GDN out projection 输入宽度溢出"))?;
    if width != expected {
        return Err(invalid(format!(
            "GDN out projection 输入宽度不匹配：实际 {width}，期望 {expected}"
        )));
    }
    tensor.permute_columns(v_head_permutation(key_heads, value_heads, value_dim)?)
}

fn adjust_norm(tensor: &WeightTensor) -> Result<WeightTensor> {
    let values = tensor
        .to_f32()?
        .into_iter()
        .map(|value| value - 1.0)
        .collect();
    WeightTensor::from_f32(tensor.shape().to_vec(), values)
}

fn restore_a_log(tensor: &WeightTensor) -> Result<WeightTensor> {
    let mut values = tensor.to_f32()?;
    for value in &mut values {
        if !value.is_finite() || *value >= 0.0 {
            return Err(invalid("GGUF ssm_a 必须是有限的负数编码 -exp(A_log)"));
        }
        *value = (-*value).ln();
    }
    WeightTensor::from_f32(tensor.shape().to_vec(), values)
}

fn reshape_conv(tensor: &WeightTensor) -> Result<WeightTensor> {
    let shape = tensor.shape();
    match shape {
        [channels, kernel] => tensor.reshape(vec![*channels, 1, *kernel]),
        [_, 1, _] => Ok(tensor.clone()),
        _ => Err(invalid(format!(
            "GDN conv1d 形状 {:?} 无法恢复为 [channels,1,kernel]",
            shape
        ))),
    }
}

fn map_tensors(
    store: &WeightStore,
    architecture: &str,
    spec: &ModelSpec,
) -> Result<BTreeMap<String, WeightTensor>> {
    let mut mapped = BTreeMap::new();
    for source_name in store.names() {
        let Some((target_name, transform)) = mapped_name(source_name, architecture, spec)? else {
            continue;
        };
        let source = store.get(source_name)?;
        let value = match transform {
            TensorTransform::None => source,
            TensorTransform::Norm => adjust_norm(&source)?,
            TensorTransform::ALog => {
                let g = spec
                    .gdn
                    .as_ref()
                    .ok_or_else(|| invalid("GDN A_log 张量缺少 GDN 配置"))?;
                restore_a_log(&v_vector(&source, g.key_heads, g.value_heads)?)?
            }
            TensorTransform::VRows => {
                let g = spec
                    .gdn
                    .as_ref()
                    .ok_or_else(|| invalid("GDN V-head 张量缺少 GDN 配置"))?;
                let (rows, _) = tensor_rows_width(&source)?;
                let value_width = g
                    .value_heads
                    .checked_mul(g.value_dim)
                    .ok_or_else(|| invalid("GDN value 宽度溢出"))?;
                if rows == value_width {
                    v_rows(&source, g.key_heads, g.value_heads, g.value_dim)?
                } else if rows == g.value_heads {
                    v_rows(&source, g.key_heads, g.value_heads, 1)?
                } else {
                    return Err(invalid(format!(
                        "GDN V-head 行数不匹配：实际 {rows}，期望 {} 或 {}",
                        value_width, g.value_heads
                    )));
                }
            }
            TensorTransform::VRowsQkv => {
                let g = spec
                    .gdn
                    .as_ref()
                    .ok_or_else(|| invalid("GDN QKV 张量缺少 GDN 配置"))?;
                v_rows_qkv(&source, g.key_heads, g.key_dim, g.value_heads, g.value_dim)?
            }
            TensorTransform::VRowsConv => {
                let g = spec
                    .gdn
                    .as_ref()
                    .ok_or_else(|| invalid("GDN conv1d 张量缺少 GDN 配置"))?;
                let reordered =
                    v_rows_conv(&source, g.key_heads, g.key_dim, g.value_heads, g.value_dim)?;
                reshape_conv(&reordered)?
            }
            TensorTransform::VVector => {
                let g = spec
                    .gdn
                    .as_ref()
                    .ok_or_else(|| invalid("GDN 向量张量缺少 GDN 配置"))?;
                v_vector(&source, g.key_heads, g.value_heads)?
            }
            TensorTransform::VColumns => {
                let g = spec
                    .gdn
                    .as_ref()
                    .ok_or_else(|| invalid("GDN out projection 缺少 GDN 配置"))?;
                v_columns(&source, g.key_heads, g.value_heads, g.value_dim)?
            }
        };
        if mapped.insert(target_name.clone(), value).is_some() {
            return Err(invalid(format!("GGUF 张量映射后名称重复：{target_name}")));
        }
    }
    Ok(mapped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::model::native::NativeModel;
    use crate::model::spec::ModelFamily;
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    struct FixtureTensor {
        name: String,
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_meta_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_meta_f32(bytes: &mut Vec<u8>, key: &str, value: f32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_meta_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        push_string(bytes, value);
    }

    fn push_meta_u32_array(bytes: &mut Vec<u8>, key: &str, values: &[u32]) {
        push_string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        push_u64(bytes, values.len() as u64);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_meta_bool_array(bytes: &mut Vec<u8>, key: &str, values: &[bool]) {
        push_string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        push_u64(bytes, values.len() as u64);
        for value in values {
            bytes.push(if *value { 1 } else { 0 });
        }
    }

    fn fixture_tensor(name: &str, shape: &[usize], fill: f32) -> FixtureTensor {
        let elements = shape.iter().product();
        FixtureTensor {
            name: name.to_owned(),
            shape: shape.to_vec(),
            values: vec![fill; elements],
        }
    }

    fn patterned(len: usize, seed: f32) -> Vec<f32> {
        (0..len).map(|index| seed + index as f32 * 0.0007).collect()
    }

    fn tiled_rows(
        logical: &[f32],
        rows: usize,
        width: usize,
        prefix_rows: usize,
        key_heads: usize,
        value_heads: usize,
        head_width: usize,
    ) -> Vec<f32> {
        assert_eq!(logical.len(), rows * width);
        let permutation = v_head_permutation(key_heads, value_heads, head_width).unwrap();
        assert_eq!(prefix_rows + permutation.len(), rows);
        let mut physical = logical.to_vec();
        for (logical_row, &physical_row) in permutation.iter().enumerate() {
            let source = (prefix_rows + logical_row) * width;
            let target = (prefix_rows + physical_row) * width;
            physical[target..target + width].copy_from_slice(&logical[source..source + width]);
        }
        physical
    }

    fn tiled_columns(
        logical: &[f32],
        rows: usize,
        key_heads: usize,
        value_heads: usize,
        head_width: usize,
    ) -> Vec<f32> {
        let width = value_heads * head_width;
        assert_eq!(logical.len(), rows * width);
        let permutation = v_head_permutation(key_heads, value_heads, head_width).unwrap();
        let mut physical = logical.to_vec();
        for (logical_column, &physical_column) in permutation.iter().enumerate() {
            for row in 0..rows {
                physical[row * width + physical_column] = logical[row * width + logical_column];
            }
        }
        physical
    }

    fn tiled_vector(logical: &[f32], key_heads: usize, value_heads: usize) -> Vec<f32> {
        let permutation = v_head_permutation(key_heads, value_heads, 1).unwrap();
        assert_eq!(logical.len(), permutation.len());
        let mut physical = logical.to_vec();
        for (logical_index, &physical_index) in permutation.iter().enumerate() {
            physical[physical_index] = logical[logical_index];
        }
        physical
    }

    fn tiled_a_log(logical: &[f32], key_heads: usize, value_heads: usize) -> Vec<f32> {
        tiled_vector(
            &logical.iter().map(|value| -value.exp()).collect::<Vec<_>>(),
            key_heads,
            value_heads,
        )
    }

    fn insert_grouped(
        tensors: &mut BTreeMap<String, WeightTensor>,
        name: &str,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) {
        tensors.insert(
            name.to_owned(),
            WeightTensor::from_f32(shape, values).unwrap(),
        );
    }

    fn build_qwen2_fixture() -> Vec<u8> {
        let mut tensors = Vec::new();
        let mut add = |name: &str, shape: &[usize], fill: f32| {
            tensors.push(fixture_tensor(name, shape, fill));
        };
        add("token_embd.weight", &[4, 4], 0.1);
        add("output.weight", &[4, 4], 0.0);
        add("output_norm.weight", &[4], 1.0);
        add("blk.0.attn_norm.weight", &[4], 1.0);
        add("blk.0.post_attention_norm.weight", &[4], 1.0);
        add("blk.0.attn_q.weight", &[4, 4], 0.0);
        add("blk.0.attn_q.bias", &[4], 0.0);
        add("blk.0.attn_k.weight", &[2, 4], 0.0);
        add("blk.0.attn_k.bias", &[2], 0.0);
        add("blk.0.attn_v.weight", &[2, 4], 0.0);
        add("blk.0.attn_v.bias", &[2], 0.0);
        add("blk.0.attn_output.weight", &[4, 4], 0.0);
        add("blk.0.ffn_gate.weight", &[6, 4], 0.0);
        add("blk.0.ffn_up.weight", &[6, 4], 0.0);
        add("blk.0.ffn_down.weight", &[4, 6], 0.0);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, 12);
        push_meta_string(&mut bytes, "general.architecture", "qwen2");
        push_meta_u32(&mut bytes, "qwen2.embedding_length", 4);
        push_meta_u32(&mut bytes, "qwen2.feed_forward_length", 6);
        push_meta_u32(&mut bytes, "qwen2.block_count", 1);
        push_meta_u32(&mut bytes, "qwen2.attention.head_count", 2);
        push_meta_u32(&mut bytes, "qwen2.attention.head_count_kv", 1);
        push_meta_u32(&mut bytes, "qwen2.context_length", 8);
        push_meta_u32(&mut bytes, "qwen2.vocab_size", 4);
        push_meta_f32(&mut bytes, "qwen2.attention.layer_norm_rms_epsilon", 1e-6);
        push_meta_u32(&mut bytes, "qwen2.rope.dimension_count", 2);
        push_meta_f32(&mut bytes, "qwen2.rope.freq_base", 1_000_000.0);
        push_meta_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 3);

        let payloads: Vec<Vec<u8>> = tensors
            .iter()
            .map(|tensor| {
                tensor
                    .values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            })
            .collect();
        let mut offset = 0u64;
        for (tensor, payload) in tensors.iter().zip(&payloads) {
            push_string(&mut bytes, &tensor.name);
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dimension in tensor.shape.iter().rev() {
                push_u64(&mut bytes, *dimension as u64);
            }
            bytes.extend_from_slice(&0u32.to_le_bytes());
            push_u64(&mut bytes, offset);
            offset += payload.len() as u64;
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        for payload in payloads {
            bytes.extend_from_slice(&payload);
        }
        bytes
    }

    fn build_qwen35_fixture() -> Vec<u8> {
        let mut tensors = Vec::new();
        let mut add = |name: &str, shape: &[usize], fill: f32| {
            tensors.push(fixture_tensor(name, shape, fill));
        };
        add("token_embd.weight", &[4, 4], 0.1);
        add("output.weight", &[4, 4], 0.0);
        add("output_norm.weight", &[4], 2.0);

        add("blk.0.attn_norm.weight", &[4], 2.0);
        add("blk.0.post_attention_norm.weight", &[4], 2.0);
        add("blk.0.attn_qkv.weight", &[6, 4], 0.0);
        add("blk.0.attn_gate.weight", &[2, 4], 0.0);
        add("blk.0.ssm_alpha.weight", &[1, 4], 0.0);
        add("blk.0.ssm_beta.weight", &[1, 4], 0.0);
        add("blk.0.ssm_out.weight", &[4, 2], 0.0);
        add("blk.0.ssm_conv1d.weight", &[6, 1, 2], 0.0);
        add("blk.0.ssm_a", &[1], -1.0);
        add("blk.0.ssm_dt.bias", &[1], 0.0);
        add("blk.0.ssm_norm.weight", &[2], 1.0);
        add("blk.0.ffn_gate.weight", &[6, 4], 0.0);
        add("blk.0.ffn_up.weight", &[6, 4], 0.0);
        add("blk.0.ffn_down.weight", &[4, 6], 0.0);

        add("blk.1.attn_norm.weight", &[4], 2.0);
        add("blk.1.post_attention_norm.weight", &[4], 2.0);
        add("blk.1.attn_q.weight", &[8, 4], 0.0);
        add("blk.1.attn_k.weight", &[2, 4], 0.0);
        add("blk.1.attn_v.weight", &[2, 4], 0.0);
        add("blk.1.attn_output.weight", &[4, 4], 0.0);
        add("blk.1.attn_q_norm.weight", &[2], 2.0);
        add("blk.1.attn_k_norm.weight", &[2], 2.0);
        add("blk.1.ffn_gate.weight", &[6, 4], 0.0);
        add("blk.1.ffn_up.weight", &[6, 4], 0.0);
        add("blk.1.ffn_down.weight", &[4, 6], 0.0);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, 19);
        push_meta_string(&mut bytes, "general.architecture", "qwen35");
        push_meta_u32(&mut bytes, "qwen35.embedding_length", 4);
        push_meta_u32(&mut bytes, "qwen35.feed_forward_length", 6);
        push_meta_u32(&mut bytes, "qwen35.block_count", 2);
        push_meta_u32(&mut bytes, "qwen35.attention.head_count", 2);
        push_meta_u32(&mut bytes, "qwen35.attention.head_count_kv", 1);
        push_meta_u32(&mut bytes, "qwen35.context_length", 8);
        push_meta_u32(&mut bytes, "qwen35.vocab_size", 4);
        push_meta_f32(&mut bytes, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        push_meta_u32(&mut bytes, "qwen35.rope.dimension_count", 2);
        push_meta_f32(&mut bytes, "qwen35.rope.freq_base", 10_000_000.0);
        push_meta_u32(&mut bytes, "qwen35.ssm.group_count", 1);
        push_meta_u32(&mut bytes, "qwen35.ssm.time_step_rank", 1);
        push_meta_u32(&mut bytes, "qwen35.ssm.state_size", 2);
        push_meta_u32(&mut bytes, "qwen35.ssm.inner_size", 2);
        push_meta_u32(&mut bytes, "qwen35.ssm.conv_kernel", 2);
        push_meta_bool_array(
            &mut bytes,
            "qwen35.attention.recurrent_layers",
            &[true, false],
        );
        push_meta_u32_array(&mut bytes, "qwen35.rope.dimension_sections", &[1, 0, 0, 0]);
        push_meta_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 3);

        let payloads: Vec<Vec<u8>> = tensors
            .iter()
            .map(|tensor| {
                tensor
                    .values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            })
            .collect();
        let mut offset = 0u64;
        for (tensor, payload) in tensors.iter().zip(&payloads) {
            push_string(&mut bytes, &tensor.name);
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dimension in tensor.shape.iter().rev() {
                push_u64(&mut bytes, *dimension as u64);
            }
            bytes.extend_from_slice(&0u32.to_le_bytes());
            push_u64(&mut bytes, offset);
            offset += payload.len() as u64;
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        for payload in payloads {
            bytes.extend_from_slice(&payload);
        }
        bytes
    }

    fn build_qwen35_with_gdn_metadata(
        tensors: &[FixtureTensor],
        key_heads: u32,
        value_heads: u32,
        key_dim: u32,
        value_dim: u32,
        conv_kernel: u32,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, 19);
        push_meta_string(&mut bytes, "general.architecture", "qwen35");
        push_meta_u32(&mut bytes, "qwen35.embedding_length", 4);
        push_meta_u32(&mut bytes, "qwen35.feed_forward_length", 6);
        push_meta_u32(&mut bytes, "qwen35.block_count", 2);
        push_meta_u32(&mut bytes, "qwen35.attention.head_count", 2);
        push_meta_u32(&mut bytes, "qwen35.attention.head_count_kv", 1);
        push_meta_u32(&mut bytes, "qwen35.context_length", 8);
        push_meta_u32(&mut bytes, "qwen35.vocab_size", 4);
        push_meta_f32(&mut bytes, "qwen35.attention.layer_norm_rms_epsilon", 1e-6);
        push_meta_u32(&mut bytes, "qwen35.rope.dimension_count", 2);
        push_meta_f32(&mut bytes, "qwen35.rope.freq_base", 10_000_000.0);
        push_meta_u32(&mut bytes, "qwen35.ssm.group_count", key_heads);
        push_meta_u32(&mut bytes, "qwen35.ssm.time_step_rank", value_heads);
        push_meta_u32(&mut bytes, "qwen35.ssm.state_size", key_dim);
        push_meta_u32(&mut bytes, "qwen35.ssm.inner_size", value_heads * value_dim);
        push_meta_u32(&mut bytes, "qwen35.ssm.conv_kernel", conv_kernel);
        push_meta_bool_array(
            &mut bytes,
            "qwen35.attention.recurrent_layers",
            &[true, false],
        );
        push_meta_u32_array(&mut bytes, "qwen35.rope.dimension_sections", &[1, 0, 0, 0]);
        push_meta_u32(&mut bytes, "tokenizer.ggml.eos_token_id", 3);

        let payloads: Vec<Vec<u8>> = tensors
            .iter()
            .map(|tensor| {
                tensor
                    .values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            })
            .collect();
        let mut offset = 0u64;
        for (tensor, payload) in tensors.iter().zip(&payloads) {
            push_string(&mut bytes, &tensor.name);
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dimension in tensor.shape.iter().rev() {
                push_u64(&mut bytes, *dimension as u64);
            }
            bytes.extend_from_slice(&0u32.to_le_bytes());
            push_u64(&mut bytes, offset);
            offset += payload.len() as u64;
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        for payload in payloads {
            bytes.extend_from_slice(&payload);
        }
        bytes
    }

    fn asymmetric_qwen35_pair() -> (Vec<u8>, WeightStore, ModelSpec) {
        let key_heads = 2;
        let value_heads = 6;
        let key_dim = 1;
        let value_dim = 1;
        let conv_kernel = 2;
        let channels = 2 * key_heads * key_dim + value_heads * value_dim;
        let mut raw = Vec::new();
        let mut grouped = BTreeMap::new();
        let mut add_pair = |raw_name: &str,
                            grouped_name: &str,
                            shape: Vec<usize>,
                            logical: Vec<f32>,
                            physical: Vec<f32>| {
            raw.push(FixtureTensor {
                name: raw_name.to_owned(),
                shape,
                values: physical,
            });
            insert_grouped(
                &mut grouped,
                grouped_name,
                raw.last().unwrap().shape.clone(),
                logical,
            );
        };
        let norm = |seed: f32| patterned(4, seed);
        let raw_norm = |values: &[f32]| values.iter().map(|value| value + 1.0).collect();

        let embeddings = patterned(16, 0.01);
        add_pair(
            "token_embd.weight",
            "model.embed_tokens.weight",
            vec![4, 4],
            embeddings.clone(),
            embeddings,
        );
        let output = patterned(16, 0.03);
        add_pair(
            "output.weight",
            "lm_head.weight",
            vec![4, 4],
            output.clone(),
            output,
        );
        let output_norm = norm(0.2);
        add_pair(
            "output_norm.weight",
            "model.norm.weight",
            vec![4],
            output_norm.clone(),
            raw_norm(&output_norm),
        );

        for (raw_index, grouped_name) in [
            (
                "blk.0.attn_norm.weight",
                "model.layers.0.input_layernorm.weight",
            ),
            (
                "blk.0.post_attention_norm.weight",
                "model.layers.0.post_attention_layernorm.weight",
            ),
        ] {
            let values = norm(0.25);
            add_pair(
                raw_index,
                grouped_name,
                vec![4],
                values.clone(),
                raw_norm(&values),
            );
        }

        let qkv = patterned(channels * 4, 0.05);
        add_pair(
            "blk.0.attn_qkv.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            vec![channels, 4],
            qkv.clone(),
            tiled_rows(
                &qkv,
                channels,
                4,
                2 * key_heads * key_dim,
                key_heads,
                value_heads,
                value_dim,
            ),
        );
        let z = patterned(value_heads * 4, 0.08);
        add_pair(
            "blk.0.attn_gate.weight",
            "model.layers.0.linear_attn.in_proj_z.weight",
            vec![value_heads, 4],
            z.clone(),
            tiled_rows(&z, value_heads, 4, 0, key_heads, value_heads, value_dim),
        );
        let alpha = patterned(value_heads * 4, 0.11);
        add_pair(
            "blk.0.ssm_alpha.weight",
            "model.layers.0.linear_attn.in_proj_a.weight",
            vec![value_heads, 4],
            alpha.clone(),
            tiled_rows(&alpha, value_heads, 4, 0, key_heads, value_heads, 1),
        );
        let beta = patterned(value_heads * 4, 0.14);
        add_pair(
            "blk.0.ssm_beta.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            vec![value_heads, 4],
            beta.clone(),
            tiled_rows(&beta, value_heads, 4, 0, key_heads, value_heads, 1),
        );
        let output_projection = patterned(4 * value_heads, 0.17);
        add_pair(
            "blk.0.ssm_out.weight",
            "model.layers.0.linear_attn.out_proj.weight",
            vec![4, value_heads],
            output_projection.clone(),
            tiled_columns(&output_projection, 4, key_heads, value_heads, value_dim),
        );
        let convolution = patterned(channels * conv_kernel, 0.20);
        add_pair(
            "blk.0.ssm_conv1d.weight",
            "model.layers.0.linear_attn.conv1d.weight",
            vec![channels, 1, conv_kernel],
            convolution.clone(),
            tiled_rows(
                &convolution,
                channels,
                conv_kernel,
                2 * key_heads * key_dim,
                key_heads,
                value_heads,
                value_dim,
            ),
        );
        let a_log = patterned(value_heads, 0.23);
        add_pair(
            "blk.0.ssm_a",
            "model.layers.0.linear_attn.A_log",
            vec![value_heads],
            a_log.clone(),
            tiled_a_log(&a_log, key_heads, value_heads),
        );
        let dt_bias = patterned(value_heads, 0.26);
        add_pair(
            "blk.0.ssm_dt.bias",
            "model.layers.0.linear_attn.dt_bias",
            vec![value_heads],
            dt_bias.clone(),
            tiled_vector(&dt_bias, key_heads, value_heads),
        );
        let gdn_norm = vec![0.9];
        add_pair(
            "blk.0.ssm_norm.weight",
            "model.layers.0.linear_attn.norm.weight",
            vec![value_dim],
            gdn_norm.clone(),
            gdn_norm,
        );
        for (raw_index, grouped_name, seed) in [
            (
                "blk.0.ffn_gate.weight",
                "model.layers.0.mlp.gate_proj.weight",
                0.29,
            ),
            (
                "blk.0.ffn_up.weight",
                "model.layers.0.mlp.up_proj.weight",
                0.32,
            ),
        ] {
            let values = patterned(24, seed);
            add_pair(raw_index, grouped_name, vec![6, 4], values.clone(), values);
        }
        let down = patterned(24, 0.35);
        add_pair(
            "blk.0.ffn_down.weight",
            "model.layers.0.mlp.down_proj.weight",
            vec![4, 6],
            down.clone(),
            down,
        );

        for (raw_index, grouped_name) in [
            (
                "blk.1.attn_norm.weight",
                "model.layers.1.input_layernorm.weight",
            ),
            (
                "blk.1.post_attention_norm.weight",
                "model.layers.1.post_attention_layernorm.weight",
            ),
        ] {
            let values = norm(0.38);
            add_pair(
                raw_index,
                grouped_name,
                vec![4],
                values.clone(),
                raw_norm(&values),
            );
        }
        let full_q = patterned(32, 0.41);
        add_pair(
            "blk.1.attn_q.weight",
            "model.layers.1.self_attn.q_proj.weight",
            vec![8, 4],
            full_q.clone(),
            full_q,
        );
        for (raw_index, grouped_name, seed) in [
            (
                "blk.1.attn_k.weight",
                "model.layers.1.self_attn.k_proj.weight",
                0.44,
            ),
            (
                "blk.1.attn_v.weight",
                "model.layers.1.self_attn.v_proj.weight",
                0.47,
            ),
        ] {
            let values = patterned(8, seed);
            add_pair(raw_index, grouped_name, vec![2, 4], values.clone(), values);
        }
        let full_o = patterned(16, 0.50);
        add_pair(
            "blk.1.attn_output.weight",
            "model.layers.1.self_attn.o_proj.weight",
            vec![4, 4],
            full_o.clone(),
            full_o,
        );
        for (raw_index, grouped_name) in [
            (
                "blk.1.attn_q_norm.weight",
                "model.layers.1.self_attn.q_norm.weight",
            ),
            (
                "blk.1.attn_k_norm.weight",
                "model.layers.1.self_attn.k_norm.weight",
            ),
        ] {
            let values = vec![0.8, 0.81];
            add_pair(
                raw_index,
                grouped_name,
                vec![2],
                values.clone(),
                raw_norm(&values),
            );
        }
        for (raw_index, grouped_name, seed) in [
            (
                "blk.1.ffn_gate.weight",
                "model.layers.1.mlp.gate_proj.weight",
                0.53,
            ),
            (
                "blk.1.ffn_up.weight",
                "model.layers.1.mlp.up_proj.weight",
                0.56,
            ),
        ] {
            let values = patterned(24, seed);
            add_pair(raw_index, grouped_name, vec![6, 4], values.clone(), values);
        }
        let full_down = patterned(24, 0.59);
        add_pair(
            "blk.1.ffn_down.weight",
            "model.layers.1.mlp.down_proj.weight",
            vec![4, 6],
            full_down.clone(),
            full_down,
        );

        let config = json!({
            "model_type": "qwen3_5_text",
            "architectures": ["Qwen3_5ForCausalLM"],
            "hidden_size": 4,
            "intermediate_size": 6,
            "vocab_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "max_position_embeddings": 8,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "rope_parameters": {"rope_type": "default", "partial_rotary_factor": 1.0, "rope_theta": 10_000_000.0, "mrope_section": [1, 0, 0]},
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": key_heads,
            "linear_num_value_heads": value_heads,
            "linear_key_head_dim": key_dim,
            "linear_value_head_dim": value_dim,
            "linear_conv_kernel_dim": conv_kernel
        });
        let spec = ModelSpec::from_hf(&config).unwrap();
        let store = WeightStore::from_tensors(config, grouped).unwrap();
        (
            build_qwen35_with_gdn_metadata(
                &raw,
                key_heads as u32,
                value_heads as u32,
                key_dim as u32,
                value_dim as u32,
                conv_kernel as u32,
            ),
            store,
            spec,
        )
    }

    fn fixture_path() -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rsinfer-gguf-normalize-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    fn write_fixture(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn a_log_inverse_and_norm_restore_are_finite() {
        let a = WeightTensor::from_f32(vec![2], vec![-1.0, -2.0]).unwrap();
        let restored = restore_a_log(&a).unwrap().to_f32().unwrap();
        assert!((restored[0] - 0.0).abs() < 1e-6);
        assert!((restored[1] - 2.0_f32.ln()).abs() < 1e-6);

        let norm = WeightTensor::from_f32(vec![2], vec![1.5, 0.5]).unwrap();
        assert_eq!(
            adjust_norm(&norm).unwrap().to_f32().unwrap(),
            vec![0.5, -0.5]
        );
    }

    #[test]
    fn v_head_permutation_restores_grouped_order_for_asymmetric_heads() {
        assert_eq!(
            v_head_permutation(2, 6, 2).unwrap(),
            vec![0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
        );

        let physical =
            WeightTensor::from_f32(vec![12, 1], (0..12).map(|value| value as f32).collect())
                .unwrap();
        let grouped = v_rows(&physical, 2, 6, 2).unwrap();
        assert!(!grouped.is_contiguous());
        assert_eq!(
            grouped.to_f32().unwrap(),
            vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0]
        );

        let qkv = WeightTensor::from_f32(vec![10, 1], (0..10).map(|value| value as f32).collect())
            .unwrap();
        let grouped_qkv = v_rows_qkv(&qkv, 2, 1, 6, 1).unwrap();
        assert_eq!(
            grouped_qkv.to_f32().unwrap(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 5.0, 7.0, 9.0]
        );

        let vector =
            WeightTensor::from_f32(vec![6], (0..6).map(|value| value as f32).collect()).unwrap();
        assert_eq!(
            v_vector(&vector, 2, 6).unwrap().to_f32().unwrap(),
            vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]
        );
    }

    #[test]
    fn unsupported_a_log_sign_is_rejected() {
        let a = WeightTensor::from_f32(vec![1], vec![0.0]).unwrap();
        assert!(restore_a_log(&a).is_err());
    }

    #[test]
    fn model_type_config_accepts_qwen35_defaults() {
        let _ = json!({"model_type": "qwen3_5_text"});
    }

    #[test]
    fn mrope_sections_drop_gguf_trailing_axis() {
        let normalized =
            mrope_sections(&json!([11, 11, 10, 0]), "qwen35.rope.dimension_sections").unwrap();
        assert_eq!(normalized, json!([11, 11, 10]));
        assert!(mrope_sections(&json!([11, 11, 10, 1]), "qwen35.rope.dimension_sections").is_err());
    }

    #[test]
    fn qwen2_gguf_normalizes_into_native_model() {
        let path = fixture_path();
        write_fixture(&path, &build_qwen2_fixture());
        let raw = WeightStore::open(&path).unwrap();
        let (normalized, spec) = normalize(&raw).unwrap();
        assert_eq!(spec.family, ModelFamily::Qwen2);
        assert_eq!(
            normalized.get("model.embed_tokens.weight").unwrap().shape(),
            &[4, 4]
        );
        assert_eq!(
            normalized
                .get("model.layers.0.self_attn.k_proj.bias")
                .unwrap()
                .shape(),
            &[2]
        );

        let model = NativeModel::from_weights(&normalized, spec, Backend::Cpu).unwrap();
        let mut state = model.create_state(4).unwrap();
        let output = model.forward(&[1], &mut state, true).unwrap();
        assert_eq!(output.logits.len(), 4);
        assert!(output.logits.iter().all(|value| value.is_finite()));

        drop(model);
        drop(normalized);
        drop(raw);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn qwen35_gguf_normalizes_gdn_and_full_attention_layers() {
        let path = fixture_path();
        write_fixture(&path, &build_qwen35_fixture());
        let raw = WeightStore::open(&path).unwrap();
        let (normalized, spec) = normalize(&raw).unwrap();
        assert_eq!(spec.family, ModelFamily::Qwen35);
        assert_eq!(spec.layers.len(), 2);
        assert_eq!(
            normalized
                .get("model.layers.0.linear_attn.A_log")
                .unwrap()
                .to_f32()
                .unwrap(),
            vec![0.0]
        );
        assert_eq!(
            normalized
                .get("model.layers.0.linear_attn.dt_bias")
                .unwrap()
                .shape(),
            &[1]
        );
        assert_eq!(
            normalized
                .get("model.layers.0.linear_attn.norm.weight")
                .unwrap()
                .to_f32()
                .unwrap(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            normalized
                .get("model.layers.0.linear_attn.conv1d.weight")
                .unwrap()
                .shape(),
            &[6, 1, 2]
        );

        let model = NativeModel::from_weights(&normalized, spec, Backend::Cpu).unwrap();
        let mut state = model.create_state(4).unwrap();
        let output = model.forward(&[1, 2], &mut state, true).unwrap();
        assert_eq!(output.logits.len(), 8);
        assert!(output.logits.iter().all(|value| value.is_finite()));

        drop(model);
        drop(normalized);
        drop(raw);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn qwen35_asymmetric_gdn_gguf_matches_grouped_hf_across_chunks() {
        let path = fixture_path();
        let (bytes, grouped, grouped_spec) = asymmetric_qwen35_pair();
        write_fixture(&path, &bytes);
        let raw = WeightStore::open(&path).unwrap();
        let (normalized, gguf_spec) = normalize(&raw).unwrap();
        assert_eq!(gguf_spec.gdn.as_ref().unwrap().key_heads, 2);
        assert_eq!(gguf_spec.gdn.as_ref().unwrap().value_heads, 6);
        assert!(!normalized
            .get("model.layers.0.linear_attn.in_proj_qkv.weight")
            .unwrap()
            .is_contiguous());
        assert!(!normalized
            .get("model.layers.0.linear_attn.out_proj.weight")
            .unwrap()
            .is_contiguous());
        assert_eq!(
            normalized
                .get("model.layers.0.linear_attn.in_proj_qkv.weight")
                .unwrap()
                .bytes(),
            raw.get("blk.0.attn_qkv.weight").unwrap().bytes()
        );

        let gguf_model = NativeModel::from_weights(&normalized, gguf_spec, Backend::Cpu).unwrap();
        let grouped_model =
            NativeModel::from_weights(&grouped, grouped_spec, Backend::Cpu).unwrap();
        let mut gguf_state = gguf_model.create_state(4).unwrap();
        let mut grouped_state = grouped_model.create_state(4).unwrap();
        for tokens in [&[0, 1][..], &[2, 3][..]] {
            let gguf = gguf_model.forward(tokens, &mut gguf_state, true).unwrap();
            let hf = grouped_model
                .forward(tokens, &mut grouped_state, true)
                .unwrap();
            assert_eq!(gguf.rows, hf.rows);
            assert_eq!(gguf.logits.len(), hf.logits.len());
            assert_eq!(gguf.hidden.len(), hf.hidden.len());
            for (actual, expected) in gguf.logits.iter().zip(&hf.logits) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "logit mismatch: {actual} vs {expected}"
                );
            }
            for (actual, expected) in gguf.hidden.iter().zip(&hf.hidden) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "hidden mismatch: {actual} vs {expected}"
                );
            }
        }

        drop(gguf_model);
        drop(grouped_model);
        drop(normalized);
        drop(raw);
        drop(grouped);
        fs::remove_file(path).unwrap();
    }
}
