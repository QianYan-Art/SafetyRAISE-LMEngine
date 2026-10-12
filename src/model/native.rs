//! 稠密 Qwen 与 Qwen3.5 混合注意力的统一执行器。
//!
//! 权重只读共享；每个请求单独持有 KV、卷积和递归状态。CPU 路径提供可核对的
//! 模型计算，线性算子可通过 Backend 替换；这里不包装第三方推理进程。

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crate::artifact::{WeightStore, WeightTensor};
use crate::backend::Backend;
use crate::engine::state::{LayerState, SequenceState};
use crate::model::math::{
    apply_rope, causal_attention, causal_conv_silu, gated_delta_step, rms_norm, sigmoid, silu,
};
use crate::model::spec::{LayerKind, ModelFamily, ModelSpec};
use crate::{Result, RsinferError};

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn allocate_model_id() -> Result<u64> {
    NEXT_MODEL_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| bad("模型实例 ID 已耗尽"))
}

pub struct NativeModel {
    pub(crate) spec: ModelSpec,
    pub(crate) backend: Backend,
    owner: u64,
    embedding: WeightTensor,
    head: WeightTensor,
    norm: Vec<f32>,
    layers: Vec<Block>,
}

impl NativeModel {
    /// 架构在装载后保持不变，防止配置与已验证的权重、状态形状脱节。
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
    pub fn backend(&self) -> &Backend {
        &self.backend
    }
}

pub(crate) struct Projection {
    weight: WeightTensor,
    bias: Option<Vec<f32>>,
}

pub(crate) struct Block {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    gate: Projection,
    up: Projection,
    down: Projection,
}

enum Mixer {
    Attention(Attention),
    Gdn(Gdn),
}

struct Attention {
    query: Projection,
    key: Projection,
    value: Projection,
    output: Projection,
    query_norm: Option<Vec<f32>>,
    key_norm: Option<Vec<f32>>,
}

struct Gdn {
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    output: Projection,
    convolution: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

pub struct ForwardOutput {
    /// 最后一行或所有行的目标 logits，由调用方明确选择。
    pub logits: Vec<f32>,
    /// 所有输入行的最终归一化 hidden，供 MTP 对齐或最后状态保留。
    pub hidden: Vec<f32>,
    pub rows: usize,
}

fn bad(message: impl Into<String>) -> RsinferError {
    RsinferError::WeightError(message.into())
}

pub(crate) fn tensor(store: &WeightStore, name: &str, shape: &[usize]) -> Result<WeightTensor> {
    let t = store.get(name)?;
    if t.shape() != shape {
        return Err(bad(format!(
            "权重 {name} 形状 {:?}，期望 {shape:?}",
            t.shape()
        )));
    }
    Ok(t)
}

pub(crate) fn vector(store: &WeightStore, name: &str, size: usize) -> Result<Vec<f32>> {
    let values = tensor(store, name, &[size])?.to_f32()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(bad(format!("权重 {name} 包含非有限数值")));
    }
    Ok(values)
}

impl Projection {
    pub(crate) fn load(
        store: &WeightStore,
        name: &str,
        output: usize,
        input: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: tensor(store, &format!("{name}.weight"), &[output, input])?,
            bias: if bias {
                Some(vector(store, &format!("{name}.bias"), output)?)
            } else {
                None
            },
        })
    }

    pub(crate) fn forward(
        &self,
        backend: &Backend,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let mut result = backend.linear(&self.weight, input, rows)?;
        if let Some(bias) = &self.bias {
            for row in result.chunks_exact_mut(bias.len()) {
                for (x, bias) in row.iter_mut().zip(bias) {
                    *x += bias;
                }
            }
        }
        Ok(result)
    }
}

impl NativeModel {
    pub fn from_weights(store: &WeightStore, spec: ModelSpec, backend: Backend) -> Result<Self> {
        spec.validate()?;
        let prefix = if store
            .names()
            .any(|n| n == "model.language_model.embed_tokens.weight")
        {
            "model.language_model"
        } else {
            "model"
        };
        let embedding = tensor(
            store,
            &format!("{prefix}.embed_tokens.weight"),
            &[spec.vocab_size, spec.hidden_size],
        )?;
        let head = if spec.tie_word_embeddings {
            embedding.clone()
        } else {
            tensor(
                store,
                "lm_head.weight",
                &[spec.vocab_size, spec.hidden_size],
            )?
        };
        let norm = vector(store, &format!("{prefix}.norm.weight"), spec.hidden_size)?;
        let layers = spec
            .layers
            .iter()
            .enumerate()
            .map(|(i, kind)| Block::load(store, &format!("{prefix}.layers.{i}"), *kind, &spec))
            .collect::<Result<Vec<_>>>()?;
        let owner = allocate_model_id()?;
        Ok(Self {
            spec,
            backend,
            owner,
            embedding,
            head,
            norm,
            layers,
        })
    }

    pub fn model_id(&self) -> u64 {
        self.owner
    }

    pub(crate) fn project_logits(&self, hidden: &[f32], rows: usize) -> Result<Vec<f32>> {
        self.backend.linear(&self.head, hidden, rows)
    }

    pub fn create_state(&self, context: usize) -> Result<SequenceState> {
        let mut state = SequenceState::new(&self.spec, context)?;
        state.owner = self.owner;
        Ok(state)
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let size = tokens
            .len()
            .checked_mul(self.spec.hidden_size)
            .ok_or_else(|| bad("embedding 大小溢出"))?;
        let mut embedding = vec![0.0; size];
        for (token, row) in tokens
            .iter()
            .zip(embedding.chunks_exact_mut(self.spec.hidden_size))
        {
            if *token as usize >= self.spec.vocab_size {
                return Err(bad("输入 token 不属于模型词表"));
            }
            self.embedding.read_row(*token as usize, row)?;
        }
        Ok(embedding)
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        state: &mut SequenceState,
        all_logits: bool,
    ) -> Result<ForwardOutput> {
        state.ensure_room(tokens.len())?;
        let positions = (0..tokens.len())
            .map(|i| {
                state
                    .next_position
                    .checked_add(i)
                    .map(|p| [p; 3])
                    .ok_or_else(|| bad("位置索引溢出"))
            })
            .collect::<Result<Vec<_>>>()?;
        self.forward_embeddings(&self.embed(tokens)?, &positions, state, all_logits)
    }

    /// 接收前端已替换视觉占位符的输入。只有整个 chunk 成功才提交请求状态。
    pub fn forward_embeddings(
        &self,
        embeddings: &[f32],
        positions: &[[usize; 3]],
        state: &mut SequenceState,
        all_logits: bool,
    ) -> Result<ForwardOutput> {
        state.ensure_room(positions.len())?;
        if state.owner != self.owner || state.layers.len() != self.layers.len() {
            return Err(bad("请求状态不属于当前模型实例"));
        }
        let rows = positions.len();
        if rows.checked_mul(self.spec.hidden_size) != Some(embeddings.len())
            || embeddings.iter().any(|v| !v.is_finite())
            || positions
                .iter()
                .flatten()
                .any(|p| *p >= self.spec.max_position_embeddings)
        {
            return Err(bad("输入嵌入或位置超出模型合同"));
        }
        let mut candidate = state.checkpoint();
        let mut hidden = embeddings.to_vec();
        for (block, layer_state) in self.layers.iter().zip(&mut candidate.layers) {
            hidden = block.forward(&hidden, positions, layer_state, &self.spec, &self.backend)?;
            if hidden.iter().any(|v| !v.is_finite()) {
                return Err(bad("模型计算产生非有限激活，状态未提交"));
            }
        }
        let hidden = rms_norm(
            &hidden,
            &self.norm,
            self.spec.rms_norm_eps,
            self.spec.family == ModelFamily::Qwen35,
        )?;
        let last = &hidden[(rows - 1) * self.spec.hidden_size..];
        let logits = self.backend.linear(
            &self.head,
            if all_logits { &hidden } else { last },
            if all_logits { rows } else { 1 },
        )?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(bad("输出 logits 包含非有限值，状态未提交"));
        }
        candidate.len += rows;
        candidate.next_position = positions
            .iter()
            .flatten()
            .copied()
            .max()
            .ok_or_else(|| bad("输入位置为空"))?
            .checked_add(1)
            .ok_or_else(|| bad("位置溢出"))?
            .max(state.next_position);
        candidate.last_hidden = last.to_vec();
        *state = candidate;
        Ok(ForwardOutput {
            logits,
            hidden,
            rows,
        })
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    /// 小几何合成权重只验证执行合同，不代表真实模型质量或性能。
    pub fn tiny(family: ModelFamily) -> (WeightStore, ModelSpec) {
        let hybrid = family == ModelFamily::Qwen35;
        let mut config = json!({
            "model_type": match family { ModelFamily::Qwen2 => "qwen2", ModelFamily::Qwen3 => "qwen3", ModelFamily::Qwen35 => "qwen3_5_text" },
            "hidden_size":4,"intermediate_size":6,"vocab_size":8,"num_hidden_layers":2,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,
            "max_position_embeddings":128,"tie_word_embeddings":true,
            "rope_parameters":{"rope_type":"default","partial_rotary_factor":1.0}
        });
        if hybrid {
            config["layer_types"] = json!(["linear_attention", "full_attention"]);
            config["linear_num_key_heads"] = json!(1);
            config["linear_num_value_heads"] = json!(2);
            config["linear_key_head_dim"] = json!(2);
            config["linear_value_head_dim"] = json!(2);
            config["linear_conv_kernel_dim"] = json!(3);
        }
        let spec = ModelSpec::from_hf(&config).unwrap();
        let mut tensors = BTreeMap::new();
        let mut insert = |name: String, shape: Vec<usize>, constant: Option<f32>| {
            let n: usize = shape.iter().product();
            let values = (0..n)
                .map(|i| constant.unwrap_or(((i * 7 + 3) % 17) as f32 / 40.0 - 0.2))
                .collect();
            tensors.insert(name, WeightTensor::from_f32(shape, values).unwrap());
        };
        insert("model.embed_tokens.weight".into(), vec![8, 4], None);
        insert(
            "model.norm.weight".into(),
            vec![4],
            Some(if hybrid { 0.0 } else { 1.0 }),
        );
        for (i, kind) in spec.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            for name in ["input_layernorm", "post_attention_layernorm"] {
                insert(
                    format!("{p}.{name}.weight"),
                    vec![4],
                    Some(if hybrid { 0.0 } else { 1.0 }),
                );
            }
            for (name, out, input) in [("gate_proj", 6, 4), ("up_proj", 6, 4), ("down_proj", 4, 6)]
            {
                insert(format!("{p}.mlp.{name}.weight"), vec![out, input], None);
            }
            if *kind == LayerKind::FullAttention {
                for (name, out, input) in [
                    ("q_proj", if hybrid { 8 } else { 4 }, 4),
                    ("k_proj", 2, 4),
                    ("v_proj", 2, 4),
                    ("o_proj", 4, 4),
                ] {
                    insert(
                        format!("{p}.self_attn.{name}.weight"),
                        vec![out, input],
                        None,
                    );
                    if family == ModelFamily::Qwen2 && name != "o_proj" {
                        insert(format!("{p}.self_attn.{name}.bias"), vec![out], Some(0.02));
                    }
                }
                if family != ModelFamily::Qwen2 {
                    for name in ["q_norm", "k_norm"] {
                        insert(
                            format!("{p}.self_attn.{name}.weight"),
                            vec![2],
                            Some(if hybrid { 0.0 } else { 1.0 }),
                        );
                    }
                }
            } else {
                for (name, out, input) in [
                    ("in_proj_qkv", 8, 4),
                    ("in_proj_z", 4, 4),
                    ("in_proj_a", 2, 4),
                    ("in_proj_b", 2, 4),
                    ("out_proj", 4, 4),
                ] {
                    insert(
                        format!("{p}.linear_attn.{name}.weight"),
                        vec![out, input],
                        None,
                    );
                }
                insert(
                    format!("{p}.linear_attn.conv1d.weight"),
                    vec![8, 1, 3],
                    Some(0.2),
                );
                insert(format!("{p}.linear_attn.norm.weight"), vec![2], Some(1.0));
                insert(format!("{p}.linear_attn.A_log"), vec![2], Some(-0.5));
                insert(format!("{p}.linear_attn.dt_bias"), vec![2], Some(0.1));
            }
        }
        (WeightStore::from_tensors(config, tensors).unwrap(), spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{x} != {y}");
        }
    }

    #[test]
    fn dense_and_hybrid_chunking_and_full_state_restore_match() {
        for family in [ModelFamily::Qwen2, ModelFamily::Qwen3, ModelFamily::Qwen35] {
            let (weights, spec) = fixtures::tiny(family);
            let model = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
            let tokens: Vec<u32> = (0..70).map(|n| n % 8).collect();
            let mut whole = model.create_state(128).unwrap();
            let output = model.forward(&tokens, &mut whole, true).unwrap();
            let mut state = model.create_state(128).unwrap();
            let mut actual = Vec::new();
            for chunk in tokens.chunks(3) {
                actual.extend(model.forward(chunk, &mut state, true).unwrap().logits);
            }
            close(&actual, &output.logits);
            let saved = state.checkpoint();
            let expected = model.forward(&[2, 3], &mut state, true).unwrap();
            let mut restored = saved.checkpoint();
            model.forward(&[7, 1, 5], &mut restored, false).unwrap();
            restored = saved;
            let repeated = model.forward(&[2, 3], &mut restored, true).unwrap();
            close(&expected.logits, &repeated.logits);
            close(state.last_hidden(), restored.last_hidden());
            assert_eq!(state.len(), 72);
        }
    }

    #[test]
    fn errors_do_not_commit_and_cross_model_state_is_rejected() {
        let (weights, spec) = fixtures::tiny(ModelFamily::Qwen35);
        let model = NativeModel::from_weights(&weights, spec.clone(), Backend::Cpu).unwrap();
        let other = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
        let mut state = model.create_state(8).unwrap();
        model.forward(&[1, 2], &mut state, false).unwrap();
        let saved = state.checkpoint();
        assert!(model.forward(&[], &mut state, false).is_err());
        assert!(model.forward(&[8], &mut state, false).is_err());
        assert!(model.forward(&[1; 7], &mut state, false).is_err());
        assert!(other.forward(&[1], &mut state, false).is_err());
        assert!(model
            .forward_embeddings(&[f32::NAN; 4], &[[2; 3]], &mut state, false)
            .is_err());
        assert_eq!(state.len(), saved.len());
        assert_eq!(state.last_hidden(), saved.last_hidden());
        let mut reference = saved;
        close(
            &model.forward(&[3], &mut state, false).unwrap().logits,
            &model.forward(&[3], &mut reference, false).unwrap().logits,
        );
    }
}

impl Block {
    pub(crate) fn load(
        store: &WeightStore,
        prefix: &str,
        kind: LayerKind,
        spec: &ModelSpec,
    ) -> Result<Self> {
        let h = spec.hidden_size;
        let mixer = match kind {
            LayerKind::FullAttention => Mixer::Attention(Attention::load(store, prefix, spec)?),
            LayerKind::LinearAttention => Mixer::Gdn(Gdn::load(store, prefix, spec)?),
        };
        Ok(Self {
            input_norm: vector(store, &format!("{prefix}.input_layernorm.weight"), h)?,
            post_norm: vector(
                store,
                &format!("{prefix}.post_attention_layernorm.weight"),
                h,
            )?,
            mixer,
            gate: Projection::load(
                store,
                &format!("{prefix}.mlp.gate_proj"),
                spec.intermediate_size,
                h,
                false,
            )?,
            up: Projection::load(
                store,
                &format!("{prefix}.mlp.up_proj"),
                spec.intermediate_size,
                h,
                false,
            )?,
            down: Projection::load(
                store,
                &format!("{prefix}.mlp.down_proj"),
                h,
                spec.intermediate_size,
                false,
            )?,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &[f32],
        positions: &[[usize; 3]],
        state: &mut LayerState,
        spec: &ModelSpec,
        backend: &Backend,
    ) -> Result<Vec<f32>> {
        let offset = spec.family == ModelFamily::Qwen35;
        let normed = rms_norm(input, &self.input_norm, spec.rms_norm_eps, offset)?;
        let mixed = match (&self.mixer, state) {
            (Mixer::Attention(a), LayerState::Attention(kv)) => {
                a.forward(&normed, positions, kv, spec, backend)?
            }
            (Mixer::Gdn(g), LayerState::Recurrent(state)) => g.forward(
                &normed,
                positions.len(),
                Arc::make_mut(state),
                spec,
                backend,
            )?,
            _ => return Err(bad("模型层与请求状态类型不匹配")),
        };
        let mut residual: Vec<f32> = input.iter().zip(&mixed).map(|(a, b)| a + b).collect();
        let normed = rms_norm(&residual, &self.post_norm, spec.rms_norm_eps, offset)?;
        let rows = positions.len();
        let mut gate = self.gate.forward(backend, &normed, rows)?;
        let up = self.up.forward(backend, &normed, rows)?;
        for (g, u) in gate.iter_mut().zip(up) {
            *g = silu(*g) * u;
        }
        let output = self.down.forward(backend, &gate, rows)?;
        for (r, y) in residual.iter_mut().zip(output) {
            *r += y;
        }
        Ok(residual)
    }
}

impl Attention {
    fn load(store: &WeightStore, prefix: &str, spec: &ModelSpec) -> Result<Self> {
        let p = format!("{prefix}.self_attn");
        let a = &spec.attention;
        let q = a.heads * a.head_dim;
        let kv = a.kv_heads * a.head_dim;
        let gated = spec.family == ModelFamily::Qwen35;
        let qk_norm = spec.family != ModelFamily::Qwen2;
        let bias = spec.family == ModelFamily::Qwen2;
        Ok(Self {
            query: Projection::load(
                store,
                &format!("{p}.q_proj"),
                q * if gated { 2 } else { 1 },
                spec.hidden_size,
                bias,
            )?,
            key: Projection::load(store, &format!("{p}.k_proj"), kv, spec.hidden_size, bias)?,
            value: Projection::load(store, &format!("{p}.v_proj"), kv, spec.hidden_size, bias)?,
            output: Projection::load(store, &format!("{p}.o_proj"), spec.hidden_size, q, false)?,
            query_norm: if qk_norm {
                Some(vector(store, &format!("{p}.q_norm.weight"), a.head_dim)?)
            } else {
                None
            },
            key_norm: if qk_norm {
                Some(vector(store, &format!("{p}.k_norm.weight"), a.head_dim)?)
            } else {
                None
            },
        })
    }

    fn forward(
        &self,
        input: &[f32],
        positions: &[[usize; 3]],
        cache: &mut crate::engine::state::PagedKvCache,
        spec: &ModelSpec,
        backend: &Backend,
    ) -> Result<Vec<f32>> {
        let rows = positions.len();
        let a = &spec.attention;
        let q_width = a.heads * a.head_dim;
        let kv_width = a.kv_heads * a.head_dim;
        let gated = spec.family == ModelFamily::Qwen35;
        let q_raw = self.query.forward(backend, input, rows)?;
        let k_raw = self.key.forward(backend, input, rows)?;
        let values = self.value.forward(backend, input, rows)?;
        let mut attended = Vec::with_capacity(rows * q_width);
        for (row, position) in positions.iter().enumerate() {
            let mut query = Vec::with_capacity(q_width);
            let mut gate = Vec::with_capacity(if gated { q_width } else { 0 });
            if gated {
                // Qwen3.5 的 Q/gate 是逐 head 交错排列，不能按整个矩阵二等分。
                let row = &q_raw[row * q_width * 2..(row + 1) * q_width * 2];
                for head in row.chunks_exact(2 * a.head_dim) {
                    query.extend_from_slice(&head[..a.head_dim]);
                    gate.extend_from_slice(&head[a.head_dim..]);
                }
            } else {
                query.extend_from_slice(&q_raw[row * q_width..(row + 1) * q_width]);
            }
            let mut key = k_raw[row * kv_width..(row + 1) * kv_width].to_vec();
            if let Some(norm) = &self.query_norm {
                query = rms_norm(&query, norm, spec.rms_norm_eps, gated)?;
            }
            if let Some(norm) = &self.key_norm {
                key = rms_norm(&key, norm, spec.rms_norm_eps, gated)?;
            }
            apply_rope(&mut query, a.head_dim, *position, &spec.rope)?;
            apply_rope(&mut key, a.head_dim, *position, &spec.rope)?;
            cache.append(&key, &values[row * kv_width..(row + 1) * kv_width])?;
            let mut output = causal_attention(&query, cache, a.heads, a.kv_heads, a.head_dim)?;
            if gated {
                for (x, g) in output.iter_mut().zip(gate) {
                    *x *= sigmoid(g);
                }
            }
            attended.extend(output);
        }
        self.output.forward(backend, &attended, rows)
    }
}

impl Gdn {
    fn load(store: &WeightStore, prefix: &str, spec: &ModelSpec) -> Result<Self> {
        let g = spec.gdn.as_ref().ok_or_else(|| bad("缺少 GDN 参数"))?;
        let p = format!("{prefix}.linear_attn");
        let channels = 2 * g.key_heads * g.key_dim + g.value_heads * g.value_dim;
        let value_width = g.value_heads * g.value_dim;
        Ok(Self {
            qkv: Projection::load(
                store,
                &format!("{p}.in_proj_qkv"),
                channels,
                spec.hidden_size,
                false,
            )?,
            z: Projection::load(
                store,
                &format!("{p}.in_proj_z"),
                value_width,
                spec.hidden_size,
                false,
            )?,
            a: Projection::load(
                store,
                &format!("{p}.in_proj_a"),
                g.value_heads,
                spec.hidden_size,
                false,
            )?,
            b: Projection::load(
                store,
                &format!("{p}.in_proj_b"),
                g.value_heads,
                spec.hidden_size,
                false,
            )?,
            output: Projection::load(
                store,
                &format!("{p}.out_proj"),
                spec.hidden_size,
                value_width,
                false,
            )?,
            convolution: tensor(
                store,
                &format!("{p}.conv1d.weight"),
                &[channels, 1, g.conv_kernel],
            )?
            .to_f32()?,
            a_log: vector(store, &format!("{p}.A_log"), g.value_heads)?,
            dt_bias: vector(store, &format!("{p}.dt_bias"), g.value_heads)?,
            norm: vector(store, &format!("{p}.norm.weight"), g.value_dim)?,
        })
    }

    fn forward(
        &self,
        input: &[f32],
        rows: usize,
        state: &mut crate::engine::state::RecurrentState,
        spec: &ModelSpec,
        backend: &Backend,
    ) -> Result<Vec<f32>> {
        let g = spec.gdn.as_ref().ok_or_else(|| bad("缺少 GDN 参数"))?;
        let channels = 2 * g.key_heads * g.key_dim + g.value_heads * g.value_dim;
        let width = g.value_heads * g.value_dim;
        let qkv = self.qkv.forward(backend, input, rows)?;
        let z = self.z.forward(backend, input, rows)?;
        let a = self.a.forward(backend, input, rows)?;
        let b = self.b.forward(backend, input, rows)?;
        let mut mixed = Vec::with_capacity(rows * width);
        for row in 0..rows {
            let conv = causal_conv_silu(
                &qkv[row * channels..(row + 1) * channels],
                &self.convolution,
                g.conv_kernel,
                &mut state.convolution,
            )?;
            let controls = row * g.value_heads..(row + 1) * g.value_heads;
            let recurrent = gated_delta_step(
                &conv,
                &a[controls.clone()],
                &b[controls],
                &self.a_log,
                &self.dt_bias,
                g,
                state,
            )?;
            let mut normed = rms_norm(&recurrent, &self.norm, spec.rms_norm_eps, false)?;
            for (x, z) in normed.iter_mut().zip(&z[row * width..(row + 1) * width]) {
                *x *= silu(*z);
            }
            mixed.extend(normed);
        }
        self.output.forward(backend, &mixed, rows)
    }
}
