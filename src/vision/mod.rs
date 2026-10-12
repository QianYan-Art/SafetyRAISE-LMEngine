//! Qwen3.5 静态视觉塔的 CPU 参考实现。
//!
//! 数学实现依据 Hugging Face Transformers Qwen3.5 视觉代码的固定提交
//! `812a950d7037ba93bb624b154030e9068afdce4d`（Apache-2.0）：3D patch
//! projection、learned position interpolation、全局非因果 attention、轴向
//! RoPE、LayerNorm、GELU MLP 与 2×2 patch merger。图像排列依据
//! Qwen2VLImageProcessor 固定提交 `a12a224f759d4457e1a2fd4ba7ced46f9a6cc430`。

mod preprocess;

use crate::artifact::{WeightStore, WeightTensor};
use crate::backend::Backend;
use crate::model::spec::VisionSpec;
use crate::{Result, RsinferError};

use preprocess::{add_interpolated_positions, patch_coordinates, PreparedImage};

/// 单路图像编码的资源上限；所有上限都必须大于零。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionLimits {
    pub max_pixels: usize,
    pub max_patches: usize,
    pub max_tokens: usize,
    pub max_bytes: usize,
}

impl Default for VisionLimits {
    fn default() -> Self {
        Self {
            // 约 1024 个 16×16 patch，适合 CPU 参考 attention。
            max_pixels: 1_048_576,
            max_patches: 1_024,
            max_tokens: 256,
            max_bytes: 32 * 1024 * 1024,
        }
    }
}

/// 视觉塔输出的合并 token，行主序为 `[grid_t, grid_h, grid_w, hidden]`。
#[derive(Debug, Clone, PartialEq)]
pub struct VisualEmbedding {
    pub values: Vec<f32>,
    pub grid: [usize; 3],
    pub hidden_size: usize,
}

struct VisionBlockWeights {
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    qkv: WeightTensor,
    qkv_bias: Vec<f32>,
    proj: WeightTensor,
    proj_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    fc1: WeightTensor,
    fc1_bias: Vec<f32>,
    fc2: WeightTensor,
    fc2_bias: Vec<f32>,
}

struct VisionWeights {
    patch_projection: Vec<f32>,
    patch_bias: Vec<f32>,
    position_embedding: Vec<f32>,
    blocks: Vec<VisionBlockWeights>,
    merger_norm_weight: Vec<f32>,
    merger_norm_bias: Vec<f32>,
    merger_fc1: WeightTensor,
    merger_fc1_bias: Vec<f32>,
    merger_fc2: WeightTensor,
    merger_fc2_bias: Vec<f32>,
}

/// Qwen3.5 vision tower，所有权重只读并可与后端共享。
pub struct VisionTower {
    spec: VisionSpec,
    backend: Backend,
    weights: VisionWeights,
    inv_freq: Vec<f32>,
}

impl VisionTower {
    /// 从权重仓库建立视觉塔。前缀以完整权重集合为单位选择，避免跨包装混拼。
    pub fn from_weights(store: &WeightStore, spec: &VisionSpec, backend: Backend) -> Result<Self> {
        validate_spec(spec)?;
        let mut last_error = None;
        // Qwen checkpoints 常见的包装层；只接受能完整加载所有视觉参数的前缀。
        for prefix in [
            "model.visual.",
            "visual.",
            "model.vision.",
            "vision.",
            "model.vision_tower.",
            "vision_tower.",
            "model.vision_model.",
            "vision_model.",
            "",
        ] {
            match load_weights(store, spec, prefix) {
                Ok(weights) => {
                    let inv_freq = axial_inv_freq(spec)?;
                    return Ok(Self {
                        spec: spec.clone(),
                        backend,
                        weights,
                        inv_freq,
                    });
                }
                Err(error) => last_error = Some(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            RsinferError::WeightError("找不到完整的 Qwen3.5 视觉权重集合".into())
        }))
    }

    /// 编码 RGB8 静态图像；静态帧会复制到 temporal patch 的各时间槽。
    pub fn encode_rgb(
        &self,
        width: usize,
        height: usize,
        rgb: &[u8],
        limits: &VisionLimits,
    ) -> Result<VisualEmbedding> {
        let limits = self.tightened_limits(limits)?;
        let prepared = preprocess::prepare_rgb(
            width,
            height,
            rgb,
            &limits,
            self.spec.patch_size,
            self.spec.spatial_merge_size,
            self.spec.temporal_patch_size,
        )?;
        self.encode_prepared(prepared, &limits)
    }

    /// 解码并编码 PNG/JPEG 静态图像，先检查压缩字节数和图像尺寸。
    pub fn encode_bytes(&self, bytes: &[u8], limits: &VisionLimits) -> Result<VisualEmbedding> {
        let limits = self.tightened_limits(limits)?;
        let prepared = preprocess::prepare_bytes(
            bytes,
            &limits,
            self.spec.patch_size,
            self.spec.spatial_merge_size,
            self.spec.temporal_patch_size,
        )?;
        self.encode_prepared(prepared, &limits)
    }

    fn tightened_limits(&self, limits: &VisionLimits) -> Result<VisionLimits> {
        let attention_budget = 128_000_000usize
            .checked_div(self.spec.num_heads)
            .ok_or_else(|| {
                RsinferError::DimensionError("视觉 attention head 数必须大于零".into())
            })?;
        let safe = attention_budget.isqrt();
        if safe == 0 {
            return Err(RsinferError::DimensionError(
                "视觉 attention patch 安全上限为零".into(),
            ));
        }
        let mut tightened = *limits;
        tightened.max_patches = tightened.max_patches.min(safe);
        Ok(tightened)
    }

    fn encode_prepared(
        &self,
        prepared: PreparedImage,
        limits: &VisionLimits,
    ) -> Result<VisualEmbedding> {
        let hidden = self.spec.hidden_size;
        let patch_count = prepared
            .patches
            .len()
            .checked_div(prepared.patch_width)
            .ok_or_else(|| RsinferError::DimensionError("视觉 patch 宽度为零".into()))?;
        if patch_count == 0 || patch_count > limits.max_patches {
            return Err(RsinferError::DimensionError(format!(
                "视觉 patch 数量 {} 超过上限 {}",
                patch_count, limits.max_patches
            )));
        }
        let merged_h = prepared.grid_h / self.spec.spatial_merge_size;
        let merged_w = prepared.grid_w / self.spec.spatial_merge_size;
        let merged_tokens = merged_h
            .checked_mul(merged_w)
            .ok_or_else(|| RsinferError::DimensionError("合并 token 数量溢出".into()))?;
        if merged_tokens == 0 || merged_tokens > limits.max_tokens {
            return Err(RsinferError::DimensionError(format!(
                "合并 token 数量 {} 超过上限 {}",
                merged_tokens, limits.max_tokens
            )));
        }
        // attention 不建立 N×N 的二次矩阵，但仍限制计算量，避免误配导致长时间阻塞。
        let attention_work = patch_count
            .checked_mul(patch_count)
            .and_then(|n| n.checked_mul(self.spec.num_heads))
            .ok_or_else(|| RsinferError::DimensionError("attention 工作量溢出".into()))?;
        if attention_work > 128_000_000 {
            return Err(RsinferError::DimensionError(format!(
                "视觉 attention 工作量 {} 超过 CPU 参考上限 128000000",
                attention_work
            )));
        }

        let mut hidden_states = self.project_patches(&prepared)?;
        add_interpolated_positions(
            &mut hidden_states,
            &self.weights.position_embedding,
            self.spec.num_position_embeddings,
            hidden,
            prepared.grid_h,
            prepared.grid_w,
            self.spec.spatial_merge_size,
        )?;

        for block in &self.weights.blocks {
            hidden_states =
                self.forward_block(hidden_states, block, prepared.grid_h, prepared.grid_w)?;
        }

        let values = self.merge(&hidden_states, prepared.grid_h, prepared.grid_w)?;
        Ok(VisualEmbedding {
            values,
            grid: [1, merged_h, merged_w],
            hidden_size: self.spec.out_hidden_size,
        })
    }

    fn project_patches(&self, prepared: &PreparedImage) -> Result<Vec<f32>> {
        let hidden = self.spec.hidden_size;
        let rows = prepared.patches.len() / prepared.patch_width;
        let mut output = vec![0.0; rows * hidden];
        for row in 0..rows {
            let input =
                &prepared.patches[row * prepared.patch_width..(row + 1) * prepared.patch_width];
            let dst = &mut output[row * hidden..(row + 1) * hidden];
            for (out, (dst_value, weights)) in dst
                .iter_mut()
                .zip(
                    self.weights
                        .patch_projection
                        .chunks_exact(prepared.patch_width),
                )
                .enumerate()
            {
                let mut value = self.weights.patch_bias[out];
                for (weight, input) in weights.iter().zip(input) {
                    value += weight * input;
                }
                *dst_value = value;
            }
        }
        Ok(output)
    }

    fn forward_block(
        &self,
        mut hidden_states: Vec<f32>,
        block: &VisionBlockWeights,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.spec.hidden_size;
        let rows = hidden_states.len() / hidden;
        let normed = layer_norm(
            &hidden_states,
            &block.norm1_weight,
            &block.norm1_bias,
            hidden,
            1e-6,
        )?;
        let qkv = linear_with_bias(
            &self.backend,
            &block.qkv,
            &block.qkv_bias,
            &normed,
            rows,
            hidden * 3,
        )?;
        let head_dim = hidden / self.spec.num_heads;
        let mut query = vec![0.0; rows * hidden];
        let mut key = vec![0.0; rows * hidden];
        let mut value = vec![0.0; rows * hidden];
        for row in 0..rows {
            for head in 0..self.spec.num_heads {
                let dst = (row * self.spec.num_heads + head) * head_dim;
                let src = row * hidden * 3 + head * head_dim;
                query[dst..dst + head_dim].copy_from_slice(&qkv[src..src + head_dim]);
                let key_src = row * hidden * 3 + hidden + head * head_dim;
                key[dst..dst + head_dim].copy_from_slice(&qkv[key_src..key_src + head_dim]);
                let value_src = row * hidden * 3 + hidden * 2 + head * head_dim;
                value[dst..dst + head_dim].copy_from_slice(&qkv[value_src..value_src + head_dim]);
            }
        }
        apply_axial_rope(
            &mut query,
            &mut key,
            rows,
            self.spec.num_heads,
            head_dim,
            grid_h,
            grid_w,
            self.spec.spatial_merge_size,
            &self.inv_freq,
        );
        let attended = attention(&query, &key, &value, rows, self.spec.num_heads, head_dim)?;
        let projected = linear_with_bias(
            &self.backend,
            &block.proj,
            &block.proj_bias,
            &attended,
            rows,
            hidden,
        )?;
        for (dst, value) in hidden_states.iter_mut().zip(projected) {
            *dst += value;
        }

        let normed = layer_norm(
            &hidden_states,
            &block.norm2_weight,
            &block.norm2_bias,
            hidden,
            1e-6,
        )?;
        let mut feed_forward = linear_with_bias(
            &self.backend,
            &block.fc1,
            &block.fc1_bias,
            &normed,
            rows,
            self.spec.intermediate_size,
        )?;
        for value in &mut feed_forward {
            *value = gelu_tanh(*value);
        }
        let feed_forward = linear_with_bias(
            &self.backend,
            &block.fc2,
            &block.fc2_bias,
            &feed_forward,
            rows,
            hidden,
        )?;
        for (dst, value) in hidden_states.iter_mut().zip(feed_forward) {
            *dst += value;
        }
        Ok(hidden_states)
    }

    fn merge(&self, hidden_states: &[f32], grid_h: usize, grid_w: usize) -> Result<Vec<f32>> {
        let hidden = self.spec.hidden_size;
        let merge = self.spec.spatial_merge_size;
        let merged_h = grid_h / merge;
        let merged_w = grid_w / merge;
        let merged_hidden = hidden
            .checked_mul(merge)
            .and_then(|n| n.checked_mul(merge))
            .ok_or_else(|| RsinferError::DimensionError("merger 隐藏宽度溢出".into()))?;
        let rows = merged_h * merged_w;
        let mut merger_input = vec![0.0; rows * merged_hidden];
        let normed = layer_norm(
            hidden_states,
            &self.weights.merger_norm_weight,
            &self.weights.merger_norm_bias,
            hidden,
            1e-6,
        )?;
        for row in 0..rows {
            let block_h = row / merged_w;
            let block_w = row % merged_w;
            let dst = &mut merger_input[row * merged_hidden..(row + 1) * merged_hidden];
            let mut offset = 0;
            for inner_h in 0..merge {
                for inner_w in 0..merge {
                    let token = (block_h * merge + inner_h) * grid_w + block_w * merge + inner_w;
                    let source = &normed[token * hidden..(token + 1) * hidden];
                    dst[offset..offset + hidden].copy_from_slice(source);
                    offset += hidden;
                }
            }
        }
        let mut first = linear_with_bias(
            &self.backend,
            &self.weights.merger_fc1,
            &self.weights.merger_fc1_bias,
            &merger_input,
            rows,
            merged_hidden,
        )?;
        for value in &mut first {
            *value = gelu_exact(*value);
        }
        linear_with_bias(
            &self.backend,
            &self.weights.merger_fc2,
            &self.weights.merger_fc2_bias,
            &first,
            rows,
            self.spec.out_hidden_size,
        )
    }
}

fn validate_spec(spec: &VisionSpec) -> Result<()> {
    if spec.depth == 0
        || spec.hidden_size == 0
        || spec.intermediate_size == 0
        || spec.num_heads == 0
        || spec.in_channels != 3
        || spec.patch_size == 0
        || spec.spatial_merge_size == 0
        || spec.temporal_patch_size == 0
        || spec.out_hidden_size == 0
        || spec.num_position_embeddings == 0
    {
        return Err(RsinferError::ConfigError(
            "视觉配置包含零维度或非 RGB 通道".into(),
        ));
    }
    if !spec.hidden_size.is_multiple_of(spec.num_heads) {
        return Err(RsinferError::ConfigError(
            "视觉 hidden_size 必须整除 num_heads".into(),
        ));
    }
    let head_dim = spec.hidden_size / spec.num_heads;
    if !head_dim.is_multiple_of(4) {
        return Err(RsinferError::ConfigError(
            "视觉 head_dim 必须是 4 的倍数以使用轴向 RoPE".into(),
        ));
    }
    let side = (spec.num_position_embeddings as f64).sqrt() as usize;
    if side * side != spec.num_position_embeddings {
        return Err(RsinferError::ConfigError(
            "视觉 num_position_embeddings 必须是平方数".into(),
        ));
    }
    Ok(())
}

fn axial_inv_freq(spec: &VisionSpec) -> Result<Vec<f32>> {
    let head_dim = spec.hidden_size / spec.num_heads;
    let spatial_dim = head_dim / 2;
    // Qwen3.5VisionConfig does not serialize rope_theta; Transformers'
    // RotaryEmbeddingConfigMixin supplies its default_theta = 10_000.0.
    let theta = 10_000.0f32;
    Ok((0..spatial_dim)
        .step_by(2)
        .map(|index| 1.0 / theta.powf(index as f32 / spatial_dim as f32))
        .collect())
}

fn load_weights(store: &WeightStore, spec: &VisionSpec, prefix: &str) -> Result<VisionWeights> {
    let name = |key: &str| format!("{prefix}{key}");
    let patch_shape = [
        spec.hidden_size,
        spec.in_channels,
        spec.temporal_patch_size,
        spec.patch_size,
        spec.patch_size,
    ];
    let patch = load_tensor(store, &name("patch_embed.proj.weight"), &patch_shape)?;
    let patch_projection = patch.to_f32()?;
    let patch_bias = load_vector(store, &name("patch_embed.proj.bias"), spec.hidden_size)?;
    let position_embedding = load_matrix(
        store,
        &name("pos_embed.weight"),
        spec.num_position_embeddings,
        spec.hidden_size,
    )?;

    let mut blocks = Vec::with_capacity(spec.depth);
    for index in 0..spec.depth {
        let block = format!("blocks.{index}.");
        blocks.push(VisionBlockWeights {
            norm1_weight: load_vector(
                store,
                &name(&format!("{block}norm1.weight")),
                spec.hidden_size,
            )?,
            norm1_bias: load_vector(
                store,
                &name(&format!("{block}norm1.bias")),
                spec.hidden_size,
            )?,
            qkv: load_tensor(
                store,
                &name(&format!("{block}attn.qkv.weight")),
                &[spec.hidden_size * 3, spec.hidden_size],
            )?,
            qkv_bias: load_vector(
                store,
                &name(&format!("{block}attn.qkv.bias")),
                spec.hidden_size * 3,
            )?,
            proj: load_tensor(
                store,
                &name(&format!("{block}attn.proj.weight")),
                &[spec.hidden_size, spec.hidden_size],
            )?,
            proj_bias: load_vector(
                store,
                &name(&format!("{block}attn.proj.bias")),
                spec.hidden_size,
            )?,
            norm2_weight: load_vector(
                store,
                &name(&format!("{block}norm2.weight")),
                spec.hidden_size,
            )?,
            norm2_bias: load_vector(
                store,
                &name(&format!("{block}norm2.bias")),
                spec.hidden_size,
            )?,
            fc1: load_tensor(
                store,
                &name(&format!("{block}mlp.linear_fc1.weight")),
                &[spec.intermediate_size, spec.hidden_size],
            )?,
            fc1_bias: load_vector(
                store,
                &name(&format!("{block}mlp.linear_fc1.bias")),
                spec.intermediate_size,
            )?,
            fc2: load_tensor(
                store,
                &name(&format!("{block}mlp.linear_fc2.weight")),
                &[spec.hidden_size, spec.intermediate_size],
            )?,
            fc2_bias: load_vector(
                store,
                &name(&format!("{block}mlp.linear_fc2.bias")),
                spec.hidden_size,
            )?,
        });
    }

    let merger_hidden = spec
        .hidden_size
        .checked_mul(spec.spatial_merge_size)
        .and_then(|n| n.checked_mul(spec.spatial_merge_size))
        .ok_or_else(|| RsinferError::ConfigError("merger 隐藏宽度溢出".into()))?;
    Ok(VisionWeights {
        patch_projection,
        patch_bias,
        position_embedding,
        blocks,
        merger_norm_weight: load_vector(store, &name("merger.norm.weight"), spec.hidden_size)?,
        merger_norm_bias: load_vector(store, &name("merger.norm.bias"), spec.hidden_size)?,
        merger_fc1: load_tensor(
            store,
            &name("merger.linear_fc1.weight"),
            &[merger_hidden, merger_hidden],
        )?,
        merger_fc1_bias: load_vector(store, &name("merger.linear_fc1.bias"), merger_hidden)?,
        merger_fc2: load_tensor(
            store,
            &name("merger.linear_fc2.weight"),
            &[spec.out_hidden_size, merger_hidden],
        )?,
        merger_fc2_bias: load_vector(store, &name("merger.linear_fc2.bias"), spec.out_hidden_size)?,
    })
}

fn load_tensor(store: &WeightStore, name: &str, expected: &[usize]) -> Result<WeightTensor> {
    let tensor = store.get(name)?;
    if tensor.shape() != expected {
        return Err(RsinferError::ShapeMismatch {
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(tensor)
}

fn load_vector(store: &WeightStore, name: &str, len: usize) -> Result<Vec<f32>> {
    let tensor = load_tensor(store, name, &[len])?;
    tensor.to_f32()
}

fn load_matrix(store: &WeightStore, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>> {
    let tensor = load_tensor(store, name, &[rows, cols])?;
    tensor.to_f32()
}

fn linear_with_bias(
    backend: &Backend,
    weight: &WeightTensor,
    bias: &[f32],
    input: &[f32],
    rows: usize,
    output_dim: usize,
) -> Result<Vec<f32>> {
    let mut output = backend.linear(weight, input, rows)?;
    if output.len() != rows * output_dim || bias.len() != output_dim {
        return Err(RsinferError::DimensionError(
            "线性层输出或 bias 长度不匹配".into(),
        ));
    }
    for row in 0..rows {
        for col in 0..output_dim {
            output[row * output_dim + col] += bias[col];
        }
    }
    Ok(output)
}

fn layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    hidden: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    if hidden == 0
        || !input.len().is_multiple_of(hidden)
        || weight.len() != hidden
        || bias.len() != hidden
    {
        return Err(RsinferError::DimensionError(
            "LayerNorm 输入形状不匹配".into(),
        ));
    }
    let rows = input.len() / hidden;
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let source = &input[row * hidden..(row + 1) * hidden];
        let mean = source.iter().sum::<f32>() / hidden as f32;
        let variance = source
            .iter()
            .map(|value| {
                let delta = *value - mean;
                delta * delta
            })
            .sum::<f32>()
            / hidden as f32;
        let inv = (variance + eps).sqrt().recip();
        let dst = &mut output[row * hidden..(row + 1) * hidden];
        for col in 0..hidden {
            dst[col] = (source[col] - mean) * inv * weight[col] + bias[col];
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn apply_axial_rope(
    query: &mut [f32],
    key: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    grid_h: usize,
    grid_w: usize,
    merge: usize,
    inv_freq: &[f32],
) {
    let half = head_dim / 2;
    for token in 0..rows {
        let (row, col) = patch_coordinates(token, grid_h, grid_w, merge);
        let mut cos = vec![0.0f32; half];
        let mut sin = vec![0.0f32; half];
        for index in 0..inv_freq.len() {
            let h_angle = row as f32 * inv_freq[index];
            let w_angle = col as f32 * inv_freq[index];
            cos[index] = h_angle.cos();
            sin[index] = h_angle.sin();
            cos[inv_freq.len() + index] = w_angle.cos();
            sin[inv_freq.len() + index] = w_angle.sin();
        }
        for head in 0..heads {
            let offset = (token * heads + head) * head_dim;
            for index in 0..half {
                let q0 = query[offset + index];
                let q1 = query[offset + half + index];
                query[offset + index] = q0 * cos[index] - q1 * sin[index];
                query[offset + half + index] = q1 * cos[index] + q0 * sin[index];
                let k0 = key[offset + index];
                let k1 = key[offset + half + index];
                key[offset + index] = k0 * cos[index] - k1 * sin[index];
                key[offset + half + index] = k1 * cos[index] + k0 * sin[index];
            }
        }
    }
}

fn attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let hidden = heads
        .checked_mul(head_dim)
        .ok_or_else(|| RsinferError::DimensionError("attention 隐藏宽度溢出".into()))?;
    if query.len() != rows * hidden || key.len() != query.len() || value.len() != query.len() {
        return Err(RsinferError::DimensionError(
            "attention 张量形状不匹配".into(),
        ));
    }
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![0.0f32; rows];
    let scale = (head_dim as f32).sqrt().recip();
    for row in 0..rows {
        for head in 0..heads {
            let q_offset = (row * heads + head) * head_dim;
            let mut max_score = f32::NEG_INFINITY;
            for (other, score_slot) in scores.iter_mut().enumerate() {
                let k_offset = (other * heads + head) * head_dim;
                let mut dot = 0.0;
                for index in 0..head_dim {
                    dot += query[q_offset + index] * key[k_offset + index];
                }
                let score = dot * scale;
                *score_slot = score;
                max_score = max_score.max(score);
            }
            let mut normalizer = 0.0;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                normalizer += *score;
            }
            let out_offset = q_offset;
            for index in 0..head_dim {
                let mut weighted = 0.0;
                for (other, score) in scores.iter().enumerate() {
                    let v_offset = (other * heads + head) * head_dim;
                    weighted += *score * value[v_offset + index];
                }
                output[out_offset + index] = weighted / normalizer;
            }
        }
    }
    Ok(output)
}

fn gelu_tanh(value: f32) -> f32 {
    0.5 * value
        * (1.0 + (std::f32::consts::FRAC_2_SQRT_PI * (value + 0.044715 * value.powi(3))).tanh())
}

fn gelu_exact(value: f32) -> f32 {
    0.5 * value * (1.0 + erf_approx(value / std::f32::consts::SQRT_2))
}

fn erf_approx(value: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26; max absolute error is about 1.5e-7.
    let sign = if value < 0.0 { -1.0f64 } else { 1.0f64 };
    let x = f64::from(value.abs());
    let t = 1.0f64 / (1.0 + 0.3275911f64 * x);
    let polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
        * t
        + 0.254829592)
        * t;
    (sign * (1.0 - polynomial * (-x * x).exp())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_matches_two_key_softmax() {
        let query = [1.0, 0.0, 0.0, 1.0];
        let key = [1.0, 0.0, 0.0, 1.0];
        let value = [2.0, 0.0, 0.0, 4.0];
        let output = attention(&query, &key, &value, 2, 1, 2).unwrap();
        assert!((output[0] - 1.3395231).abs() < 1e-5);
        assert!((output[1] - 1.3209538).abs() < 1e-5);
    }

    #[test]
    fn gelu_values_are_finite() {
        assert!((gelu_tanh(0.0)).abs() < 1e-7);
        assert!((gelu_exact(0.0)).abs() < 1e-7);
        assert!(gelu_tanh(3.0).is_finite());
        assert!(gelu_exact(-3.0).is_finite());
    }
}
