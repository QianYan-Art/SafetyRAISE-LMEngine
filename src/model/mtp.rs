//! Qwen3.5 的单层多 token 预测（MTP）执行器。
//!
//! MTP 只复用目标模型的 embedding 与 lm head；它自己的 stem、单层全注意力、
//! 归一化和 KV 状态完全独立。目标模型的 hidden 是已经过目标 final norm 的值，
//! 因此 MTP 不会再次执行目标层，而是在进入 stem 前分别归一化 embedding 与
//! hidden。

use crate::artifact::WeightStore;
use crate::backend::Backend;
use crate::engine::state::{LayerState, SequenceState};
use crate::model::math::rms_norm;
use crate::model::native::{allocate_model_id, vector, Block, NativeModel, Projection};
use crate::model::spec::{LayerKind, ModelSpec};
use crate::{Result, RsinferError};

fn invalid(message: impl Into<String>) -> RsinferError {
    RsinferError::DimensionError(message.into())
}

fn unsupported(message: impl Into<String>) -> RsinferError {
    RsinferError::Unsupported(message.into())
}

fn next_position(position: [usize; 3]) -> Result<[usize; 3]> {
    Ok([
        position[0]
            .checked_add(1)
            .ok_or_else(|| invalid("MTP 位置溢出"))?,
        position[1]
            .checked_add(1)
            .ok_or_else(|| invalid("MTP 位置溢出"))?,
        position[2]
            .checked_add(1)
            .ok_or_else(|| invalid("MTP 位置溢出"))?,
    ])
}

/// MTP 请求的私有状态。
///
/// `sequence` 保存单层注意力的 KV；`tail_hidden` 是尚未和下一个 token 配对的
/// hidden。prefill 时它是目标模型的 final hidden，单 token MTP 前向后它会变成
/// MTP 自己的 final hidden。两种值都带有同一个“下一 token 应使用的位置”。
#[derive(Clone)]
pub struct MtpState {
    pub(crate) sequence: SequenceState,
    pub(crate) tail_hidden: Option<Vec<f32>>,
    pub(crate) tail_position: Option<[usize; 3]>,
    pub(crate) last_logits: Vec<f32>,
}

impl MtpState {
    /// 返回已经处理的 MTP 行数。
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }

    pub fn max_context(&self) -> usize {
        self.sequence.max_context()
    }

    pub fn next_position(&self) -> usize {
        self.tail_position
            .map(|position| position[0])
            .unwrap_or_else(|| self.sequence.next_position())
    }

    /// 最近一次 MTP 行产生的完整目标词表 logits。
    pub fn last_logits(&self) -> &[f32] {
        &self.last_logits
    }

    /// 返回下一 token 配对前保留的 hidden。
    pub fn tail_hidden(&self) -> Option<&[f32]> {
        self.tail_hidden.as_deref()
    }

    /// COW KV 与尾部 hidden 的完整快照。
    pub fn checkpoint(&self) -> Self {
        self.clone()
    }
}

/// Qwen3.5 MTP 执行器。
pub struct MtpModel {
    spec: ModelSpec,
    backend: Backend,
    owner: u64,
    target_owner: u64,
    fc: Projection,
    pre_fc_norm_embedding: Vec<f32>,
    pre_fc_norm_hidden: Vec<f32>,
    norm: Vec<f32>,
    layer: Block,
}

impl MtpModel {
    /// 从目标模型权重中加载 MTP。
    ///
    /// 仓库可能不包含 MTP，此时返回 `Ok(None)`。只要出现任何 MTP 前缀，便
    /// 要求整个必要参数集合存在，避免把不完整的草稿模型当成可用模型。
    pub fn from_weights(store: &WeightStore, target: &NativeModel) -> Result<Option<Self>> {
        let prefix = match find_mtp_prefix(store)? {
            None => return Ok(None),
            Some(prefix) => prefix,
        };
        if target.spec.family != crate::model::spec::ModelFamily::Qwen35 {
            return Err(unsupported("MTP 目前只支持 Qwen3.5"));
        }
        let mut spec = target.spec.clone();
        spec.layers = vec![LayerKind::FullAttention];
        let h = spec.hidden_size;
        let fc_input = h
            .checked_mul(2)
            .ok_or_else(|| invalid("MTP fc 输入维度溢出"))?;
        let fc = Projection::load(store, &format!("{prefix}.fc"), h, fc_input, false)?;
        let pre_fc_norm_embedding =
            vector(store, &format!("{prefix}.pre_fc_norm_embedding.weight"), h)?;
        let pre_fc_norm_hidden = vector(store, &format!("{prefix}.pre_fc_norm_hidden.weight"), h)?;
        let norm = vector(store, &format!("{prefix}.norm.weight"), h)?;
        let layer = Block::load(
            store,
            &format!("{prefix}.layers.0"),
            LayerKind::FullAttention,
            &spec,
        )?;
        let owner = allocate_model_id()?;
        Ok(Some(Self {
            spec,
            backend: target.backend.clone(),
            owner,
            target_owner: target.model_id(),
            fc,
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            norm,
            layer,
        }))
    }

    pub fn model_id(&self) -> u64 {
        self.owner
    }

    pub(crate) fn validate_target(&self, target: &NativeModel) -> Result<()> {
        if target.model_id() != self.target_owner {
            return Err(invalid("MTP 与目标模型实例不匹配"));
        }
        Ok(())
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// 创建只属于当前 MTP 实例的单层状态。
    pub fn create_state(&self, context: usize) -> Result<MtpState> {
        let mut sequence = SequenceState::new(&self.spec, context)?;
        sequence.owner = self.owner;
        Ok(MtpState {
            sequence,
            tail_hidden: None,
            tail_position: None,
            last_logits: Vec::new(),
        })
    }

    fn check_state(&self, state: &MtpState) -> Result<()> {
        if state.sequence.owner != self.owner
            || state.sequence.layers.len() != 1
            || !matches!(state.sequence.layers[0], LayerState::Attention(_))
        {
            return Err(invalid("MTP 状态不属于当前 MTP 实例"));
        }
        Ok(())
    }

    fn stem(
        &self,
        target: &NativeModel,
        embeddings: &[f32],
        hidden: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let h = self.spec.hidden_size;
        if rows == 0
            || embeddings.len()
                != rows
                    .checked_mul(h)
                    .ok_or_else(|| invalid("MTP stem 大小溢出"))?
            || hidden.len() != embeddings.len()
            || embeddings.iter().chain(hidden).any(|v| !v.is_finite())
        {
            return Err(invalid("MTP stem 输入形状或数值非法"));
        }
        let embed = rms_norm(
            embeddings,
            &self.pre_fc_norm_embedding,
            self.spec.rms_norm_eps,
            true,
        )?;
        let hidden = rms_norm(
            hidden,
            &self.pre_fc_norm_hidden,
            self.spec.rms_norm_eps,
            true,
        )?;
        let mut concat = Vec::with_capacity(rows * h * 2);
        for (e, x) in embed.chunks_exact(h).zip(hidden.chunks_exact(h)) {
            concat.extend_from_slice(e);
            concat.extend_from_slice(x);
        }
        let output = self.fc.forward(&self.backend, &concat, rows)?;
        if output.iter().any(|v| !v.is_finite()) {
            return Err(invalid("MTP fc 产生非有限激活"));
        }
        // 保留 target 参数在签名中，确保 MTP 的 head/embedding 只通过目标模型
        // 访问；这里的 stem 本身不需要调用 target。
        let _ = target;
        Ok(output)
    }

    fn run_rows(
        &self,
        target: &NativeModel,
        embeddings: &[f32],
        hidden: &[f32],
        positions: &[[usize; 3]],
        state: &mut MtpState,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.check_state(state)?;
        let rows = positions.len();
        if rows == 0 {
            return Err(invalid("MTP 输入不能为空"));
        }
        if positions
            .iter()
            .flatten()
            .any(|p| *p >= self.spec.max_position_embeddings)
        {
            return Err(invalid("MTP 位置超出模型容量"));
        }
        state.sequence.ensure_room(rows)?;
        let mut candidate = state.sequence.checkpoint();
        let stem = self.stem(target, embeddings, hidden, rows)?;
        let mixed = self.layer.forward(
            &stem,
            positions,
            &mut candidate.layers[0],
            &self.spec,
            &self.backend,
        )?;
        let output = rms_norm(&mixed, &self.norm, self.spec.rms_norm_eps, true)?;
        if output.iter().any(|v| !v.is_finite()) {
            return Err(invalid("MTP norm 产生非有限激活"));
        }
        // 后续草稿只消费末位置分布；prefill 不必为每行做一次完整词表投影。
        let last_hidden = &output[(rows - 1) * self.spec.hidden_size..];
        let logits = target.project_logits(last_hidden, 1)?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(invalid("MTP logits 包含非有限值"));
        }
        candidate.len = candidate
            .len
            .checked_add(rows)
            .ok_or_else(|| invalid("MTP 状态长度溢出"))?;
        let last_position = positions
            .iter()
            .flatten()
            .copied()
            .max()
            .ok_or_else(|| invalid("MTP 位置为空"))?;
        candidate.next_position = candidate.next_position.max(
            last_position
                .checked_add(1)
                .ok_or_else(|| invalid("MTP 位置溢出"))?,
        );
        candidate.last_hidden = last_hidden.to_vec();
        state.sequence = candidate;
        state.last_logits = logits.clone();
        Ok((output, logits))
    }

    /// 用目标模型的 prompt hidden 预填充 MTP。
    ///
    /// 第一个 token 没有前一位置的 hidden，因此会被跳过；随后每一行使用
    /// `token[i]` 与 `target_hidden[i-1]`。如果已有尾部 hidden，则当前 chunk
    /// 的第一个 token 会先与该尾部配对。chunk 末尾的目标 hidden 永远保留为
    /// 下一 token 的未对齐尾部，不会误用刚刚生成的 MTP hidden。
    pub fn prefill(
        &self,
        target: &NativeModel,
        tokens: &[u32],
        target_hidden: &[f32],
        positions: &[[usize; 3]],
        state: &mut MtpState,
    ) -> Result<()> {
        self.validate_target(target)?;
        self.check_state(state)?;
        let h = self.spec.hidden_size;
        if tokens.is_empty()
            || positions.len() != tokens.len()
            || target_hidden.len()
                != tokens
                    .len()
                    .checked_mul(h)
                    .ok_or_else(|| invalid("MTP hidden 大小溢出"))?
            || target_hidden.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("MTP prefill 输入形状或数值非法"));
        }
        if tokens
            .iter()
            .any(|token| *token as usize >= self.spec.vocab_size)
        {
            return Err(invalid("MTP token 不属于目标词表"));
        }
        if positions
            .iter()
            .flatten()
            .any(|p| *p >= self.spec.max_position_embeddings)
        {
            return Err(invalid("MTP prefill 位置超出模型容量"));
        }
        if let Some(tail_position) = state.tail_position {
            if positions[0] != next_position(tail_position)? {
                return Err(invalid("MTP chunk 与未对齐尾部的位置不连续"));
            }
        }

        let has_tail = state.tail_hidden.is_some();
        let pair_count = if has_tail {
            tokens.len()
        } else {
            tokens.len().saturating_sub(1)
        };
        let mut pair_tokens = Vec::with_capacity(pair_count);
        let mut pair_hidden = Vec::with_capacity(pair_count * h);
        let mut pair_positions = Vec::with_capacity(pair_count);
        if has_tail {
            pair_tokens.push(tokens[0]);
            pair_hidden.extend_from_slice(
                state
                    .tail_hidden
                    .as_deref()
                    .ok_or_else(|| invalid("MTP 尾部状态不完整"))?,
            );
            pair_positions.push(
                state
                    .tail_position
                    .ok_or_else(|| invalid("MTP 尾部位置不完整"))?,
            );
        }
        let start = 1usize;
        for i in start..tokens.len() {
            pair_tokens.push(tokens[i]);
            pair_hidden.extend_from_slice(&target_hidden[(i - 1) * h..i * h]);
            pair_positions.push(positions[i - 1]);
        }

        if !pair_tokens.is_empty() {
            let mut embeddings = Vec::with_capacity(pair_tokens.len() * h);
            for token in &pair_tokens {
                embeddings.extend_from_slice(&target.embed(&[*token])?);
            }
            self.run_rows(target, &embeddings, &pair_hidden, &pair_positions, state)?;
        } else {
            state.last_logits.clear();
        }

        // 无论是否产生 MTP 行，最后一个目标 hidden 都属于下一 token 的输入。
        state.tail_hidden = Some(target_hidden[(tokens.len() - 1) * h..].to_vec());
        state.tail_position = Some(positions[tokens.len() - 1]);
        Ok(())
    }

    /// 用当前尾部 hidden 消费一个 token，返回该 token 之后的目标词表 logits。
    pub fn step(&self, target: &NativeModel, token: u32, state: &mut MtpState) -> Result<Vec<f32>> {
        self.step_with_hidden(target, token, None, state)
    }

    /// 消费一个 token；`hidden` 仅用于恢复/重放时覆盖当前尾部 hidden。
    pub(crate) fn step_with_hidden(
        &self,
        target: &NativeModel,
        token: u32,
        hidden: Option<&[f32]>,
        state: &mut MtpState,
    ) -> Result<Vec<f32>> {
        self.validate_target(target)?;
        self.check_state(state)?;
        if token as usize >= self.spec.vocab_size {
            return Err(invalid("MTP token 不属于目标词表"));
        }
        let source = match hidden {
            Some(hidden) => hidden.to_vec(),
            None => state
                .tail_hidden
                .clone()
                .ok_or_else(|| invalid("MTP 尚未准备可配对的 hidden"))?,
        };
        if source.len() != self.spec.hidden_size || source.iter().any(|v| !v.is_finite()) {
            return Err(invalid("MTP 配对 hidden 形状或数值非法"));
        }
        let position = state
            .tail_position
            .ok_or_else(|| invalid("MTP 尚未准备可配对的位置"))?;
        let embedding = target.embed(&[token])?;
        let (_, logits) = self.run_rows(target, &embedding, &source, &[position], state)?;
        let next = next_position(position)?;
        state.tail_hidden = Some(state.sequence.last_hidden().to_vec());
        state.tail_position = Some(next);
        Ok(logits)
    }

    /// 将目标模型真正提交的 token 与目标 hidden 重放到 MTP。
    ///
    /// 第一 token 使用快照中的尾部 hidden；后续 token 使用目标模型对应的
    /// 前一行 hidden，绝不沿用被拒绝草稿的 MTP hidden。
    pub(crate) fn replay(
        &self,
        target: &NativeModel,
        tokens: &[u32],
        target_hidden: &[f32],
        state: &mut MtpState,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let h = self.spec.hidden_size;
        if target_hidden.len()
            != tokens
                .len()
                .checked_mul(h)
                .ok_or_else(|| invalid("MTP replay hidden 大小溢出"))?
        {
            return Err(invalid("MTP replay hidden 形状错误"));
        }
        for (i, token) in tokens.iter().copied().enumerate() {
            if i == 0 {
                self.step_with_hidden(target, token, None, state)?;
            } else {
                self.step_with_hidden(
                    target,
                    token,
                    Some(&target_hidden[(i - 1) * h..i * h]),
                    state,
                )?;
            }
        }
        Ok(())
    }

    /// 在尚未调用 prefill 的请求上，以目标模型末行 hidden 建立首个尾部。
    pub(crate) fn seed_from_target(
        &self,
        target_state: &SequenceState,
        state: &mut MtpState,
    ) -> Result<()> {
        self.check_state(state)?;
        if target_state.owner != self.target_owner {
            return Err(invalid("MTP 尾部来自不同的目标模型实例"));
        }
        if state.tail_hidden.is_some() {
            return Ok(());
        }
        if target_state.last_hidden().len() != self.spec.hidden_size {
            return Err(invalid("目标状态缺少 MTP 所需的 last hidden"));
        }
        state.tail_hidden = Some(target_state.last_hidden().to_vec());
        let position = target_state
            .next_position()
            .checked_sub(1)
            .ok_or_else(|| invalid("目标状态缺少上一 token 位置"))?;
        state.tail_position = Some([position, position, position]);
        Ok(())
    }
}

fn find_mtp_prefix(store: &WeightStore) -> Result<Option<String>> {
    let candidates = ["mtp", "model.mtp", "model.language_model.mtp"];
    let mut found = None;
    for prefix in candidates {
        let marker = format!("{prefix}.fc.weight");
        if store.names().any(|name| name == marker) {
            found = Some(prefix.to_owned());
            break;
        }
    }
    let Some(prefix) = found else {
        let has_partial = store.names().any(|name| {
            name == "mtp"
                || name.starts_with("mtp.")
                || name.starts_with("model.mtp.")
                || name.starts_with("model.language_model.mtp.")
        });
        if has_partial {
            return Err(RsinferError::WeightError(
                "检测到 MTP 权重前缀，但缺少 mtp.fc.weight".into(),
            ));
        }
        return Ok(None);
    };
    let required = [
        "fc.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "norm.weight",
    ];
    for suffix in required {
        let name = format!("{prefix}.{suffix}");
        if !store.names().any(|candidate| candidate == name) {
            return Err(RsinferError::WeightError(format!("缺少 MTP 权重 {name}")));
        }
    }
    let layer_prefix = format!("{prefix}.layers.0.");
    if !store.names().any(|name| name.starts_with(&layer_prefix)) {
        return Err(RsinferError::WeightError(format!(
            "缺少 MTP 单层权重前缀 {layer_prefix}"
        )));
    }
    Ok(Some(prefix))
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::artifact::{WeightStore, WeightTensor};
    use crate::model::native::fixtures;
    use crate::model::spec::{ModelFamily, ModelSpec};
    use std::collections::BTreeMap;

    pub fn with_mtp() -> (WeightStore, ModelSpec) {
        let (base, spec) = fixtures::tiny(ModelFamily::Qwen35);
        let config = base.hf_config().cloned().expect("tiny fixture config");
        let mut tensors = BTreeMap::new();
        for name in base.names() {
            tensors.insert(name.to_owned(), base.get(name).unwrap());
        }
        let mut insert = |name: &str, shape: Vec<usize>, value: f32| {
            let count: usize = shape.iter().product();
            let data = (0..count)
                .map(|index| value + ((index % 11) as f32 - 5.0) / 100.0)
                .collect();
            tensors.insert(
                name.to_owned(),
                WeightTensor::from_f32(shape, data).unwrap(),
            );
        };
        insert("mtp.fc.weight", vec![4, 8], 0.02);
        insert("mtp.pre_fc_norm_embedding.weight", vec![4], 0.0);
        insert("mtp.pre_fc_norm_hidden.weight", vec![4], 0.0);
        insert("mtp.norm.weight", vec![4], 0.0);
        let p = "mtp.layers.0";
        for name in ["input_layernorm", "post_attention_layernorm"] {
            insert(&format!("{p}.{name}.weight"), vec![4], 0.0);
        }
        for (name, out, input) in [
            ("q_proj", 8, 4),
            ("k_proj", 2, 4),
            ("v_proj", 2, 4),
            ("o_proj", 4, 4),
        ] {
            insert(
                &format!("{p}.self_attn.{name}.weight"),
                vec![out, input],
                0.01,
            );
        }
        for name in ["q_norm", "k_norm"] {
            insert(&format!("{p}.self_attn.{name}.weight"), vec![2], 0.0);
        }
        for (name, out, input) in [("gate_proj", 6, 4), ("up_proj", 6, 4), ("down_proj", 4, 6)] {
            insert(&format!("{p}.mlp.{name}.weight"), vec![out, input], 0.01);
        }
        (WeightStore::from_tensors(config, tensors).unwrap(), spec)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::with_mtp;
    use super::*;
    use crate::backend::Backend;
    use crate::model::native::fixtures;
    use crate::model::spec::ModelFamily;

    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (left, right) in a.iter().zip(b) {
            assert!((left - right).abs() < 1e-5, "{left} != {right}");
        }
    }

    #[test]
    fn missing_mtp_is_optional() {
        let (weights, spec) = fixtures::tiny(ModelFamily::Qwen35);
        let target = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
        assert!(MtpModel::from_weights(&weights, &target).unwrap().is_none());
    }

    #[test]
    fn prefill_right_shift_and_chunk_tail_match() {
        let (weights, spec) = with_mtp();
        let target = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
        let mtp = MtpModel::from_weights(&weights, &target).unwrap().unwrap();
        let tokens = [0, 1, 2, 3];
        let positions: Vec<[usize; 3]> = (0..tokens.len()).map(|p| [p; 3]).collect();
        let mut target_state = target.create_state(32).unwrap();
        let target_output = target.forward(&tokens, &mut target_state, true).unwrap();

        let mut whole = mtp.create_state(32).unwrap();
        mtp.prefill(
            &target,
            &tokens,
            &target_output.hidden,
            &positions,
            &mut whole,
        )
        .unwrap();
        assert_eq!(whole.len(), 3);
        // MTP 行使用 hidden 的原始位置：x1/h0 在 0，x2/h1 在 1，x3/h2 在 2；
        // 尾部 h3 仍保留在目标位置 3，下一步才会消费该尾部。
        assert_eq!(whole.next_position(), 3);
        assert_eq!(whole.tail_hidden().unwrap(), &target_output.hidden[12..]);

        let mut chunked = mtp.create_state(32).unwrap();
        mtp.prefill(
            &target,
            &tokens[..2],
            &target_output.hidden[..8],
            &positions[..2],
            &mut chunked,
        )
        .unwrap();
        assert_eq!(chunked.len(), 1);
        assert_eq!(chunked.next_position(), 1);
        mtp.prefill(
            &target,
            &tokens[2..],
            &target_output.hidden[8..],
            &positions[2..],
            &mut chunked,
        )
        .unwrap();
        assert_eq!(chunked.len(), whole.len());
        assert_eq!(chunked.next_position(), whole.next_position());
        close(chunked.last_logits(), whole.last_logits());
        close(chunked.tail_hidden().unwrap(), whole.tail_hidden().unwrap());

        let whole_next = mtp.step(&target, 4, &mut whole).unwrap();
        let chunked_next = mtp.step(&target, 4, &mut chunked).unwrap();
        close(&whole_next, &chunked_next);
        assert_eq!(whole.next_position(), 4);
        assert_eq!(whole.len(), chunked.len());
    }

    #[test]
    fn mtp_owner_isolated() {
        let (weights, spec) = with_mtp();
        let target = NativeModel::from_weights(&weights, spec.clone(), Backend::Cpu).unwrap();
        let other_target = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
        let mtp = MtpModel::from_weights(&weights, &target).unwrap().unwrap();
        let other_mtp = MtpModel::from_weights(&weights, &other_target)
            .unwrap()
            .unwrap();
        let mut state = mtp.create_state(16).unwrap();
        assert!(other_mtp.step(&other_target, 0, &mut state).is_err());
        assert!(mtp
            .prefill(&other_target, &[1], &[0.0; 4], &[[0; 3]], &mut state)
            .is_err());
        assert!(state.is_empty());
        assert!(state.tail_hidden().is_none());
        assert!(mtp.validate_target(&other_target).is_err());
        assert!(mtp.validate_target(&target).is_ok());
        let mut wrong_state = other_target.create_state(16).unwrap();
        other_target.forward(&[1], &mut wrong_state, false).unwrap();
        assert!(mtp.seed_from_target(&wrong_state, &mut state).is_err());
    }
}
