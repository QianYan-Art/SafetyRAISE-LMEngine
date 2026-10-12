//! 目标模型完整词表验证的 greedy speculative decoding。
//!
//! 草稿模型只负责提出候选；每个候选都由目标模型以完整词表重新计算并验证。
//! 目标模型与 MTP 的快照一起事务提交，Qwen3.5 的 KV 与 GDN 状态不会通过
//! truncate 伪造回滚。

use crate::engine::state::SequenceState;
use crate::model::mtp::{MtpModel, MtpState};
use crate::model::native::NativeModel;
use crate::{Result, RsinferError};

const HARD_DRAFT_LIMIT: usize = 8;

fn invalid(message: impl Into<String>) -> RsinferError {
    RsinferError::DimensionError(message.into())
}

/// speculative 的预算与提议策略。
///
/// 当前实现只接受 greedy。`draft_vocab` 仅限制 MTP 提议，目标模型验证和
/// correction token 永远访问完整词表。
#[derive(Debug, Clone)]
pub struct SpeculativePolicy {
    pub max_draft_tokens: usize,
    pub acceptance_rate: f32,
    pub draft_vocab: Option<Vec<u32>>,
    pub sampling: bool,
}

impl Default for SpeculativePolicy {
    fn default() -> Self {
        Self {
            max_draft_tokens: HARD_DRAFT_LIMIT,
            acceptance_rate: 1.0,
            draft_vocab: None,
            sampling: false,
        }
    }
}

impl SpeculativePolicy {
    pub fn greedy() -> Self {
        Self::default()
    }

    pub fn with_draft_vocab(mut self, vocab: Vec<u32>) -> Self {
        self.draft_vocab = Some(vocab);
        self
    }

    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        if self.sampling {
            return Err(RsinferError::Unsupported(
                "随机采样、重复惩罚和 speculative 暂不同时启用".into(),
            ));
        }
        if !self.acceptance_rate.is_finite() || !(0.0..=1.0).contains(&self.acceptance_rate) {
            return Err(invalid("speculative 接受率必须在 0 到 1 之间"));
        }
        if let Some(vocab) = &self.draft_vocab {
            if vocab.is_empty() {
                return Err(invalid("草稿词表子集不能为空"));
            }
            if vocab.iter().any(|token| *token as usize >= vocab_size) {
                return Err(invalid("草稿词表子集包含越界 token"));
            }
        }
        Ok(())
    }

    fn draft_budget(&self, max_tokens: usize, active_count: usize) -> usize {
        let mut budget = self.max_draft_tokens.min(HARD_DRAFT_LIMIT).min(max_tokens);
        if budget == 0 {
            return 0;
        }
        // 活跃请求越多，越保守地缩短验证批，避免一个请求独占工作区。
        let active_cap = (HARD_DRAFT_LIMIT / active_count.max(1)).max(1);
        budget = budget.min(active_cap);
        // 低接受率时及时收缩草稿长度；保留至少一个 target token 的正常路径。
        if self.acceptance_rate < 0.4 {
            budget = 1;
        } else if self.acceptance_rate < 0.75 {
            budget = budget.min(budget.div_ceil(2));
        }
        budget
    }
}

/// 一轮 speculative 的已提交结果。
#[derive(Debug, Clone)]
pub struct Round {
    /// 实际提交给目标状态并可发布的 token；包含 correction token（若预算允许）。
    pub tokens: Vec<u32>,
    /// 提交后最后一行的完整目标词表 logits，可直接作为下一轮输入。
    pub logits: Vec<f32>,
    /// 被目标模型接受的草稿 token 数，不包含 correction token。
    pub accepted: usize,
    /// MTP 实际提出并送入目标验证的草稿 token 数。
    pub proposed: usize,
}

/// 可复用的 speculative 解码器配置包装。
#[derive(Debug, Clone, Default)]
pub struct SpeculativeDecoder {
    pub policy: SpeculativePolicy,
}

impl SpeculativeDecoder {
    pub fn new(policy: SpeculativePolicy) -> Result<Self> {
        // 词表大小在目标模型上才可知，构造时只检查策略的静态范围。
        if !policy.acceptance_rate.is_finite() || !(0.0..=1.0).contains(&policy.acceptance_rate) {
            return Err(invalid("speculative 接受率必须在 0 到 1 之间"));
        }
        Ok(Self { policy })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_round(
        &self,
        target: &NativeModel,
        mtp: &MtpModel,
        target_state: &mut SequenceState,
        mtp_state: &mut MtpState,
        next_logits: &[f32],
        max_tokens: usize,
        active_count: usize,
    ) -> Result<Round> {
        decode_round(
            target,
            mtp,
            target_state,
            mtp_state,
            next_logits,
            max_tokens,
            active_count,
            &self.policy,
        )
    }
}

/// 执行一轮 greedy speculative。
#[allow(clippy::too_many_arguments)]
pub fn decode_round(
    target: &NativeModel,
    mtp: &MtpModel,
    target_state: &mut SequenceState,
    mtp_state: &mut MtpState,
    next_logits: &[f32],
    max_tokens: usize,
    active_count: usize,
    policy: &SpeculativePolicy,
) -> Result<Round> {
    let vocab = target.spec.vocab_size;
    policy.validate(vocab)?;
    if next_logits.len() != vocab || next_logits.iter().any(|value| !value.is_finite()) {
        return Err(invalid("目标 next_logits 必须是完整词表且全部有限"));
    }
    if max_tokens == 0 {
        return Err(invalid("speculative 的输出预算不能为空"));
    }
    if target_state.is_empty() || target_state.last_hidden().len() != target.spec.hidden_size {
        return Err(invalid("目标状态缺少用于 speculative 的 prompt hidden"));
    }
    if target_state.max_context() != mtp_state.max_context() {
        return Err(invalid("目标与 MTP 的请求上下文窗口不一致"));
    }
    let room = target_state
        .max_context()
        .checked_sub(target_state.len())
        .ok_or_else(|| invalid("目标状态上下文长度非法"))?;
    if room == 0 {
        return Err(invalid("目标状态已达到上下文上限"));
    }

    // 所有工作都在快照上进行；目标与 MTP 只有最后的完整状态同时提交。
    let target_before = target_state.checkpoint();
    let mtp_before = mtp_state.checkpoint();
    let mut draft_state = mtp_before.clone();
    mtp.seed_from_target(target_state, &mut draft_state)?;
    let mtp_seeded = draft_state.checkpoint();

    let budget = policy
        .draft_budget(max_tokens, active_count)
        .min(room)
        .max(1);
    let first = argmax(next_logits);
    let mut proposed = vec![first];
    if budget > 1 && !target.spec.eos_token_ids.contains(&first) {
        let mut draft_logits = mtp.step(target, first, &mut draft_state)?;
        while proposed.len() < budget {
            let token = argmax_restricted(&draft_logits, policy.draft_vocab.as_deref())?;
            proposed.push(token);
            if target.spec.eos_token_ids.contains(&token) || proposed.len() == budget {
                break;
            }
            draft_logits = mtp.step(target, token, &mut draft_state)?;
        }
    }

    // 对所有候选一次性执行目标模型，all_logits=true 保证验证使用完整词表。
    let mut verify_state = target_before.checkpoint();
    let verified = target.forward(&proposed, &mut verify_state, true)?;
    if verified.rows != proposed.len()
        || verified.logits.len() != proposed.len().saturating_mul(vocab)
    {
        return Err(invalid("目标 speculative 验证 logits 行数错误"));
    }

    let mut accepted = 0usize;
    let mut expected = first;
    while accepted < proposed.len() && proposed[accepted] == expected {
        accepted += 1;
        if accepted < proposed.len() {
            expected = argmax(&verified.logits[(accepted - 1) * vocab..accepted * vocab]);
        }
    }

    // 目标拒绝点给出 correction；若全部接受，则最后一行验证 logits 给出 bonus。
    let correction = if accepted == 0 {
        first
    } else {
        argmax(&verified.logits[(accepted - 1) * vocab..accepted * vocab])
    };
    let hit_eos = accepted > 0 && target.spec.eos_token_ids.contains(&proposed[accepted - 1]);
    let can_add_correction = accepted < max_tokens && accepted < room && !hit_eos;
    let mut committed = proposed[..accepted].to_vec();
    if can_add_correction {
        committed.push(correction);
    }
    if committed.is_empty() {
        // max_tokens 至少为一且首 token 一定是目标 greedy 结果；这个分支只保护
        // 未知的预算组合，不会发布未经目标验证的 token。
        committed.push(first);
    }
    if committed.len() > max_tokens {
        committed.truncate(max_tokens);
    }
    if committed.len() > room {
        committed.truncate(room);
    }

    // 重新执行实际提交序列，不能用 verify_state 的 truncate 代替混合状态恢复。
    let mut committed_target = target_before;
    let committed_output = target.forward(&committed, &mut committed_target, false)?;
    if committed_output.rows != committed.len() || committed_output.logits.len() != vocab {
        return Err(invalid("目标提交 logits 行数错误"));
    }

    let mut committed_mtp = mtp_seeded;
    mtp.replay(
        target,
        &committed,
        &committed_output.hidden,
        &mut committed_mtp,
    )?;

    let logits = committed_output.logits;
    *target_state = committed_target;
    *mtp_state = committed_mtp;
    Ok(Round {
        tokens: committed,
        logits,
        accepted,
        proposed: proposed.len(),
    })
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate().skip(1) {
        if *value > logits[best] {
            best = index;
        }
    }
    best as u32
}

fn argmax_restricted(logits: &[f32], allowed: Option<&[u32]>) -> Result<u32> {
    let Some(allowed) = allowed else {
        return Ok(argmax(logits));
    };
    if allowed.is_empty() {
        return Err(invalid("草稿词表子集不能为空"));
    }
    let mut best = None;
    for &token in allowed {
        let index = token as usize;
        if index >= logits.len() {
            return Err(invalid("草稿词表子集包含越界 token"));
        }
        best = match best {
            None => Some(index),
            Some(current)
                if logits[index] > logits[current]
                    || (logits[index] == logits[current] && index < current) =>
            {
                Some(index)
            }
            Some(current) => Some(current),
        };
    }
    Ok(best.expect("allowed 已检查非空") as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::model::mtp::test_support::with_mtp;
    use crate::model::native::NativeModel;
    use crate::model::spec::ModelFamily;

    fn setup() -> (
        NativeModel,
        MtpModel,
        SequenceState,
        MtpState,
        Vec<f32>,
        usize,
    ) {
        let (weights, spec) = with_mtp();
        let target = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
        let mtp = MtpModel::from_weights(&weights, &target).unwrap().unwrap();
        let prompt = [0, 1, 2, 3];
        let mut target_state = target.create_state(32).unwrap();
        let output = target.forward(&prompt, &mut target_state, true).unwrap();
        let vocab = target.spec.vocab_size;
        let next_logits = output.logits[(prompt.len() - 1) * vocab..].to_vec();
        let positions: Vec<[usize; 3]> = (0..prompt.len()).map(|p| [p; 3]).collect();
        let mut mtp_state = mtp.create_state(32).unwrap();
        mtp.prefill(&target, &prompt, &output.hidden, &positions, &mut mtp_state)
            .unwrap();
        (target, mtp, target_state, mtp_state, next_logits, vocab)
    }

    fn greedy(logits: &[f32]) -> u32 {
        argmax(logits)
    }

    #[test]
    fn tied_logits_match_plain_greedy_and_ignore_subset_order() {
        let logits = [0.0, 2.0, 2.0, -1.0];
        let mut sampler = crate::frontend::Sampler::new(crate::frontend::SamplingParams {
            temperature: 0.0,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(argmax(&logits), sampler.sample(&logits, &[]).unwrap());
        assert_eq!(argmax_restricted(&logits, Some(&[2, 1])).unwrap(), 1);
    }

    #[test]
    fn speculative_greedy_matches_target_token_for_token() {
        let (target, mtp, target_state, mtp_state, next_logits, vocab) = setup();
        let mut reference_state = target_state.checkpoint();
        let mut reference_logits = next_logits.clone();
        let mut expected = Vec::new();
        for _ in 0..6 {
            let token = greedy(&reference_logits);
            expected.push(token);
            let output = target
                .forward(&[token], &mut reference_state, true)
                .unwrap();
            reference_logits = output.logits[(output.rows - 1) * vocab..].to_vec();
        }

        let mut actual_state = target_state;
        let mut actual_mtp = mtp_state;
        let mut logits = next_logits;
        let mut actual = Vec::new();
        while actual.len() < expected.len() {
            let remaining = expected.len() - actual.len();
            let round = decode_round(
                &target,
                &mtp,
                &mut actual_state,
                &mut actual_mtp,
                &logits,
                remaining,
                1,
                &SpeculativePolicy::default(),
            )
            .unwrap();
            assert!(round.proposed <= HARD_DRAFT_LIMIT);
            assert!(!round.tokens.is_empty());
            logits = round.logits;
            actual.extend(round.tokens);
        }
        actual.truncate(expected.len());
        assert_eq!(actual, expected);
        assert_eq!(actual_state.len(), reference_state.len());
        assert_eq!(
            actual_state.next_position(),
            reference_state.next_position()
        );
        assert_eq!(actual_state.last_hidden(), reference_state.last_hidden());
    }

    #[test]
    fn rejection_commits_correction_and_restores_hybrid_state() {
        let (target, mtp, target_state, mtp_state, next_logits, vocab) = setup();
        let baseline_state = target_state.checkpoint();
        let first = argmax(&next_logits);
        let mut reference_state = baseline_state.checkpoint();
        let first_output = target
            .forward(&[first], &mut reference_state, true)
            .unwrap();
        let target_next = argmax(&first_output.logits[..vocab]);
        let wrong = (target_next as usize + 1) as u32 % vocab as u32;
        let policy = SpeculativePolicy {
            max_draft_tokens: 2,
            acceptance_rate: 1.0,
            draft_vocab: Some(vec![wrong]),
            sampling: false,
        };
        let mut actual_state = target_state;
        let mut actual_mtp = mtp_state;
        let round = decode_round(
            &target,
            &mtp,
            &mut actual_state,
            &mut actual_mtp,
            &next_logits,
            3,
            1,
            &policy,
        )
        .unwrap();
        assert_eq!(round.proposed, 2);
        assert_eq!(round.accepted, 1);
        assert_eq!(round.tokens, vec![first, target_next]);

        let mut expected_state = baseline_state;
        let expected_output = target
            .forward(&round.tokens, &mut expected_state, true)
            .unwrap();
        assert_eq!(actual_state.len(), expected_state.len());
        assert_eq!(actual_state.last_hidden(), expected_state.last_hidden());
        assert_eq!(round.logits, expected_output.logits[vocab..].to_vec());
    }

    #[test]
    fn random_policy_is_rejected_and_first_eos_commits_once() {
        let (target, mtp, target_state, mtp_state, next_logits, _) = setup();
        let mut rejected_state = target_state.checkpoint();
        let mut rejected_mtp = mtp_state.checkpoint();
        let random_policy = SpeculativePolicy {
            sampling: true,
            ..SpeculativePolicy::default()
        };
        assert!(decode_round(
            &target,
            &mtp,
            &mut rejected_state,
            &mut rejected_mtp,
            &next_logits,
            4,
            1,
            &random_policy,
        )
        .is_err());

        let first = argmax(&next_logits);
        let mut eos_target = target;
        eos_target.spec.eos_token_ids = vec![first];
        let baseline_len = target_state.len();
        let mut eos_state = target_state;
        let mut eos_mtp = mtp_state;
        let round = decode_round(
            &eos_target,
            &mtp,
            &mut eos_state,
            &mut eos_mtp,
            &next_logits,
            4,
            1,
            &SpeculativePolicy::default(),
        )
        .unwrap();
        assert_eq!(round.tokens, vec![first]);
        assert_eq!(round.proposed, 1);
        assert_eq!(round.accepted, 1);
        assert_eq!(eos_state.len(), baseline_len + 1);
    }

    #[test]
    fn absent_mtp_fixture_is_not_required_for_non_mtp_model() {
        let (weights, spec) = crate::model::native::fixtures::tiny(ModelFamily::Qwen3);
        let target = NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap();
        assert!(crate::model::mtp::MtpModel::from_weights(&weights, &target)
            .unwrap()
            .is_none());
    }
}
