//! 单一模型所有者的有界调度。按 chunk 预填充，按 token 轮转，不并行写同一请求状态。

use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc as events;

use crate::engine::speculative::{decode_round, SpeculativePolicy};
use crate::engine::state::SequenceState;
use crate::frontend::{Sampler, SamplingParams, ThinkingAction, ThinkingController, ThinkingSpec};
use crate::model::mtp::{MtpModel, MtpState};
use crate::model::native::NativeModel;
use crate::{Result, RsinferError};

fn invalid(message: impl Into<String>) -> RsinferError {
    RsinferError::ConfigError(message.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    pub max_context: usize,
    pub max_active: usize,
    pub queue_capacity: usize,
    pub prefill_chunk: usize,
    pub event_capacity: usize,
    /// 包括活动请求、事务副本、prefix 和保守激活估算；不包含模型权重及插件副本。
    pub working_memory_bytes: usize,
    pub prefix_cache_bytes: usize,
    pub max_input_bytes: usize,
    pub max_request_ms: u64,
    pub max_stall_ms: u64,
    /// 0 关闭；只对纯文本、无思考预算且无重复惩罚的 greedy 请求启用。
    pub max_draft_tokens: usize,
    pub draft_vocab: Option<Vec<u32>>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_context: 4096,
            max_active: 1,
            queue_capacity: 8,
            prefill_chunk: 32,
            event_capacity: 32,
            working_memory_bytes: 2 * 1024 * 1024 * 1024,
            prefix_cache_bytes: 0,
            max_input_bytes: 32 * 1024 * 1024,
            max_request_ms: 300_000,
            max_stall_ms: 10_000,
            max_draft_tokens: 0,
            draft_vocab: None,
        }
    }
}

impl EngineConfig {
    fn reservation(&self, model: &NativeModel) -> Result<usize> {
        let s = &model.spec;
        let chunk = self
            .prefill_chunk
            .max(self.max_draft_tokens + 1)
            .min(self.max_context);
        // 包括两份状态、输入/残差及线性投影临时量；显式留出词表 logits。
        let gdn_width = match &s.gdn {
            Some(g) => g
                .key_heads
                .checked_mul(g.key_dim)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(g.value_heads.checked_mul(g.value_dim)?))
                .ok_or_else(|| invalid("GDN 工作区预算溢出"))?,
            None => 0,
        };
        let widths = s
            .hidden_size
            .checked_mul(8)
            .and_then(|n| n.checked_add(s.intermediate_size.checked_mul(3)?))
            .and_then(|n| {
                n.checked_add(
                    s.attention
                        .heads
                        .checked_mul(s.attention.head_dim)?
                        .checked_mul(6)?,
                )
            })
            .and_then(|n| n.checked_add(gdn_width.checked_mul(4)?))
            .ok_or_else(|| invalid("工作区预算溢出"))?;
        s.state_bytes(self.max_context)?
            .checked_mul(if self.max_draft_tokens > 0 { 5 } else { 2 })
            .and_then(|n| n.checked_add(widths.checked_mul(chunk)?.checked_mul(4)?))
            .and_then(|n| {
                n.checked_add(
                    s.vocab_size
                        .checked_mul(8)?
                        .checked_mul(chunk.checked_add(2)?)?,
                )
            })
            .and_then(|n| {
                if self.max_draft_tokens == 0 {
                    Some(n)
                } else {
                    n.checked_add(
                        self.max_context
                            .checked_mul(s.attention.kv_heads)?
                            .checked_mul(s.attention.head_dim)?
                            .checked_mul(2 * 4 * 4)?,
                    )
                }
            })
            .and_then(|n| n.checked_add(self.max_input_bytes))
            .ok_or_else(|| invalid("请求预算溢出"))
    }

    pub fn validate(&self, model: &NativeModel) -> Result<()> {
        if self.max_draft_tokens > 8 {
            return Err(invalid("MTP 草稿长度不能超过 8"));
        }
        if [
            self.max_context,
            self.max_active,
            self.queue_capacity,
            self.prefill_chunk,
            self.event_capacity,
            self.max_input_bytes,
        ]
        .contains(&0)
            || self.max_context > model.spec.max_position_embeddings
            || self.max_request_ms == 0
            || self.max_stall_ms == 0
            || self.max_active > 1024
            || self.queue_capacity > 65536
            || self.event_capacity > 65536
        {
            return Err(invalid("调度容量必须非零且在模型及调度器上限内"));
        }
        let reserved = self
            .reservation(model)?
            .checked_mul(self.max_active)
            .and_then(|n| n.checked_add(self.prefix_cache_bytes))
            .and_then(|n| n.checked_add(self.queue_capacity.checked_mul(self.max_input_bytes)?))
            .ok_or_else(|| invalid("调度器总预算溢出"))?;
        if reserved > self.working_memory_bytes {
            return Err(invalid(format!(
                "配置至少需要 {reserved} 字节工作内存，预算为 {}",
                self.working_memory_bytes
            )));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Prompt {
    pub tokens: Vec<u32>,
    /// 视觉输入的完整文本/图像嵌入及多轴位置；文本请求留空。
    pub embeddings: Option<(Vec<f32>, Vec<[usize; 3]>)>,
}

impl Prompt {
    pub fn text(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            embeddings: None,
        }
    }
    fn bytes(&self) -> Option<usize> {
        let n = self.tokens.len().checked_mul(4)?;
        match &self.embeddings {
            None => Some(n),
            Some((values, positions)) => n.checked_add(values.len().checked_mul(4)?)?.checked_add(
                positions
                    .len()
                    .checked_mul(std::mem::size_of::<[usize; 3]>())?,
            ),
        }
    }
}

pub struct GenerationRequest {
    pub prompt: Prompt,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub deadline: Option<Duration>,
    /// 仅在外层确认同一模型模板时允许前缀复用；媒体请求自动禁用。
    pub cache_prefix: bool,
    pub thinking: Option<ThinkingSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
    Deadline,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub queue_ms: u64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
pub enum GenerationEvent {
    Token { id: u32, index: usize },
    Finished { reason: FinishReason, usage: Usage },
    Error(String),
}

#[derive(Default)]
struct Counters {
    submitted: AtomicU64,
    rejected: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    generated_tokens: AtomicU64,
    prefix_hits: AtomicU64,
    mtp_rounds: AtomicU64,
    draft_proposed: AtomicU64,
    draft_accepted: AtomicU64,
    active: AtomicU64,
    queued: AtomicU64,
}

#[derive(Debug, Serialize)]
pub struct Metrics {
    pub submitted: u64,
    pub rejected: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub generated_tokens: u64,
    pub prefix_hits: u64,
    pub mtp_rounds: u64,
    /// 不含目标模型直接给出的首个 token。
    pub draft_proposed: u64,
    pub draft_accepted: u64,
    pub active: u64,
    pub queued: u64,
}

impl Counters {
    fn snapshot(&self) -> Metrics {
        let read = |c: &AtomicU64| c.load(Ordering::Relaxed);
        Metrics {
            submitted: read(&self.submitted),
            rejected: read(&self.rejected),
            completed: read(&self.completed),
            failed: read(&self.failed),
            cancelled: read(&self.cancelled),
            generated_tokens: read(&self.generated_tokens),
            prefix_hits: read(&self.prefix_hits),
            mtp_rounds: read(&self.mtp_rounds),
            draft_proposed: read(&self.draft_proposed),
            draft_accepted: read(&self.draft_accepted),
            active: read(&self.active),
            queued: read(&self.queued),
        }
    }
}

pub struct GenerationHandle {
    pub receiver: events::Receiver<GenerationEvent>,
    cancelled: Arc<AtomicBool>,
}
impl GenerationHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
impl Drop for GenerationHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

struct Job {
    request: GenerationRequest,
    sender: events::Sender<GenerationEvent>,
    cancelled: Arc<AtomicBool>,
    submitted: Instant,
    deadline: Instant,
}

struct Inner {
    sender: mpsc::SyncSender<Job>,
    stop: Arc<AtomicBool>,
    join: Mutex<Option<JoinHandle<()>>>,
    model: Arc<NativeModel>,
    config: EngineConfig,
    counters: Arc<Counters>,
    mtp: Option<Arc<MtpModel>>,
}
impl Drop for Inner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Ok(join) = self.join.get_mut() {
            if let Some(thread) = join.take() {
                let _ = thread.join();
            }
        }
    }
}

#[derive(Clone)]
pub struct Engine(Arc<Inner>);
impl Engine {
    pub fn new(model: Arc<NativeModel>, config: EngineConfig) -> Result<Self> {
        Self::with_mtp(model, config, None)
    }
    pub fn with_mtp(
        model: Arc<NativeModel>,
        config: EngineConfig,
        mtp: Option<Arc<MtpModel>>,
    ) -> Result<Self> {
        config.validate(&model)?;
        if let Some(mtp) = &mtp {
            mtp.validate_target(&model)?;
        }
        if config.max_draft_tokens > 0 {
            if mtp.is_none() {
                return Err(invalid("启用 MTP 但模型没有完整 MTP 权重"));
            }
            SpeculativePolicy {
                max_draft_tokens: config.max_draft_tokens,
                acceptance_rate: 1.0,
                draft_vocab: config.draft_vocab.clone(),
                sampling: false,
            }
            .validate(model.spec.vocab_size)?;
        }
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let counters = Arc::new(Counters::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (worker_model, worker_config, worker_counters, worker_stop) = (
            model.clone(),
            config.clone(),
            counters.clone(),
            stop.clone(),
        );
        let worker_mtp = mtp.clone();
        let join = thread::Builder::new()
            .name("safetyraise-engine".into())
            .spawn(move || {
                worker(
                    worker_model,
                    worker_config,
                    receiver,
                    worker_counters,
                    worker_stop,
                    worker_mtp,
                )
            })?;
        Ok(Self(Arc::new(Inner {
            sender,
            stop,
            join: Mutex::new(Some(join)),
            model,
            config,
            counters,
            mtp,
        })))
    }
    pub fn model(&self) -> &Arc<NativeModel> {
        &self.0.model
    }
    pub fn config(&self) -> &EngineConfig {
        &self.0.config
    }
    pub fn metrics(&self) -> Metrics {
        self.0.counters.snapshot()
    }
    pub fn has_mtp(&self) -> bool {
        self.0.mtp.is_some()
    }

    pub fn submit(&self, request: GenerationRequest) -> Result<GenerationHandle> {
        let config = &self.0.config;
        let count = request.prompt.tokens.len();
        if count == 0
            || request.max_new_tokens == 0
            || count
                .checked_add(request.max_new_tokens)
                .is_none_or(|n| n > config.max_context)
            || request
                .prompt
                .tokens
                .iter()
                .any(|id| *id as usize >= self.0.model.spec.vocab_size)
            || request
                .prompt
                .bytes()
                .is_none_or(|n| n > config.max_input_bytes)
        {
            return Err(invalid("请求 token、上下文或输入字节预算非法"));
        }
        if let Some((values, positions)) = &request.prompt.embeddings {
            if count.checked_mul(self.0.model.spec.hidden_size) != Some(values.len())
                || positions.len() != count
                || values.iter().any(|v| !v.is_finite())
                || positions
                    .iter()
                    .flatten()
                    .any(|p| *p >= self.0.model.spec.max_position_embeddings)
            {
                return Err(invalid("媒体嵌入与 token/位置合同不匹配"));
            }
        }
        request.sampling.validate()?;
        if let Some(thinking) = &request.thinking {
            if thinking
                .start_tokens
                .iter()
                .chain(&thinking.end_tokens)
                .any(|id| *id as usize >= self.0.model.spec.vocab_size)
                || (thinking.budget.is_some()
                    && (!thinking.supported
                        || !thinking.enabled
                        || thinking.start_tokens.is_empty()
                        || thinking.end_tokens.is_empty()
                        || thinking
                            .end_tokens
                            .iter()
                            .any(|id| self.0.model.spec.eos_token_ids.contains(id))))
                || thinking.budget.is_some_and(|budget| {
                    budget
                        .checked_add(thinking.end_tokens.len())
                        .is_none_or(|n| n > request.max_new_tokens)
                })
            {
                return Err(invalid("思考预算和结束标记必须落在输出 token 预算内"));
            }
        }
        let submitted = Instant::now();
        let timeout = request
            .deadline
            .unwrap_or(Duration::from_millis(config.max_request_ms));
        if timeout.is_zero() || timeout > Duration::from_millis(config.max_request_ms) {
            return Err(invalid("请求截止时间超出调度器范围"));
        }
        let deadline = submitted
            .checked_add(timeout)
            .ok_or_else(|| invalid("截止时间溢出"))?;
        let (sender, receiver) = events::channel(config.event_capacity);
        let cancelled = Arc::new(AtomicBool::new(false));
        let job = Job {
            request,
            sender,
            cancelled: cancelled.clone(),
            submitted,
            deadline,
        };
        self.0.counters.queued.fetch_add(1, Ordering::Relaxed);
        match self.0.sender.try_send(job) {
            Ok(()) => {
                self.0.counters.submitted.fetch_add(1, Ordering::Relaxed);
                Ok(GenerationHandle {
                    receiver,
                    cancelled,
                })
            }
            Err(error) => {
                self.0.counters.queued.fetch_sub(1, Ordering::Relaxed);
                self.0.counters.rejected.fetch_add(1, Ordering::Relaxed);
                Err(match error {
                    mpsc::TrySendError::Full(_) => {
                        RsinferError::ResourceExhausted("推理队列已满".into())
                    }
                    mpsc::TrySendError::Disconnected(_) => {
                        RsinferError::Unavailable("推理工作线程已关闭".into())
                    }
                })
            }
        }
    }
}

struct Prefix {
    tokens: Vec<u32>,
    state: SequenceState,
    logits: Vec<f32>,
    bytes: usize,
}
struct PrefixCache {
    entries: VecDeque<Prefix>,
    used: usize,
    capacity: usize,
}
impl PrefixCache {
    fn lookup(&mut self, tokens: &[u32]) -> Option<Prefix> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, p)| tokens.starts_with(&p.tokens))
            .max_by_key(|(_, p)| p.tokens.len())
            .map(|(i, _)| i)?;
        let entry = self.entries.remove(index)?;
        let copy = Prefix {
            tokens: entry.tokens.clone(),
            state: entry.state.checkpoint(),
            logits: entry.logits.clone(),
            bytes: entry.bytes,
        };
        self.entries.push_back(entry);
        Some(copy)
    }
    fn insert(&mut self, tokens: &[u32], state: &SequenceState, logits: &[f32]) {
        let Some(bytes) = state
            .resident_bytes()
            .checked_add(tokens.len().saturating_add(logits.len()).saturating_mul(4))
        else {
            return;
        };
        if bytes > self.capacity {
            return;
        }
        if let Some(i) = self.entries.iter().position(|p| p.tokens == tokens) {
            if let Some(old) = self.entries.remove(i) {
                self.used -= old.bytes;
            }
        }
        while self.used > self.capacity - bytes {
            if let Some(old) = self.entries.pop_front() {
                self.used -= old.bytes;
            } else {
                break;
            }
        }
        self.entries.push_back(Prefix {
            tokens: tokens.to_vec(),
            state: state.checkpoint(),
            logits: logits.to_vec(),
            bytes,
        });
        self.used += bytes;
    }
}

struct Active {
    job: Job,
    state: SequenceState,
    logits: Vec<f32>,
    history: Vec<u32>,
    sampler: Sampler,
    thinking: Option<ThinkingController>,
    forced: VecDeque<u32>,
    mtp_state: Option<MtpState>,
    policy: SpeculativePolicy,
    prefetched: usize,
    generated: usize,
    cached: usize,
    queue_ms: u64,
    pending: VecDeque<GenerationEvent>,
    terminal: bool,
    interrupted: bool,
    stalled_since: Option<Instant>,
}
impl Active {
    fn new(
        job: Job,
        model: &NativeModel,
        config: &EngineConfig,
        prefix: &mut PrefixCache,
        mtp: Option<&MtpModel>,
    ) -> Result<Self> {
        let use_mtp = config.max_draft_tokens > 0
            && mtp.is_some()
            && job.request.sampling.temperature == 0.0
            && job.request.sampling.repetition_penalty == 1.0
            && job.request.prompt.embeddings.is_none()
            && job
                .request
                .thinking
                .as_ref()
                .is_none_or(|t| t.budget.is_none());
        let saved =
            if !use_mtp && job.request.cache_prefix && job.request.prompt.embeddings.is_none() {
                prefix.lookup(&job.request.prompt.tokens)
            } else {
                None
            };
        let (state, logits, cached) = match saved {
            Some(p) => (p.state, p.logits, p.tokens.len()),
            None => (model.create_state(config.max_context)?, vec![], 0),
        };
        let sampler = Sampler::new(job.request.sampling.clone())?;
        let queue_ms = job.submitted.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let history = job.request.prompt.tokens.clone();
        let thinking = job.request.thinking.clone().map(ThinkingController::new);
        let mtp_state = if use_mtp {
            Some(
                mtp.ok_or_else(|| invalid("缺少 MTP"))?
                    .create_state(config.max_context)?,
            )
        } else {
            None
        };
        Ok(Self {
            job,
            state,
            logits,
            history,
            sampler,
            thinking,
            forced: VecDeque::new(),
            mtp_state,
            policy: SpeculativePolicy {
                max_draft_tokens: config.max_draft_tokens.max(1),
                acceptance_rate: 1.0,
                draft_vocab: config.draft_vocab.clone(),
                sampling: false,
            },
            prefetched: cached,
            generated: 0,
            cached,
            queue_ms,
            pending: VecDeque::new(),
            terminal: false,
            interrupted: false,
            stalled_since: None,
        })
    }
    fn finish(&mut self, reason: FinishReason) {
        if self.terminal {
            return;
        }
        self.pending.push_back(GenerationEvent::Finished {
            reason,
            usage: Usage {
                prompt_tokens: self.job.request.prompt.tokens.len(),
                completion_tokens: self.generated,
                cached_tokens: self.cached,
                queue_ms: self.queue_ms,
                elapsed_ms: self
                    .job
                    .submitted
                    .elapsed()
                    .as_millis()
                    .min(u64::MAX as u128) as u64,
            },
        });
        self.terminal = true;
    }
    /// 取消与截止时间也覆盖尚未发出的正常终态，避免背压队列继续发布 token。
    fn interrupt_if_needed(&mut self, counters: &Counters) {
        if self.interrupted {
            return;
        }
        let reason = if self.job.cancelled.load(Ordering::Acquire) {
            counters.cancelled.fetch_add(1, Ordering::Relaxed);
            Some(FinishReason::Cancelled)
        } else if Instant::now() >= self.job.deadline {
            Some(FinishReason::Deadline)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.interrupted = true;
            self.pending.clear();
            self.terminal = false;
            self.finish(reason);
        }
    }
    fn step(
        &mut self,
        model: &NativeModel,
        config: &EngineConfig,
        prefix: &mut PrefixCache,
        counters: &Counters,
        mtp: Option<&MtpModel>,
        active_count: usize,
    ) -> Result<()> {
        let total = self.job.request.prompt.tokens.len();
        if self.prefetched < total {
            let end = (self.prefetched + config.prefill_chunk).min(total);
            let result = match &self.job.request.prompt.embeddings {
                Some((values, positions)) => model.forward_embeddings(
                    &values[self.prefetched * model.spec.hidden_size..end * model.spec.hidden_size],
                    &positions[self.prefetched..end],
                    &mut self.state,
                    false,
                ),
                None => model.forward(
                    &self.job.request.prompt.tokens[self.prefetched..end],
                    &mut self.state,
                    false,
                ),
            }?;
            if let (Some(mtp), Some(state)) = (mtp, self.mtp_state.as_mut()) {
                let positions = (self.prefetched..end).map(|p| [p; 3]).collect::<Vec<_>>();
                mtp.prefill(
                    model,
                    &self.job.request.prompt.tokens[self.prefetched..end],
                    &result.hidden,
                    &positions,
                    state,
                )?;
            }
            self.prefetched = end;
            self.logits = result.logits;
            if end == total
                && self.mtp_state.is_none()
                && self.job.request.cache_prefix
                && self.job.request.prompt.embeddings.is_none()
            {
                prefix.insert(&self.job.request.prompt.tokens, &self.state, &self.logits);
            }
            return Ok(());
        }
        if let (Some(mtp), Some(state)) = (mtp, self.mtp_state.as_mut()) {
            let round = decode_round(
                model,
                mtp,
                &mut self.state,
                state,
                &self.logits,
                self.job.request.max_new_tokens - self.generated,
                active_count,
                &self.policy,
            )?;
            counters.mtp_rounds.fetch_add(1, Ordering::Relaxed);
            counters
                .draft_proposed
                .fetch_add(round.proposed.saturating_sub(1) as u64, Ordering::Relaxed);
            counters
                .draft_accepted
                .fetch_add(round.accepted.saturating_sub(1) as u64, Ordering::Relaxed);
            self.logits = round.logits;
            if round.proposed > 1 {
                self.policy.acceptance_rate = 0.8 * self.policy.acceptance_rate
                    + 0.2 * (round.accepted.saturating_sub(1) as f32 / (round.proposed - 1) as f32);
            }
            for token in round.tokens {
                if model.spec.eos_token_ids.contains(&token) {
                    self.finish(FinishReason::Stop);
                    break;
                }
                self.publish_token(token, counters);
            }
            if !self.terminal && self.generated == self.job.request.max_new_tokens {
                self.finish(FinishReason::Length);
            }
            return Ok(());
        }
        let remaining = self.job.request.max_new_tokens - self.generated;
        if self.forced.is_empty() {
            if let Some(thinking) = &mut self.thinking {
                if thinking.spec().budget.is_some()
                    && thinking.in_thinking()
                    && remaining <= thinking.spec().end_tokens.len()
                {
                    self.forced.extend(thinking.spec().end_tokens.clone());
                } else if let Some(tokens) = thinking.take_force_close() {
                    self.forced.extend(tokens);
                }
            }
        }
        let was_forced = !self.forced.is_empty();
        let mut token = match self.forced.pop_front() {
            Some(token) => token,
            None => self.sampler.sample(&self.logits, &self.history)?,
        };
        if !was_forced {
            if let Some(thinking) = &self.thinking {
                if thinking.spec().budget.is_some() {
                    if !thinking.in_thinking()
                        && thinking.spec().start_tokens.first() == Some(&token)
                        && remaining
                            < thinking
                                .spec()
                                .start_tokens
                                .len()
                                .saturating_add(thinking.spec().end_tokens.len())
                    {
                        self.finish(FinishReason::Length);
                        return Ok(());
                    }
                    if thinking.in_thinking() && model.spec.eos_token_ids.contains(&token) {
                        self.forced.extend(thinking.spec().end_tokens.clone());
                        token = self
                            .forced
                            .pop_front()
                            .ok_or_else(|| invalid("思考结束标记为空"))?;
                    }
                }
            }
        }
        if model.spec.eos_token_ids.contains(&token) {
            self.finish(FinishReason::Stop);
            return Ok(());
        }
        self.logits = model.forward(&[token], &mut self.state, false)?.logits;
        if let Some(thinking) = &mut self.thinking {
            if let ThinkingAction::ForceClose { end_tokens } = thinking.observe(token) {
                if !was_forced && self.forced.is_empty() {
                    self.forced.extend(end_tokens);
                }
            }
        }
        self.publish_token(token, counters);
        if self.generated == self.job.request.max_new_tokens {
            self.finish(FinishReason::Length);
        }
        Ok(())
    }
    fn publish_token(&mut self, token: u32, counters: &Counters) {
        self.history.push(token);
        self.generated += 1;
        counters.generated_tokens.fetch_add(1, Ordering::Relaxed);
        self.pending.push_back(GenerationEvent::Token {
            id: token,
            index: self.generated - 1,
        });
    }
}

fn worker(
    model: Arc<NativeModel>,
    config: EngineConfig,
    queue: mpsc::Receiver<Job>,
    counters: Arc<Counters>,
    stop: Arc<AtomicBool>,
    mtp: Option<Arc<MtpModel>>,
) {
    let mut active: VecDeque<Active> = VecDeque::new();
    let mut prefix = PrefixCache {
        entries: VecDeque::new(),
        used: 0,
        capacity: config.prefix_cache_bytes,
    };
    while !stop.load(Ordering::Acquire) {
        while active.len() < config.max_active {
            let job = match queue.try_recv() {
                Ok(job) => job,
                Err(_) => break,
            };
            counters.queued.fetch_sub(1, Ordering::Relaxed);
            let sender = job.sender.clone();
            match Active::new(job, &model, &config, &mut prefix, mtp.as_deref()) {
                Ok(mut request) => {
                    if request.cached > 0 {
                        counters.prefix_hits.fetch_add(1, Ordering::Relaxed);
                    }
                    if Instant::now() >= request.job.deadline {
                        request.finish(FinishReason::Deadline);
                    }
                    counters.active.fetch_add(1, Ordering::Relaxed);
                    active.push_back(request);
                }
                Err(error) => {
                    let _ = sender.try_send(GenerationEvent::Error(error.to_string()));
                    counters.failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let mut progressed = false;
        let turns = active.len();
        for _ in 0..turns {
            let Some(mut request) = active.pop_front() else {
                break;
            };
            if request.job.sender.is_closed() {
                counters.cancelled.fetch_add(1, Ordering::Relaxed);
                counters.active.fetch_sub(1, Ordering::Relaxed);
                continue;
            }
            request.interrupt_if_needed(&counters);
            if request.pending.is_empty() && !request.terminal {
                if let Err(error) = request.step(
                    &model,
                    &config,
                    &mut prefix,
                    &counters,
                    mtp.as_deref(),
                    turns,
                ) {
                    request
                        .pending
                        .push_back(GenerationEvent::Error(error.to_string()));
                    request.finish(FinishReason::Error);
                    counters.failed.fetch_add(1, Ordering::Relaxed);
                }
                progressed = true;
            }
            let mut dropped = false;
            loop {
                request.interrupt_if_needed(&counters);
                let Some(event) = request.pending.pop_front() else {
                    break;
                };
                match request.job.sender.try_send(event) {
                    Ok(()) => {
                        request.stalled_since = None;
                        progressed = true;
                    }
                    Err(events::error::TrySendError::Full(event)) => {
                        request.pending.push_front(event);
                        let stalled = *request.stalled_since.get_or_insert_with(Instant::now);
                        if stalled.elapsed() >= Duration::from_millis(config.max_stall_ms) {
                            dropped = true;
                            counters.cancelled.fetch_add(1, Ordering::Relaxed);
                        }
                        break;
                    }
                    Err(events::error::TrySendError::Closed(_)) => {
                        dropped = true;
                        break;
                    }
                }
            }
            if dropped || (request.terminal && request.pending.is_empty()) {
                counters.active.fetch_sub(1, Ordering::Relaxed);
                if !dropped {
                    counters.completed.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                active.push_back(request);
            }
        }
        if !progressed {
            thread::sleep(Duration::from_millis(1));
        }
    }
    // 停止时释放所有请求和前缀；消费者收到通道关闭，不能将其误判为成功结束。
    counters.active.store(0, Ordering::Relaxed);
    counters.queued.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::Backend,
        model::{native::fixtures, spec::ModelFamily},
    };
    fn engine(active: usize) -> Engine {
        let (weights, spec) = fixtures::tiny(ModelFamily::Qwen35);
        let model = Arc::new(NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap());
        Engine::new(
            model,
            EngineConfig {
                max_context: 64,
                max_active: active,
                queue_capacity: 2,
                prefill_chunk: 2,
                event_capacity: 1,
                max_input_bytes: 1024,
                working_memory_bytes: 16 * 1024 * 1024,
                prefix_cache_bytes: 1024 * 1024,
                max_stall_ms: 1000,
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn request() -> GenerationRequest {
        GenerationRequest {
            prompt: Prompt::text(vec![1, 2, 3, 4, 5, 6]),
            max_new_tokens: 4,
            sampling: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            deadline: None,
            cache_prefix: true,
            thinking: None,
        }
    }
    async fn collect(mut handle: GenerationHandle) -> (Vec<u32>, Usage, FinishReason) {
        let mut tokens = vec![];
        loop {
            match tokio::time::timeout(Duration::from_secs(3), handle.receiver.recv())
                .await
                .unwrap()
                .unwrap()
            {
                GenerationEvent::Token { id, .. } => tokens.push(id),
                GenerationEvent::Finished { usage, reason } => return (tokens, usage, reason),
                GenerationEvent::Error(e) => panic!("{e}"),
            }
        }
    }
    #[tokio::test]
    async fn cached_hybrid_prefix_preserves_output_and_slow_consumer_does_not_block_peer() {
        let engine = engine(2);
        let first = collect(engine.submit(request()).unwrap()).await;
        let mut slow_request = request();
        slow_request.max_new_tokens = 20;
        let slow = engine.submit(slow_request).unwrap();
        let second = collect(engine.submit(request()).unwrap()).await;
        assert_eq!(first.0, second.0);
        assert_eq!(second.1.cached_tokens, 6);
        assert_eq!(second.2, FinishReason::Length);
        slow.cancel();
        drop(slow);
        tokio::time::timeout(Duration::from_secs(2), async {
            while engine.metrics().active != 0 {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .unwrap();
        assert!(engine.metrics().prefix_hits >= 1);
    }
    #[tokio::test]
    async fn queued_deadline_and_budget_fail_before_generation() {
        let engine = engine(1);
        let mut long = request();
        long.max_new_tokens = 40;
        let slow = engine.submit(long).unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        let mut expiring = request();
        expiring.deadline = Some(Duration::from_millis(5));
        let expired = engine.submit(expiring).unwrap();
        tokio::time::sleep(Duration::from_millis(15)).await;
        drop(slow);
        let (_, usage, reason) = collect(expired).await;
        assert_eq!(reason, FinishReason::Deadline);
        assert_eq!(usage.completion_tokens, 0);
        let mut bad = request();
        bad.max_new_tokens = 64;
        assert!(engine.submit(bad).is_err());
        let mut config = engine.config().clone();
        config.working_memory_bytes = 1;
        assert!(config.validate(engine.model()).is_err());
    }

    #[tokio::test]
    async fn mtp_scheduler_matches_normal_greedy_with_chunked_prefill() {
        let (weights, spec) = crate::model::mtp::test_support::with_mtp();
        let model = Arc::new(NativeModel::from_weights(&weights, spec, Backend::Cpu).unwrap());
        let mtp = Arc::new(MtpModel::from_weights(&weights, &model).unwrap().unwrap());
        let config = engine(1).config().clone();
        let normal = Engine::new(model.clone(), config.clone()).unwrap();
        let speculative = Engine::with_mtp(
            model,
            EngineConfig {
                max_draft_tokens: 4,
                ..config
            },
            Some(mtp),
        )
        .unwrap();
        let mut input = request();
        input.max_new_tokens = 12;
        let expected = collect(normal.submit(input).unwrap()).await;
        let mut input = request();
        input.max_new_tokens = 12;
        let actual = collect(speculative.submit(input).unwrap()).await;
        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.2, expected.2);
        assert_eq!(actual.1.cached_tokens, 0);
        assert!(speculative.metrics().mtp_rounds > 0);
        assert!(speculative.metrics().draft_accepted <= speculative.metrics().draft_proposed);
    }

    #[test]
    fn thinking_closes_within_remaining_budget_and_cancel_replaces_buffered_terminal() {
        let engine = engine(1);
        let config = engine.config();
        let model = engine.model();
        let mut input = request();
        input.max_new_tokens = 6;
        input.thinking = Some(ThinkingSpec {
            supported: true,
            enabled: true,
            budget: Some(4),
            start_tokens: vec![1],
            end_tokens: vec![2, 3],
            initially_in_thinking: false,
            disabled_reason: None,
        });
        let (sender, _receiver) = events::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let now = Instant::now();
        let job = Job {
            request: input,
            sender,
            cancelled: cancelled.clone(),
            submitted: now,
            deadline: now + Duration::from_secs(10),
        };
        let mut prefix = PrefixCache {
            entries: VecDeque::new(),
            used: 0,
            capacity: 0,
        };
        let counters = Counters::default();
        let mut active = Active::new(job, model, config, &mut prefix, None).unwrap();
        while active.prefetched < 6 {
            active
                .step(model, config, &mut prefix, &counters, None, 1)
                .unwrap();
        }
        // 先生成普通内容，再晚进入 thinking；剩余两格必须完整容纳结束标记。
        for token in [4, 4, 1, 5] {
            active.logits = vec![-100.0; 8];
            active.logits[token] = 100.0;
            active
                .step(model, config, &mut prefix, &counters, None, 1)
                .unwrap();
        }
        active
            .step(model, config, &mut prefix, &counters, None, 1)
            .unwrap();
        active
            .step(model, config, &mut prefix, &counters, None, 1)
            .unwrap();
        assert_eq!(&active.history[6..], &[4, 4, 1, 5, 2, 3]);
        assert!(!active.thinking.as_ref().unwrap().in_thinking());
        assert!(active.terminal);
        cancelled.store(true, Ordering::Release);
        active.interrupt_if_needed(&counters);
        assert_eq!(active.pending.len(), 1);
        assert!(matches!(
            active.pending.front(),
            Some(GenerationEvent::Finished {
                reason: FinishReason::Cancelled,
                ..
            })
        ));
        active.interrupt_if_needed(&counters);
        assert_eq!(counters.cancelled.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn zero_thinking_budget_emits_complete_closer() {
        let engine = engine(1);
        let mut input = request();
        input.max_new_tokens = 2;
        input.thinking = Some(ThinkingSpec {
            supported: true,
            enabled: true,
            budget: Some(0),
            start_tokens: vec![1],
            end_tokens: vec![2, 3],
            initially_in_thinking: true,
            disabled_reason: None,
        });
        let (tokens, usage, reason) = collect(engine.submit(input).unwrap()).await;
        assert_eq!(tokens, vec![2, 3]);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(reason, FinishReason::Length);
    }
}
