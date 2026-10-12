//! 可复现的 logits 采样器。

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::{Result, RsinferError};

/// 生成阶段的采样参数。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// 温度为零时使用贪心；正数越大越平坦。
    pub temperature: f32,
    /// nucleus sampling 累积概率，范围为 `(0, 1]`。
    pub top_p: f32,
    /// 保留的最高 logits 数量；零表示不限制。
    pub top_k: usize,
    /// 重复 token 的惩罚系数，必须为正数。
    pub repetition_penalty: f32,
    /// 可选固定种子；设置后同一 logits/history 序列可复现。
    pub seed: Option<u64>,
    /// 单次请求允许生成的最大 token 数。
    pub max_new_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
            max_new_tokens: 512,
        }
    }
}

impl SamplingParams {
    /// 校验参数，避免把无效值传入 softmax 或生成循环。
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(RsinferError::ConfigError(
                "temperature 必须是有限的非负数".into(),
            ));
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(RsinferError::ConfigError("top_p 必须位于 (0, 1]".into()));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(RsinferError::ConfigError(
                "repetition_penalty 必须是有限的正数".into(),
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(RsinferError::ConfigError(
                "max_new_tokens 必须大于零".into(),
            ));
        }
        Ok(())
    }
}

/// 具有显式参数和随机源的采样器。
pub struct Sampler {
    params: SamplingParams,
    rng: StdRng,
}

impl Sampler {
    /// 创建采样器；无 seed 时使用系统随机源初始化。
    pub fn new(params: SamplingParams) -> Result<Self> {
        params.validate()?;
        let rng = match params.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Ok(Self { params, rng })
    }

    pub fn params(&self) -> &SamplingParams {
        &self.params
    }

    /// 当前采样器是否严格等价于 argmax，可用于需要确定验证分布的路径。
    pub fn is_greedy(&self) -> bool {
        self.params.temperature == 0.0 && self.params.repetition_penalty == 1.0
    }

    /// 从 logits 选择 token。history 只用于 repetition penalty。
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(RsinferError::DimensionError("采样 logits 不能为空".into()));
        }
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(RsinferError::WeightError("采样 logits 包含非有限值".into()));
        }

        let mut adjusted = logits.to_vec();
        apply_repetition_penalty(&mut adjusted, history, self.params.repetition_penalty);
        if self.params.temperature == 0.0 {
            return Ok(argmax(&adjusted));
        }

        let temperature = self.params.temperature;
        let mut candidates: Vec<(usize, f32)> = adjusted
            .iter()
            .enumerate()
            .map(|(index, value)| (index, *value / temperature))
            .collect();
        candidates.sort_unstable_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        if self.params.top_k > 0 && candidates.len() > self.params.top_k {
            candidates.truncate(self.params.top_k);
        }

        let max_logit = candidates
            .iter()
            .map(|(_, value)| *value)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = candidates
            .iter()
            .map(|(_, value)| (*value - max_logit).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(RsinferError::WeightError("采样 softmax 权重无效".into()));
        }
        for weight in &mut weights {
            *weight /= total;
        }

        let mut cumulative = 0.0;
        let mut cutoff = weights.len();
        for (index, weight) in weights.iter().enumerate() {
            cumulative += *weight;
            if cumulative >= self.params.top_p {
                cutoff = index + 1;
                break;
            }
        }
        cutoff = cutoff.max(1).min(weights.len());
        let kept_sum: f32 = weights[..cutoff].iter().sum();
        let mut draw = self.rng.gen::<f32>() * kept_sum;
        for (candidate, weight) in candidates[..cutoff].iter().zip(&weights[..cutoff]) {
            draw -= *weight;
            if draw <= 0.0 {
                return Ok(candidate.0 as u32);
            }
        }
        Ok(candidates[cutoff - 1].0 as u32)
    }
}

fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = HashSet::with_capacity(history.len());
    for &token in history {
        if !seen.insert(token) {
            continue;
        }
        let index = token as usize;
        if let Some(value) = logits.get_mut(index) {
            if *value < 0.0 {
                *value *= penalty;
            } else {
                *value /= penalty;
            }
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_sampling_boundaries() {
        assert!(SamplingParams::default().validate().is_ok());
        assert!(SamplingParams {
            top_p: 0.0,
            ..SamplingParams::default()
        }
        .validate()
        .is_err());
        assert!(SamplingParams {
            temperature: -1.0,
            ..SamplingParams::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let params = SamplingParams {
            seed: Some(7),
            temperature: 1.0,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        let mut first = Sampler::new(params.clone()).unwrap();
        let mut second = Sampler::new(params).unwrap();
        let logits = [0.2, 1.1, 0.7, -0.3];
        let a: Vec<_> = (0..16)
            .map(|_| first.sample(&logits, &[]).unwrap())
            .collect();
        let b: Vec<_> = (0..16)
            .map(|_| second.sample(&logits, &[]).unwrap())
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn greedy_and_repetition_penalty_are_applied() {
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..SamplingParams::default()
        };
        let mut sampler = Sampler::new(params).unwrap();
        assert_eq!(sampler.sample(&[1.0, 1.5], &[1]).unwrap(), 0);
    }

    #[test]
    fn repetition_penalty_applies_once_per_unique_token() {
        let mut logits = [4.0, -4.0];
        apply_repetition_penalty(&mut logits, &[0, 0, 1, 1], 2.0);
        assert_eq!(logits, [2.0, -8.0]);
    }
}
