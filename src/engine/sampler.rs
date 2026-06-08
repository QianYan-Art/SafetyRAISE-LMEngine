//! 从 logits 选择下一个 token 的采样策略。

use crate::tensor::Tensor;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub trait Sampler: Send + Sync {
    /// logits: [vocab_size] 一维张量
    fn sample(&self, logits: &Tensor) -> u32;

    fn is_greedy(&self) -> bool {
        false
    }
}

/// 贪心采样：取 argmax，确定性输出。
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &Tensor) -> u32 {
        argmax(logits.as_slice())
    }

    fn is_greedy(&self) -> bool {
        true
    }
}

/// temperature -> top-k -> top-p -> 多项式采样。
///
/// `temperature <= 0` 时退化为贪心。
pub struct CombinedSampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl CombinedSampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32) -> Self {
        Self {
            temperature,
            top_k,
            top_p,
        }
    }
}

impl Sampler for CombinedSampler {
    fn sample(&self, logits: &Tensor) -> u32 {
        let data = logits.as_slice();
        if self.temperature <= 0.0 {
            return argmax(data);
        }

        let inv_t = 1.0 / self.temperature;
        if self.top_k > 0 && self.top_k < data.len() {
            let candidates = top_k_indices(data, self.top_k);
            return sample_ordered_candidates(data, &candidates, inv_t, self.top_p);
        }

        let scaled: Vec<f32> = data.iter().map(|&x| x * inv_t).collect();
        let probs = softmax(&scaled);
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_unstable_by(|&a, &b| cmp_desc(probs[a], probs[b]));

        let mut cumulative = 0.0;
        let mut cutoff = order.len();
        for (rank, &i) in order.iter().enumerate() {
            cumulative += probs[i];
            if cumulative >= self.top_p {
                cutoff = rank + 1;
                break;
            }
        }

        let kept = &order[..cutoff];
        let sum: f32 = kept.iter().map(|&i| probs[i]).sum();
        let r: f32 = rand::thread_rng().gen::<f32>() * sum;
        let mut acc = 0.0;
        for &i in kept {
            acc += probs[i];
            if r < acc {
                return i as u32;
            }
        }
        *kept.last().unwrap() as u32
    }

    fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

fn top_k_indices(data: &[f32], top_k: usize) -> Vec<usize> {
    if top_k == 0 {
        return Vec::new();
    }
    if top_k > 64 {
        return top_k_indices_heap(data, top_k);
    }
    let mut candidates = Vec::with_capacity(top_k.min(data.len()));
    for i in 0..data.len() {
        if candidates.len() < top_k {
            insert_desc(&mut candidates, i, data);
        } else if cmp_desc(data[i], data[*candidates.last().unwrap()]).is_lt() {
            insert_desc(&mut candidates, i, data);
            candidates.pop();
        }
    }
    candidates
}

fn insert_desc(candidates: &mut Vec<usize>, idx: usize, data: &[f32]) {
    let pos = candidates
        .iter()
        .position(|&candidate| cmp_desc(data[idx], data[candidate]).is_lt())
        .unwrap_or(candidates.len());
    candidates.insert(pos, idx);
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    index: usize,
    value: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.value.to_bits() == other.value.to_bits()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value
            .partial_cmp(&other.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn top_k_indices_heap(data: &[f32], top_k: usize) -> Vec<usize> {
    let mut candidates = BinaryHeap::with_capacity(top_k.min(data.len()));
    for (index, &value) in data.iter().enumerate() {
        let candidate = Reverse(Candidate { index, value });
        if candidates.len() < top_k {
            candidates.push(candidate);
        } else if candidate.0 > candidates.peek().unwrap().0 {
            candidates.pop();
            candidates.push(candidate);
        }
    }

    let mut indices: Vec<usize> = candidates
        .into_iter()
        .map(|candidate| candidate.0.index)
        .collect();
    indices.sort_unstable_by(|&a, &b| cmp_desc(data[a], data[b]));
    indices
}

fn sample_ordered_candidates(data: &[f32], ordered: &[usize], inv_t: f32, top_p: f32) -> u32 {
    let max = ordered
        .iter()
        .map(|&i| data[i] * inv_t)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = ordered
        .iter()
        .map(|&i| (data[i] * inv_t - max).exp())
        .collect();
    let sum: f32 = exp.iter().sum();

    let mut cumulative = 0.0;
    let mut cutoff = ordered.len();
    for (rank, &prob) in exp.iter().enumerate() {
        cumulative += prob / sum;
        if cumulative >= top_p {
            cutoff = rank + 1;
            break;
        }
    }

    let kept = &ordered[..cutoff];
    let kept_exp = &exp[..cutoff];
    let kept_sum: f32 = kept_exp.iter().sum();
    let r: f32 = rand::thread_rng().gen::<f32>() * kept_sum;
    let mut acc = 0.0;
    for (&i, &weight) in kept.iter().zip(kept_exp) {
        acc += weight;
        if r < acc {
            return i as u32;
        }
    }
    *kept.last().unwrap() as u32
}

pub fn default_sampler() -> Box<dyn Sampler> {
    Box::new(CombinedSampler::new(0.6, 20, 0.95))
}

fn argmax(data: &[f32]) -> u32 {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn cmp_desc(a: f32, b: f32) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&x| x / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let logits = Tensor::from_f32_slice(&[5], &[1.0, 2.0, 5.0, 3.0, 4.0]).unwrap();
        assert_eq!(GreedySampler.sample(&logits), 2);
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let logits = Tensor::from_f32_slice(&[5], &[1.0, 2.0, 5.0, 3.0, 4.0]).unwrap();
        assert_eq!(CombinedSampler::new(0.0, 0, 1.0).sample(&logits), 2);
    }

    #[test]
    fn top_k_indices_keep_only_highest_logits_in_order() {
        let logits = [0.1, 3.0, -1.0, 2.5, 5.0, 4.0];

        let indices = top_k_indices(&logits, 3);

        assert_eq!(indices, vec![4, 5, 1]);
    }

    #[test]
    fn top_k_indices_handle_zero_and_full_vocab() {
        let logits = [0.1, 3.0, -1.0, 2.5];

        assert!(top_k_indices(&logits, 0).is_empty());
        assert_eq!(top_k_indices(&logits, 4), vec![1, 3, 0, 2]);
    }

    #[test]
    fn top_k_indices_large_k_uses_heap_path() {
        let logits: Vec<f32> = (0..128).map(|i| ((i * 37) % 128) as f32).collect();

        let indices = top_k_indices(&logits, 80);

        assert_eq!(indices.len(), 80);
        assert_eq!(indices[0], 83);
        assert_eq!(logits[indices[79]], 48.0);
        for pair in indices.windows(2) {
            assert!(logits[pair[0]] >= logits[pair[1]]);
        }
    }

    #[test]
    fn top_k_sampling_never_returns_filtered_token() {
        let logits = Tensor::from_f32_slice(&[5], &[100.0, 90.0, 80.0, -1000.0, -1000.0]).unwrap();
        let sampler = CombinedSampler::new(0.6, 2, 1.0);

        for _ in 0..64 {
            let token = sampler.sample(&logits);
            assert!(token == 0 || token == 1);
        }
    }

    #[test]
    fn top_p_can_cut_top_k_candidates_to_first_token() {
        let logits = Tensor::from_f32_slice(&[4], &[20.0, 1.0, 0.0, -1.0]).unwrap();
        let sampler = CombinedSampler::new(1.0, 3, 0.5);

        for _ in 0..16 {
            assert_eq!(sampler.sample(&logits), 0);
        }
    }
}
