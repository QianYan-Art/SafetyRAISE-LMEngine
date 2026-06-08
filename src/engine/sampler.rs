//! 从 logits 选择下一个 token 的采样策略。

use crate::tensor::Tensor;
use rand::Rng;

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
        let mut scaled: Vec<f32> = data.iter().map(|&x| x * inv_t).collect();

        if self.top_k > 0 && self.top_k < scaled.len() {
            let mut idx: Vec<usize> = (0..scaled.len()).collect();
            idx.sort_unstable_by(|&a, &b| cmp_desc(scaled[a], scaled[b]));
            for &i in idx.iter().skip(self.top_k) {
                scaled[i] = f32::NEG_INFINITY;
            }
        }

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
}
