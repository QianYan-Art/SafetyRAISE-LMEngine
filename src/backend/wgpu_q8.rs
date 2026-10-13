//! 复用既有 WGSL Q8 matvec 的统一后端。量化与驻留有界，所有请求共享只读权重。

use super::{ensure_finite_output, validate_linear_arguments, WeightKey};
use crate::artifact::{WeightFormat, WeightTensor};
use crate::gpu::{GpuContext, GpuQ8MatVec};
use crate::tensor::Q8LinearWeight;
use crate::{Result, RsinferError};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;

struct Prepared {
    _source: WeightTensor,
    quantized: Q8LinearWeight,
    gpu: GpuQ8MatVec,
}
struct Inner {
    weights: HashMap<WeightKey, Option<Prepared>>,
    stats: WgpuStats,
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct WgpuStats {
    pub estimated_resident_bytes: usize,
    pub prepared_weights: usize,
    pub gpu_calls: u64,
    pub cpu_calls: u64,
}

pub struct WgpuQ8Backend {
    context: GpuContext,
    budget: usize,
    inner: Mutex<Inner>,
}
impl WgpuQ8Backend {
    pub fn new(resident_budget_bytes: usize) -> Result<Self> {
        if resident_budget_bytes == 0 {
            return Err(RsinferError::ConfigError("wgpu 驻留预算不能为零".into()));
        }
        Ok(Self {
            context: GpuContext::new().map_err(RsinferError::Unsupported)?,
            budget: resident_budget_bytes,
            inner: Mutex::new(Inner {
                weights: HashMap::new(),
                stats: WgpuStats::default(),
            }),
        })
    }
    pub fn stats(&self) -> Result<WgpuStats> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| RsinferError::WeightError("wgpu 权重锁失效".into()))?
            .stats
            .clone())
    }
    pub fn linear(&self, weight: &WeightTensor, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        validate_linear_arguments(weight, input, rows)?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| RsinferError::WeightError("wgpu 权重锁失效".into()))?;
        // 低位 GGUF 保持其原量化，不再做一次有损 Q8 转换。
        if !matches!(
            weight.format(),
            WeightFormat::F32 | WeightFormat::F16 | WeightFormat::BF16
        ) {
            inner.stats.cpu_calls += 1;
            return super::Backend::Cpu.linear(weight, input, rows);
        }
        let key = WeightKey::new(weight, weight.format());
        if !inner.weights.contains_key(&key) {
            let (out, width) = (weight.shape()[0], weight.shape()[1]);
            let bytes = width
                .div_ceil(4)
                .checked_mul(4)
                .and_then(|w| w.checked_mul(out))
                .and_then(|n| n.checked_add(out.checked_add(width)?.checked_mul(16)?))
                .and_then(|n| n.checked_add(256 * 1024))
                .ok_or_else(|| RsinferError::DimensionError("wgpu 驻留预算溢出".into()))?;
            let prepared = if bytes
                <= self
                    .budget
                    .saturating_sub(inner.stats.estimated_resident_bytes)
            {
                let mut quantized = Q8LinearWeight {
                    qweight: Vec::with_capacity(out * width),
                    scales: Vec::with_capacity(out),
                    out_features: out,
                    in_features: width,
                };
                let mut row = vec![0.0; width];
                for i in 0..out {
                    weight.read_row(i, &mut row)?;
                    if row.iter().any(|x| !x.is_finite()) {
                        return Err(RsinferError::WeightError("Q8 输入权重含非有限值".into()));
                    }
                    let maximum = row.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                    let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
                    quantized.scales.push(scale);
                    quantized.qweight.extend(
                        row.iter()
                            .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8),
                    );
                }
                let gpu = GpuQ8MatVec::from_q8_weight_with_context_and_argmax(
                    &self.context,
                    &quantized,
                    false,
                )
                .map_err(|e| RsinferError::WeightError(format!("wgpu 权重驻留失败：{e}")))?;
                inner.stats.estimated_resident_bytes += bytes;
                inner.stats.prepared_weights += 1;
                Some(Prepared {
                    _source: weight.clone(),
                    quantized,
                    gpu,
                })
            } else {
                None
            };
            inner.weights.insert(key.clone(), prepared);
        }
        let result = if let Some(prepared) = inner.weights.get(&key).and_then(Option::as_ref) {
            if rows == 1 {
                let output = prepared
                    .gpu
                    .forward_raw(input)
                    .map_err(|e| RsinferError::WeightError(format!("wgpu 执行失败：{e}")))?;
                inner.stats.gpu_calls += 1;
                output
            } else {
                // prefill 复用相同 Q8 权重，避免 CPU/GPU 切换改变层的量化合同。
                let q = &prepared.quantized;
                let mut output = vec![0.0; rows * q.out_features];
                output
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(index, value)| {
                        let (r, o) = (index / q.out_features, index % q.out_features);
                        *value = input[r * q.in_features..(r + 1) * q.in_features]
                            .iter()
                            .zip(&q.qweight[o * q.in_features..(o + 1) * q.in_features])
                            .map(|(x, w)| x * (*w as f32))
                            .sum::<f32>()
                            * q.scales[o];
                    });
                inner.stats.cpu_calls += 1;
                output
            }
        } else {
            inner.stats.cpu_calls += 1;
            weight.matmul(input, rows)?
        };
        ensure_finite_output(&result, "wgpu-q8")?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "需要启用 gpu-tests 并提供可用 GPU 后端"
    )]
    fn real_q8_decode_matches_quantized_prefill_and_budget_fallback() {
        let _guard = crate::gpu::gpu_test_guard();
        let weight = WeightTensor::from_f32(
            vec![3, 4],
            vec![
                1.0, -0.5, 0.2, 0.0, 0.7, 0.8, -0.9, 0.1, 0.0, 0.3, 0.2, -1.0,
            ],
        )
        .unwrap();
        let backend = WgpuQ8Backend::new(4 * 1024 * 1024).unwrap();
        eprintln!("[rsinfer-gpu] stage=q8_backend_ready");
        let single = backend.linear(&weight, &[0.2, 0.4, -0.1, 0.8], 1).unwrap();
        eprintln!("[rsinfer-gpu] stage=q8_single_ready");
        let batch = backend
            .linear(&weight, &[0.2, 0.4, -0.1, 0.8, 0.2, 0.4, -0.1, 0.8], 2)
            .unwrap();
        for (a, b) in single.iter().zip(&batch) {
            assert!((a - b).abs() < 1e-5);
        }
        assert_eq!(backend.stats().unwrap().prepared_weights, 1);
        assert_eq!(backend.stats().unwrap().gpu_calls, 1);
        eprintln!("[rsinfer-gpu] stage=q8_prefill_ready");
        let limited = WgpuQ8Backend::new(1).unwrap();
        let expected = weight.matmul(&[0.2, 0.4, -0.1, 0.8], 1).unwrap();
        assert_eq!(
            limited.linear(&weight, &[0.2, 0.4, -0.1, 0.8], 1).unwrap(),
            expected
        );
        assert_eq!(limited.stats().unwrap().prepared_weights, 0);
    }
}
