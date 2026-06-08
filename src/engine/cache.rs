//! KV Cache 管理
//!
//! 用于缓存注意力层的 Key 和 Value，避免重复计算。

use crate::error::{Result, RsinferError};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// KV Cache 数据结构
///
/// 为每一层 Transformer 维护独立的 Key 和 Value 缓存。
#[derive(Clone)]
pub struct KVCache {
    /// 每层的 Key 缓存: [num_heads, seq_len, head_dim]
    key_cache: Vec<Option<Tensor>>,

    /// 每层的 Value 缓存: [num_heads, seq_len, head_dim]
    value_cache: Vec<Option<Tensor>>,

    /// 当前序列长度 (用于跟踪)
    current_len: usize,

    /// 最大支持的序列长度 (保留用于未来扩展)
    #[allow(dead_code)]
    max_len: usize,
}

impl KVCache {
    /// 创建新的 KV Cache
    ///
    /// # Arguments
    /// * `num_layers` - Transformer 层数
    /// * `max_len` - 最大序列长度
    pub fn new(num_layers: usize, max_len: usize) -> Self {
        Self {
            key_cache: vec![None; num_layers],
            value_cache: vec![None; num_layers],
            current_len: 0,
            max_len,
        }
    }

    /// 追加新的 Key 和 Value 到指定层的缓存
    ///
    /// # Arguments
    /// * `layer_idx` - 层索引
    /// * `new_k` - 新的 Key: [num_heads, new_seq_len, head_dim]
    /// * `new_v` - 新的 Value: [num_heads, new_seq_len, head_dim]
    pub fn append(&mut self, layer_idx: usize, new_k: &Tensor, new_v: &Tensor) -> Result<()> {
        if layer_idx >= self.key_cache.len() {
            return Err(RsinferError::DimensionError(format!(
                "Layer index {} out of range (max {})",
                layer_idx,
                self.key_cache.len()
            )));
        }

        // 如果缓存为空，直接设置
        if self.key_cache[layer_idx].is_none() {
            self.key_cache[layer_idx] = Some(new_k.clone());
            self.value_cache[layer_idx] = Some(new_v.clone());

            // 更新当前长度 (只在第一层更新)
            if layer_idx == 0 {
                self.current_len = new_k.shape()[1];
            }
            return Ok(());
        }

        // 否则，拼接新的 K, V
        let old_k = self.key_cache[layer_idx].as_ref().unwrap();
        let old_v = self.value_cache[layer_idx].as_ref().unwrap();

        let concat_k = concat_along_seq(old_k, new_k)?;
        let concat_v = concat_along_seq(old_v, new_v)?;

        self.key_cache[layer_idx] = Some(concat_k);
        self.value_cache[layer_idx] = Some(concat_v);

        // 更新当前长度
        if layer_idx == 0 {
            self.current_len = self.key_cache[0].as_ref().unwrap().shape()[1];
        }

        Ok(())
    }

    /// 获取指定层的缓存
    ///
    /// 返回 (key_cache, value_cache)，形状均为 [num_heads, seq_len, head_dim]
    pub fn get(&self, layer_idx: usize) -> Result<(&Tensor, &Tensor)> {
        let k = self
            .key_cache
            .get(layer_idx)
            .and_then(|x| x.as_ref())
            .ok_or_else(|| {
                RsinferError::DimensionError(format!("No cache for layer {}", layer_idx))
            })?;

        let v = self
            .value_cache
            .get(layer_idx)
            .and_then(|x| x.as_ref())
            .ok_or_else(|| {
                RsinferError::DimensionError(format!("No cache for layer {}", layer_idx))
            })?;

        Ok((k, v))
    }

    /// 获取当前序列长度
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// 重置缓存 (用于新的生成会话)
    pub fn reset(&mut self) {
        for k in self.key_cache.iter_mut() {
            *k = None;
        }
        for v in self.value_cache.iter_mut() {
            *v = None;
        }
        self.current_len = 0;
    }
}

/// 沿序列维度 (axis=1) 拼接两个张量
///
/// a: [num_heads, seq_len_a, head_dim]
/// b: [num_heads, seq_len_b, head_dim]
/// result: [num_heads, seq_len_a + seq_len_b, head_dim]
fn concat_along_seq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[0] != b_shape[0] || a_shape[2] != b_shape[2] {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![a_shape[0], a_shape[2]],
            actual: vec![b_shape[0], b_shape[2]],
        });
    }

    let num_heads = a_shape[0];
    let seq_len_a = a_shape[1];
    let seq_len_b = b_shape[1];
    let head_dim = a_shape[2];
    let total_seq_len = seq_len_a + seq_len_b;

    let a_std = a.data.as_standard_layout();
    let b_std = b.data.as_standard_layout();
    let a_slice = a_std.as_slice().ok_or_else(|| {
        RsinferError::DimensionError("K cache old tensor is not contiguous".into())
    })?;
    let b_slice = b_std.as_slice().ok_or_else(|| {
        RsinferError::DimensionError("K cache new tensor is not contiguous".into())
    })?;

    let old_head_len = seq_len_a * head_dim;
    let new_head_len = seq_len_b * head_dim;
    let total_head_len = total_seq_len * head_dim;
    let mut output = vec![0f32; num_heads * total_head_len];
    for h in 0..num_heads {
        let out_start = h * total_head_len;
        let old_start = h * old_head_len;
        let new_start = h * new_head_len;
        output[out_start..out_start + old_head_len]
            .copy_from_slice(&a_slice[old_start..old_start + old_head_len]);
        output[out_start + old_head_len..out_start + total_head_len]
            .copy_from_slice(&b_slice[new_start..new_start + new_head_len]);
    }

    Ok(Tensor {
        data: ArrayD::from_shape_vec(IxDyn(&[num_heads, total_seq_len, head_dim]), output)
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4, 1024);
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCache::new(2, 1024);

        let k = Tensor::from_f32_slice(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
            .unwrap();
        let v = Tensor::from_f32_slice(&[2, 2, 2], &[5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0])
            .unwrap();

        cache.append(0, &k, &v).unwrap();
        assert_eq!(cache.current_len(), 2);

        let k2 = Tensor::from_f32_slice(&[2, 1, 2], &[9.0, 10.0, 90.0, 100.0]).unwrap();
        let v2 = Tensor::from_f32_slice(&[2, 1, 2], &[11.0, 12.0, 110.0, 120.0]).unwrap();
        cache.append(0, &k2, &v2).unwrap();
        assert_eq!(cache.current_len(), 3);

        let (cached_k, cached_v) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[2, 3, 2]);
        assert_eq!(
            cached_k.as_slice(),
            &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 10.0, 20.0, 30.0, 40.0, 90.0, 100.0]
        );
        assert_eq!(
            cached_v.as_slice(),
            &[5.0, 6.0, 7.0, 8.0, 11.0, 12.0, 50.0, 60.0, 70.0, 80.0, 110.0, 120.0]
        );
    }
}
