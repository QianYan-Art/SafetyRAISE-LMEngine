//! KV Cache 管理
//!
//! 用于缓存注意力层的 Key 和 Value，避免重复计算。

use crate::error::{Result, RsinferError};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

#[derive(Clone)]
struct KVCacheLayer {
    key: Vec<f32>,
    value: Vec<f32>,
    num_heads: usize,
    head_dim: usize,
    current_len: usize,
    capacity_len: usize,
}

pub struct CachedKV<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub num_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub capacity_len: usize,
}

#[derive(Clone, Debug)]
pub struct KVCacheSnapshot {
    layer_lengths: Vec<Option<usize>>,
    current_len: usize,
}

impl KVCacheLayer {
    fn ensure_capacity(&mut self, required_len: usize, max_len: usize) -> Result<()> {
        if required_len <= self.capacity_len {
            return Ok(());
        }
        let mut next_capacity = self.capacity_len.max(64);
        while next_capacity < required_len {
            next_capacity = next_capacity.saturating_mul(2);
        }
        next_capacity = next_capacity.min(max_len);
        if next_capacity < required_len {
            return Err(RsinferError::DimensionError(format!(
                "KV cache required length {required_len} exceeds max_len {max_len}"
            )));
        }

        let old_capacity = self.capacity_len;
        let old_head_len = old_capacity * self.head_dim;
        let new_head_len = next_capacity * self.head_dim;
        let mut key = vec![0.0; self.num_heads * new_head_len];
        let mut value = vec![0.0; key.len()];
        if old_capacity > 0 {
            let used_head_len = self.current_len * self.head_dim;
            for h in 0..self.num_heads {
                let old_start = h * old_head_len;
                let new_start = h * new_head_len;
                key[new_start..new_start + used_head_len]
                    .copy_from_slice(&self.key[old_start..old_start + used_head_len]);
                value[new_start..new_start + used_head_len]
                    .copy_from_slice(&self.value[old_start..old_start + used_head_len]);
            }
        }
        self.key = key;
        self.value = value;
        self.capacity_len = next_capacity;
        Ok(())
    }
}

/// KV Cache 数据结构
///
/// 为每一层 Transformer 维护独立的 Key 和 Value 缓存。
#[derive(Clone)]
pub struct KVCache {
    layers: Vec<Option<KVCacheLayer>>,

    /// 当前序列长度 (用于跟踪)
    current_len: usize,

    /// 最大支持的序列长度
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
            layers: vec![None; num_layers],
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
        if layer_idx >= self.layers.len() {
            return Err(RsinferError::DimensionError(format!(
                "Layer index {} out of range (max {})",
                layer_idx,
                self.layers.len()
            )));
        }

        let k_shape = new_k.shape();
        let v_shape = new_v.shape();
        if k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(RsinferError::DimensionError(
                "KV cache append expects [num_heads, seq_len, head_dim] tensors".into(),
            ));
        }
        if k_shape != v_shape {
            return Err(RsinferError::ShapeMismatch {
                expected: k_shape.to_vec(),
                actual: v_shape.to_vec(),
            });
        }

        let num_heads = k_shape[0];
        let new_seq_len = k_shape[1];
        let head_dim = k_shape[2];
        let layer = self.layers[layer_idx].get_or_insert_with(|| KVCacheLayer {
            key: Vec::new(),
            value: Vec::new(),
            num_heads,
            head_dim,
            current_len: 0,
            capacity_len: 0,
        });

        if layer.num_heads != num_heads || layer.head_dim != head_dim {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![layer.num_heads, layer.current_len, layer.head_dim],
                actual: k_shape.to_vec(),
            });
        }
        if layer.current_len + new_seq_len > self.max_len {
            return Err(RsinferError::DimensionError(format!(
                "KV cache length {} exceeds max_len {}",
                layer.current_len + new_seq_len,
                self.max_len
            )));
        }
        layer.ensure_capacity(layer.current_len + new_seq_len, self.max_len)?;

        let k_std = new_k.data.as_standard_layout();
        let v_std = new_v.data.as_standard_layout();
        let k_slice = k_std
            .as_slice()
            .ok_or_else(|| RsinferError::DimensionError("new K tensor is not contiguous".into()))?;
        let v_slice = v_std
            .as_slice()
            .ok_or_else(|| RsinferError::DimensionError("new V tensor is not contiguous".into()))?;
        let new_head_len = new_seq_len * head_dim;
        let cache_head_len = layer.capacity_len * head_dim;
        let dst_seq_offset = layer.current_len * head_dim;
        for h in 0..num_heads {
            let src_start = h * new_head_len;
            let dst_start = h * cache_head_len + dst_seq_offset;
            layer.key[dst_start..dst_start + new_head_len]
                .copy_from_slice(&k_slice[src_start..src_start + new_head_len]);
            layer.value[dst_start..dst_start + new_head_len]
                .copy_from_slice(&v_slice[src_start..src_start + new_head_len]);
        }

        layer.current_len += new_seq_len;
        if layer_idx == 0 {
            self.current_len = layer.current_len;
        }
        Ok(())
    }

    pub fn get_cached(&self, layer_idx: usize) -> Result<CachedKV<'_>> {
        let layer = self
            .layers
            .get(layer_idx)
            .and_then(|x| x.as_ref())
            .ok_or_else(|| {
                RsinferError::DimensionError(format!("No cache for layer {}", layer_idx))
            })?;

        Ok(CachedKV {
            key: &layer.key,
            value: &layer.value,
            num_heads: layer.num_heads,
            seq_len: layer.current_len,
            head_dim: layer.head_dim,
            capacity_len: layer.capacity_len,
        })
    }

    /// 获取指定层的缓存，返回紧凑 Tensor。该兼容路径会复制当前缓存内容。
    pub fn get(&self, layer_idx: usize) -> Result<(Tensor, Tensor)> {
        let cached = self.get_cached(layer_idx)?;
        let mut key = vec![0f32; cached.num_heads * cached.seq_len * cached.head_dim];
        let mut value = vec![0f32; key.len()];
        let compact_head_len = cached.seq_len * cached.head_dim;
        let cache_head_len = cached.capacity_len * cached.head_dim;
        for h in 0..cached.num_heads {
            let src_start = h * cache_head_len;
            let dst_start = h * compact_head_len;
            key[dst_start..dst_start + compact_head_len]
                .copy_from_slice(&cached.key[src_start..src_start + compact_head_len]);
            value[dst_start..dst_start + compact_head_len]
                .copy_from_slice(&cached.value[src_start..src_start + compact_head_len]);
        }

        Ok((
            Tensor {
                data: ArrayD::from_shape_vec(
                    IxDyn(&[cached.num_heads, cached.seq_len, cached.head_dim]),
                    key,
                )
                .map_err(|e| RsinferError::DimensionError(e.to_string()))?,
            },
            Tensor {
                data: ArrayD::from_shape_vec(
                    IxDyn(&[cached.num_heads, cached.seq_len, cached.head_dim]),
                    value,
                )
                .map_err(|e| RsinferError::DimensionError(e.to_string()))?,
            },
        ))
    }

    /// 获取当前序列长度
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn snapshot(&self) -> KVCacheSnapshot {
        KVCacheSnapshot {
            layer_lengths: self
                .layers
                .iter()
                .map(|layer| layer.as_ref().map(|layer| layer.current_len))
                .collect(),
            current_len: self.current_len,
        }
    }

    pub fn restore(&mut self, snapshot: KVCacheSnapshot) -> Result<()> {
        if snapshot.layer_lengths.len() != self.layers.len() {
            return Err(RsinferError::DimensionError(format!(
                "KV cache snapshot layer count {} does not match cache layer count {}",
                snapshot.layer_lengths.len(),
                self.layers.len()
            )));
        }
        for (layer, length) in self.layers.iter_mut().zip(snapshot.layer_lengths) {
            match (layer.as_mut(), length) {
                (Some(layer), Some(length)) if length <= layer.capacity_len => {
                    layer.current_len = length;
                }
                (Some(_), Some(length)) => {
                    return Err(RsinferError::DimensionError(format!(
                        "KV cache snapshot length {length} exceeds layer capacity"
                    )));
                }
                (Some(layer), None) => layer.current_len = 0,
                (None, Some(0)) => {}
                (None, Some(_)) => {
                    return Err(RsinferError::DimensionError(
                        "KV cache snapshot references missing layer".into(),
                    ));
                }
                (None, None) => {}
            }
        }
        self.current_len = snapshot.current_len;
        Ok(())
    }

    /// 重置缓存 (用于新的生成会话)
    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut().flatten() {
            layer.current_len = 0;
        }
        self.current_len = 0;
    }
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

        let cached = cache.get_cached(0).unwrap();
        assert_eq!(cached.num_heads, 2);
        assert_eq!(cached.seq_len, 3);
        assert_eq!(cached.head_dim, 2);
        assert_eq!(cached.capacity_len, 64);
        assert_eq!(&cached.key[0..6], &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]);
        assert_eq!(
            &cached.key[128..134],
            &[10.0, 20.0, 30.0, 40.0, 90.0, 100.0]
        );

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

    #[test]
    fn snapshot_restore_truncates_lengths_without_losing_prior_cache() {
        let mut cache = KVCache::new(1, 16);
        let k = Tensor::from_f32_slice(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = Tensor::from_f32_slice(&[1, 2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        cache.append(0, &k, &v).unwrap();
        let snapshot = cache.snapshot();

        let speculative_k = Tensor::from_f32_slice(&[1, 1, 2], &[9.0, 10.0]).unwrap();
        let speculative_v = Tensor::from_f32_slice(&[1, 1, 2], &[11.0, 12.0]).unwrap();
        cache.append(0, &speculative_k, &speculative_v).unwrap();
        assert_eq!(cache.current_len(), 3);

        cache.restore(snapshot).unwrap();
        assert_eq!(cache.current_len(), 2);
        let replacement_k = Tensor::from_f32_slice(&[1, 1, 2], &[13.0, 14.0]).unwrap();
        let replacement_v = Tensor::from_f32_slice(&[1, 1, 2], &[15.0, 16.0]).unwrap();
        cache.append(0, &replacement_k, &replacement_v).unwrap();

        let (cached_k, cached_v) = cache.get(0).unwrap();
        assert_eq!(cached_k.as_slice(), &[1.0, 2.0, 3.0, 4.0, 13.0, 14.0]);
        assert_eq!(cached_v.as_slice(), &[5.0, 6.0, 7.0, 8.0, 15.0, 16.0]);
    }
}
