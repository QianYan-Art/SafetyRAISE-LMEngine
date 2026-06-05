//! KV Cache 管理
//!
//! 用于缓存注意力层的 Key 和 Value，避免重复计算。

use crate::error::{Result, RsinferError};
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// KV Cache 数据结构
///
/// 为每一层 Transformer 维护独立的 Key 和 Value 缓存。
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
        let k = self.key_cache.get(layer_idx)
            .and_then(|x| x.as_ref())
            .ok_or_else(|| RsinferError::DimensionError(format!(
                "No cache for layer {}",
                layer_idx
            )))?;
        
        let v = self.value_cache.get(layer_idx)
            .and_then(|x| x.as_ref())
            .ok_or_else(|| RsinferError::DimensionError(format!(
                "No cache for layer {}",
                layer_idx
            )))?;
        
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

    let mut result = ArrayD::zeros(IxDyn(&[num_heads, total_seq_len, head_dim]));

    // 复制 a
    for h in 0..num_heads {
        for s in 0..seq_len_a {
            for d in 0..head_dim {
                result[[h, s, d]] = a.data[[h, s, d]];
            }
        }
    }

    // 复制 b
    for h in 0..num_heads {
        for s in 0..seq_len_b {
            for d in 0..head_dim {
                result[[h, seq_len_a + s, d]] = b.data[[h, s, d]];
            }
        }
    }

    Ok(Tensor { data: result })
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
        
        let k = Tensor::zeros(&[4, 3, 64]); // [num_heads, seq_len, head_dim]
        let v = Tensor::zeros(&[4, 3, 64]);
        
        cache.append(0, &k, &v).unwrap();
        assert_eq!(cache.current_len(), 3);
        
        // 追加更多
        let k2 = Tensor::zeros(&[4, 2, 64]);
        let v2 = Tensor::zeros(&[4, 2, 64]);
        cache.append(0, &k2, &v2).unwrap();
        assert_eq!(cache.current_len(), 5);
    }
}
