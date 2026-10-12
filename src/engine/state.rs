//! 请求私有的可恢复状态。快照共享只读页，后续写入采用 copy-on-write。

use std::sync::Arc;

use crate::model::spec::{checked_elements, LayerKind, ModelSpec};
use crate::{Result, RsinferError};

fn invalid(message: &str) -> RsinferError {
    RsinferError::DimensionError(message.into())
}

#[derive(Debug, Clone)]
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    pages: Vec<Arc<KvPage>>,
    width: usize,
    page_tokens: usize,
    max_tokens: usize,
    len: usize,
}

impl PagedKvCache {
    pub fn new(width: usize, max_tokens: usize, page_tokens: usize) -> Result<Self> {
        if width == 0 || max_tokens == 0 || page_tokens == 0 {
            return Err(invalid("KV 几何不能为零"));
        }
        checked_elements(&[width, page_tokens, 2, 4])?;
        Ok(Self {
            pages: vec![],
            width,
            page_tokens,
            max_tokens,
            len: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != self.width || value.len() != self.width || self.len == self.max_tokens {
            return Err(invalid("KV 写入宽度错误或上下文已满"));
        }
        if key.iter().chain(value).any(|v| !v.is_finite()) {
            return Err(invalid("KV 写入含非有限值"));
        }
        if self.len.is_multiple_of(self.page_tokens) {
            let capacity = self.page_tokens.min(self.max_tokens - self.len) * self.width;
            let mut keys = Vec::new();
            let mut values = Vec::new();
            keys.try_reserve_exact(capacity)
                .map_err(|_| invalid("KV key 页分配失败"))?;
            values
                .try_reserve_exact(capacity)
                .map_err(|_| invalid("KV value 页分配失败"))?;
            self.pages.push(Arc::new(KvPage { keys, values }));
        }
        let page = Arc::make_mut(self.pages.last_mut().ok_or_else(|| invalid("缺少 KV 页"))?);
        // COW 克隆可能收缩 capacity；恢复本页的精确上限，避免 Vec 自动倍增突破预算。
        let page_start = self.len / self.page_tokens * self.page_tokens;
        let capacity = self.page_tokens.min(self.max_tokens - page_start) * self.width;
        page.keys
            .try_reserve_exact(capacity.saturating_sub(page.keys.len()))
            .map_err(|_| invalid("KV key 页扩展失败"))?;
        page.values
            .try_reserve_exact(capacity.saturating_sub(page.values.len()))
            .map_err(|_| invalid("KV value 页扩展失败"))?;
        page.keys.extend_from_slice(key);
        page.values.extend_from_slice(value);
        self.len += 1;
        Ok(())
    }

    pub fn get(&self, token: usize) -> Result<(&[f32], &[f32])> {
        if token >= self.len {
            return Err(invalid("KV token 索引越界"));
        }
        let page = &self.pages[token / self.page_tokens];
        let start = token % self.page_tokens * self.width;
        Ok((
            &page.keys[start..start + self.width],
            &page.values[start..start + self.width],
        ))
    }

    /// 仅适用于全注意力。混合模型必须恢复完整 SequenceState 快照。
    pub fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.len {
            return Err(invalid("不能以 truncate 扩展 KV"));
        }
        if len == self.len {
            return Ok(());
        }
        self.pages.truncate(len.div_ceil(self.page_tokens));
        let tail = len % self.page_tokens;
        if tail != 0 {
            let page = Arc::make_mut(
                self.pages
                    .last_mut()
                    .ok_or_else(|| invalid("缺少 KV 尾页"))?,
            );
            page.keys.truncate(tail * self.width);
            page.values.truncate(tail * self.width);
        }
        self.len = len;
        Ok(())
    }

    pub fn resident_bytes(&self) -> usize {
        self.pages
            .iter()
            .map(|p| (p.keys.capacity() + p.values.capacity()) * 4)
            .sum()
    }
}

#[derive(Debug, Clone)]
pub struct RecurrentState {
    /// [value_head, value_dim, key_dim]，与 GDN 更新中的 S[Dv,Dk] 对应。
    pub matrix: Vec<f32>,
    /// [qkv_channel, conv_kernel - 1]，旧 token 在前。
    pub convolution: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum LayerState {
    Attention(PagedKvCache),
    Recurrent(Arc<RecurrentState>),
}

#[derive(Debug, Clone)]
pub struct SequenceState {
    pub(crate) owner: u64,
    pub(crate) layers: Vec<LayerState>,
    pub(crate) len: usize,
    pub(crate) next_position: usize,
    pub(crate) last_hidden: Vec<f32>,
    max_context: usize,
}

impl SequenceState {
    pub fn new(spec: &ModelSpec, max_context: usize) -> Result<Self> {
        spec.validate()?;
        if max_context == 0 || max_context > spec.max_position_embeddings {
            return Err(invalid("请求窗口超出模型位置容量"));
        }
        let width = checked_elements(&[spec.attention.kv_heads, spec.attention.head_dim])?;
        let layers = spec
            .layers
            .iter()
            .map(|kind| match kind {
                LayerKind::FullAttention => Ok(LayerState::Attention(PagedKvCache::new(
                    width,
                    max_context,
                    64,
                )?)),
                LayerKind::LinearAttention => {
                    let g = spec.gdn.as_ref().ok_or_else(|| invalid("缺少 GDN 配置"))?;
                    let matrix = checked_elements(&[g.value_heads, g.key_dim, g.value_dim])?;
                    let channels = checked_elements(&[g.key_heads, g.key_dim, 2])?
                        .checked_add(checked_elements(&[g.value_heads, g.value_dim])?)
                        .ok_or_else(|| invalid("卷积通道溢出"))?;
                    let history = checked_elements(&[channels, g.conv_kernel - 1])?;
                    Ok(LayerState::Recurrent(Arc::new(RecurrentState {
                        matrix: zeros(matrix)?,
                        convolution: zeros(history)?,
                    })))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            owner: 0,
            layers,
            len: 0,
            next_position: 0,
            last_hidden: vec![],
            max_context,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn max_context(&self) -> usize {
        self.max_context
    }
    pub fn next_position(&self) -> usize {
        self.next_position
    }
    pub fn last_hidden(&self) -> &[f32] {
        &self.last_hidden
    }

    pub fn ensure_room(&self, tokens: usize) -> Result<()> {
        if tokens == 0
            || self
                .len
                .checked_add(tokens)
                .is_none_or(|n| n > self.max_context)
        {
            return Err(invalid("输入为空或超过请求上下文窗口"));
        }
        Ok(())
    }

    /// 克隆的是一组相同提交点的 KV 页和递归状态，不允许跨模型恢复。
    pub fn checkpoint(&self) -> Self {
        self.clone()
    }

    pub fn resident_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|state| match state {
                LayerState::Attention(kv) => kv.resident_bytes(),
                LayerState::Recurrent(s) => (s.matrix.len() + s.convolution.len()) * 4,
            })
            .sum::<usize>()
            + self.last_hidden.len() * 4
    }
}

fn zeros(len: usize) -> Result<Vec<f32>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| invalid("递归状态分配失败"))?;
    data.resize(len, 0.0);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_isolated_across_partial_and_full_page_boundaries() {
        let mut cache = PagedKvCache::new(2, 9, 4).unwrap();
        for n in 0..3 {
            cache.append(&[n as f32, 1.0], &[2.0, n as f32]).unwrap();
        }
        let saved = cache.clone();
        for n in 3..9 {
            cache.append(&[n as f32, 1.0], &[2.0, n as f32]).unwrap();
        }
        assert!(cache.append(&[1.0; 2], &[1.0; 2]).is_err());
        cache.truncate(2).unwrap();
        cache.append(&[99.0; 2], &[88.0; 2]).unwrap();
        assert_eq!(saved.get(2).unwrap().0, &[2.0, 1.0]);
        assert_eq!(cache.get(2).unwrap().0, &[99.0; 2]);
        assert!(saved.get(3).is_err());
        assert!(cache.truncate(4).is_err());
        cache.truncate(0).unwrap();
        assert_eq!(cache.resident_bytes(), 0);
    }
}
