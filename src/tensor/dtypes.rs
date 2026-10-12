//! 数据类型定义
//!
//! 支持 f32、f16、bf16 数据类型的转换。

use half::{bf16, f16};
use rayon::prelude::*;

const PARALLEL_CONVERT_THRESHOLD: usize = 1 << 20;

/// 支持的数值类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    /// 返回每种类型的字节大小
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
        }
    }
}

/// 将 f16 字节切片转换为 f32 向量
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    f16_slice_to_f32(&f16_bytes_to_f16(bytes))
}

/// 将 f16 字节切片转换为 f16 向量
pub fn f16_bytes_to_f16(bytes: &[u8]) -> Vec<f16> {
    debug_assert_eq!(bytes.len() % 2, 0);
    if bytes.len() / 2 >= PARALLEL_CONVERT_THRESHOLD {
        bytes
            .par_chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    }
}

/// 将 f16 切片转换为 f32 向量
pub fn f16_slice_to_f32(values: &[f16]) -> Vec<f32> {
    if values.len() >= PARALLEL_CONVERT_THRESHOLD {
        values.par_iter().map(|x| x.to_f32()).collect()
    } else {
        values.iter().map(|x| x.to_f32()).collect()
    }
}

/// 将 bf16 字节切片转换为 f32 向量
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 2, 0);
    let convert = |b: &[u8]| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
    if bytes.len() / 2 >= PARALLEL_CONVERT_THRESHOLD {
        bytes.par_chunks_exact(2).map(convert).collect()
    } else {
        bytes.chunks_exact(2).map(convert).collect()
    }
}

/// 将 f32 字节切片转换为 f32 向量
pub fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unaligned_little_endian_weights_are_safe() {
        let f32_data = [0xff, 0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0];
        assert_eq!(f32_bytes_to_f32(&f32_data[1..]), vec![1.0, -2.0]);
        let bf16_data = [0xff, 0x80, 0x3f, 0, 0xc0];
        assert_eq!(bf16_bytes_to_f32(&bf16_data[1..]), vec![1.0, -2.0]);
    }
}
