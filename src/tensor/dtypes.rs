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
    let bf16_slice =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const bf16, bytes.len() / 2) };
    if bf16_slice.len() >= PARALLEL_CONVERT_THRESHOLD {
        bf16_slice.par_iter().map(|x| x.to_f32()).collect()
    } else {
        bf16_slice.iter().map(|x| x.to_f32()).collect()
    }
}

/// 将 f32 字节切片转换为 f32 向量
pub fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let f32_slice =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
    f32_slice.to_vec()
}
