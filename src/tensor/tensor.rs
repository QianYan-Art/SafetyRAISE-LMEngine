//! Tensor 结构体与基础操作
//!
//! 封装 ndarray 的动态维度数组，提供类型安全的张量操作。

use crate::error::{Result, RsinferError};
use ndarray::{s, ArrayD, IxDyn};

/// 张量数据结构
///
/// 封装 ndarray 的 ArrayD<f32>，统一使用 f32 进行计算。
/// 支持任意维度的张量。
#[derive(Clone, Debug)]
pub struct Tensor {
    /// 底层数据存储
    pub data: ArrayD<f32>,
}

impl Tensor {
    /// 从 f32 切片和形状创建张量
    pub fn from_f32_slice(shape: &[usize], data: &[f32]) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(RsinferError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![data.len()],
            });
        }
        let array = ArrayD::from_shape_vec(IxDyn(shape), data.to_vec())
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?;
        Ok(Self { data: array })
    }

    /// 从 f16 字节数据创建张量
    pub fn from_f16_bytes(shape: &[usize], bytes: &[u8]) -> Result<Self> {
        let f32_data = super::f16_bytes_to_f32(bytes);
        Self::from_f32_slice(shape, &f32_data)
    }

    /// 从 bf16 字节数据创建张量
    pub fn from_bf16_bytes(shape: &[usize], bytes: &[u8]) -> Result<Self> {
        let f32_data = super::bf16_bytes_to_f32(bytes);
        Self::from_f32_slice(shape, &f32_data)
    }

    /// 从 f32 字节数据创建张量
    pub fn from_f32_bytes(shape: &[usize], bytes: &[u8]) -> Result<Self> {
        let f32_data = super::f32_bytes_to_f32(bytes);
        Self::from_f32_slice(shape, &f32_data)
    }

    /// 创建全零张量
    pub fn zeros(shape: &[usize]) -> Self {
        let array = ArrayD::zeros(IxDyn(shape));
        Self { data: array }
    }

    /// 返回张量形状
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 返回张量维度数
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// 返回张量元素总数
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// 重塑张量形状
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.numel() {
            return Err(RsinferError::ShapeMismatch {
                expected: new_shape.to_vec(),
                actual: self.shape().to_vec(),
            });
        }
        let reshaped = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(new_shape))
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?;
        Ok(Self { data: reshaped })
    }

    /// 矩阵乘法 (2D x 2D)
    ///
    /// self: [m, k], other: [k, n] -> result: [m, n]
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(RsinferError::DimensionError(format!(
                "matmul requires 2D tensors, got {}D and {}D",
                self.ndim(),
                other.ndim()
            )));
        }
        let shape_a = self.shape();
        let shape_b = other.shape();
        if shape_a[1] != shape_b[0] {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![shape_a[1]],
                actual: vec![shape_b[0]],
            });
        }

        // 使用 ndarray 的 dot 进行矩阵乘法
        let a = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?;
        let b = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?;
        let result = a.dot(&b);
        Ok(Tensor {
            data: result.into_dyn(),
        })
    }

    /// 逐元素加法 (支持广播)
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data + &other.data;
        Ok(Tensor { data: result })
    }

    /// 逐元素乘法 (支持广播)
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data * &other.data;
        Ok(Tensor { data: result })
    }

    /// 标量乘法
    pub fn scale(&self, scalar: f32) -> Tensor {
        Tensor {
            data: &self.data * scalar,
        }
    }

    /// 转置 (2D 张量)
    pub fn t(&self) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(RsinferError::DimensionError(format!(
                "transpose requires 2D tensor, got {}D",
                self.ndim()
            )));
        }
        let transposed = self.data.t().into_owned().into_dyn();
        Ok(Tensor { data: transposed })
    }

    /// 沿指定维度切片
    ///
    /// 对于 2D 张量，返回指定行的子集
    pub fn slice_rows(&self, start: usize, end: usize) -> Result<Tensor> {
        if self.ndim() < 1 {
            return Err(RsinferError::DimensionError(
                "Cannot slice 0D tensor".into(),
            ));
        }
        let sliced = self.data.slice(s![start..end, ..]).to_owned().into_dyn();
        Ok(Tensor { data: sliced })
    }

    /// 获取最后一个元素 (用于生成)
    pub fn last_row(&self) -> Result<Tensor> {
        if self.ndim() < 1 {
            return Err(RsinferError::DimensionError(
                "Cannot get last row of 0D tensor".into(),
            ));
        }
        let shape = self.shape();
        let last_idx = shape[0] - 1;

        // 根据维度数动态构造切片索引
        let sliced = match self.ndim() {
            1 => self
                .data
                .slice(s![last_idx..last_idx + 1])
                .to_owned()
                .into_dyn(),
            2 => self
                .data
                .slice(s![last_idx..last_idx + 1, ..])
                .to_owned()
                .into_dyn(),
            3 => self
                .data
                .slice(s![last_idx..last_idx + 1, .., ..])
                .to_owned()
                .into_dyn(),
            _ => {
                return Err(RsinferError::DimensionError(
                    "Unsupported dimension for last_row".into(),
                ))
            }
        };
        Ok(Tensor { data: sliced })
    }

    /// 转换为 1D 视图 (用于采样)
    pub fn to_1d(&self) -> Result<Tensor> {
        let flat = self
            .data
            .clone()
            .into_shape_with_order(IxDyn(&[self.numel()]))
            .map_err(|e| RsinferError::DimensionError(e.to_string()))?;
        Ok(Tensor { data: flat })
    }

    /// 获取数据引用
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice().unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_f32_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_f32_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_f32_slice(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }
}
