//! SafeTensors 权重加载
//!
//! 支持加载单文件和分片的 SafeTensors 模型权重。

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use half::f16;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::error::{Result, RsinferError};
use crate::tensor::{f16_bytes_to_f16, f16_slice_to_f32, Tensor};

/// 权重字典类型
pub type WeightMap = HashMap<String, LoadedTensor>;

#[derive(Clone, Debug)]
pub enum LoadedTensor {
    Tensor(Tensor),
    F16 { shape: Vec<usize>, data: Vec<f16> },
}

impl LoadedTensor {
    pub fn to_tensor(&self) -> Result<Tensor> {
        match self {
            LoadedTensor::Tensor(tensor) => Ok(tensor.clone()),
            LoadedTensor::F16 { shape, data } => {
                let f32_data = f16_slice_to_f32(data);
                Tensor::from_f32_slice(shape, &f32_data)
            }
        }
    }

    fn as_f16_2d(&self) -> Option<(&[usize], &[f16])> {
        match self {
            LoadedTensor::F16 { shape, data } if shape.len() == 2 => Some((shape, data)),
            _ => None,
        }
    }
}

/// SafeTensors 索引文件格式
#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// 从模型目录加载所有权重
///
/// 自动检测是单文件还是分片格式：
/// - 单文件: model.safetensors
/// - 分片: model.safetensors.index.json + model-*.safetensors
pub fn load_weights<P: AsRef<Path>>(model_dir: P) -> Result<WeightMap> {
    let model_dir = model_dir.as_ref();
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    if index_path.exists() {
        // 分片格式
        load_sharded_weights(model_dir, &index_path)
    } else if single_path.exists() {
        // 单文件格式
        load_single_weights(&single_path)
    } else {
        Err(RsinferError::WeightError(format!(
            "No safetensors file found in {:?}",
            model_dir
        )))
    }
}

/// 加载单文件 SafeTensors
fn load_single_weights<P: AsRef<Path>>(path: P) -> Result<WeightMap> {
    let data = fs::read(path.as_ref())?;
    let safetensors =
        SafeTensors::deserialize(&data).map_err(|e| RsinferError::SafeTensors(e.to_string()))?;

    let mut weights = HashMap::new();

    for (name, tensor_info) in safetensors.tensors() {
        let shape: Vec<usize> = tensor_info.shape().to_vec();
        let dtype = tensor_info.dtype();
        let data_bytes = tensor_info.data();

        let tensor = convert_tensor(&shape, dtype, data_bytes, &name)?;
        weights.insert(name.to_string(), tensor);
    }

    Ok(weights)
}

/// 加载分片 SafeTensors
fn load_sharded_weights<P: AsRef<Path>>(model_dir: P, index_path: P) -> Result<WeightMap> {
    let index_content = fs::read_to_string(index_path.as_ref())?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    // 收集所有需要加载的文件
    let shard_files: std::collections::HashSet<_> = index.weight_map.values().cloned().collect();

    let mut weights = HashMap::new();

    for shard_file in shard_files {
        let shard_path = model_dir.as_ref().join(&shard_file);
        let data = fs::read(&shard_path)?;
        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| RsinferError::SafeTensors(format!("{}: {}", shard_file, e)))?;

        for (name, tensor_info) in safetensors.tensors() {
            let shape: Vec<usize> = tensor_info.shape().to_vec();
            let dtype = tensor_info.dtype();
            let data_bytes = tensor_info.data();

            let tensor = convert_tensor(&shape, dtype, data_bytes, &name)?;
            weights.insert(name.to_string(), tensor);
        }
    }

    Ok(weights)
}

/// 将 SafeTensors 的原始数据转换为 Tensor
fn convert_tensor(
    shape: &[usize],
    dtype: safetensors::Dtype,
    data: &[u8],
    name: &str,
) -> Result<LoadedTensor> {
    match dtype {
        safetensors::Dtype::F32 => Tensor::from_f32_bytes(shape, data).map(LoadedTensor::Tensor),
        safetensors::Dtype::F16 => Ok(LoadedTensor::F16 {
            shape: shape.to_vec(),
            data: f16_bytes_to_f16(data),
        }),
        safetensors::Dtype::BF16 => Tensor::from_bf16_bytes(shape, data).map(LoadedTensor::Tensor),
        _ => Err(RsinferError::Unsupported(format!(
            "Unsupported dtype {:?} for tensor '{}'",
            dtype, name
        ))),
    }
}

/// 从权重字典中获取指定名称的权重
pub fn get_weight(weights: &WeightMap, name: &str) -> Result<Tensor> {
    weights
        .get(name)
        .ok_or_else(|| RsinferError::WeightError(format!("Weight '{}' not found", name)))?
        .to_tensor()
}

pub fn get_linear_weight_f16(weights: &WeightMap, name: &str) -> Result<(Vec<usize>, Vec<f16>)> {
    let weight = weights
        .get(name)
        .ok_or_else(|| RsinferError::WeightError(format!("Weight '{}' not found", name)))?;

    if let Some((shape, data)) = weight.as_f16_2d() {
        return Ok((shape.to_vec(), data.to_vec()));
    }

    let tensor = weight.to_tensor()?;
    let data = tensor
        .as_slice()
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    Ok((tensor.shape().to_vec(), data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_f32_tensor() {
        let shape = vec![2, 3];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensor = convert_tensor(&shape, safetensors::Dtype::F32, &bytes, "test")
            .unwrap()
            .to_tensor()
            .unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_convert_f16_keeps_f16_storage() {
        let shape = vec![2, 2];
        let data = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|f| f.to_bits().to_le_bytes())
            .collect();

        let tensor = convert_tensor(&shape, safetensors::Dtype::F16, &bytes, "test").unwrap();
        match tensor {
            LoadedTensor::F16 { shape, data } => {
                assert_eq!(shape, vec![2, 2]);
                assert_eq!(data.len(), 4);
                assert_eq!(data[2], f16::from_f32(3.0));
            }
            LoadedTensor::Tensor(_) => panic!("f16 tensor should stay in f16 storage"),
        }
    }
}
