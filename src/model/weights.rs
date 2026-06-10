//! SafeTensors 权重加载
//!
//! 支持加载单文件和分片的 SafeTensors 模型权重。

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
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

#[derive(Debug, Deserialize)]
struct SafetensorsHeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
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

pub fn load_weights_filtered<P, F>(model_dir: P, keep: F) -> Result<WeightMap>
where
    P: AsRef<Path>,
    F: Fn(&str) -> bool,
{
    let model_dir = model_dir.as_ref();
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    if index_path.exists() {
        load_sharded_weights_filtered(model_dir, &index_path, &keep)
    } else if single_path.exists() {
        load_single_weights_filtered(&single_path, &keep)
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

fn load_single_weights_filtered<P, F>(path: P, keep: &F) -> Result<WeightMap>
where
    P: AsRef<Path>,
    F: Fn(&str) -> bool,
{
    let mut file = fs::File::open(path.as_ref())?;
    let (data_start, header) = read_safetensors_header(&mut file, path.as_ref())?;
    let selected_names: Vec<String> = header.keys().filter(|name| keep(name)).cloned().collect();
    let mut weights = HashMap::new();
    load_selected_tensors(
        &mut file,
        path.as_ref(),
        data_start,
        &header,
        &selected_names,
        &mut weights,
    )?;
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

fn load_sharded_weights_filtered<P, F>(model_dir: P, index_path: P, keep: &F) -> Result<WeightMap>
where
    P: AsRef<Path>,
    F: Fn(&str) -> bool,
{
    let index_content = fs::read_to_string(index_path.as_ref())?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    let mut shard_map: HashMap<String, HashSet<String>> = HashMap::new();
    for (name, shard_file) in index.weight_map {
        if keep(&name) {
            shard_map.entry(shard_file).or_default().insert(name);
        }
    }

    let mut weights = HashMap::new();
    for (shard_file, selected_names) in shard_map {
        let shard_path = model_dir.as_ref().join(&shard_file);
        let mut file = fs::File::open(&shard_path)?;
        let (data_start, header) = read_safetensors_header(&mut file, &shard_path)?;
        let selected_names: Vec<String> = selected_names.into_iter().collect();
        load_selected_tensors(
            &mut file,
            &shard_path,
            data_start,
            &header,
            &selected_names,
            &mut weights,
        )?;
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

fn read_safetensors_header(
    file: &mut fs::File,
    path: &Path,
) -> Result<(u64, HashMap<String, SafetensorsHeaderEntry>)> {
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    let header_len_usize = usize::try_from(header_len).map_err(|_| {
        RsinferError::WeightError(format!(
            "Safetensors header too large in {}",
            path.display()
        ))
    })?;
    let mut header_bytes = vec![0u8; header_len_usize];
    file.read_exact(&mut header_bytes)?;

    let header_value: serde_json::Value = serde_json::from_slice(&header_bytes)?;
    let header_object = header_value.as_object().ok_or_else(|| {
        RsinferError::SafeTensors(format!(
            "Invalid safetensors header object in {}",
            path.display()
        ))
    })?;

    let mut header = HashMap::new();
    for (name, value) in header_object {
        if name == "__metadata__" {
            continue;
        }
        let entry: SafetensorsHeaderEntry = serde_json::from_value(value.clone())?;
        header.insert(name.clone(), entry);
    }

    Ok((8 + header_len, header))
}

fn load_selected_tensors(
    file: &mut fs::File,
    path: &Path,
    data_start: u64,
    header: &HashMap<String, SafetensorsHeaderEntry>,
    selected_names: &[String],
    weights: &mut WeightMap,
) -> Result<()> {
    let mut selected_entries = Vec::with_capacity(selected_names.len());
    for name in selected_names {
        let entry = header.get(name).ok_or_else(|| {
            RsinferError::WeightError(format!(
                "Tensor '{}' not found in safetensors header {}",
                name,
                path.display()
            ))
        })?;
        selected_entries.push((name.as_str(), entry));
    }
    selected_entries.sort_by_key(|(_, entry)| entry.data_offsets[0]);

    for (name, entry) in selected_entries {
        let start = data_start + entry.data_offsets[0];
        let len = entry.data_offsets[1]
            .checked_sub(entry.data_offsets[0])
            .ok_or_else(|| {
                RsinferError::SafeTensors(format!(
                    "Invalid data offsets for tensor '{}' in {}",
                    name,
                    path.display()
                ))
            })?;
        let len = usize::try_from(len).map_err(|_| {
            RsinferError::WeightError(format!(
                "Tensor '{}' is too large to load on this platform",
                name
            ))
        })?;

        file.seek(SeekFrom::Start(start))?;
        let mut data = vec![0u8; len];
        file.read_exact(&mut data)?;

        let dtype = parse_safetensors_dtype(&entry.dtype, name)?;
        let tensor = convert_tensor(&entry.shape, dtype, &data, name)?;
        weights.insert(name.to_string(), tensor);
    }

    Ok(())
}

fn parse_safetensors_dtype(dtype: &str, name: &str) -> Result<safetensors::Dtype> {
    match dtype {
        "F16" => Ok(safetensors::Dtype::F16),
        "F32" => Ok(safetensors::Dtype::F32),
        "BF16" => Ok(safetensors::Dtype::BF16),
        _ => Err(RsinferError::Unsupported(format!(
            "Unsupported dtype '{}' for tensor '{}'",
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
