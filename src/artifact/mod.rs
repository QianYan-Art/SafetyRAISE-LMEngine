//! 本地模型权重资源层。
//!
//! 本模块以只读内存映射保存 SafeTensors 与 GGUF 权重，提供统一的形状、
//! dtype、行读取和矩阵乘合同。映射建立后，调用方必须保证源文件在整个
//! `WeightStore` 生命周期内不被修改；文件变更不会被本模块同步或检测。
//!
//! GGUF 量化布局改编自 ggml（MIT License），固定参考提交：
//! `3612c0af93f2cfc8f048295300a61187723b12ae`。详情见 [`quant`] 模块。

mod quant;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};

use crate::error::{Result, RsinferError};

const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_METADATA_ENTRIES: u64 = 1_000_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_DIMS: u32 = 64;
const MAX_ARRAY_ITEMS: u64 = 1_000_000;

/// 支持的权重存储格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q4K,
    Q6K,
}

#[derive(Debug)]
enum BackingStorage {
    Mmap(Mmap),
    Owned(Arc<[u8]>),
}

#[derive(Debug)]
struct Mapping {
    storage: BackingStorage,
    path: PathBuf,
}

impl Mapping {
    fn bytes(&self) -> &[u8] {
        match &self.storage {
            BackingStorage::Mmap(mmap) => mmap,
            BackingStorage::Owned(bytes) => bytes,
        }
    }
}

#[derive(Debug)]
struct TensorSpec {
    backing: Arc<Mapping>,
    offset: usize,
    len: usize,
    /// 逻辑形状。映射视图可以与物理形状相同，也可以插入大小为 1 的维度。
    shape: Vec<usize>,
    /// 原始字节对应的物理形状。
    physical_shape: Vec<usize>,
    format: WeightFormat,
    /// 逻辑行到物理行的映射；`None` 表示行顺序不变。
    row_map: Option<Arc<[usize]>>,
    /// 逻辑列到物理列的映射；`None` 表示列顺序不变。
    col_map: Option<Arc<[usize]>>,
}

struct StoreInner {
    tensors: HashMap<String, Arc<TensorSpec>>,
    names: Vec<String>,
    metadata: Map<String, Value>,
    hf_config: Option<Value>,
}

/// 只读权重仓库。
pub struct WeightStore {
    inner: Arc<StoreInner>,
}

/// 一个共享底层映射的权重张量句柄。
#[derive(Clone)]
pub struct WeightTensor {
    spec: Arc<TensorSpec>,
}

impl WeightStore {
    /// 打开目录中的 SafeTensors，或打开 GGUF v3 文件。
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_dir() {
            open_safetensors_dir(path)
        } else if path.is_file() {
            open_gguf_file(path)
        } else {
            Err(RsinferError::WeightError(format!(
                "权重路径不存在或不是普通文件/目录：{}",
                path.display()
            )))
        }
    }

    /// 返回 GGUF 元数据；SafeTensors 返回空对象。
    pub fn metadata(&self) -> &Map<String, Value> {
        &self.inner.metadata
    }

    /// 返回目录中的 Hugging Face `config.json`，GGUF 返回 `None`。
    pub fn hf_config(&self) -> Option<&Value> {
        self.inner.hf_config.as_ref()
    }

    /// 获取指定名称的共享张量句柄。
    pub fn get(&self, name: &str) -> Result<WeightTensor> {
        self.inner
            .tensors
            .get(name)
            .cloned()
            .map(|spec| WeightTensor { spec })
            .ok_or_else(|| RsinferError::WeightError(format!("找不到权重张量：{name}")))
    }

    /// 迭代张量名称。名称顺序在一次打开期间稳定。
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.inner.names.iter().map(String::as_str)
    }

    /// 从内存中的张量句柄组装一个只读权重仓库。
    pub fn from_tensors(config: Value, tensors: BTreeMap<String, WeightTensor>) -> Result<Self> {
        let specs = tensors
            .into_iter()
            .map(|(name, tensor)| (name, tensor.spec))
            .collect();
        make_store(specs, empty_metadata(), Some(config))
    }
}

impl WeightTensor {
    /// 从 f32 数据创建内存权重。数据会被编码为小端 F32，并与句柄共享所有权。
    pub fn from_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let elements = shape_elements(&shape, "<memory>")?;
        if elements != data.len() {
            return Err(RsinferError::ShapeMismatch {
                expected: shape,
                actual: vec![data.len()],
            });
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(RsinferError::WeightError(
                "内存 F32 权重包含非有限值".into(),
            ));
        }
        let mut bytes = Vec::with_capacity(
            elements
                .checked_mul(4)
                .ok_or_else(|| RsinferError::WeightError("内存 F32 权重字节长度溢出".into()))?,
        );
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let backing = Arc::new(Mapping {
            storage: BackingStorage::Owned(Arc::<[u8]>::from(bytes)),
            path: PathBuf::from("<memory>"),
        });
        Ok(Self {
            spec: Arc::new(TensorSpec {
                backing,
                offset: 0,
                len: elements * 4,
                physical_shape: shape.clone(),
                shape,
                format: WeightFormat::F32,
                row_map: None,
                col_map: None,
            }),
        })
    }

    /// 统一后的行主序形状；GGUF 原始维度已反转。
    pub fn shape(&self) -> &[usize] {
        &self.spec.shape
    }

    pub fn format(&self) -> WeightFormat {
        self.spec.format
    }

    pub fn byte_len(&self) -> usize {
        self.spec.len
    }

    /// 返回张量是否可按其原始字节连续读取。
    ///
    /// 行/列重排视图保留原始 backing 但不能把逻辑顺序作为一段连续字节
    /// 交给原生后端，因此会返回 `false`。
    pub fn is_contiguous(&self) -> bool {
        self.spec.row_map.is_none() && self.spec.col_map.is_none()
    }

    /// 返回当前张量视图的稳定标识。
    ///
    /// 同一个底层视图的克隆句柄共享标识；不同视图即使暂时拥有相同的
    /// 字节、形状和格式，也拥有不同标识。标识只保证在该句柄及其克隆仍
    /// 存活期间稳定，适合用作进程内权重缓存键，不应序列化或跨进程保存。
    pub fn view_id(&self) -> usize {
        Arc::as_ptr(&self.spec) as usize
    }

    /// 返回已经通过边界校验的原始张量字节，只读且不包含文件头。
    pub fn bytes(&self) -> &[u8] {
        &self.spec.backing.bytes()[self.spec.offset..self.spec.offset + self.spec.len]
    }

    /// 按逻辑行建立惰性重排视图。映射中的每个元素是该逻辑行在原始
    /// 物理张量中的行号；不会复制或反量化 backing。
    pub(crate) fn permute_rows(&self, logical_to_physical: Vec<usize>) -> Result<Self> {
        let (rows, _) = shape_rows_width(&self.spec.shape)?;
        if logical_to_physical.len() != rows {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![rows],
                actual: vec![logical_to_physical.len()],
            });
        }
        let (physical_rows, _) = shape_rows_width(&self.spec.physical_shape)?;
        let row_map = compose_view_map(
            self.spec.row_map.as_deref(),
            &logical_to_physical,
            physical_rows,
            "行",
        )?;
        Ok(Self {
            spec: Arc::new(TensorSpec {
                backing: Arc::clone(&self.spec.backing),
                offset: self.spec.offset,
                len: self.spec.len,
                shape: self.spec.shape.clone(),
                physical_shape: self.spec.physical_shape.clone(),
                format: self.spec.format,
                row_map: Some(Arc::<[usize]>::from(row_map)),
                col_map: self.spec.col_map.clone(),
            }),
        })
    }

    /// 按逻辑列建立惰性重排视图。映射中的每个元素是该逻辑列在原始
    /// 物理张量中的列号；不会复制或反量化 backing。
    pub(crate) fn permute_columns(&self, logical_to_physical: Vec<usize>) -> Result<Self> {
        let (_, width) = shape_rows_width(&self.spec.shape)?;
        if logical_to_physical.len() != width {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![width],
                actual: vec![logical_to_physical.len()],
            });
        }
        let (_, physical_width) = shape_rows_width(&self.spec.physical_shape)?;
        let col_map = compose_view_map(
            self.spec.col_map.as_deref(),
            &logical_to_physical,
            physical_width,
            "列",
        )?;
        Ok(Self {
            spec: Arc::new(TensorSpec {
                backing: Arc::clone(&self.spec.backing),
                offset: self.spec.offset,
                len: self.spec.len,
                shape: self.spec.shape.clone(),
                physical_shape: self.spec.physical_shape.clone(),
                format: self.spec.format,
                row_map: self.spec.row_map.clone(),
                col_map: Some(Arc::<[usize]>::from(col_map)),
            }),
        })
    }

    /// 在不改变行主序元素数与最后一维宽度的前提下建立逻辑 reshape 视图。
    /// 该操作只用于恢复 GGUF 卷积权重的大小为 1 维度，不复制 backing。
    pub(crate) fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        let old = shape_rows_width(&self.spec.shape)?;
        let new = shape_rows_width(&shape)?;
        if old != new {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![old.0, old.1],
                actual: vec![new.0, new.1],
            });
        }
        Ok(Self {
            spec: Arc::new(TensorSpec {
                backing: Arc::clone(&self.spec.backing),
                offset: self.spec.offset,
                len: self.spec.len,
                shape,
                physical_shape: self.spec.physical_shape.clone(),
                format: self.spec.format,
                row_map: self.spec.row_map.clone(),
                col_map: self.spec.col_map.clone(),
            }),
        })
    }

    /// 将一行转成 f32。高维张量按最后一维展平为行，向量视作一行。
    pub fn read_row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        let (rows, width) = self.rows_width()?;
        if row >= rows {
            return Err(RsinferError::WeightError(format!(
                "行索引越界：row={row}，rows={rows}"
            )));
        }
        if out.len() != width {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![width],
                actual: vec![out.len()],
            });
        }
        let (physical_rows, physical_width) = shape_rows_width(&self.spec.physical_shape)?;
        if self.is_contiguous() && rows == physical_rows && width == physical_width {
            return self.read_physical_row(row, out);
        }
        let physical_row = self
            .spec
            .row_map
            .as_ref()
            .map(|map| map[row])
            .unwrap_or(row);
        if physical_row >= physical_rows {
            return Err(RsinferError::WeightError("行映射越过物理张量边界".into()));
        }
        let mut physical = vec![0.0f32; physical_width];
        self.read_physical_row(physical_row, &mut physical)?;
        if let Some(columns) = &self.spec.col_map {
            if columns.len() != width {
                return Err(RsinferError::WeightError(
                    "列映射宽度与逻辑张量不一致".into(),
                ));
            }
            for (dst, &src) in out.iter_mut().zip(columns.iter()) {
                *dst = *physical
                    .get(src)
                    .ok_or_else(|| RsinferError::WeightError("列映射越过物理张量边界".into()))?;
            }
        } else if physical_width == width {
            out.copy_from_slice(&physical);
        } else {
            return Err(RsinferError::WeightError(
                "逻辑与物理张量的最后一维不一致且缺少列映射".into(),
            ));
        }
        Ok(())
    }

    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let (rows, width) = self.rows_width()?;
        let total = rows
            .checked_mul(width)
            .ok_or_else(|| RsinferError::WeightError("f32 输出长度溢出".into()))?;
        let mut out = vec![0.0f32; total];
        for row in 0..rows {
            let start = row * width;
            self.read_row(row, &mut out[start..start + width])?;
        }
        Ok(out)
    }

    /// 计算 `input[rows, in] * weight[out, in]^T`，量化权重逐行解码。
    pub fn matmul(&self, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        let (weight_rows, width) = self.rows_width()?;
        let input_len = rows
            .checked_mul(width)
            .ok_or_else(|| RsinferError::WeightError("matmul 输入长度溢出".into()))?;
        if input.len() != input_len {
            return Err(RsinferError::ShapeMismatch {
                expected: vec![rows, width],
                actual: vec![input.len()],
            });
        }
        let output_len = rows
            .checked_mul(weight_rows)
            .ok_or_else(|| RsinferError::WeightError("matmul 输出长度溢出".into()))?;
        let mut output = vec![0.0f32; output_len];
        let mut weight_row = vec![0.0f32; width];
        for out_col in 0..weight_rows {
            self.read_row(out_col, &mut weight_row)?;
            for batch in 0..rows {
                let input_row = &input[batch * width..(batch + 1) * width];
                let mut sum = 0.0f32;
                for (x, w) in input_row.iter().zip(&weight_row) {
                    sum += x * w;
                }
                output[batch * weight_rows + out_col] = sum;
            }
        }
        Ok(output)
    }

    fn rows_width(&self) -> Result<(usize, usize)> {
        shape_rows_width(&self.spec.shape)
    }

    fn read_physical_row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        let (rows, width) = shape_rows_width(&self.spec.physical_shape)?;
        if row >= rows || out.len() != width {
            return Err(RsinferError::WeightError(
                "物理张量行索引或宽度不匹配".into(),
            ));
        }
        let row_bytes = self
            .spec
            .len
            .checked_div(rows)
            .ok_or_else(|| RsinferError::WeightError("张量行字节长度除法溢出".into()))?;
        let start = row
            .checked_mul(row_bytes)
            .ok_or_else(|| RsinferError::WeightError("行偏移溢出".into()))?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| RsinferError::WeightError("行结束偏移溢出".into()))?;
        let bytes = &self.bytes()[start..end];
        match self.spec.format {
            WeightFormat::F32 => read_f32_row(bytes, out),
            WeightFormat::F16 => read_f16_row(bytes, out),
            WeightFormat::BF16 => read_bf16_row(bytes, out),
            WeightFormat::Q4_0 => quant::q4_0_row(bytes, out),
            WeightFormat::Q8_0 => quant::q8_0_row(bytes, out),
            WeightFormat::Q4K => quant::q4_k_row(bytes, out),
            WeightFormat::Q6K => quant::q6_k_row(bytes, out),
        }
    }
}

fn shape_rows_width(shape: &[usize]) -> Result<(usize, usize)> {
    let width = *shape
        .last()
        .ok_or_else(|| RsinferError::WeightError("张量形状不能为空".into()))?;
    let mut rows = 1usize;
    for &dim in &shape[..shape.len() - 1] {
        rows = rows
            .checked_mul(dim)
            .ok_or_else(|| RsinferError::WeightError("张量行数溢出".into()))?;
    }
    Ok((rows, width))
}

fn compose_view_map(
    existing: Option<&[usize]>,
    mapping: &[usize],
    physical_len: usize,
    axis: &str,
) -> Result<Vec<usize>> {
    let logical_len = existing.map_or(physical_len, |map| map.len());
    let mut composed = Vec::with_capacity(mapping.len());
    for &index in mapping {
        if index >= logical_len {
            return Err(RsinferError::WeightError(format!(
                "{axis}映射索引越过逻辑张量边界"
            )));
        }
        let physical = existing.map_or(index, |map| map[index]);
        if physical >= physical_len {
            return Err(RsinferError::WeightError(format!(
                "{axis}映射索引越过物理张量边界"
            )));
        }
        composed.push(physical);
    }
    Ok(composed)
}

fn read_f32_row(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = out
        .len()
        .checked_mul(4)
        .ok_or_else(|| RsinferError::WeightError("F32 行长度溢出".into()))?;
    if bytes.len() != expected {
        return Err(RsinferError::WeightError("F32 行字节长度不匹配".into()));
    }
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn read_f16_row(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = out
        .len()
        .checked_mul(2)
        .ok_or_else(|| RsinferError::WeightError("F16 行长度溢出".into()))?;
    if bytes.len() != expected {
        return Err(RsinferError::WeightError("F16 行字节长度不匹配".into()));
    }
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *dst = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
    }
    Ok(())
}

fn read_bf16_row(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = out
        .len()
        .checked_mul(2)
        .ok_or_else(|| RsinferError::WeightError("BF16 行长度溢出".into()))?;
    if bytes.len() != expected {
        return Err(RsinferError::WeightError("BF16 行字节长度不匹配".into()));
    }
    for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *dst = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
    }
    Ok(())
}

fn map_readonly(path: &Path) -> Result<Arc<Mapping>> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let len = usize::try_from(metadata.len()).map_err(|_| {
        RsinferError::WeightError(format!("文件过大，无法在当前平台映射：{}", path.display()))
    })?;
    if len == 0 {
        return Err(RsinferError::WeightError(format!(
            "权重文件为空：{}",
            path.display()
        )));
    }
    // SAFETY（安全性说明）：映射仅用于只读切片；文件句柄在 mmap 建立后仍由操作系统保证映射有效。
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    if mmap.len() != len {
        return Err(RsinferError::WeightError(format!(
            "映射长度与文件长度不一致：{}",
            path.display()
        )));
    }
    Ok(Arc::new(Mapping {
        storage: BackingStorage::Mmap(mmap),
        path: path.to_path_buf(),
    }))
}

fn empty_metadata() -> Map<String, Value> {
    Map::new()
}

fn make_store(
    specs: Vec<(String, Arc<TensorSpec>)>,
    metadata: Map<String, Value>,
    hf_config: Option<Value>,
) -> Result<WeightStore> {
    let mut tensors = HashMap::with_capacity(specs.len());
    let mut names = Vec::with_capacity(specs.len());
    for (name, spec) in specs {
        if tensors.insert(name.clone(), spec).is_some() {
            return Err(RsinferError::WeightError(format!("重复的张量名称：{name}")));
        }
        names.push(name);
    }
    names.sort();
    Ok(WeightStore {
        inner: Arc::new(StoreInner {
            tensors,
            names,
            metadata,
            hf_config,
        }),
    })
}

fn open_safetensors_dir(dir: &Path) -> Result<WeightStore> {
    let index_path = dir.join("model.safetensors.index.json");
    let single_path = dir.join("model.safetensors");
    let hf_config = read_config(dir)?;
    if index_path.is_file() && single_path.is_file() {
        return Err(RsinferError::WeightError(
            "目录同时包含单文件和分片 SafeTensors".into(),
        ));
    }
    if index_path.is_file() {
        open_sharded_safetensors(dir, &index_path, hf_config)
    } else if single_path.is_file() {
        let backing = map_readonly(&single_path)?;
        let names = parse_safetensors_header(&backing)?;
        let specs = names
            .into_iter()
            .map(|(name, spec)| (name, Arc::new(spec)))
            .collect();
        make_store(specs, empty_metadata(), hf_config)
    } else {
        Err(RsinferError::WeightError(format!(
            "目录中找不到 model.safetensors 或 model.safetensors.index.json：{}",
            dir.display()
        )))
    }
}

fn read_config(dir: &Path) -> Result<Option<Value>> {
    let path = dir.join("config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let metadata = fs::metadata(&path)?;
    let len = usize::try_from(metadata.len())
        .map_err(|_| RsinferError::WeightError(format!("config.json 过大：{}", path.display())))?;
    if len > MAX_JSON_BYTES {
        return Err(RsinferError::WeightError(format!(
            "config.json 超过大小限制：{}",
            path.display()
        )));
    }
    let bytes = fs::read(&path)?;
    Ok(Some(serde_json::from_slice(&bytes)?))
}

fn open_sharded_safetensors(
    dir: &Path,
    index_path: &Path,
    hf_config: Option<Value>,
) -> Result<WeightStore> {
    let index_bytes = read_limited(index_path, MAX_JSON_BYTES)?;
    let index = parse_no_dup_object(&index_bytes)
        .map_err(|e| RsinferError::SafeTensors(format!("分片索引 JSON：{e}")))?;
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| RsinferError::WeightError("分片索引缺少 object 类型 weight_map".into()))?;
    if weight_map.is_empty() {
        return Err(RsinferError::WeightError("分片索引 weight_map 为空".into()));
    }

    let mut grouped: HashMap<String, HashSet<String>> = HashMap::new();
    for (name, shard) in weight_map {
        let shard = shard.as_str().ok_or_else(|| {
            RsinferError::WeightError(format!("张量 {name} 的分片路径不是字符串"))
        })?;
        validate_relative_shard_path(shard)?;
        grouped
            .entry(shard.to_owned())
            .or_default()
            .insert(name.clone());
    }

    let mut specs = Vec::new();
    for (shard, expected) in grouped {
        let shard_path = dir.join(&shard);
        if !shard_path.is_file() {
            return Err(RsinferError::WeightError(format!(
                "索引引用的分片不存在：{}",
                shard_path.display()
            )));
        }
        let backing = map_readonly(&shard_path)?;
        let entries = parse_safetensors_header(&backing)?;
        let actual: HashSet<&str> = entries.iter().map(|(name, _)| name.as_str()).collect();
        for name in &expected {
            if !actual.contains(name.as_str()) {
                return Err(RsinferError::WeightError(format!(
                    "索引中的张量 {name} 不在分片 {shard} 中"
                )));
            }
        }
        for (name, spec) in entries {
            if !expected.contains(&name) {
                return Err(RsinferError::WeightError(format!(
                    "分片 {shard} 中的张量 {name} 未出现在索引"
                )));
            }
            specs.push((name, Arc::new(spec)));
        }
    }
    make_store(specs, empty_metadata(), hf_config)
}

fn validate_relative_shard_path(path: &str) -> Result<()> {
    let p = Path::new(path);
    if p.as_os_str().is_empty() || p.is_absolute() {
        return Err(RsinferError::WeightError(format!("非法分片路径：{path}")));
    }
    for component in p.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(RsinferError::WeightError(format!(
                    "分片路径穿越模型目录：{path}"
                )))
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct SafetensorsHeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

fn parse_safetensors_header(backing: &Arc<Mapping>) -> Result<Vec<(String, TensorSpec)>> {
    let data = backing.bytes();
    if data.len() < 8 {
        return Err(RsinferError::SafeTensors(format!(
            "SafeTensors 文件截断：{}",
            backing.path.display()
        )));
    }
    let header_len_u64 = u64::from_le_bytes(data[..8].try_into().unwrap());
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| RsinferError::SafeTensors("SafeTensors header 长度超过平台 usize".into()))?;
    if header_len > MAX_HEADER_BYTES {
        return Err(RsinferError::SafeTensors(format!(
            "SafeTensors header 超过大小限制：{}",
            header_len
        )));
    }
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or_else(|| RsinferError::SafeTensors("SafeTensors header 偏移溢出".into()))?;
    let header_end = data_start;
    let header_bytes = data
        .get(8..header_end)
        .ok_or_else(|| RsinferError::SafeTensors("SafeTensors header 截断".into()))?;
    let header = parse_no_dup_object(header_bytes)
        .map_err(|e| RsinferError::SafeTensors(format!("SafeTensors header JSON：{e}")))?;
    let mut entries = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let object = value.as_object().ok_or_else(|| {
            RsinferError::SafeTensors(format!("张量 {name} 的 header 项不是 object"))
        })?;
        let dtype = object
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| RsinferError::SafeTensors(format!("张量 {name} 缺少 dtype")))?
            .to_owned();
        let shape_value = object
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| RsinferError::SafeTensors(format!("张量 {name} 缺少 shape")))?;
        if shape_value.is_empty() {
            return Err(RsinferError::SafeTensors(format!(
                "张量 {name} 的 shape 不能为空"
            )));
        }
        let mut shape = Vec::with_capacity(shape_value.len());
        for dim in shape_value {
            let dim = dim.as_u64().ok_or_else(|| {
                RsinferError::SafeTensors(format!("张量 {name} 的 shape 不是非负整数"))
            })?;
            let dim = usize::try_from(dim).map_err(|_| {
                RsinferError::SafeTensors(format!("张量 {name} 的 shape 超过平台 usize"))
            })?;
            if dim == 0 {
                return Err(RsinferError::SafeTensors(format!(
                    "张量 {name} 包含零维 shape"
                )));
            }
            shape.push(dim);
        }
        let offsets = object
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| RsinferError::SafeTensors(format!("张量 {name} 缺少 data_offsets")))?;
        if offsets.len() != 2 {
            return Err(RsinferError::SafeTensors(format!(
                "张量 {name} 的 data_offsets 长度不是 2"
            )));
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| RsinferError::SafeTensors(format!("张量 {name} 的起始偏移非法")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| RsinferError::SafeTensors(format!("张量 {name} 的结束偏移非法")))?;
        entries.push((
            name,
            SafetensorsHeaderEntry {
                dtype,
                shape,
                start,
                end,
            },
        ));
    }

    let mut ranges: Vec<(u64, u64, &str)> = entries
        .iter()
        .map(|(name, entry)| (entry.start, entry.end, name.as_str()))
        .collect();
    ranges.sort_by_key(|(start, _, _)| *start);
    let payload_len = u64::try_from(data.len() - data_start)
        .map_err(|_| RsinferError::SafeTensors("SafeTensors payload 长度超过 u64".into()))?;
    let mut previous_end = 0u64;
    for (start, end, name) in &ranges {
        if end < start || *end > payload_len {
            return Err(RsinferError::SafeTensors(format!(
                "张量 {name} 的 data_offsets 越界或倒置"
            )));
        }
        if *start < previous_end {
            return Err(RsinferError::SafeTensors(format!(
                "张量 {name} 的数据区与其他张量重叠"
            )));
        }
        previous_end = *end;
    }

    let mut specs = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        let format = parse_safetensors_format(&entry.dtype, &name)?;
        let elements = shape_elements(&entry.shape, &name)?;
        let expected_len = plain_byte_len(format, elements, &name)?;
        let actual_len = usize::try_from(entry.end - entry.start).map_err(|_| {
            RsinferError::SafeTensors(format!("张量 {name} 数据长度超过平台 usize"))
        })?;
        if actual_len != expected_len {
            return Err(RsinferError::SafeTensors(format!(
                "张量 {name} 数据长度错误：期望 {expected_len}，实际 {actual_len}"
            )));
        }
        let rel_start = usize::try_from(entry.start).map_err(|_| {
            RsinferError::SafeTensors(format!("张量 {name} 起始偏移超过平台 usize"))
        })?;
        let offset = data_start
            .checked_add(rel_start)
            .ok_or_else(|| RsinferError::SafeTensors("张量绝对偏移溢出".into()))?;
        specs.push((
            name,
            TensorSpec {
                backing: Arc::clone(backing),
                offset,
                len: actual_len,
                physical_shape: entry.shape.clone(),
                shape: entry.shape,
                format,
                row_map: None,
                col_map: None,
            },
        ));
    }
    Ok(specs)
}

fn parse_safetensors_format(dtype: &str, name: &str) -> Result<WeightFormat> {
    match dtype {
        "F32" => Ok(WeightFormat::F32),
        "F16" => Ok(WeightFormat::F16),
        "BF16" => Ok(WeightFormat::BF16),
        _ => Err(RsinferError::Unsupported(format!(
            "SafeTensors 张量 {name} 使用不支持的 dtype：{dtype}"
        ))),
    }
}

fn shape_elements(shape: &[usize], name: &str) -> Result<usize> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(RsinferError::WeightError(format!(
            "张量 {name} 的 shape 非法"
        )));
    }
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| RsinferError::WeightError(format!("张量 {name} 的元素数量溢出")))
    })
}

fn plain_byte_len(format: WeightFormat, elements: usize, name: &str) -> Result<usize> {
    let bytes = match format {
        WeightFormat::F32 => 4,
        WeightFormat::F16 | WeightFormat::BF16 => 2,
        _ => {
            return Err(RsinferError::Unsupported(format!(
                "SafeTensors 张量 {name} 不能使用量化 dtype"
            )))
        }
    };
    elements
        .checked_mul(bytes)
        .ok_or_else(|| RsinferError::WeightError(format!("张量 {name} 的字节长度溢出")))
}

fn open_gguf_file(path: &Path) -> Result<WeightStore> {
    let backing = map_readonly(path)?;
    let (metadata, specs) = parse_gguf(&backing)?;
    make_store(specs, metadata, None)
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| RsinferError::WeightError("GGUF 读取偏移溢出".into()))?;
        let bytes = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| RsinferError::WeightError("GGUF 文件截断".into()))?;
        self.pos = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes([self.u8()?]))
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(RsinferError::WeightError("GGUF 字符串超过大小限制".into()));
        }
        let len = usize::try_from(len)
            .map_err(|_| RsinferError::WeightError("GGUF 字符串长度超过平台 usize".into()))?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| RsinferError::WeightError(format!("GGUF 字符串不是 UTF-8：{e}")))
    }
}

type ParsedGguf = (Map<String, Value>, Vec<(String, Arc<TensorSpec>)>);

fn parse_gguf(backing: &Arc<Mapping>) -> Result<ParsedGguf> {
    let mut reader = Reader::new(backing.bytes());
    if reader.take(4)? != b"GGUF" {
        return Err(RsinferError::WeightError("GGUF magic 不正确".into()));
    }
    let version = reader.u32()?;
    if version != 3 {
        return Err(RsinferError::Unsupported(format!(
            "仅支持 GGUF v3，实际版本 {version}"
        )));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count > MAX_TENSORS || metadata_count > MAX_METADATA_ENTRIES {
        return Err(RsinferError::WeightError(
            "GGUF tensor/metadata 数量超过限制".into(),
        ));
    }

    let mut metadata = Map::new();
    for _ in 0..metadata_count {
        let key = reader.string()?;
        if metadata.contains_key(&key) {
            return Err(RsinferError::WeightError(format!(
                "GGUF metadata key 重复：{key}"
            )));
        }
        let value_type = reader.u32()?;
        let value = read_gguf_value(&mut reader, value_type, 0)?;
        metadata.insert(key, value);
    }

    let tensor_count_usize = usize::try_from(tensor_count)
        .map_err(|_| RsinferError::WeightError("GGUF tensor_count 超过平台 usize".into()))?;
    let mut raw_tensors = Vec::with_capacity(tensor_count_usize);
    let mut names = HashSet::with_capacity(tensor_count_usize);
    for _ in 0..tensor_count_usize {
        let name = reader.string()?;
        if !names.insert(name.clone()) {
            return Err(RsinferError::WeightError(format!(
                "GGUF tensor 名称重复：{name}"
            )));
        }
        let n_dims = reader.u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(RsinferError::WeightError(format!(
                "GGUF 张量 {name} 的维度数非法：{n_dims}"
            )));
        }
        let n_dims_usize = usize::try_from(n_dims).unwrap();
        let mut raw_shape = Vec::with_capacity(n_dims_usize);
        for _ in 0..n_dims_usize {
            let dim = reader.u64()?;
            let dim = usize::try_from(dim).map_err(|_| {
                RsinferError::WeightError(format!("GGUF 张量 {name} 维度超过平台 usize"))
            })?;
            if dim == 0 {
                return Err(RsinferError::WeightError(format!(
                    "GGUF 张量 {name} 含零维 shape"
                )));
            }
            raw_shape.push(dim);
        }
        let dtype = reader.u32()?;
        let offset = reader.u64()?;
        raw_tensors.push((name, raw_shape, dtype, offset));
    }

    let alignment = gguf_alignment(&metadata)?;
    let data_start = align_up(reader.pos, alignment)?;
    if data_start > backing.bytes().len() {
        return Err(RsinferError::WeightError(
            "GGUF 数据起点超过文件长度".into(),
        ));
    }
    let mut specs_raw = Vec::with_capacity(raw_tensors.len());
    for (name, raw_shape, dtype, offset) in raw_tensors {
        let (format, block_elems, block_bytes) = gguf_format(dtype, &name)?;
        let mut shape = raw_shape;
        shape.reverse();
        let elements = shape_elements(&shape, &name)?;
        let width = *shape.last().unwrap();
        if let Some(block_elems) = block_elems {
            if !width.is_multiple_of(block_elems) {
                return Err(RsinferError::WeightError(format!(
                    "GGUF 量化张量 {name} 最后一维 {width} 未按 {block_elems} 对齐"
                )));
            }
            if !elements.is_multiple_of(block_elems) {
                return Err(RsinferError::WeightError(format!(
                    "GGUF 量化张量 {name} 元素数未按块对齐"
                )));
            }
            let blocks = elements / block_elems;
            let len = blocks.checked_mul(block_bytes).ok_or_else(|| {
                RsinferError::WeightError(format!("GGUF 张量 {name} 字节长度溢出"))
            })?;
            specs_raw.push((name, shape, format, offset, len));
        } else {
            let len = plain_byte_len(format, elements, &name)?;
            specs_raw.push((name, shape, format, offset, len));
        }
    }

    let mut ranges: Vec<(usize, usize, &str)> = Vec::with_capacity(specs_raw.len());
    for (name, _, _, offset, len) in &specs_raw {
        let offset = usize::try_from(*offset).map_err(|_| {
            RsinferError::WeightError(format!("GGUF 张量 {name} 偏移超过平台 usize"))
        })?;
        let start = data_start
            .checked_add(offset)
            .ok_or_else(|| RsinferError::WeightError(format!("GGUF 张量 {name} 起始偏移溢出")))?;
        let end = start
            .checked_add(*len)
            .ok_or_else(|| RsinferError::WeightError(format!("GGUF 张量 {name} 结束偏移溢出")))?;
        if end > backing.bytes().len() {
            return Err(RsinferError::WeightError(format!(
                "GGUF 张量 {name} 数据区越过文件末尾"
            )));
        }
        ranges.push((start, end, name.as_str()));
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = data_start;
    for (start, end, name) in &ranges {
        if *start < previous_end {
            return Err(RsinferError::WeightError(format!(
                "GGUF 张量 {name} 数据区重叠"
            )));
        }
        previous_end = *end;
    }

    let mut specs = Vec::with_capacity(specs_raw.len());
    for (name, shape, format, offset, len) in specs_raw {
        let offset = usize::try_from(offset).map_err(|_| {
            RsinferError::WeightError(format!("GGUF 张量 {name} 偏移超过平台 usize"))
        })?;
        let absolute = data_start
            .checked_add(offset)
            .ok_or_else(|| RsinferError::WeightError(format!("GGUF 张量 {name} 偏移溢出")))?;
        specs.push((
            name,
            Arc::new(TensorSpec {
                backing: Arc::clone(backing),
                offset: absolute,
                len,
                physical_shape: shape.clone(),
                shape,
                format,
                row_map: None,
                col_map: None,
            }),
        ));
    }
    Ok((metadata, specs))
}

fn gguf_alignment(metadata: &Map<String, Value>) -> Result<usize> {
    let Some(value) = metadata.get("general.alignment") else {
        return Ok(32);
    };
    let value = value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|v| u64::try_from(v).ok()))
        .ok_or_else(|| RsinferError::WeightError("general.alignment 不是正整数".into()))?;
    let alignment = usize::try_from(value)
        .map_err(|_| RsinferError::WeightError("general.alignment 超过平台 usize".into()))?;
    if alignment == 0 || alignment > (1 << 30) {
        return Err(RsinferError::WeightError(
            "general.alignment 超出允许范围".into(),
        ));
    }
    Ok(alignment)
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| RsinferError::WeightError("GGUF 对齐偏移溢出".into()))
    }
}

fn gguf_format(dtype: u32, name: &str) -> Result<(WeightFormat, Option<usize>, usize)> {
    match dtype {
        0 => Ok((WeightFormat::F32, None, 0)),
        1 => Ok((WeightFormat::F16, None, 0)),
        2 => Ok((WeightFormat::Q4_0, Some(32), 18)),
        8 => Ok((WeightFormat::Q8_0, Some(32), 34)),
        12 => Ok((WeightFormat::Q4K, Some(256), 144)),
        14 => Ok((WeightFormat::Q6K, Some(256), 210)),
        30 => Ok((WeightFormat::BF16, None, 0)),
        _ => Err(RsinferError::Unsupported(format!(
            "GGUF 张量 {name} 使用不支持的 ggml type：{dtype}"
        ))),
    }
}

fn read_gguf_value(reader: &mut Reader<'_>, value_type: u32, depth: usize) -> Result<Value> {
    if depth > 8 {
        return Err(RsinferError::WeightError(
            "GGUF metadata 数组嵌套过深".into(),
        ));
    }
    match value_type {
        0 => Ok(Value::Number(Number::from(reader.u8()?))),
        1 => Ok(Value::Number(Number::from(reader.i8()?))),
        2 => Ok(Value::Number(Number::from(reader.u16()?))),
        3 => Ok(Value::Number(Number::from(reader.i16()?))),
        4 => Ok(Value::Number(Number::from(reader.u32()?))),
        5 => Ok(Value::Number(Number::from(reader.i32()?))),
        6 => finite_f64_value(f64::from(reader.f32()?)),
        7 => match reader.u8()? {
            0 => Ok(Value::Bool(false)),
            1 => Ok(Value::Bool(true)),
            other => Err(RsinferError::WeightError(format!(
                "GGUF bool 值非法：{other}"
            ))),
        },
        8 => Ok(Value::String(reader.string()?)),
        9 => {
            let subtype = reader.u32()?;
            let len = reader.u64()?;
            if len > MAX_ARRAY_ITEMS {
                return Err(RsinferError::WeightError("GGUF metadata 数组过大".into()));
            }
            let len = usize::try_from(len)
                .map_err(|_| RsinferError::WeightError("GGUF metadata 数组长度溢出".into()))?;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(read_gguf_value(reader, subtype, depth + 1)?);
            }
            Ok(Value::Array(values))
        }
        10 => Ok(Value::Number(Number::from(reader.u64()?))),
        11 => Ok(Value::Number(Number::from(reader.i64()?))),
        12 => finite_f64_value(reader.f64()?),
        _ => Err(RsinferError::Unsupported(format!(
            "GGUF metadata value type 不支持：{value_type}"
        ))),
    }
}

fn finite_f64_value(value: f64) -> Result<Value> {
    if !value.is_finite() {
        return Err(RsinferError::WeightError(
            "GGUF metadata 浮点值不是有限数".into(),
        ));
    }
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| RsinferError::WeightError("GGUF metadata 浮点值无法编码为 JSON".into()))
}

fn read_limited(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path)?;
    let len = usize::try_from(metadata.len())
        .map_err(|_| RsinferError::WeightError(format!("文件过大：{}", path.display())))?;
    if len > limit {
        return Err(RsinferError::WeightError(format!(
            "文件超过大小限制：{}",
            path.display()
        )));
    }
    Ok(fs::read(path)?)
}

struct NoDupMap(Map<String, Value>);

impl<'de> Deserialize<'de> for NoDupMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct NoDupVisitor;

        impl<'de> Visitor<'de> for NoDupVisitor {
            type Value = NoDupMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("JSON object without duplicate keys")
            }

            fn visit_map<A>(self, mut access: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut map = Map::new();
                while let Some((key, value)) = access.next_entry::<String, Value>()? {
                    if map.contains_key(&key) {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON key: {key}"
                        )));
                    }
                    map.insert(key, value);
                }
                Ok(NoDupMap(map))
            }
        }

        deserializer.deserialize_map(NoDupVisitor)
    }
}

fn parse_no_dup_object(bytes: &[u8]) -> std::result::Result<Map<String, Value>, String> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let NoDupMap(map) = NoDupMap::deserialize(&mut deserializer).map_err(|e| e.to_string())?;
    deserializer.end().map_err(|e| e.to_string())?;
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(suffix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rsinfer-artifact-{}-{}{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            suffix
        ));
        path
    }

    fn push_u64(dst: &mut Vec<u8>, value: u64) {
        dst.extend_from_slice(&value.to_le_bytes());
    }

    fn push_str(dst: &mut Vec<u8>, value: &str) {
        push_u64(dst, value.len() as u64);
        dst.extend_from_slice(value.as_bytes());
    }

    fn build_gguf(tensors: &[(&str, &[u64], u32, u64)], payload: &[(u64, &[u8])]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, 0);
        for (name, dims, dtype, offset) in tensors {
            push_str(&mut bytes, name);
            bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in *dims {
                push_u64(&mut bytes, *dim);
            }
            bytes.extend_from_slice(&dtype.to_le_bytes());
            push_u64(&mut bytes, *offset);
        }
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        let max_end = payload
            .iter()
            .map(|(offset, data)| usize::try_from(*offset).unwrap() + data.len())
            .max()
            .unwrap_or(0);
        bytes.resize(bytes.len() + max_end, 0);
        let data_start = bytes.len() - max_end;
        for (offset, data) in payload {
            let start = data_start + usize::try_from(*offset).unwrap();
            bytes[start..start + data.len()].copy_from_slice(data);
        }
        bytes
    }

    fn write_fixture(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(bytes).unwrap();
        drop(file);
    }

    #[test]
    fn gguf_v3_f32_fixture_and_matmul() {
        let path = temp_path(".gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        push_u64(&mut bytes, 1); // 张量数量
        push_u64(&mut bytes, 0); // metadata 数量
        push_str(&mut bytes, "w");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        push_u64(&mut bytes, 2); // 原始维度 [2, 2]，反转后仍为 [2, 2]
        push_u64(&mut bytes, 2);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32 类型
        push_u64(&mut bytes, 0);
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut file = File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        let store = WeightStore::open(&path).unwrap();
        let tensor = store.get("w").unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.to_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.matmul(&[1.0, 1.0], 1).unwrap(), vec![3.0, 7.0]);
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_duplicate_metadata_key() {
        let parsed = parse_no_dup_object(br#"{"x":1,"x":2}"#);
        assert!(parsed.is_err());
    }

    #[test]
    fn rejects_duplicate_tensor_name_and_overlapping_data() {
        let duplicate = build_gguf(
            &[("w", &[1], 0, 0), ("w", &[1], 0, 4)],
            &[(0, &[0, 0, 0, 0, 0, 0, 128, 63])],
        );
        let duplicate_path = temp_path("-duplicate.gguf");
        write_fixture(&duplicate_path, &duplicate);
        assert!(WeightStore::open(&duplicate_path).is_err());
        fs::remove_file(duplicate_path).unwrap();

        let overlap = build_gguf(
            &[("a", &[1], 0, 0), ("b", &[1], 0, 0)],
            &[(0, &[0, 0, 128, 63])],
        );
        let overlap_path = temp_path("-overlap.gguf");
        write_fixture(&overlap_path, &overlap);
        assert!(WeightStore::open(&overlap_path).is_err());
        fs::remove_file(overlap_path).unwrap();
    }

    #[test]
    fn high_dimensional_rows_are_flattened_by_last_dimension() {
        let values: Vec<u8> = (0..24)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect();
        let bytes = build_gguf(&[("w", &[2, 3, 4], 0, 0)], &[(0, &values)]);
        let path = temp_path("-highdim.gguf");
        write_fixture(&path, &bytes);
        let store = WeightStore::open(&path).unwrap();
        let tensor = store.get("w").unwrap();
        assert_eq!(tensor.shape(), &[4, 3, 2]);
        let mut row = [0.0; 2];
        tensor.read_row(5, &mut row).unwrap();
        assert_eq!(row, [10.0, 11.0]);
        drop(store);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn tensor_view_identity_is_stable_and_distinguishes_handles() {
        let first = WeightTensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap();
        let first_clone = first.clone();
        let second = WeightTensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap();

        assert!(first.is_contiguous());
        assert_eq!(first.view_id(), first_clone.view_id());
        assert_ne!(first.view_id(), second.view_id());
    }

    #[test]
    fn tensor_views_compose_without_copying_backing() {
        let source =
            WeightTensor::from_f32(vec![3, 4], (0..12).map(|value| value as f32).collect())
                .unwrap();
        let rows = source.permute_rows(vec![2, 0, 1]).unwrap();
        assert!(!rows.is_contiguous());
        assert_eq!(rows.bytes(), source.bytes());
        assert_eq!(
            rows.to_f32().unwrap(),
            vec![8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );

        let composed = rows.permute_rows(vec![1, 2, 0]).unwrap();
        assert_eq!(composed.to_f32().unwrap(), source.to_f32().unwrap());

        let columns = rows.permute_columns(vec![3, 1, 0, 2]).unwrap();
        assert_eq!(
            columns.to_f32().unwrap(),
            vec![11.0, 9.0, 8.0, 10.0, 3.0, 1.0, 0.0, 2.0, 7.0, 5.0, 4.0, 6.0]
        );

        let reshaped = rows.reshape(vec![3, 1, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 1, 4]);
        assert_eq!(reshaped.to_f32().unwrap(), rows.to_f32().unwrap());
    }
}
