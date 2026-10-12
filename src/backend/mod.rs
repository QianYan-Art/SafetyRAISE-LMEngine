//! 可选原生算子后端。
//!
//! Rust 侧只负责资源生命周期、形状校验和插件 ABI 边界。原生插件必须把权重
//! 创建为驻留句柄，后续 `linear` 调用只传入输入和输出缓冲区；插件不支持的
//! 权重格式会显式计入回退次数并使用 `WeightTensor::matmul`。

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use libloading::Library;

use crate::artifact::{WeightFormat, WeightTensor};
use crate::{Result, RsinferError};

mod wgpu_q8;
pub use wgpu_q8::{WgpuQ8Backend, WgpuStats};

/// 当前原生算子插件 ABI 版本。
pub const NATIVE_ABI_VERSION: u32 = 1;

const STATUS_OK: i32 = 0;
const DTYPE_F32: u32 = 1;
const DTYPE_F16: u32 = 2;
const DTYPE_BF16: u32 = 3;
const DTYPE_Q4_0: u32 = 4;
const DTYPE_Q8_0: u32 = 5;
const DTYPE_Q4K: u32 = 6;
const DTYPE_Q6K: u32 = 7;

/// 可用于执行线性层的后端。
#[derive(Clone)]
pub enum Backend {
    /// 直接使用 Rust 权重实现的 CPU 正确性路径。
    Cpu,
    /// 共享一个动态库上下文和驻留权重缓存的原生插件路径。
    Native(Arc<NativeBackend>),
    WgpuQ8(Arc<WgpuQ8Backend>),
}

impl Backend {
    /// 执行 `weight[out, in] * input[rows, in]`，返回 `[rows, out]`。
    pub fn linear(&self, weight: &WeightTensor, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        match self {
            Self::Cpu => {
                validate_linear_arguments(weight, input, rows)?;
                let output = weight.matmul(input, rows)?;
                ensure_finite_output(&output, "CPU")?;
                Ok(output)
            }
            Self::Native(native) => native.linear(weight, input, rows),
            Self::WgpuQ8(wgpu) => wgpu.linear(weight, input, rows),
        }
    }

    /// 返回后端的稳定名称，适合日志和能力报告。
    pub fn name(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Native(native) => native.name(),
            Self::WgpuQ8(_) => "wgpu-q8",
        }
    }
}

/// 插件声明的设备类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum NativeDeviceKind {
    Host = 1,
    Cuda = 2,
    Other = 3,
}

impl NativeDeviceKind {
    fn from_raw(value: u32) -> Self {
        match value {
            1 => Self::Host,
            2 => Self::Cuda,
            _ => Self::Other,
        }
    }
}

/// 动态插件的能力声明。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeCapabilities {
    /// 插件 ABI 版本。
    pub abi_version: u32,
    /// 支持的权重格式位图，位号等于 ABI 中的 dtype 数值。
    pub dtype_mask: u64,
    /// 插件实际运行设备。
    pub device: NativeDeviceKind,
    /// 插件自定义标志位。
    pub flags: u32,
    /// 非零时为单次调用允许的最大输入行数。
    pub max_rows: u64,
    /// 非零时为单次调用允许的最大输入/权重列数。
    pub max_cols: u64,
}

impl NativeCapabilities {
    /// 判断插件是否声明支持给定权重格式。
    pub fn supports_format(&self, format: WeightFormat) -> bool {
        let dtype = native_dtype(format);
        self.dtype_mask & (1u64 << dtype) != 0
    }
}

/// 原生后端运行统计。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NativeStats {
    /// 因插件能力不足而走 CPU 的调用次数。
    pub fallback_count: u64,
    /// 当前插件中已准备并缓存的权重句柄数量。
    pub prepared_weights: u64,
}

/// 已加载的原生插件后端。
pub struct NativeBackend {
    inner: Arc<NativeInner>,
}

impl Clone for NativeBackend {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl NativeBackend {
    /// 加载插件并创建一个常驻执行上下文。
    pub fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }.map_err(|error| {
            RsinferError::Io(std::io::Error::other(format!(
                "加载原生算子插件失败 '{}': {error}",
                path.display()
            )))
        })?;

        let api = unsafe { NativeApi::load(&library) }?;
        let options = NativeContextOptions {
            abi_version: NATIVE_ABI_VERSION,
            struct_size: std::mem::size_of::<NativeContextOptions>() as u32,
            flags: 0,
        };
        let mut context = ptr::null_mut();
        let status = unsafe { (api.create_context)(&options, &mut context) };
        if status != STATUS_OK {
            if !context.is_null() {
                unsafe {
                    (api.destroy_context)(context);
                }
            }
            return Err(plugin_error_without_context(status, "创建原生插件上下文"));
        }
        if context.is_null() {
            return Err(RsinferError::WeightError(
                "原生插件创建上下文返回空句柄".into(),
            ));
        }

        let mut raw_capabilities = NativeCapabilitiesRaw::default();
        let status = unsafe { (api.get_capabilities)(context, &mut raw_capabilities) };
        if status != STATUS_OK {
            // 必须在 destroy_context 前读取插件错误文本；插件通常把文本存放在 context 中。
            let error = plugin_error_with_context(&api, context, status, "读取原生插件能力");
            unsafe {
                (api.destroy_context)(context);
            }
            return Err(error);
        }
        if raw_capabilities.abi_version != NATIVE_ABI_VERSION {
            unsafe {
                (api.destroy_context)(context);
            }
            return Err(RsinferError::Unsupported(format!(
                "原生插件能力 ABI 版本不匹配：需要 {}，插件声明 {}",
                NATIVE_ABI_VERSION, raw_capabilities.abi_version
            )));
        }
        let expected_size = std::mem::size_of::<NativeCapabilitiesRaw>() as u32;
        if raw_capabilities.struct_size < expected_size {
            unsafe {
                (api.destroy_context)(context);
            }
            return Err(RsinferError::Unsupported(format!(
                "原生插件能力结构体过短：需要至少 {expected_size} 字节，得到 {}",
                raw_capabilities.struct_size
            )));
        }

        let name = unsafe {
            let pointer = (api.name)();
            if pointer.is_null() {
                "native".to_owned()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        if name.is_empty() {
            unsafe {
                (api.destroy_context)(context);
            }
            return Err(RsinferError::Unsupported("原生插件返回空名称".into()));
        }

        let capabilities = NativeCapabilities {
            abi_version: raw_capabilities.abi_version,
            dtype_mask: raw_capabilities.dtype_mask,
            device: NativeDeviceKind::from_raw(raw_capabilities.device_kind),
            flags: raw_capabilities.flags,
            max_rows: raw_capabilities.max_rows,
            max_cols: raw_capabilities.max_cols,
        };
        let inner = NativeInner {
            library,
            api,
            context,
            name,
            capabilities,
            weights: Mutex::new(HashMap::new()),
            fallback_count: AtomicU64::new(0),
            prepared_weights: AtomicU64::new(0),
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// 返回插件公开的名称。
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// 返回插件能力快照。
    pub fn capabilities(&self) -> NativeCapabilities {
        self.inner.capabilities
    }

    /// 返回回退和驻留句柄统计。
    pub fn stats(&self) -> NativeStats {
        NativeStats {
            fallback_count: self.inner.fallback_count.load(Ordering::Relaxed),
            prepared_weights: self.inner.prepared_weights.load(Ordering::Relaxed),
        }
    }

    /// 使用原生插件执行线性层，必要时显式回退到 CPU。
    pub fn linear(&self, weight: &WeightTensor, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        let (output_dim, input_dim, output_len) = validate_linear_arguments(weight, input, rows)?;
        let format = weight.format();
        let dtype = native_dtype(format);
        if !weight.is_contiguous() || !self.inner.capabilities.supports_format(format) {
            self.inner.fallback_count.fetch_add(1, Ordering::Relaxed);
            let output = weight.matmul(input, rows)?;
            ensure_finite_output(&output, "CPU 回退")?;
            return Ok(output);
        }
        if self.inner.capabilities.max_rows != 0
            && u64::try_from(rows)
                .ok()
                .is_none_or(|value| value > self.inner.capabilities.max_rows)
        {
            return Err(RsinferError::DimensionError(format!(
                "原生插件 '{}' 不支持 {} 行输入（上限 {}）",
                self.name(),
                rows,
                self.inner.capabilities.max_rows
            )));
        }
        if self.inner.capabilities.max_cols != 0
            && u64::try_from(input_dim)
                .ok()
                .is_none_or(|value| value > self.inner.capabilities.max_cols)
        {
            return Err(RsinferError::DimensionError(format!(
                "原生插件 '{}' 不支持 {} 列输入（上限 {}）",
                self.name(),
                input_dim,
                self.inner.capabilities.max_cols
            )));
        }

        let key = WeightKey::new(weight, format);
        let mut state = self
            .inner
            .weights
            .lock()
            .map_err(|_| RsinferError::WeightError("原生插件权重缓存锁已中毒".into()))?;
        let handle = if let Some(prepared) = state.get(&key) {
            prepared.handle
        } else {
            let handle = self.prepare_weight(weight, output_dim, input_dim, dtype)?;
            state.insert(
                key,
                PreparedWeight {
                    handle,
                    _weight: weight.clone(),
                },
            );
            self.inner.prepared_weights.fetch_add(1, Ordering::Relaxed);
            handle
        };

        let mut output = vec![0.0f32; output_len];
        let input_pointer = if input.is_empty() {
            ptr::null()
        } else {
            input.as_ptr()
        };
        let output_pointer = if output.is_empty() {
            ptr::null_mut()
        } else {
            output.as_mut_ptr()
        };
        let status = unsafe {
            (self.inner.api.linear)(
                self.inner.context,
                handle,
                input_pointer,
                rows as u64,
                input_dim as u64,
                output_pointer,
                rows as u64,
                output_dim as u64,
            )
        };
        if status != STATUS_OK {
            return Err(self.plugin_error(status, "执行原生线性算子"));
        }
        ensure_finite_output(&output, "原生插件")?;
        Ok(output)
    }

    fn prepare_weight(
        &self,
        weight: &WeightTensor,
        output_dim: usize,
        input_dim: usize,
        dtype: u32,
    ) -> Result<*mut c_void> {
        let bytes = weight.bytes();
        let byte_len = u64::try_from(bytes.len()).map_err(|_| {
            RsinferError::DimensionError("权重字节长度超过原生 ABI 的 u64 范围".into())
        })?;
        let output_dim = u64::try_from(output_dim)
            .map_err(|_| RsinferError::DimensionError("权重输出维度溢出 u64".into()))?;
        let input_dim = u64::try_from(input_dim)
            .map_err(|_| RsinferError::DimensionError("权重输入维度溢出 u64".into()))?;
        let bytes_pointer = if bytes.is_empty() {
            ptr::null()
        } else {
            bytes.as_ptr()
        };
        let mut handle = ptr::null_mut();
        let status = unsafe {
            (self.inner.api.create_weight)(
                self.inner.context,
                bytes_pointer,
                byte_len,
                output_dim,
                input_dim,
                dtype,
                &mut handle,
            )
        };
        if status != STATUS_OK {
            return Err(self.plugin_error(status, "准备原生权重"));
        }
        if handle.is_null() {
            return Err(RsinferError::WeightError(
                "原生插件准备权重返回空句柄".into(),
            ));
        }
        Ok(handle)
    }

    fn plugin_error(&self, status: i32, operation: &str) -> RsinferError {
        plugin_error_with_context(&self.inner.api, self.inner.context, status, operation)
    }
}

struct NativeInner {
    // 必须保持在函数指针和上下文之后存活，Drop 中先销毁插件资源再卸载库。
    library: Library,
    api: NativeApi,
    context: *mut c_void,
    name: String,
    capabilities: NativeCapabilities,
    weights: Mutex<HashMap<WeightKey, PreparedWeight>>,
    fallback_count: AtomicU64,
    prepared_weights: AtomicU64,
}

// 原生句柄只在 weights 锁保护下使用；插件调用因此串行化，避免插件未声明线程安全时
// 发生并发访问。动态库句柄与上下文在同一 Arc 中保持生命周期。
unsafe impl Send for NativeInner {}
unsafe impl Sync for NativeInner {}

impl Drop for NativeInner {
    fn drop(&mut self) {
        if let Ok(mut weights) = self.weights.lock() {
            for (_, prepared) in weights.drain() {
                if !prepared.handle.is_null() {
                    unsafe {
                        (self.api.destroy_weight)(self.context, prepared.handle);
                    }
                }
            }
        }
        if !self.context.is_null() {
            unsafe {
                (self.api.destroy_context)(self.context);
            }
        }
        // `library` 有意保留到 Drop 结束：插件函数和上下文的最后一次访问必须先完成。
        let _ = &self.library;
    }
}

struct PreparedWeight {
    handle: *mut c_void,
    // 保持原始权重 backing 存活，确保缓存键中的 bytes 指针在句柄生命周期内有效。
    _weight: WeightTensor,
}

#[derive(Clone, Eq)]
struct WeightKey {
    view_id: usize,
    bytes_pointer: usize,
    byte_len: usize,
    shape: Vec<usize>,
    format: u32,
}

impl WeightKey {
    fn new(weight: &WeightTensor, format: WeightFormat) -> Self {
        let bytes = weight.bytes();
        Self {
            bytes_pointer: bytes.as_ptr() as usize,
            view_id: weight.view_id(),
            byte_len: bytes.len(),
            shape: weight.shape().to_vec(),
            format: native_dtype(format),
        }
    }
}

impl PartialEq for WeightKey {
    fn eq(&self, other: &Self) -> bool {
        self.bytes_pointer == other.bytes_pointer
            && self.view_id == other.view_id
            && self.byte_len == other.byte_len
            && self.shape == other.shape
            && self.format == other.format
    }
}

impl Hash for WeightKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes_pointer.hash(state);
        self.view_id.hash(state);
        self.byte_len.hash(state);
        self.shape.hash(state);
        self.format.hash(state);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeContextOptions {
    abi_version: u32,
    struct_size: u32,
    flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeCapabilitiesRaw {
    abi_version: u32,
    struct_size: u32,
    dtype_mask: u64,
    device_kind: u32,
    flags: u32,
    max_rows: u64,
    max_cols: u64,
}

impl Default for NativeCapabilitiesRaw {
    fn default() -> Self {
        Self {
            abi_version: 0,
            struct_size: std::mem::size_of::<Self>() as u32,
            dtype_mask: 0,
            device_kind: 0,
            flags: 0,
            max_rows: 0,
            max_cols: 0,
        }
    }
}

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateContextFn = unsafe extern "C" fn(
    options: *const NativeContextOptions,
    out_context: *mut *mut c_void,
) -> i32;
type DestroyContextFn = unsafe extern "C" fn(context: *mut c_void) -> i32;
type GetCapabilitiesFn =
    unsafe extern "C" fn(context: *mut c_void, out_capabilities: *mut NativeCapabilitiesRaw) -> i32;
type CreateWeightFn = unsafe extern "C" fn(
    context: *mut c_void,
    bytes: *const u8,
    byte_len: u64,
    rows: u64,
    cols: u64,
    dtype: u32,
    out_weight: *mut *mut c_void,
) -> i32;
type DestroyWeightFn = unsafe extern "C" fn(context: *mut c_void, weight: *mut c_void) -> i32;
type LinearFn = unsafe extern "C" fn(
    context: *mut c_void,
    weight: *mut c_void,
    input: *const f32,
    input_rows: u64,
    input_cols: u64,
    output: *mut f32,
    output_rows: u64,
    output_cols: u64,
) -> i32;
type LastErrorFn = unsafe extern "C" fn(context: *mut c_void) -> *const c_char;
type NameFn = unsafe extern "C" fn() -> *const c_char;

#[derive(Clone, Copy)]
struct NativeApi {
    abi_version: AbiVersionFn,
    create_context: CreateContextFn,
    destroy_context: DestroyContextFn,
    get_capabilities: GetCapabilitiesFn,
    create_weight: CreateWeightFn,
    destroy_weight: DestroyWeightFn,
    linear: LinearFn,
    last_error: LastErrorFn,
    name: NameFn,
}

impl NativeApi {
    unsafe fn load(library: &Library) -> Result<Self> {
        let api = Self {
            abi_version: load_symbol(library, b"safetyraise_kernel_abi_version\0")?,
            create_context: load_symbol(library, b"safetyraise_kernel_create_context\0")?,
            destroy_context: load_symbol(library, b"safetyraise_kernel_destroy_context\0")?,
            get_capabilities: load_symbol(library, b"safetyraise_kernel_get_capabilities\0")?,
            create_weight: load_symbol(library, b"safetyraise_kernel_create_weight\0")?,
            destroy_weight: load_symbol(library, b"safetyraise_kernel_destroy_weight\0")?,
            linear: load_symbol(library, b"safetyraise_kernel_linear\0")?,
            last_error: load_symbol(library, b"safetyraise_kernel_last_error\0")?,
            name: load_symbol(library, b"safetyraise_kernel_name\0")?,
        };
        let version = (api.abi_version)();
        if version != NATIVE_ABI_VERSION {
            return Err(RsinferError::Unsupported(format!(
                "原生插件 ABI 版本不匹配：需要 {}，插件导出 {}",
                NATIVE_ABI_VERSION, version
            )));
        }
        Ok(api)
    }
}

unsafe fn load_symbol<T: Copy>(library: &Library, name: &'static [u8]) -> Result<T> {
    library
        .get::<T>(name)
        .map(|symbol| *symbol)
        .map_err(|error| {
            let symbol_name = String::from_utf8_lossy(name.strip_suffix(&[0]).unwrap_or(name));
            RsinferError::Unsupported(format!("原生插件缺少符号 '{symbol_name}': {error}"))
        })
}

fn plugin_error_without_context(status: i32, operation: &str) -> RsinferError {
    // create_context 失败时尚无可传给插件的 context；避免假定插件接受空句柄，
    // 仍将非零状态码原样暴露给调用方。
    plugin_error_message(status, operation, String::new())
}

fn plugin_error_with_context(
    api: &NativeApi,
    context: *mut c_void,
    status: i32,
    operation: &str,
) -> RsinferError {
    let message = unsafe { c_string_lossy((api.last_error)(context)) };
    plugin_error_message(status, operation, message)
}

fn plugin_error_message(status: i32, operation: &str, message: String) -> RsinferError {
    let detail = if message.is_empty() {
        format!("状态码 {status}")
    } else {
        format!("状态码 {status}: {message}")
    };
    RsinferError::WeightError(format!("{operation}失败（{detail}）"))
}

unsafe fn c_string_lossy(pointer: *const c_char) -> String {
    if pointer.is_null() {
        String::new()
    } else {
        CStr::from_ptr(pointer).to_string_lossy().into_owned()
    }
}

fn native_dtype(format: WeightFormat) -> u32 {
    match format {
        WeightFormat::F32 => DTYPE_F32,
        WeightFormat::F16 => DTYPE_F16,
        WeightFormat::BF16 => DTYPE_BF16,
        WeightFormat::Q4_0 => DTYPE_Q4_0,
        WeightFormat::Q8_0 => DTYPE_Q8_0,
        WeightFormat::Q4K => DTYPE_Q4K,
        WeightFormat::Q6K => DTYPE_Q6K,
    }
}

fn validate_linear_arguments(
    weight: &WeightTensor,
    input: &[f32],
    rows: usize,
) -> Result<(usize, usize, usize)> {
    let shape = weight.shape();
    if shape.len() != 2 {
        return Err(RsinferError::DimensionError(format!(
            "线性权重必须是二维 [out, in]，实际维度为 {shape:?}"
        )));
    }
    let output_dim = shape[0];
    let input_dim = shape[1];
    let expected_input = rows
        .checked_mul(input_dim)
        .ok_or_else(|| RsinferError::DimensionError("输入行数与列数相乘溢出 usize".into()))?;
    if input.len() != expected_input {
        return Err(RsinferError::ShapeMismatch {
            expected: vec![rows, input_dim],
            actual: vec![input.len()],
        });
    }
    let output_len = rows
        .checked_mul(output_dim)
        .ok_or_else(|| RsinferError::DimensionError("输出行数与列数相乘溢出 usize".into()))?;
    Ok((output_dim, input_dim, output_len))
}

fn ensure_finite_output(output: &[f32], source: &str) -> Result<()> {
    if let Some((index, value)) = output
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(RsinferError::WeightError(format!(
            "{source}线性算子返回非有限值：位置 {index}，值 {value:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_dtype_bits_are_stable() {
        assert_eq!(native_dtype(WeightFormat::F32), DTYPE_F32);
        assert_eq!(native_dtype(WeightFormat::F16), DTYPE_F16);
        assert_eq!(native_dtype(WeightFormat::Q6K), DTYPE_Q6K);
        let caps = NativeCapabilities {
            abi_version: NATIVE_ABI_VERSION,
            dtype_mask: (1u64 << DTYPE_F32) | (1u64 << DTYPE_F16),
            device: NativeDeviceKind::Host,
            flags: 0,
            max_rows: 0,
            max_cols: 0,
        };
        assert!(caps.supports_format(WeightFormat::F32));
        assert!(caps.supports_format(WeightFormat::F16));
        assert!(!caps.supports_format(WeightFormat::Q4_0));
    }

    #[test]
    fn finite_output_rejects_nan_and_infinity() {
        assert!(ensure_finite_output(&[0.0, 1.0], "测试").is_ok());
        assert!(ensure_finite_output(&[0.0, f32::NAN], "测试").is_err());
        assert!(ensure_finite_output(&[f32::INFINITY], "测试").is_err());
    }

    /// 通过显式环境变量运行真实 CPU 动态插件；默认不自动跳过，而是以 ignored
    /// 测试形式要求调用者在本机已构建插件后显式执行命令。
    #[test]
    #[ignore = "需要 RSINFER_NATIVE_CPU_PLUGIN 指向已构建的 CPU 插件"]
    fn native_cpu_plugin_roundtrip_caches_weight_and_propagates_error() {
        let path = std::env::var_os("RSINFER_NATIVE_CPU_PLUGIN")
            .expect("请设置 RSINFER_NATIVE_CPU_PLUGIN 指向 safetyraise_cpu 动态库");
        let backend = NativeBackend::load(Path::new(&path)).expect("加载 CPU 原生插件失败");
        assert_eq!(backend.capabilities().device, NativeDeviceKind::Host);
        assert!(backend.capabilities().supports_format(WeightFormat::F32));

        let weight = WeightTensor::from_f32(vec![2, 3], vec![1.0, 0.0, -1.0, 2.0, 3.0, 4.0])
            .expect("构造测试权重失败");
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected = [-2.0, 20.0, -2.0, 47.0];
        assert_eq!(backend.linear(&weight, &input, 2).unwrap(), expected);
        assert_eq!(backend.linear(&weight, &input, 2).unwrap(), expected);
        assert_eq!(backend.stats().prepared_weights, 1);

        // 插件主动返回错误时必须向上传播，不能被错误地视为 dtype 回退。
        let error = backend.linear(&weight, &[f32::NAN, 2.0, 3.0], 1);
        assert!(error.is_err());
        assert_eq!(backend.stats().fallback_count, 0);
    }
}
