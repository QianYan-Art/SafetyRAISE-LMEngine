//! 运行时设备规划。
//!
//! 当前计算内核仍是 CPU 路径；本模块只负责把 CPU/GPU 混合推理的选择、
//! GPU 探测和层放置计划固定下来，避免后续接 GPU kernel 时改动 CLI/API。

use std::fmt;
use std::path::PathBuf;
use std::process::Command;

use crate::model::Qwen3Config;

const MAX_ACTIVE_TRANSFORMER_GPU_LAYERS: usize = 16;
const MAX_AUTO_TRANSFORMER_GPU_LAYERS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DevicePreference {
    Cpu,
    Auto,
    Hybrid,
}

impl DevicePreference {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerDevice {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationMode {
    None,
    Q8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationCacheMode {
    Auto,
    Off,
}

impl QuantizationCacheMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }
}

impl QuantizationMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Q8 => "q8",
        }
    }
}

impl fmt::Display for LayerDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("CPU"),
            Self::Gpu => f.write_str("GPU"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeOptions {
    pub device: DevicePreference,
    pub gpu_layers: Option<usize>,
    pub quantization: QuantizationMode,
    pub quantization_cache: QuantizationCacheMode,
    pub q8_cache_dir: Option<PathBuf>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            device: DevicePreference::Cpu,
            gpu_layers: None,
            quantization: QuantizationMode::None,
            quantization_cache: QuantizationCacheMode::Auto,
            q8_cache_dir: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    pub memory_total_mib: Option<usize>,
    pub driver_version: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePlan {
    pub requested_device: DevicePreference,
    pub compute_backend: String,
    pub gpu: Option<GpuInfo>,
    pub layer_devices: Vec<LayerDevice>,
    pub transformer_decode_gpu_layers: usize,
    pub lm_head_device: LayerDevice,
    pub quantization: QuantizationMode,
    pub notes: Vec<String>,
}

impl RuntimePlan {
    pub fn cpu_only(num_layers: usize, requested_device: DevicePreference, note: String) -> Self {
        Self {
            requested_device,
            compute_backend: "cpu".to_string(),
            gpu: None,
            layer_devices: vec![LayerDevice::Cpu; num_layers],
            transformer_decode_gpu_layers: 0,
            lm_head_device: LayerDevice::Cpu,
            quantization: QuantizationMode::None,
            notes: vec![note],
        }
    }

    pub fn gpu_layer_count(&self) -> usize {
        self.layer_devices
            .iter()
            .filter(|&&device| device == LayerDevice::Gpu)
            .count()
    }
}

impl fmt::Display for RuntimePlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "runtime.device: {}", self.requested_device.as_str())?;
        writeln!(f, "runtime.compute_backend: {}", self.compute_backend)?;
        if let Some(gpu) = &self.gpu {
            write!(f, "runtime.gpu: {}", gpu.name)?;
            if let Some(memory) = gpu.memory_total_mib {
                write!(f, " ({memory} MiB)")?;
            }
            if let Some(driver) = &gpu.driver_version {
                write!(f, ", driver {driver}")?;
            }
            writeln!(f)?;
        } else {
            writeln!(f, "runtime.gpu: none")?;
        }
        writeln!(
            f,
            "runtime.layers: {} GPU / {} CPU",
            self.gpu_layer_count(),
            self.layer_devices
                .len()
                .saturating_sub(self.gpu_layer_count())
        )?;
        writeln!(
            f,
            "runtime.transformer_decode_gpu_layers: {}",
            self.transformer_decode_gpu_layers
        )?;
        writeln!(f, "runtime.lm_head: {}", self.lm_head_device)?;
        writeln!(f, "runtime.quantization: {}", self.quantization.as_str())?;
        for note in &self.notes {
            writeln!(f, "runtime.note: {note}")?;
        }
        Ok(())
    }
}

pub fn build_runtime_plan(config: &Qwen3Config, options: &RuntimeOptions) -> RuntimePlan {
    match options.device {
        DevicePreference::Cpu => RuntimePlan::cpu_only(
            config.num_hidden_layers,
            DevicePreference::Cpu,
            "CPU-only execution selected; no GPU probing performed.".to_string(),
        ),
        DevicePreference::Auto | DevicePreference::Hybrid => {
            let gpu = detect_nvidia_gpu();
            let Some(gpu) = gpu else {
                return RuntimePlan::cpu_only(
                    config.num_hidden_layers,
                    options.device,
                    "No NVIDIA GPU was detected through nvidia-smi; falling back to CPU execution."
                        .to_string(),
                );
            };

            let estimated = estimate_gpu_layers(config, &gpu, options.quantization);
            let requested = options.gpu_layers;
            let requested_or_estimated = requested.unwrap_or(estimated);
            let gpu_layers = requested_or_estimated
                .min(config.num_hidden_layers)
                .min(MAX_ACTIVE_TRANSFORMER_GPU_LAYERS);
            let mut layer_devices = vec![LayerDevice::Cpu; config.num_hidden_layers];
            for device in layer_devices.iter_mut().take(gpu_layers) {
                *device = LayerDevice::Gpu;
            }

            let mut notes = vec![
                "Transformer attention/KV/prefill GPU kernels are not implemented yet; only selected decode linear matvecs can be active GPU work."
                    .to_string(),
                "Layer placement controls optional decode linear GPU offload and remains CPU fallback compatible."
                    .to_string(),
            ];
            if requested_or_estimated > config.num_hidden_layers {
                notes.push(format!(
                    "--gpu-layers requested {requested_or_estimated}, clamped to {} transformer layers.",
                    config.num_hidden_layers
                ));
            }
            if requested_or_estimated > gpu_layers {
                notes.push(format!(
                    "--gpu-layers requested {requested_or_estimated}, active transformer GPU layers capped to {gpu_layers} for this wgpu/f32 backend."
                ));
            }
            if options.gpu_layers.is_none() {
                if options.quantization == QuantizationMode::Q8 {
                    notes.push(
                        "Q8 hybrid defaults transformer decode GPU layers to 16 on this local decode profile; pass --gpu-layers to override for explicit GPU+CPU decode testing."
                            .to_string(),
                    );
                } else {
                    notes.push(
                        "GPU layer count was conservatively estimated from visible GPU memory."
                            .to_string(),
                    );
                }
            }

            RuntimePlan {
                requested_device: options.device,
                compute_backend: "cpu-execution-with-planned-gpu-placement".to_string(),
                gpu: Some(gpu),
                layer_devices,
                transformer_decode_gpu_layers: 0,
                lm_head_device: LayerDevice::Cpu,
                quantization: options.quantization,
                notes,
            }
        }
    }
}

impl RuntimePlan {
    pub fn mark_lm_head_gpu(&mut self) {
        self.lm_head_device = LayerDevice::Gpu;
        self.refresh_compute_backend();
        self.notes
            .push("lm_head matvec is using the optional wgpu backend.".to_string());
    }

    pub fn mark_lm_head_q8_gpu(&mut self) {
        self.lm_head_device = LayerDevice::Gpu;
        self.refresh_compute_backend();
        self.notes
            .push("lm_head Q8 matvec is using the optional wgpu backend; greedy temperature<=0 can use fused Q8 GPU argmax.".to_string());
    }

    pub fn mark_lm_head_gpu_fallback(&mut self, reason: impl Into<String>) {
        self.lm_head_device = LayerDevice::Cpu;
        self.notes.push(format!(
            "lm_head GPU backend unavailable; using CPU fallback: {}",
            reason.into()
        ));
    }

    pub fn should_try_lm_head_gpu(&self) -> bool {
        self.requested_device != DevicePreference::Cpu && self.gpu.is_some()
    }

    pub fn should_try_gpu_backend(&self) -> bool {
        self.should_try_lm_head_gpu() || self.planned_transformer_gpu_layers() > 0
    }

    pub fn planned_transformer_gpu_layers(&self) -> usize {
        self.gpu_layer_count()
    }

    pub fn mark_transformer_decode_gpu_layers(&mut self, active_layers: usize, attached: usize) {
        self.transformer_decode_gpu_layers = active_layers;
        self.refresh_compute_backend();
        if active_layers > 0 {
            self.notes.push(format!(
                "decode matvec for {active_layers} transformer layer(s) attached {attached} linear GPU kernels through the shared wgpu context; q/k/v same-input groups are batched and MLP can fuse gate/up SwiGLU into GPU down_proj; prefill still falls back to CPU."
            ));
        }
    }

    pub fn mark_transformer_decode_q8_gpu_layers(&mut self, active_layers: usize, attached: usize) {
        self.transformer_decode_gpu_layers = active_layers;
        self.refresh_compute_backend();
        if active_layers > 0 {
            self.notes.push(format!(
                "decode Q8 matvec for {active_layers} transformer layer(s) attached {attached} linear GPU kernels through the shared wgpu context; q/k/v can use one shared input upload, MLP can fuse gate/up SwiGLU into Q8 GPU down_proj, and prefill still falls back to CPU."
            ));
        }
    }

    pub fn mark_transformer_gpu_fallback(&mut self, reason: impl Into<String>) {
        self.notes.push(format!(
            "transformer decode GPU backend unavailable or partial; using CPU fallback where needed: {}",
            reason.into()
        ));
    }

    pub fn mark_q8_quantization(&mut self, attached_linears: usize) {
        self.quantization = QuantizationMode::Q8;
        self.notes.push(format!(
            "Q8 linear path enabled for {attached_linears} bias-free linear layer(s); Q8 GPU matvecs or existing GPU matvecs take priority where attached, then CPU Q8 is used as fallback."
        ));
    }

    pub fn mark_q8_sidecar_disabled(&mut self) {
        self.notes
            .push("Q8 sidecar cache disabled; deriving Q8 weights in memory.".to_string());
    }

    pub fn mark_q8_sidecar_unavailable(&mut self, reason: impl Into<String>) {
        self.notes.push(format!(
            "Q8 sidecar cache unavailable; deriving Q8 weights in memory: {}",
            reason.into()
        ));
    }

    pub fn mark_q8_sidecar_report(
        &mut self,
        dir: impl fmt::Display,
        hits: usize,
        writes: usize,
        fallbacks: usize,
    ) {
        self.notes.push(format!(
            "Q8 sidecar cache at {dir}: {hits} hit(s), {writes} write(s), {fallbacks} fallback(s)."
        ));
    }

    fn refresh_compute_backend(&mut self) {
        self.compute_backend = match (
            self.transformer_decode_gpu_layers > 0,
            self.lm_head_device == LayerDevice::Gpu,
        ) {
            (true, true) => "cpu-prefill-gpu-decode-linears-gpu-lm-head".to_string(),
            (true, false) => "cpu-prefill-gpu-decode-linears".to_string(),
            (false, true) => "cpu-transformer-gpu-lm-head".to_string(),
            (false, false) if self.requested_device == DevicePreference::Cpu => "cpu".to_string(),
            (false, false) => "cpu-execution-with-planned-gpu-placement".to_string(),
        };
    }
}

fn estimate_gpu_layers(
    config: &Qwen3Config,
    gpu: &GpuInfo,
    quantization: QuantizationMode,
) -> usize {
    if quantization == QuantizationMode::Q8 {
        return MAX_AUTO_TRANSFORMER_GPU_LAYERS.min(config.num_hidden_layers);
    }

    let Some(memory_mib) = gpu.memory_total_mib else {
        return 0;
    };
    if memory_mib < 4096 {
        return 0;
    }

    let reserve_mib = 2048usize;
    let lm_head_mib = estimate_lm_head_bytes(config) / (1024 * 1024);
    let usable_mib = memory_mib.saturating_sub(reserve_mib + lm_head_mib);
    let bytes_per_layer = estimate_transformer_layer_bytes(config);
    if bytes_per_layer == 0 {
        return 0;
    }

    let fit = (usable_mib * 1024 * 1024) / bytes_per_layer;
    fit.min(config.num_hidden_layers)
        .min(MAX_ACTIVE_TRANSFORMER_GPU_LAYERS)
        .min(MAX_AUTO_TRANSFORMER_GPU_LAYERS)
}

fn estimate_transformer_layer_bytes(config: &Qwen3Config) -> usize {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let kv_hidden = config.num_key_value_heads * config.head_dim();

    let q_proj = hidden * hidden;
    let k_proj = kv_hidden * hidden;
    let v_proj = kv_hidden * hidden;
    let o_proj = hidden * hidden;
    let gate_proj = intermediate * hidden;
    let up_proj = intermediate * hidden;
    let down_proj = hidden * intermediate;

    (q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj) * 4
}

fn estimate_lm_head_bytes(config: &Qwen3Config) -> usize {
    config.vocab_size * config.hidden_size * 4
}

fn detect_nvidia_gpu() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(parse_nvidia_smi_line)
}

fn parse_nvidia_smi_line(line: &str) -> Option<GpuInfo> {
    let mut fields = line.split(',').map(str::trim);
    let name = fields.next()?.to_string();
    if name.is_empty() {
        return None;
    }
    let memory_total_mib = fields.next().and_then(|value| value.parse().ok());
    let driver_version = fields
        .next()
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    Some(GpuInfo {
        name,
        memory_total_mib,
        driver_version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Qwen3Config {
        Qwen3Config {
            hidden_size: 256,
            intermediate_size: 512,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            explicit_head_dim: Some(32),
            vocab_size: 1024,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            model_type: "qwen3".to_string(),
            torch_dtype: "float16".to_string(),
            tie_word_embeddings: true,
            eos_token_id: 151645,
            bos_token_id: None,
        }
    }

    #[test]
    fn cpu_plan_never_probes_gpu() {
        let plan = build_runtime_plan(&test_config(), &RuntimeOptions::default());
        assert_eq!(plan.compute_backend, "cpu");
        assert_eq!(plan.gpu_layer_count(), 0);
        assert!(plan.layer_devices.iter().all(|&d| d == LayerDevice::Cpu));
    }

    #[test]
    fn nvidia_smi_parser_accepts_first_gpu_line() {
        let gpu =
            parse_nvidia_smi_line("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 595.79").unwrap();
        assert_eq!(gpu.name, "NVIDIA GeForce RTX 4060 Laptop GPU");
        assert_eq!(gpu.memory_total_mib, Some(8188));
        assert_eq!(gpu.driver_version.as_deref(), Some("595.79"));
    }

    #[test]
    fn runtime_plan_display_is_honest_about_cpu_execution() {
        let config = test_config();
        let gpu = GpuInfo {
            name: "Test GPU".to_string(),
            memory_total_mib: Some(8192),
            driver_version: Some("1.0".to_string()),
        };
        let layers = estimate_gpu_layers(&config, &gpu, QuantizationMode::None);
        assert!(layers <= config.num_hidden_layers);

        let text = RuntimePlan {
            requested_device: DevicePreference::Hybrid,
            compute_backend: "cpu-execution-with-planned-gpu-placement".to_string(),
            gpu: Some(gpu),
            layer_devices: vec![
                LayerDevice::Gpu,
                LayerDevice::Gpu,
                LayerDevice::Cpu,
                LayerDevice::Cpu,
            ],
            transformer_decode_gpu_layers: 0,
            lm_head_device: LayerDevice::Cpu,
            quantization: QuantizationMode::None,
            notes: vec!["GPU kernels are not implemented yet.".to_string()],
        }
        .to_string();
        assert!(text.contains("cpu-execution-with-planned-gpu-placement"));
        assert!(text.contains("2 GPU / 2 CPU"));
        assert!(text.contains("runtime.quantization: none"));
    }

    #[test]
    fn q8_auto_estimate_uses_sixteen_transformer_gpu_layers() {
        let mut config = test_config();
        config.num_hidden_layers = 24;
        let gpu = GpuInfo {
            name: "Test GPU".to_string(),
            memory_total_mib: Some(8192),
            driver_version: Some("1.0".to_string()),
        };

        assert_eq!(estimate_gpu_layers(&config, &gpu, QuantizationMode::Q8), 16);
    }
}
