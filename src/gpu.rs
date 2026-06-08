//! 可选 GPU 计算内核。
//!
//! 目前实现单行 matvec 和 decode MLP 的 SwiGLU+down projection GPU 路径。
//! 初始化或执行失败时上层会回退 CPU。

use std::borrow::Cow;
use std::sync::{mpsc, Arc};

use half::f16;
use wgpu::util::DeviceExt;

use crate::tensor::{Q8LinearWeight, Tensor};

const WORKGROUP_SIZE: u32 = 64;

const MATVEC_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_idx = id.x;
    if (out_idx >= params.out_features) {
        return;
    }

    var sum = 0.0;
    let base = out_idx * params.in_features;
    for (var i = 0u; i < params.in_features; i = i + 1u) {
        sum = sum + input[i] * weight[base + i];
    }
    output[out_idx] = sum;
}
"#;

const Q8_MATVEC_SHADER: &str = r#"
struct Params {
    in_features: u32,
    out_features: u32,
    words_per_row: u32,
    _pad0: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> qweight: array<u32>;

@group(0) @binding(2)
var<storage, read> scales: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

fn unpack_i8(word: u32, lane: u32) -> i32 {
    let byte = (word >> (lane * 8u)) & 0xffu;
    var signed = i32(byte);
    if (byte >= 128u) {
        signed = signed - 256;
    }
    return signed;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let out_idx = id.x;
    if (out_idx >= params.out_features) {
        return;
    }

    var sum = 0.0;
    let base = out_idx * params.words_per_row;
    for (var i = 0u; i < params.in_features; i = i + 1u) {
        let word = qweight[base + (i / 4u)];
        let q = unpack_i8(word, i & 3u);
        sum = sum + input[i] * f32(q);
    }
    output[out_idx] = sum * scales[out_idx];
}
"#;

const SWIGLU_SHADER: &str = r#"
struct Params {
    offset: u32,
    count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> gate: array<f32>;

@group(0) @binding(1)
var<storage, read> up: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= params.count) {
        return;
    }

    let g = gate[idx];
    output[params.offset + idx] = (g / (1.0 + exp(-g))) * up[idx];
}
"#;

#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

struct GpuContextInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matvec_pipeline: wgpu::ComputePipeline,
    matvec_bind_group_layout: wgpu::BindGroupLayout,
    q8_matvec_pipeline: wgpu::ComputePipeline,
    q8_matvec_bind_group_layout: wgpu::BindGroupLayout,
    swiglu_pipeline: wgpu::ComputePipeline,
    swiglu_bind_group_layout: wgpu::BindGroupLayout,
    max_chunk_bytes: wgpu::BufferAddress,
}

pub struct GpuMatVec {
    context: GpuContext,
    input_buffer: wgpu::Buffer,
    chunks: Vec<GpuMatVecChunk>,
    in_features: usize,
    out_features: usize,
}

pub struct GpuQ8MatVec {
    context: GpuContext,
    input_buffer: wgpu::Buffer,
    chunks: Vec<GpuQ8MatVecChunk>,
    in_features: usize,
    out_features: usize,
}

pub struct GpuSwiGluDown {
    context: GpuContext,
    bind_groups: Vec<GpuSwiGluBindGroup>,
    _params_buffers: Vec<wgpu::Buffer>,
}

pub struct GpuQ8SwiGluDown {
    context: GpuContext,
    bind_groups: Vec<GpuSwiGluBindGroup>,
    _params_buffers: Vec<wgpu::Buffer>,
}

struct GpuSwiGluBindGroup {
    bind_group: wgpu::BindGroup,
    out_features: usize,
}

struct GpuMatVecChunk {
    bind_group: wgpu::BindGroup,
    _weight_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    out_offset: usize,
    out_features: usize,
}

struct GpuQ8MatVecChunk {
    bind_group: wgpu::BindGroup,
    _qweight_buffer: wgpu::Buffer,
    _scales_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    _params_buffer: wgpu::Buffer,
    out_offset: usize,
    out_features: usize,
}

impl GpuContext {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("request_adapter failed: {e}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rsinfer-gpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| format!("request_device failed: {e}"))?;

        let matvec_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-lm-head-matvec"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MATVEC_SHADER)),
        });
        let matvec_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-lm-head-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let matvec_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-lm-head-pipeline-layout"),
                bind_group_layouts: &[Some(&matvec_bind_group_layout)],
                immediate_size: 0,
            });
        let matvec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-lm-head-pipeline"),
            layout: Some(&matvec_pipeline_layout),
            module: &matvec_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let q8_matvec_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-q8-matvec"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(Q8_MATVEC_SHADER)),
        });
        let q8_matvec_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-q8-matvec-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let q8_matvec_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-q8-matvec-pipeline-layout"),
                bind_group_layouts: &[Some(&q8_matvec_bind_group_layout)],
                immediate_size: 0,
            });
        let q8_matvec_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-q8-matvec-pipeline"),
            layout: Some(&q8_matvec_pipeline_layout),
            module: &q8_matvec_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let swiglu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-swiglu"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SWIGLU_SHADER)),
        });
        let swiglu_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rsinfer-swiglu-bind-layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let swiglu_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rsinfer-swiglu-pipeline-layout"),
                bind_group_layouts: &[Some(&swiglu_bind_group_layout)],
                immediate_size: 0,
            });
        let swiglu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-swiglu-pipeline"),
            layout: Some(&swiglu_pipeline_layout),
            module: &swiglu_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let limits = device.limits();
        let max_chunk_bytes = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);

        Ok(Self {
            inner: Arc::new(GpuContextInner {
                device,
                queue,
                matvec_pipeline,
                matvec_bind_group_layout,
                q8_matvec_pipeline,
                q8_matvec_bind_group_layout,
                swiglu_pipeline,
                swiglu_bind_group_layout,
                max_chunk_bytes,
            }),
        })
    }
}

impl GpuMatVec {
    pub fn from_f16_weight(
        weight: &[f16],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::from_f16_weight_with_context(&context, weight, out_features, in_features)
    }

    pub fn from_f16_weight_with_context(
        context: &GpuContext,
        weight: &[f16],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self, String> {
        if weight.len() != out_features * in_features {
            return Err(format!(
                "GPU matvec weight length mismatch: got {}, expected {}",
                weight.len(),
                out_features * in_features
            ));
        }

        let input_buffer = context.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-matvec-input"),
            size: bytes_len(in_features),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let row_bytes = bytes_len(in_features);
        let max_chunk_bytes = context.inner.max_chunk_bytes;
        if row_bytes == 0 || row_bytes > max_chunk_bytes {
            return Err(format!(
                "lm_head row is too large for one GPU storage buffer binding: row={row_bytes}, limit={max_chunk_bytes}"
            ));
        }
        let rows_per_chunk = (max_chunk_bytes / row_bytes).max(1) as usize;
        let mut chunks = Vec::new();
        let mut out_offset = 0usize;
        while out_offset < out_features {
            let rows = rows_per_chunk.min(out_features - out_offset);
            chunks.push(create_chunk(
                &context.inner.device,
                &context.inner.matvec_bind_group_layout,
                &input_buffer,
                weight,
                in_features,
                out_offset,
                rows,
            ));
            out_offset += rows;
        }

        Ok(Self {
            context: context.clone(),
            input_buffer,
            chunks,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        let mut outputs = Self::forward_many_same_input(&[self], x)?;
        outputs
            .pop()
            .ok_or_else(|| "GPU matvec returned no output".to_string())
    }

    pub fn forward_many_same_input(
        matvecs: &[&GpuMatVec],
        x: &Tensor,
    ) -> Result<Vec<Tensor>, String> {
        if matvecs.is_empty() {
            return Ok(Vec::new());
        }
        let first = matvecs[0];
        if x.ndim() != 2 || x.shape() != [1, first.in_features] {
            return Err(format!(
                "GPU matvec expects [1, {}], got {:?}",
                first.in_features,
                x.shape()
            ));
        }
        if matvecs
            .iter()
            .any(|matvec| matvec.in_features != first.in_features)
        {
            return Err("GPU matvec batch requires the same input width".to_string());
        }
        if matvecs
            .iter()
            .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &first.context.inner))
        {
            return Err("GPU matvec batch requires a shared GpuContext".to_string());
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU matvec input is not contiguous".to_string())?;
        for matvec in matvecs {
            matvec.context.inner.queue.write_buffer(
                &matvec.input_buffer,
                0,
                bytemuck::cast_slice(input),
            );
        }

        let mut encoder =
            first
                .context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-lm-head-encoder"),
                });
        for matvec in matvecs {
            for chunk in &matvec.chunks {
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rsinfer-lm-head-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&first.context.inner.matvec_pipeline);
                    pass.set_bind_group(0, &chunk.bind_group, &[]);
                    let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                encoder.copy_buffer_to_buffer(
                    &chunk.output_buffer,
                    0,
                    &chunk.readback_buffer,
                    0,
                    bytes_len(chunk.out_features),
                );
            }
        }
        first.context.inner.queue.submit(Some(encoder.finish()));

        let mut receivers = Vec::new();
        for (matvec_idx, matvec) in matvecs.iter().enumerate() {
            for (chunk_idx, chunk) in matvec.chunks.iter().enumerate() {
                let slice = chunk.readback_buffer.slice(..);
                let (tx, rx) = mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send((matvec_idx, chunk_idx, result.map_err(|e| e.to_string())));
                });
                receivers.push(rx);
            }
        }
        first
            .context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, _, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut outputs = Vec::with_capacity(matvecs.len());
        for matvec in matvecs {
            let mut output = vec![0f32; matvec.out_features];
            for chunk in &matvec.chunks {
                let slice = chunk.readback_buffer.slice(..);
                let mapped = slice.get_mapped_range();
                let values = bytemuck::cast_slice::<u8, f32>(&mapped);
                output[chunk.out_offset..chunk.out_offset + chunk.out_features]
                    .copy_from_slice(values);
                drop(mapped);
                chunk.readback_buffer.unmap();
            }
            outputs.push(
                Tensor::from_f32_slice(&[1, matvec.out_features], &output)
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(outputs)
    }

    pub fn forward_swiglu_down_same_input(
        gate: &GpuMatVec,
        up: &GpuMatVec,
        down: &GpuMatVec,
        x: &Tensor,
    ) -> Result<Tensor, String> {
        let fused = GpuSwiGluDown::new(gate, up, down)?;
        fused.forward(gate, up, down, x)
    }
}

impl GpuQ8MatVec {
    pub fn from_q8_weight(weight: &Q8LinearWeight) -> Result<Self, String> {
        let context = GpuContext::new()?;
        Self::from_q8_weight_with_context(&context, weight)
    }

    pub fn from_q8_weight_with_context(
        context: &GpuContext,
        weight: &Q8LinearWeight,
    ) -> Result<Self, String> {
        let expected = weight.out_features * weight.in_features;
        if weight.qweight.len() != expected {
            return Err(format!(
                "GPU Q8 matvec qweight length mismatch: got {}, expected {}",
                weight.qweight.len(),
                expected
            ));
        }
        if weight.scales.len() != weight.out_features {
            return Err(format!(
                "GPU Q8 matvec scale length mismatch: got {}, expected {}",
                weight.scales.len(),
                weight.out_features
            ));
        }
        if weight.in_features == 0 || weight.out_features == 0 {
            return Err("GPU Q8 matvec requires non-empty dimensions".to_string());
        }

        let input_buffer = context.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rsinfer-q8-matvec-input"),
            size: bytes_len(weight.in_features),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let words_per_row = weight.in_features.div_ceil(4);
        let qrow_bytes = bytes_len_u32(words_per_row);
        let row_binding_bytes = qrow_bytes.max(bytes_len(1));
        let max_chunk_bytes = context.inner.max_chunk_bytes;
        if row_binding_bytes > max_chunk_bytes {
            return Err(format!(
                "Q8 matvec row is too large for one GPU storage buffer binding: row={row_binding_bytes}, limit={max_chunk_bytes}"
            ));
        }
        let rows_per_chunk = (max_chunk_bytes / row_binding_bytes).max(1) as usize;
        let mut chunks = Vec::new();
        let mut out_offset = 0usize;
        while out_offset < weight.out_features {
            let rows = rows_per_chunk.min(weight.out_features - out_offset);
            chunks.push(create_q8_chunk(
                &context.inner.device,
                &context.inner.q8_matvec_bind_group_layout,
                &input_buffer,
                weight,
                words_per_row,
                out_offset,
                rows,
            ));
            out_offset += rows;
        }

        Ok(Self {
            context: context.clone(),
            input_buffer,
            chunks,
            in_features: weight.in_features,
            out_features: weight.out_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, String> {
        if x.ndim() != 2 || x.shape() != [1, self.in_features] {
            return Err(format!(
                "GPU Q8 matvec expects [1, {}], got {:?}",
                self.in_features,
                x.shape()
            ));
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU Q8 matvec input is not contiguous".to_string())?;
        self.context
            .inner
            .queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(input));

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-q8-matvec-encoder"),
                });
        for chunk in &self.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-q8-matvec-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.context.inner.q8_matvec_pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &chunk.output_buffer,
                0,
                &chunk.readback_buffer,
                0,
                bytes_len(chunk.out_features),
            );
        }
        self.context.inner.queue.submit(Some(encoder.finish()));

        let mut receivers = Vec::new();
        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            let slice = chunk.readback_buffer.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send((chunk_idx, result.map_err(|e| e.to_string())));
            });
            receivers.push(rx);
        }
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut output = vec![0f32; self.out_features];
        for chunk in &self.chunks {
            let slice = chunk.readback_buffer.slice(..);
            let mapped = slice.get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&mapped);
            output[chunk.out_offset..chunk.out_offset + chunk.out_features].copy_from_slice(values);
            drop(mapped);
            chunk.readback_buffer.unmap();
        }

        Tensor::from_f32_slice(&[1, self.out_features], &output).map_err(|e| e.to_string())
    }
}

impl GpuSwiGluDown {
    pub fn new(gate: &GpuMatVec, up: &GpuMatVec, down: &GpuMatVec) -> Result<Self, String> {
        validate_swiglu_down_matvecs(gate, up, down)?;

        let device = &gate.context.inner.device;
        let mut bind_groups = Vec::with_capacity(gate.chunks.len());
        let mut params_buffers = Vec::with_capacity(gate.chunks.len());
        for (gate_chunk, up_chunk) in gate.chunks.iter().zip(&up.chunks) {
            let params = [
                gate_chunk.out_offset as u32,
                gate_chunk.out_features as u32,
                0_u32,
                0_u32,
            ];
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsinfer-swiglu-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-swiglu-bind-group"),
                layout: &gate.context.inner.swiglu_bind_group_layout,
                entries: &[
                    buffer_entry(0, &gate_chunk.output_buffer),
                    buffer_entry(1, &up_chunk.output_buffer),
                    buffer_entry(2, &down.input_buffer),
                    buffer_entry(3, &params_buffer),
                ],
            });
            params_buffers.push(params_buffer);
            bind_groups.push(GpuSwiGluBindGroup {
                bind_group,
                out_features: gate_chunk.out_features,
            });
        }

        Ok(Self {
            context: gate.context.clone(),
            bind_groups,
            _params_buffers: params_buffers,
        })
    }

    pub fn forward(
        &self,
        gate: &GpuMatVec,
        up: &GpuMatVec,
        down: &GpuMatVec,
        x: &Tensor,
    ) -> Result<Tensor, String> {
        validate_swiglu_down_inputs(gate, up, down, x)?;
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner) {
            return Err(
                "GPU fused MLP cached resources require the original GpuContext".to_string(),
            );
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU fused MLP input is not contiguous".to_string())?;
        for matvec in [gate, up] {
            matvec.context.inner.queue.write_buffer(
                &matvec.input_buffer,
                0,
                bytemuck::cast_slice(input),
            );
        }

        let device = &self.context.inner.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rsinfer-fused-mlp-encoder"),
        });
        for matvec in [gate, up] {
            for chunk in &matvec.chunks {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-fused-mlp-gate-up-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gate.context.inner.matvec_pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        for cached in &self.bind_groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-swiglu-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gate.context.inner.swiglu_pipeline);
            pass.set_bind_group(0, &cached.bind_group, &[]);
            let workgroups = (cached.out_features as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        for chunk in &down.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-fused-mlp-down-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gate.context.inner.matvec_pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &chunk.output_buffer,
                0,
                &chunk.readback_buffer,
                0,
                bytes_len(chunk.out_features),
            );
        }
        gate.context.inner.queue.submit(Some(encoder.finish()));

        let mut receivers = Vec::new();
        for (chunk_idx, chunk) in down.chunks.iter().enumerate() {
            let slice = chunk.readback_buffer.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send((chunk_idx, result.map_err(|e| e.to_string())));
            });
            receivers.push(rx);
        }
        gate.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut output = vec![0f32; down.out_features];
        for chunk in &down.chunks {
            let slice = chunk.readback_buffer.slice(..);
            let mapped = slice.get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&mapped);
            output[chunk.out_offset..chunk.out_offset + chunk.out_features].copy_from_slice(values);
            drop(mapped);
            chunk.readback_buffer.unmap();
        }

        Tensor::from_f32_slice(&[1, down.out_features], &output).map_err(|e| e.to_string())
    }
}

impl GpuQ8SwiGluDown {
    pub fn new(gate: &GpuQ8MatVec, up: &GpuQ8MatVec, down: &GpuQ8MatVec) -> Result<Self, String> {
        validate_q8_swiglu_down_matvecs(gate, up, down)?;

        let device = &gate.context.inner.device;
        let mut bind_groups = Vec::with_capacity(gate.chunks.len());
        let mut params_buffers = Vec::with_capacity(gate.chunks.len());
        for (gate_chunk, up_chunk) in gate.chunks.iter().zip(&up.chunks) {
            let params = [
                gate_chunk.out_offset as u32,
                gate_chunk.out_features as u32,
                0_u32,
                0_u32,
            ];
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rsinfer-q8-swiglu-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rsinfer-q8-swiglu-bind-group"),
                layout: &gate.context.inner.swiglu_bind_group_layout,
                entries: &[
                    buffer_entry(0, &gate_chunk.output_buffer),
                    buffer_entry(1, &up_chunk.output_buffer),
                    buffer_entry(2, &down.input_buffer),
                    buffer_entry(3, &params_buffer),
                ],
            });
            params_buffers.push(params_buffer);
            bind_groups.push(GpuSwiGluBindGroup {
                bind_group,
                out_features: gate_chunk.out_features,
            });
        }

        Ok(Self {
            context: gate.context.clone(),
            bind_groups,
            _params_buffers: params_buffers,
        })
    }

    pub fn forward(
        &self,
        gate: &GpuQ8MatVec,
        up: &GpuQ8MatVec,
        down: &GpuQ8MatVec,
        x: &Tensor,
    ) -> Result<Tensor, String> {
        validate_q8_swiglu_down_inputs(gate, up, down, x)?;
        if !Arc::ptr_eq(&self.context.inner, &gate.context.inner) {
            return Err(
                "GPU Q8 fused MLP cached resources require the original GpuContext".to_string(),
            );
        }

        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU Q8 fused MLP input is not contiguous".to_string())?;
        for matvec in [gate, up] {
            matvec.context.inner.queue.write_buffer(
                &matvec.input_buffer,
                0,
                bytemuck::cast_slice(input),
            );
        }

        let device = &self.context.inner.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rsinfer-q8-fused-mlp-encoder"),
        });
        for matvec in [gate, up] {
            for chunk in &matvec.chunks {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-q8-fused-mlp-gate-up-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gate.context.inner.q8_matvec_pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        for cached in &self.bind_groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rsinfer-q8-swiglu-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gate.context.inner.swiglu_pipeline);
            pass.set_bind_group(0, &cached.bind_group, &[]);
            let workgroups = (cached.out_features as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        for chunk in &down.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-q8-fused-mlp-down-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&gate.context.inner.q8_matvec_pipeline);
                pass.set_bind_group(0, &chunk.bind_group, &[]);
                let workgroups = (chunk.out_features as u32).div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &chunk.output_buffer,
                0,
                &chunk.readback_buffer,
                0,
                bytes_len(chunk.out_features),
            );
        }
        gate.context.inner.queue.submit(Some(encoder.finish()));

        let mut receivers = Vec::new();
        for (chunk_idx, chunk) in down.chunks.iter().enumerate() {
            let slice = chunk.readback_buffer.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send((chunk_idx, result.map_err(|e| e.to_string())));
            });
            receivers.push(rx);
        }
        gate.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        for rx in receivers {
            let (_, result) = rx.recv().map_err(|e| format!("map callback failed: {e}"))?;
            result?;
        }

        let mut output = vec![0f32; down.out_features];
        for chunk in &down.chunks {
            let slice = chunk.readback_buffer.slice(..);
            let mapped = slice.get_mapped_range();
            let values = bytemuck::cast_slice::<u8, f32>(&mapped);
            output[chunk.out_offset..chunk.out_offset + chunk.out_features].copy_from_slice(values);
            drop(mapped);
            chunk.readback_buffer.unmap();
        }

        Tensor::from_f32_slice(&[1, down.out_features], &output).map_err(|e| e.to_string())
    }
}

fn validate_swiglu_down_matvecs(
    gate: &GpuMatVec,
    up: &GpuMatVec,
    down: &GpuMatVec,
) -> Result<(), String> {
    if gate.in_features != up.in_features {
        return Err("GPU fused MLP requires gate/up input width to match".to_string());
    }
    if gate.out_features != up.out_features || gate.out_features != down.in_features {
        return Err(format!(
            "GPU fused MLP shape mismatch: gate {}, up {}, down input {}",
            gate.out_features, up.out_features, down.in_features
        ));
    }
    if gate.chunks.iter().zip(&up.chunks).count() != gate.chunks.len()
        || gate.chunks.len() != up.chunks.len()
    {
        return Err("GPU fused MLP requires gate/up chunk count to match".to_string());
    }
    for (gate_chunk, up_chunk) in gate.chunks.iter().zip(&up.chunks) {
        if gate_chunk.out_offset != up_chunk.out_offset
            || gate_chunk.out_features != up_chunk.out_features
        {
            return Err("GPU fused MLP requires gate/up chunk layout to match".to_string());
        }
        if gate_chunk.out_offset > u32::MAX as usize || gate_chunk.out_features > u32::MAX as usize
        {
            return Err("GPU fused MLP chunk is too large for u32 shader params".to_string());
        }
    }
    if [&up, &down]
        .iter()
        .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &gate.context.inner))
    {
        return Err("GPU fused MLP requires a shared GpuContext".to_string());
    }
    Ok(())
}

fn validate_swiglu_down_inputs(
    gate: &GpuMatVec,
    up: &GpuMatVec,
    down: &GpuMatVec,
    x: &Tensor,
) -> Result<(), String> {
    validate_swiglu_down_matvecs(gate, up, down)?;
    if x.ndim() != 2 || x.shape() != [1, gate.in_features] {
        return Err(format!(
            "GPU fused MLP expects [1, {}], got {:?}",
            gate.in_features,
            x.shape()
        ));
    }
    Ok(())
}

fn validate_q8_swiglu_down_matvecs(
    gate: &GpuQ8MatVec,
    up: &GpuQ8MatVec,
    down: &GpuQ8MatVec,
) -> Result<(), String> {
    if gate.in_features != up.in_features {
        return Err("GPU Q8 fused MLP requires gate/up input width to match".to_string());
    }
    if gate.out_features != up.out_features || gate.out_features != down.in_features {
        return Err(format!(
            "GPU Q8 fused MLP shape mismatch: gate {}, up {}, down input {}",
            gate.out_features, up.out_features, down.in_features
        ));
    }
    if gate.chunks.iter().zip(&up.chunks).count() != gate.chunks.len()
        || gate.chunks.len() != up.chunks.len()
    {
        return Err("GPU Q8 fused MLP requires gate/up chunk count to match".to_string());
    }
    for (gate_chunk, up_chunk) in gate.chunks.iter().zip(&up.chunks) {
        if gate_chunk.out_offset != up_chunk.out_offset
            || gate_chunk.out_features != up_chunk.out_features
        {
            return Err("GPU Q8 fused MLP requires gate/up chunk layout to match".to_string());
        }
        if gate_chunk.out_offset > u32::MAX as usize || gate_chunk.out_features > u32::MAX as usize
        {
            return Err("GPU Q8 fused MLP chunk is too large for u32 shader params".to_string());
        }
    }
    if [&up, &down]
        .iter()
        .any(|matvec| !Arc::ptr_eq(&matvec.context.inner, &gate.context.inner))
    {
        return Err("GPU Q8 fused MLP requires a shared GpuContext".to_string());
    }
    Ok(())
}

fn validate_q8_swiglu_down_inputs(
    gate: &GpuQ8MatVec,
    up: &GpuQ8MatVec,
    down: &GpuQ8MatVec,
    x: &Tensor,
) -> Result<(), String> {
    validate_q8_swiglu_down_matvecs(gate, up, down)?;
    if x.ndim() != 2 || x.shape() != [1, gate.in_features] {
        return Err(format!(
            "GPU Q8 fused MLP expects [1, {}], got {:?}",
            gate.in_features,
            x.shape()
        ));
    }
    Ok(())
}

fn bytes_len(items: usize) -> wgpu::BufferAddress {
    (items * std::mem::size_of::<f32>()) as wgpu::BufferAddress
}

fn bytes_len_u32(items: usize) -> wgpu::BufferAddress {
    (items * std::mem::size_of::<u32>()) as wgpu::BufferAddress
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_chunk(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    input_buffer: &wgpu::Buffer,
    weight: &[f16],
    in_features: usize,
    out_offset: usize,
    out_features: usize,
) -> GpuMatVecChunk {
    let start = out_offset * in_features;
    let end = start + out_features * in_features;
    let weight_f32: Vec<f32> = weight[start..end]
        .iter()
        .map(|value| value.to_f32())
        .collect();
    let weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-lm-head-weight-chunk"),
        contents: bytemuck::cast_slice(&weight_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-lm-head-output-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-lm-head-readback-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = [in_features as u32, out_features as u32, 0_u32, 0_u32];
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-lm-head-params-chunk"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rsinfer-lm-head-bind-group-chunk"),
        layout: bind_group_layout,
        entries: &[
            buffer_entry(0, input_buffer),
            buffer_entry(1, &weight_buffer),
            buffer_entry(2, &output_buffer),
            buffer_entry(3, &params_buffer),
        ],
    });
    GpuMatVecChunk {
        bind_group,
        _weight_buffer: weight_buffer,
        output_buffer,
        readback_buffer,
        _params_buffer: params_buffer,
        out_offset,
        out_features,
    }
}

fn create_q8_chunk(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    input_buffer: &wgpu::Buffer,
    weight: &Q8LinearWeight,
    words_per_row: usize,
    out_offset: usize,
    out_features: usize,
) -> GpuQ8MatVecChunk {
    let mut packed = vec![0u32; out_features * words_per_row];
    for row in 0..out_features {
        let src_row = out_offset + row;
        let src_start = src_row * weight.in_features;
        let dst_start = row * words_per_row;
        pack_i8_row_to_u32(
            &weight.qweight[src_start..src_start + weight.in_features],
            &mut packed[dst_start..dst_start + words_per_row],
        );
    }
    let qweight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-q8-matvec-weight-chunk"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scales_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-q8-matvec-scales-chunk"),
        contents: bytemuck::cast_slice(&weight.scales[out_offset..out_offset + out_features]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-q8-matvec-output-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rsinfer-q8-matvec-readback-chunk"),
        size: bytes_len(out_features),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = [
        weight.in_features as u32,
        out_features as u32,
        words_per_row as u32,
        0_u32,
    ];
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rsinfer-q8-matvec-params-chunk"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rsinfer-q8-matvec-bind-group-chunk"),
        layout: bind_group_layout,
        entries: &[
            buffer_entry(0, input_buffer),
            buffer_entry(1, &qweight_buffer),
            buffer_entry(2, &scales_buffer),
            buffer_entry(3, &output_buffer),
            buffer_entry(4, &params_buffer),
        ],
    });

    GpuQ8MatVecChunk {
        bind_group,
        _qweight_buffer: qweight_buffer,
        _scales_buffer: scales_buffer,
        output_buffer,
        readback_buffer,
        _params_buffer: params_buffer,
        out_offset,
        out_features,
    }
}

fn pack_i8_row_to_u32(input: &[i8], output: &mut [u32]) {
    output.fill(0);
    for (idx, &value) in input.iter().enumerate() {
        let word_idx = idx / 4;
        let shift = (idx % 4) * 8;
        output[word_idx] |= (value as u8 as u32) << shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{linear_forward_f16, linear_forward_q8, silu, Q8LinearWeight};

    #[test]
    fn gpu_matvec_matches_cpu_for_small_tensor_when_available() {
        let weight = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
        ];
        let x = Tensor::from_f32_slice(&[1, 3], &[0.25, -0.5, 2.0]).unwrap();
        let cpu = linear_forward_f16(&x, &weight, 2, 3).unwrap();
        let Ok(gpu) = GpuMatVec::from_f16_weight(&weight, 2, 3) else {
            eprintln!("GPU matvec test skipped: no usable wgpu adapter");
            return;
        };
        let got = gpu.forward(&x).unwrap();
        let max_abs = cpu
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU matvec max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn gpu_q8_matvec_matches_cpu_q8_when_available() {
        let weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
            f16::from_f32(1.25),
            f16::from_f32(-0.125),
            f16::from_f32(0.875),
            f16::from_f32(-1.5),
        ];
        let q8 = Q8LinearWeight::from_f16(&weight, 2, 5).unwrap();
        let x = Tensor::from_f32_slice(&[1, 5], &[0.6, -1.4, 1.0, 0.25, -0.5]).unwrap();
        let cpu = linear_forward_q8(&x, &q8).unwrap();
        let Ok(gpu) = GpuQ8MatVec::from_q8_weight(&q8) else {
            eprintln!("GPU Q8 matvec test skipped: no usable wgpu adapter");
            return;
        };
        let got = gpu.forward(&x).unwrap();
        let max_abs = cpu
            .as_slice()
            .iter()
            .zip(got.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 1e-4,
            "GPU Q8 matvec max abs diff {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn shared_context_can_drive_multiple_matvecs_when_available() {
        let Ok(context) = GpuContext::new() else {
            eprintln!("shared GPU context test skipped: no usable wgpu adapter");
            return;
        };
        let weight_a = [
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
        ];
        let weight_b = [
            f16::from_f32(-0.25),
            f16::from_f32(2.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[2.0, -1.0]).unwrap();
        let cpu_a = linear_forward_f16(&x, &weight_a, 2, 2).unwrap();
        let cpu_b = linear_forward_f16(&x, &weight_b, 2, 2).unwrap();
        let gpu_a = GpuMatVec::from_f16_weight_with_context(&context, &weight_a, 2, 2)
            .unwrap()
            .forward(&x)
            .unwrap();
        let gpu_b = GpuMatVec::from_f16_weight_with_context(&context, &weight_b, 2, 2)
            .unwrap()
            .forward(&x)
            .unwrap();

        for (expected, actual) in cpu_a.as_slice().iter().zip(gpu_a.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
        for (expected, actual) in cpu_b.as_slice().iter().zip(gpu_b.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn batched_same_input_matvecs_match_cpu_when_available() {
        let Ok(context) = GpuContext::new() else {
            eprintln!("batched GPU matvec test skipped: no usable wgpu adapter");
            return;
        };
        let weight_a = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let weight_b = [
            f16::from_f32(-1.0),
            f16::from_f32(0.5),
            f16::from_f32(2.0),
            f16::from_f32(-0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.75, -1.25]).unwrap();
        let cpu_a = linear_forward_f16(&x, &weight_a, 2, 2).unwrap();
        let cpu_b = linear_forward_f16(&x, &weight_b, 2, 2).unwrap();
        let gpu_a = GpuMatVec::from_f16_weight_with_context(&context, &weight_a, 2, 2).unwrap();
        let gpu_b = GpuMatVec::from_f16_weight_with_context(&context, &weight_b, 2, 2).unwrap();
        let outputs = GpuMatVec::forward_many_same_input(&[&gpu_a, &gpu_b], &x).unwrap();

        assert_eq!(outputs.len(), 2);
        for (expected, actual) in cpu_a.as_slice().iter().zip(outputs[0].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
        for (expected, actual) in cpu_b.as_slice().iter().zip(outputs[1].as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn fused_swiglu_down_matches_cpu_when_available() {
        let Ok(context) = GpuContext::new() else {
            eprintln!("fused GPU MLP test skipped: no usable wgpu adapter");
            return;
        };
        let gate_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let up_weight = [
            f16::from_f32(1.0),
            f16::from_f32(0.25),
            f16::from_f32(-0.5),
            f16::from_f32(0.75),
            f16::from_f32(1.25),
            f16::from_f32(-1.0),
        ];
        let down_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-0.25),
            f16::from_f32(1.0),
            f16::from_f32(-1.5),
            f16::from_f32(0.75),
            f16::from_f32(0.25),
        ];
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let gate = linear_forward_f16(&x, &gate_weight, 3, 2).unwrap();
        let up = linear_forward_f16(&x, &up_weight, 3, 2).unwrap();
        let hidden = silu(&gate).mul(&up).unwrap();
        let cpu = linear_forward_f16(&hidden, &down_weight, 2, 3).unwrap();

        let gate_gpu =
            GpuMatVec::from_f16_weight_with_context(&context, &gate_weight, 3, 2).unwrap();
        let up_gpu = GpuMatVec::from_f16_weight_with_context(&context, &up_weight, 3, 2).unwrap();
        let down_gpu =
            GpuMatVec::from_f16_weight_with_context(&context, &down_weight, 2, 3).unwrap();
        let fused = GpuSwiGluDown::new(&gate_gpu, &up_gpu, &down_gpu).unwrap();
        let got = fused.forward(&gate_gpu, &up_gpu, &down_gpu, &x).unwrap();

        for (expected, actual) in cpu.as_slice().iter().zip(got.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }

    #[test]
    fn q8_fused_swiglu_down_matches_cpu_q8_when_available() {
        let Ok(context) = GpuContext::new() else {
            eprintln!("fused GPU Q8 MLP test skipped: no usable wgpu adapter");
            return;
        };
        let gate_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-1.0),
            f16::from_f32(1.5),
            f16::from_f32(0.25),
            f16::from_f32(-0.75),
            f16::from_f32(0.5),
        ];
        let up_weight = [
            f16::from_f32(1.0),
            f16::from_f32(0.25),
            f16::from_f32(-0.5),
            f16::from_f32(0.75),
            f16::from_f32(1.25),
            f16::from_f32(-1.0),
        ];
        let down_weight = [
            f16::from_f32(0.5),
            f16::from_f32(-0.25),
            f16::from_f32(1.0),
            f16::from_f32(-1.5),
            f16::from_f32(0.75),
            f16::from_f32(0.25),
        ];
        let gate_q8 = Q8LinearWeight::from_f16(&gate_weight, 3, 2).unwrap();
        let up_q8 = Q8LinearWeight::from_f16(&up_weight, 3, 2).unwrap();
        let down_q8 = Q8LinearWeight::from_f16(&down_weight, 2, 3).unwrap();
        let x = Tensor::from_f32_slice(&[1, 2], &[0.6, -1.4]).unwrap();
        let gate = linear_forward_q8(&x, &gate_q8).unwrap();
        let up = linear_forward_q8(&x, &up_q8).unwrap();
        let hidden = silu(&gate).mul(&up).unwrap();
        let cpu = linear_forward_q8(&hidden, &down_q8).unwrap();

        let gate_gpu = GpuQ8MatVec::from_q8_weight_with_context(&context, &gate_q8).unwrap();
        let up_gpu = GpuQ8MatVec::from_q8_weight_with_context(&context, &up_q8).unwrap();
        let down_gpu = GpuQ8MatVec::from_q8_weight_with_context(&context, &down_q8).unwrap();
        let fused = GpuQ8SwiGluDown::new(&gate_gpu, &up_gpu, &down_gpu).unwrap();
        let got = fused.forward(&gate_gpu, &up_gpu, &down_gpu, &x).unwrap();

        for (expected, actual) in cpu.as_slice().iter().zip(got.as_slice()) {
            assert!((expected - actual).abs() <= 1e-4);
        }
    }
}
