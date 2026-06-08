//! 可选 GPU 计算内核。
//!
//! 目前只实现 `lm_head` 场景需要的单行 matvec：
//! `y = x @ W^T`，其中 `x` 是 `[1, in_features]`，`W` 是
//! `[out_features, in_features]`。初始化或执行失败时上层会回退 CPU。

use std::borrow::Cow;
use std::sync::{mpsc, Arc};

use half::f16;
use wgpu::util::DeviceExt;

use crate::tensor::Tensor;

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

#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

struct GpuContextInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    max_chunk_bytes: wgpu::BufferAddress,
}

pub struct GpuMatVec {
    context: GpuContext,
    input_buffer: wgpu::Buffer,
    chunks: Vec<GpuMatVecChunk>,
    in_features: usize,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rsinfer-lm-head-matvec"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(MATVEC_SHADER)),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rsinfer-lm-head-pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rsinfer-lm-head-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
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
                pipeline,
                bind_group_layout,
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
                &context.inner.bind_group_layout,
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
        if x.ndim() != 2 || x.shape() != [1, self.in_features] {
            return Err(format!(
                "GPU matvec expects [1, {}], got {:?}",
                self.in_features,
                x.shape()
            ));
        }
        let x_std = x.data.as_standard_layout();
        let input = x_std
            .as_slice()
            .ok_or_else(|| "GPU matvec input is not contiguous".to_string())?;
        self.context
            .inner
            .queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(input));

        let mut encoder =
            self.context
                .inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rsinfer-lm-head-encoder"),
                });
        for chunk in &self.chunks {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rsinfer-lm-head-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.context.inner.pipeline);
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

        let mut receivers = Vec::with_capacity(self.chunks.len());
        for chunk in &self.chunks {
            let slice = chunk.readback_buffer.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result.map_err(|e| e.to_string()));
            });
            receivers.push(rx);
        }
        self.context
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("device poll failed: {e}"))?;

        let mut output = vec![0f32; self.out_features];
        for (chunk, rx) in self.chunks.iter().zip(receivers) {
            rx.recv()
                .map_err(|e| format!("map callback failed: {e}"))??;
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

fn bytes_len(items: usize) -> wgpu::BufferAddress {
    (items * std::mem::size_of::<f32>()) as wgpu::BufferAddress
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::linear_forward_f16;

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
}
