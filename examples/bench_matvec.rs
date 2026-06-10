use std::time::Instant;

use rsinfer::gpu::{GpuQ8MatVec, GpuResidentBuffer};
use rsinfer::tensor::{Q8LinearWeight, Tensor};

const WARMUP_ITERS: usize = 10;
const MEASURE_ITERS: usize = 100;

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_i8_qweight(&mut self) -> i8 {
        let value = (self.next_u32() % 255) as i16 - 127;
        value as i8
    }

    fn next_f32_unit(&mut self) -> f32 {
        let value = self.next_u32() as f64 / u32::MAX as f64;
        (value as f32) * 2.0 - 1.0
    }
}

fn make_q8_weight(out_features: usize, in_features: usize, seed: u64) -> Q8LinearWeight {
    let mut rng = Lcg::new(seed);
    let qweight = (0..out_features * in_features)
        .map(|_| rng.next_i8_qweight())
        .collect();
    let scales = (0..out_features)
        .map(|_| 0.0025 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.02)
        .collect();
    Q8LinearWeight {
        qweight,
        scales,
        out_features,
        in_features,
    }
}

fn make_input(in_features: usize, seed: u64) -> Tensor {
    let mut rng = Lcg::new(seed);
    let values = (0..in_features).map(|_| rng.next_f32_unit()).collect();
    Tensor::from_f32_vec(&[1, in_features], values).expect("input shape should be valid")
}

fn run_dispatches(
    gpu: &GpuQ8MatVec,
    input: &GpuResidentBuffer,
    output: &GpuResidentBuffer,
    iterations: usize,
) {
    let context = gpu.shared_context();
    let mut encoder = context.create_command_encoder("rsinfer-q8-matvec-microbench");
    for _ in 0..iterations {
        gpu.encode_resident(&mut encoder, input, output)
            .expect("resident q8 matvec encode should succeed");
    }
    context.submit(encoder);
    let _ = output
        .read_back()
        .expect("microbench readback should succeed");
}

fn bench_shape(out_features: usize, in_features: usize, seed: u64) {
    let q8 = make_q8_weight(out_features, in_features, seed);
    let input_tensor = make_input(in_features, seed ^ 0x9e37_79b9_7f4a_7c15);
    let gpu = GpuQ8MatVec::from_q8_weight(&q8).expect("usable wgpu adapter required");
    let context = gpu.shared_context();
    let input =
        GpuResidentBuffer::from_tensor(&context, &input_tensor).expect("input upload should work");
    let output = GpuResidentBuffer::with_context(&context, &[1, out_features])
        .expect("output allocation should work");

    for _ in 0..WARMUP_ITERS {
        run_dispatches(&gpu, &input, &output, 1);
    }

    let start = Instant::now();
    run_dispatches(&gpu, &input, &output, MEASURE_ITERS);
    let elapsed = start.elapsed();
    let qweight_gb = (q8.qweight.len() * MEASURE_ITERS) as f64 / 1_000_000_000.0;
    let gb_s = qweight_gb / elapsed.as_secs_f64();

    println!(
        "bench.q8_matvec out_features={} in_features={} warmup_iters={} measure_iters={} elapsed_ms={:.3} qweight_gb={:.6} effective_gb_s={:.3}",
        out_features,
        in_features,
        WARMUP_ITERS,
        MEASURE_ITERS,
        elapsed.as_secs_f64() * 1000.0,
        qweight_gb,
        gb_s
    );
}

fn main() {
    bench_shape(9728, 2560, 0x2026_0610_0001);
    bench_shape(2560, 9728, 0x2026_0610_0002);
}
