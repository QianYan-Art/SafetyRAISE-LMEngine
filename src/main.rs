//! rsinfer CLI：命令行与交互式对话入口。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rsinfer::engine::{CombinedSampler, GenerationProfile, Generator, GreedySampler, Sampler};
use rsinfer::runtime::{DevicePreference, QuantizationCacheMode, QuantizationMode, RuntimeOptions};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceArg {
    Cpu,
    Auto,
    Hybrid,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QuantizationArg {
    None,
    Q8,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum QuantizationCacheArg {
    Auto,
    Off,
}

impl From<DeviceArg> for DevicePreference {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Cpu => DevicePreference::Cpu,
            DeviceArg::Auto => DevicePreference::Auto,
            DeviceArg::Hybrid => DevicePreference::Hybrid,
        }
    }
}

impl From<QuantizationArg> for QuantizationMode {
    fn from(value: QuantizationArg) -> Self {
        match value {
            QuantizationArg::None => Self::None,
            QuantizationArg::Q8 => Self::Q8,
        }
    }
}

impl From<QuantizationCacheArg> for QuantizationCacheMode {
    fn from(value: QuantizationCacheArg) -> Self {
        match value {
            QuantizationCacheArg::Auto => Self::Auto,
            QuantizationCacheArg::Off => Self::Off,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 模型目录 (含 config.json, model.safetensors, tokenizer.json)
    #[arg(short, long)]
    model_path: PathBuf,

    /// 输入提示词；省略则进入交互模式
    #[arg(short, long)]
    prompt: Option<String>,

    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,

    /// 采样温度 (0 表示贪心)。Qwen3-Thinking 推荐 0.6
    #[arg(short, long, default_value_t = 0.6)]
    temperature: f32,

    /// Qwen3-Thinking 推荐 0.95
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Qwen3-Thinking 推荐 20
    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// 用 Qwen3 chat 模板包装输入
    #[arg(long)]
    chat: bool,

    /// system 提示词 (仅 chat 模式)
    #[arg(long)]
    system: Option<String>,

    #[arg(short, long)]
    interactive: bool,

    #[arg(short, long)]
    verbose: bool,

    /// 打印 token 级生成耗时 profile；默认关闭
    #[arg(long)]
    profile_tokens: bool,

    /// 打印 Transformer 层级耗时 profile；默认关闭，通常与 --profile-tokens 一起使用
    #[arg(long)]
    profile_layers: bool,

    /// 打印 decode_forward 的 per-token 漂移摘要；默认关闭，通常与 --profile-tokens 一起使用
    #[arg(long)]
    profile_token_trace: bool,

    /// 推理设备规划: cpu 保持现有 CPU 路径；auto/hybrid 探测 GPU 并生成混合放置计划
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,

    /// 计划放到 GPU 的 Transformer 层数；省略时按模式保守选择，当前本机 Q8 hybrid 默认 35
    #[arg(long)]
    gpu_layers: Option<usize>,

    /// 权重量化路径: none 保持 f16；q8 启用行级 Q8 CPU linear fallback
    #[arg(long, value_enum, default_value_t = QuantizationArg::None)]
    quantization: QuantizationArg,

    /// Q8 派生权重缓存: auto 写入/读取模型目录旁的 sidecar；off 每次内存派生
    #[arg(long = "q8-cache", value_enum, default_value_t = QuantizationCacheArg::Auto)]
    q8_cache: QuantizationCacheArg,

    /// Q8 sidecar 目录；默认使用模型目录旁的 <model>.rsinfer-q8
    #[arg(long = "q8-cache-dir")]
    q8_cache_dir: Option<PathBuf>,
}

const IM_START: &str = "\x3c|im_start|>";
const IM_END: &str = "\x3c|im_end|>";

struct Message {
    role: &'static str,
    content: String,
}

/// 渲染 Qwen3 chat 模板。生成提示词固定以 `<think>\n` 收尾——该 thinking 模型强制思考。
fn render_chat(system: Option<&str>, history: &[Message]) -> String {
    let mut s = String::new();
    if let Some(sys) = system {
        s.push_str(&format!("{IM_START}system\n{sys}{IM_END}\n"));
    }
    for m in history {
        s.push_str(&format!("{IM_START}{}\n{}{IM_END}\n", m.role, m.content));
    }
    s.push_str(&format!("{IM_START}assistant\n\x3cthink\x3e\n"));
    s
}

/// 去掉思考段，取 `</think>` 之后的最终回答，用于写入对话历史。
fn answer_only(full: &str) -> &str {
    match full.split_once("\x3c/think\x3e") {
        Some((_, ans)) => ans.trim(),
        None => full.trim(),
    }
}

fn unescape(s: &str) -> String {
    s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\\", "\\")
}

fn main() -> rsinfer::Result<()> {
    let args = Args::parse();

    println!("正在加载模型: {:?}", args.model_path);
    let start = std::time::Instant::now();
    let runtime_options = RuntimeOptions {
        device: args.device.into(),
        gpu_layers: args.gpu_layers,
        quantization: args.quantization.into(),
        quantization_cache: args.q8_cache.into(),
        q8_cache_dir: args.q8_cache_dir.clone(),
    };
    let generator = Generator::from_pretrained_with_options(&args.model_path, &runtime_options)?;
    println!("模型加载完成，耗时: {:.2}s", start.elapsed().as_secs_f32());
    if args.verbose {
        print!("{}", generator.model.runtime_plan);
    }

    let sampler: Box<dyn Sampler> = if args.temperature <= 0.0 {
        Box::new(GreedySampler)
    } else {
        Box::new(CombinedSampler::new(
            args.temperature,
            args.top_k,
            args.top_p,
        ))
    };
    let generator = generator
        .with_sampler(sampler)
        .with_max_tokens(args.max_tokens);

    match (args.interactive, &args.prompt) {
        (false, Some(prompt)) => {
            let prompt = unescape(prompt);
            let final_prompt = if args.chat {
                render_chat(
                    args.system.as_deref(),
                    &[Message {
                        role: "user",
                        content: prompt,
                    }],
                )
            } else {
                prompt
            };
            generate_and_print(
                &generator,
                &final_prompt,
                args.verbose,
                args.profile_tokens,
                args.profile_layers,
                args.profile_token_trace,
            )?;
        }
        _ => run_interactive(
            &generator,
            args.chat,
            args.system.as_deref(),
            args.verbose,
            args.profile_tokens,
            args.profile_layers,
            args.profile_token_trace,
        )?,
    }
    Ok(())
}

fn avg_ms(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn format_trace_window(values: &[f64], start: usize, end: usize) -> String {
    values[start..end]
        .iter()
        .enumerate()
        .map(|(offset, value)| format!("t{}={value:.3}", start + offset + 1))
        .collect::<Vec<_>>()
        .join(",")
}

fn trace_window_bounds(len: usize) -> (usize, usize, usize, usize) {
    let first_end = len.min(4);
    let middle_len = len.min(4);
    let middle_start = (len.saturating_sub(middle_len)) / 2;
    let middle_end = middle_start + middle_len;
    let last_start = len.saturating_sub(4);
    (first_end, middle_start, middle_end, last_start)
}

fn print_decode_forward_trace_summary(profile: &GenerationProfile) {
    let values = &profile.decode_forward_token_ms;
    if values.is_empty() {
        println!("profile.decode_forward_trace count=0");
        return;
    }

    let (first_end, middle_start, middle_end, last_start) = trace_window_bounds(values.len());

    println!(
        "profile.decode_forward_trace count={} first=[{}] middle=[{}] last=[{}]",
        values.len(),
        format_trace_window(values, 0, first_end),
        format_trace_window(values, middle_start, middle_end),
        format_trace_window(values, last_start, values.len()),
    );
    println!(
        "profile.decode_forward_trace_avg first={:.3} middle={:.3} last={:.3}",
        avg_ms(&values[..first_end]),
        avg_ms(&values[middle_start..middle_end]),
        avg_ms(&values[last_start..]),
    );
}

fn print_final_layer_trace_avg(
    layer_index: usize,
    field: &str,
    values: &[f64],
    first_end: usize,
    middle_start: usize,
    middle_end: usize,
    last_start: usize,
) {
    println!(
        "profile.layer_trace_avg layer={} field={} first={:.3} middle={:.3} last={:.3}",
        layer_index,
        field,
        avg_ms(&values[..first_end]),
        avg_ms(&values[middle_start..middle_end]),
        avg_ms(&values[last_start..]),
    );
}

fn print_final_layer_decode_trace_summary(profile: &GenerationProfile) {
    let Some(layer_index) = profile.final_layer_index else {
        return;
    };
    let traces = &profile.final_layer_decode_trace;
    if traces.is_empty() {
        return;
    }

    let (first_end, middle_start, middle_end, last_start) = trace_window_bounds(traces.len());
    let total_values: Vec<f64> = traces.iter().map(|trace| trace.total_ms).collect();
    println!(
        "profile.layer_trace layer={} count={} total_first=[{}] middle=[{}] last=[{}]",
        layer_index,
        traces.len(),
        format_trace_window(&total_values, 0, first_end),
        format_trace_window(&total_values, middle_start, middle_end),
        format_trace_window(&total_values, last_start, total_values.len()),
    );

    let fields = [
        (
            "total",
            traces
                .iter()
                .map(|trace| trace.total_ms)
                .collect::<Vec<_>>(),
        ),
        (
            "attention",
            traces
                .iter()
                .map(|trace| trace.attention_ms)
                .collect::<Vec<_>>(),
        ),
        (
            "mlp",
            traces.iter().map(|trace| trace.mlp_ms).collect::<Vec<_>>(),
        ),
        (
            "mlp_gate_up",
            traces
                .iter()
                .map(|trace| trace.mlp_gate_up_ms)
                .collect::<Vec<_>>(),
        ),
        (
            "mlp_down_proj",
            traces
                .iter()
                .map(|trace| trace.mlp_down_proj_ms)
                .collect::<Vec<_>>(),
        ),
        (
            "mlp_q8_gate_up_dot",
            traces
                .iter()
                .map(|trace| trace.mlp_q8_gate_up_dot_ms)
                .collect::<Vec<_>>(),
        ),
        (
            "mlp_q8_down_proj_dot",
            traces
                .iter()
                .map(|trace| trace.mlp_q8_down_proj_dot_ms)
                .collect::<Vec<_>>(),
        ),
    ];
    for (field, values) in fields {
        print_final_layer_trace_avg(
            layer_index,
            field,
            &values,
            first_end,
            middle_start,
            middle_end,
            last_start,
        );
    }
}

fn generate_and_print(
    generator: &Generator,
    prompt: &str,
    verbose: bool,
    profile_tokens: bool,
    profile_layers: bool,
    profile_token_trace: bool,
) -> rsinfer::Result<String> {
    if verbose {
        println!("=== 输入 prompt ===\n{prompt}\n=== 开始生成 ===");
    }
    let start = std::time::Instant::now();
    let (full, profile) = generator.generate_stream_with_profile_options(
        prompt,
        profile_layers,
        profile_token_trace,
        |t| {
            print!("{t}");
            io::stdout().flush().ok();
        },
    )?;
    println!();
    if verbose {
        println!(
            "\n--- 生成完成 (耗时: {:.2}s) ---",
            start.elapsed().as_secs_f32()
        );
    }
    if profile_tokens {
        println!(
            "profile.tokens prompt={} generated={} fast_path={}",
            profile.prompt_tokens, profile.generated_tokens, profile.fast_path_tokens
        );
        println!(
            "profile.time_ms prefill_forward={:.3} prefill_sample={:.3} decode_forward={:.3} decode_sample={:.3} text_decode={:.3} avg_decode_forward_per_token={:.3}",
            profile.prefill_forward.as_secs_f64() * 1000.0,
            profile.prefill_sample.as_secs_f64() * 1000.0,
            profile.decode_forward.as_secs_f64() * 1000.0,
            profile.decode_sample.as_secs_f64() * 1000.0,
            profile.text_decode.as_secs_f64() * 1000.0,
            profile.avg_decode_forward_ms(),
        );
    }
    if profile_token_trace {
        print_decode_forward_trace_summary(&profile);
        print_final_layer_decode_trace_summary(&profile);
    }
    if profile_layers {
        for (idx, layer) in profile.layer_profiles.iter().enumerate() {
            println!(
                "profile.layer_ms layer={} total={:.3} input_norm={:.3} attention={:.3} post_norm={:.3} mlp={:.3} mlp_gate_up={:.3} mlp_silu_mul={:.3} mlp_down_proj={:.3} mlp_q8_gate_up_prep={:.3} mlp_q8_gate_up_dot={:.3} mlp_q8_gate_up_writeback={:.3} mlp_q8_down_prep={:.3} mlp_q8_down_dot={:.3} mlp_q8_down_writeback={:.3} residual={:.3}",
                idx,
                layer.total.as_secs_f64() * 1000.0,
                layer.input_norm.as_secs_f64() * 1000.0,
                layer.attention.as_secs_f64() * 1000.0,
                layer.post_norm.as_secs_f64() * 1000.0,
                layer.mlp.as_secs_f64() * 1000.0,
                layer.mlp_gate_up.as_secs_f64() * 1000.0,
                layer.mlp_silu_mul.as_secs_f64() * 1000.0,
                layer.mlp_down_proj.as_secs_f64() * 1000.0,
                layer.mlp_q8_gate_up_prep.as_secs_f64() * 1000.0,
                layer.mlp_q8_gate_up_dot.as_secs_f64() * 1000.0,
                layer.mlp_q8_gate_up_writeback.as_secs_f64() * 1000.0,
                layer.mlp_q8_down_proj_prep.as_secs_f64() * 1000.0,
                layer.mlp_q8_down_proj_dot.as_secs_f64() * 1000.0,
                layer.mlp_q8_down_proj_writeback.as_secs_f64() * 1000.0,
                layer.residual.as_secs_f64() * 1000.0,
            );
        }
    }
    Ok(full)
}

fn run_interactive(
    generator: &Generator,
    use_chat: bool,
    system: Option<&str>,
    verbose: bool,
    profile_tokens: bool,
    profile_layers: bool,
    profile_token_trace: bool,
) -> rsinfer::Result<()> {
    println!("\n=== rsinfer 交互式模式 ===");
    if use_chat {
        println!("Chat 模式 (Qwen3 模板, thinking 强制开启, 保留多轮历史)");
    }
    println!("命令: 'exit'/'quit' 退出, 'reset' 清空对话历史\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut history: Vec<Message> = Vec::new();

    loop {
        print!("用户: ");
        stdout.flush().ok();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();

        match input {
            "" => continue,
            "exit" | "quit" => {
                println!("再见！");
                break;
            }
            "reset" => {
                history.clear();
                println!("(已清空对话历史)");
                continue;
            }
            _ => {}
        }

        let prompt = if use_chat {
            history.push(Message {
                role: "user",
                content: input.to_string(),
            });
            render_chat(system, &history)
        } else {
            input.to_string()
        };

        print!("助手: ");
        stdout.flush().ok();
        let full = generate_and_print(
            generator,
            &prompt,
            verbose,
            profile_tokens,
            profile_layers,
            profile_token_trace,
        )?;

        if use_chat {
            history.push(Message {
                role: "assistant",
                content: answer_only(&full).to_string(),
            });
        }
        println!();
    }
    Ok(())
}
