//! rsinfer CLI：命令行与交互式对话入口。

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use rsinfer::engine::{CombinedSampler, Generator, GreedySampler, Sampler};
use rsinfer::runtime::{DevicePreference, QuantizationMode, RuntimeOptions};

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

    /// 推理设备规划: cpu 保持现有 CPU 路径；auto/hybrid 探测 GPU 并生成混合放置计划
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,

    /// 计划放到 GPU 的 Transformer 层数；当前 GPU kernel 未接入，执行仍会明确标注为 CPU
    #[arg(long)]
    gpu_layers: Option<usize>,

    /// 权重量化路径: none 保持 f16；q8 启用行级 Q8 CPU linear fallback
    #[arg(long, value_enum, default_value_t = QuantizationArg::None)]
    quantization: QuantizationArg,
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
            generate_and_print(&generator, &final_prompt, args.verbose)?;
        }
        _ => run_interactive(&generator, args.chat, args.system.as_deref(), args.verbose)?,
    }
    Ok(())
}

fn generate_and_print(
    generator: &Generator,
    prompt: &str,
    verbose: bool,
) -> rsinfer::Result<String> {
    if verbose {
        println!("=== 输入 prompt ===\n{prompt}\n=== 开始生成 ===");
    }
    let start = std::time::Instant::now();
    let full = generator.generate_stream(prompt, |t| {
        print!("{t}");
        io::stdout().flush().ok();
    })?;
    println!();
    if verbose {
        println!(
            "\n--- 生成完成 (耗时: {:.2}s) ---",
            start.elapsed().as_secs_f32()
        );
    }
    Ok(full)
}

fn run_interactive(
    generator: &Generator,
    use_chat: bool,
    system: Option<&str>,
    verbose: bool,
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
        let full = generate_and_print(generator, &prompt, verbose)?;

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
