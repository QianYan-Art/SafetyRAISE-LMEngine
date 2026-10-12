//! 新部署入口共用服务与调度器；原本的本机性能调试命令继续保留。

use crate::{
    service::{ApiRequest, Service, ServiceConfig},
    Result, RsinferError,
};
use clap::{Parser, Subcommand};
use serde_json::json;
use std::{
    io::Read,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[derive(Parser)]
#[command(name = "rsinfer", about = "SafetyRAISE 本地 Qwen 推理框架")]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// 启动有限 OpenAI 风格 HTTP/SSE 服务。
    Serve {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
    },
    /// 用共用服务链执行一个请求，输出结构化 JSON。
    Run {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, conflicts_with = "prompt")]
        request: Option<PathBuf>,
        #[arg(long)]
        prompt: Option<String>,
    },
    /// 只读解析架构、张量清单和状态预算，不运行模型。
    Inspect {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long, default_value_t = 4096)]
        context: usize,
    },
    /// 在指定模型和真实请求上测量端到端耗时，输出实验元数据。
    Benchmark {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
    },
}

pub fn selected() -> bool {
    std::env::args()
        .nth(1)
        .is_some_and(|arg| matches!(arg.as_str(), "serve" | "run" | "inspect" | "benchmark"))
}
pub fn run() -> Result<()> {
    let args = Args::parse();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(execute(args.command))
}
fn read_bounded(path: &Path, limit: usize) -> Result<String> {
    let mut data = String::new();
    std::fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_string(&mut data)?;
    if data.len() > limit {
        return Err(RsinferError::ConfigError(
            "配置或请求文件超过字节上限".into(),
        ));
    }
    Ok(data)
}
fn config(path: &Path) -> Result<ServiceConfig> {
    let mut config: ServiceConfig = serde_json::from_str(&read_bounded(path, 1024 * 1024)?)?;
    let base = path.parent().unwrap_or(Path::new("."));
    for item in [&mut config.model_path]
        .into_iter()
        .chain(config.tokenizer_path.iter_mut())
        .chain(config.native_library.iter_mut())
        .chain(config.chat_template_file.iter_mut())
    {
        if item.is_relative() {
            *item = base.join(&*item);
        }
    }
    Ok(config)
}
async fn execute(command: Command) -> Result<()> {
    match command {
        Command::Inspect {
            model_path,
            context,
        } => {
            let (weights, spec) = crate::service::open_weights(&model_path)?;
            if context == 0 || context > spec.max_position_embeddings {
                return Err(RsinferError::ConfigError(
                    "inspect 上下文超出模型容量".into(),
                ));
            }
            let mut bytes = 0usize;
            let mut count = 0usize;
            for name in weights.names() {
                bytes = bytes
                    .checked_add(weights.get(name)?.byte_len())
                    .ok_or_else(|| RsinferError::ConfigError("张量字节汇总溢出".into()))?;
                count += 1;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"spec":spec,"tensors":count,"logical_weight_bytes":bytes,"context":context,"state_bytes_per_request":spec.state_bytes(context)?,"validation":"仅配置与权重解析；没有执行完整模型"})
                )?
            );
        }
        Command::Serve {
            config: config_path,
            bind,
        } => {
            let service = Arc::new(Service::load(config(&config_path)?)?);
            let key = std::env::var("SAFETYRAISE_API_KEY").ok();
            crate::server::serve(service, bind, key).await?;
        }
        Command::Run {
            config: config_path,
            request,
            prompt,
        } => {
            let service = Service::load(config(&config_path)?)?;
            let input: ApiRequest = if let Some(path) = request {
                serde_json::from_str(&read_bounded(&path, service.config.max_body_bytes)?)?
            } else if let Some(prompt) = prompt {
                serde_json::from_value(
                    json!({"model":service.config.model_name,"messages":[{"role":"user","content":prompt}]}),
                )?
            } else {
                let mut text = String::new();
                std::io::stdin()
                    .take(service.config.max_body_bytes as u64 + 1)
                    .read_to_string(&mut text)?;
                if text.len() > service.config.max_body_bytes {
                    return Err(RsinferError::ConfigError("标准输入超出请求上限".into()));
                }
                serde_json::from_str(&text)?
            };
            let (prompt, handle, tools) = service.submit(service.prepare(&input)?)?;
            let completion =
                crate::server::complete(&prompt, handle, &tools, service.config.max_output_bytes)
                    .await?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"model":service.config.model_name,"message":completion.message,"finish_reason":completion.finish_reason,"usage":completion.usage})
                )?
            );
        }
        Command::Benchmark {
            config: config_path,
            request,
            iterations,
        } => {
            if iterations == 0 || iterations > 100 {
                return Err(RsinferError::ConfigError(
                    "benchmark iterations 必须在 1..=100".into(),
                ));
            }
            let config = config(&config_path)?;
            let input: ApiRequest =
                serde_json::from_str(&read_bounded(&request, config.max_body_bytes)?)?;
            let start = Instant::now();
            let service = Service::load(config)?;
            let load_ms = start.elapsed().as_millis();
            let mut runs = vec![];
            for _ in 0..iterations {
                let start = Instant::now();
                let (prompt, handle, tools) = service.submit(service.prepare(&input)?)?;
                let result = crate::server::complete(
                    &prompt,
                    handle,
                    &tools,
                    service.config.max_output_bytes,
                )
                .await?;
                let seconds = start.elapsed().as_secs_f64();
                runs.push(json!({"seconds":seconds,"completion_tokens":result.usage.completion_tokens,"end_to_end_tokens_per_second":result.usage.completion_tokens as f64/seconds,"usage":result.usage,"finish_reason":result.finish_reason}));
            }
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"capabilities":service.capabilities(),"load_ms":load_ms,"runs":runs,"scope":"真实指定请求的端到端计时，包含模板/视觉/队列/预填充/输出处理，不代表模型质量或纯 decode 吞吐"})
                )?
            );
        }
    }
    Ok(())
}
