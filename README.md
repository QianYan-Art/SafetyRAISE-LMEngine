# SafetyRAISE-LMEngine

`rsinfer` 是 SafetyRAISE 的本地 Qwen 推理框架。它把模型权重、tokenizer、聊天模板、请求调度、生成状态和后端执行放在同一套 Rust 运行时合同中，供 SafetyRAISE 的专家指导、事故信息提取和报告生成链路复用。

项目保留了可读的 CPU 参考路径，也提供 GGUF 桥接、Q8 线性后端、视觉输入、Qwen3.5 MTP 和有限的 HTTP/SSE 服务入口。模型回答质量、真实权重长时间稳定性和硬件吞吐仍按 [验证证据与实验边界](docs/validation.md) 单独记录。

项目从本人面向 Qwen3 专家小模型的 Rust 推理引擎演进而来，吸收了本机、工作站和 Orin 部署中关于显存、状态恢复、量化和报告链路的约束；Strata 与 NInfer 只作为分层执行、调度和 MTP 合同的参考，实际模型计算和服务协议由本仓库负责。

## 支持范围

- Hugging Face safetensors 与 GGUF 权重容器。
- Qwen2、Qwen3 和 Qwen3.5 文本模型；Qwen3.5 的混合注意力和 GDN 状态由独立状态合同管理。Qwen3.5 MoE GGUF 仍会明确拒绝加载。
- tokenizer 配置中的 `chat_template`，以及通过 `chat_template_file` 注入的外部 Jinja 模板。TS-Qwen3 等没有模板的模型需要显式提供外部模板。
- 配置声明视觉塔时的有限图像输入；服务请求只接受 PNG/JPEG base64 data URL，并受请求体、图像数量和视觉 token 上限约束。
- Qwen3.5 MTP 的完整目标分布验证和事务回滚。默认 `max_draft_tokens` 为 `0`；启用 MTP 不等于已经取得加速收益。

模型、格式和上游固定来源见 [设计参考与证据来源](docs/references.md)，模块边界见 [架构与重构约束](docs/architecture.md)。

## 构建

```powershell
cargo build --release
```

默认构建保持可移植 CPU 行为；需要本机优化时由用户显式设置 `RUSTFLAGS` 或按目标工具链配置。发布前应在目标机器重新构建并按验证文档记录结果。可选原生插件由 `native/CMakeLists.txt` 单独构建，CUDA 插件是否能编译取决于本机工具链，当前仓库不把 CUDA 编译或硬件验收当作既成事实。

## 快速使用

推荐使用共享服务链的四个子命令：

可运行 `cargo run --release -- serve --help` 查看服务参数；其余子命令同样支持 `--help`。

```powershell
# 只读检查模型架构、张量清单和状态预算
cargo run --release -- inspect --model-path <模型目录或GGUF文件> --context 4096

# 用配置加载模型并执行一个 JSON 请求
cargo run --release -- run --config configs/examples/expert.json --request configs/examples/text-request.json

# 启动 HTTP/SSE 服务
cargo run --release -- serve --config configs/examples/expert.json --bind 127.0.0.1:8080

# 对真实配置和请求记录端到端实验元数据
cargo run --release -- benchmark --config configs/examples/expert.json --request configs/examples/text-request.json --iterations 3
```

配置文件中的相对路径以配置文件所在目录为基准。专家、报告、视觉三种角色配置和文本请求示例见 [`configs/examples`](configs/examples)；部署参数、外部模板、认证、后端切换和请求边界见 [部署与服务配置](docs/deployment.md)。

服务只实现明确声明的 OpenAI 风格请求子集，不承诺完整 OpenAI API 兼容。路由为 `/health`、`/v1/models`、`/metrics` 和 `/v1/chat/completions`；流式请求返回 SSE。监听非回环地址时必须设置 `SAFETYRAISE_API_KEY`，客户端使用 `Authorization: Bearer <key>`。

仓库还保留旧的顶层本机调试参数（例如 `--model-path`、`--prompt`、`--device`、`--quantization`），用于原 Qwen3 单机生成与性能剖析。该入口的 `resident decode` 是旧执行链的实验性路径；新 `serve/run` 服务中的 `wgpu-q8` 后端只负责受能力检查保护的线性/Q8 操作，不应把两条路径的性能结论混写。

## 架构

```text
src/artifact/   safetensors、GGUF、量化布局与只读权重
src/model/      Qwen2/Qwen3/Qwen3.5 规格、数学、MTP 与模型组装
src/frontend/   tokenizer、Jinja 模板、thinking/tool 分流、停止词与采样
src/vision/     视觉预处理与视觉塔状态
src/backend/    CPU、原生 ABI、wgpu Q8 线性后端
src/engine/     有界调度、分块 prefill、decode、取消、缓存与投机事务
src/service.rs  CLI/HTTP 共用的模型装载和请求预算
src/server.rs   有限 JSON/SSE 服务协议
```

请求状态和权重只读数据分离；Qwen3.5 的 GDN 状态不能按普通 KV cache 截断；投机解码只有在完整目标分布验证并提交事务后才发布 token。详细不变量与未完成路径见 [架构文档](docs/architecture.md) 和 [验证边界](docs/validation.md)。

## 许可证

[Apache-2.0](LICENSE)
