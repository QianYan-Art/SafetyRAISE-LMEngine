# 部署与服务配置

## 入口和路径

统一部署入口是 `rsinfer serve/run/inspect/benchmark`。配置文件由 CLI 读取，`model_path`、`tokenizer_path`、`native_library` 和 `chat_template_file` 的相对路径都以该配置文件所在目录为基准；请求文件只描述 JSON 请求，不会从请求字段读取任意文件路径。

```powershell
cargo run --release -- inspect --model-path <模型目录或GGUF文件> --context 4096
cargo run --release -- run --config configs/examples/expert.json --request configs/examples/text-request.json
cargo run --release -- serve --config configs/examples/expert.json --bind 127.0.0.1:8080
cargo run --release -- serve --help
```

`inspect` 只解析配置、权重和状态预算，不执行完整模型。`run` 和 `serve` 都经过同一个 `Service`，因此模板、视觉、采样、思考和上下文预算行为一致。

## 配置结构

`ServiceConfig` 使用 `deny_unknown_fields`，拼写错误或未实现字段会被拒绝。最小配置只需 `model_path`；默认值偏保守：CPU 后端、4096 上下文、单个活动请求、8 MiB 请求体、1 MiB 输出、256 个视觉 token、1 个并发准备请求和 512 MiB 准备临时内存。

| 字段 | 说明 |
| --- | --- |
| `model_path` | safetensors 模型目录或 GGUF 文件；必填。目录需要权重和配置，GGUF 由格式 metadata 推导架构。使用 GGUF 服务时仍需单独提供 `tokenizer_path`。 |
| `tokenizer_path` | 可选 tokenizer 目录；省略时使用 `model_path`。GGUF 文件不是 tokenizer 目录，必须显式设置该字段。 |
| `chat_template_file` | 可选完整 Jinja 模板，最多 1 MiB；设置后覆盖 tokenizer 配置中的同名模板。没有模板的模型必须配置它。 |
| `model_name` | API 请求中的 `model` 值，默认 `safetyraise-local`。 |
| `backend` | `cpu`、`native` 或 `wgpu-q8`。服务中的 wgpu 后端只提供受能力检查保护的 Q8 线性操作，不能把旧调试入口的 resident decode 当成服务能力。 |
| `native_library` | `backend` 为 `native` 时的 C ABI 动态库路径；插件不是请求字段。 |
| `gpu_memory_bytes` | wgpu 后端的保守预算，默认 1 GiB。没有可用 adapter 或预算不足时服务加载失败或回退由后端明确报告。 |
| `max_body_bytes` | JSON 和媒体请求体上限，默认 8 MiB，最大 64 MiB。 |
| `max_output_bytes` | 单请求解码输出字节上限，默认 1 MiB。 |
| `vision_max_tokens` | 每张图像的视觉 token 预算，默认 256，最大 4096。 |
| `max_preparing` | 同时进行 JSON、模板和媒体准备的请求数，默认 1，范围 1..16；它与模型活动槽位分开。 |
| `preparation_memory_bytes` | 所有准备请求的保守临时内存预算，默认 512 MiB；它与 `engine.working_memory_bytes` 分开核算。 |
| `engine` | 调度和生成预算，见下节；缺失字段使用 `EngineConfig` 默认值。 |

### `engine` 预算

`engine.max_context` 默认 4096；`max_active` 默认 1；`queue_capacity`、`prefill_chunk`、`event_capacity` 默认分别为 8、32、32；`working_memory_bytes` 默认 2 GiB；`prefix_cache_bytes` 默认关闭；`max_input_bytes` 默认 32 MiB；请求和背压超时默认 300 秒和 10 秒。`max_draft_tokens` 默认为 0，只有满足纯文本、greedy、无思考预算和无重复惩罚的请求才会启用 MTP 投机；完整目标验证和事务提交仍是必需步骤，指标中的 draft 数量不等于已接受的 target token 数量。

视觉请求在服务层先估算 embedding、输入和临时空间，再调用视觉编码。每个请求最多 4 张图像，只接受 PNG/JPEG base64 data URL；服务固定 `4 × 1024²` 像素上限，CPU patch 计算还受 `sqrt(128000000 / attention_heads)` 上限约束。视觉模型没有已加载的视觉塔时，带图像请求会返回明确错误。

外部模板、模型配置和请求文件都有独立字节上限：CLI 配置文件为 1 MiB，旧 Qwen3 配置读取器的 Hugging Face `config.json` 为 2 MiB，统一服务的权重清单读取另有独立限制，外部模板为 1 MiB，请求体由 `max_body_bytes` 控制（默认 8 MiB、最大 64 MiB）。服务先取得准备许可，再在 30 秒读体超时内解析 JSON。SSE 客户端断开会主动取消生成；底层取消在调度 chunk、round 或 kernel 边界生效。

## 三种业务角色配置

`configs/examples/expert.json`、`report.json` 和 `vision.json` 分别对应专家指导、报告生成和视觉输入角色，三份配置都默认使用 CPU，并把模型路径写成相对 `configs/examples` 的示例路径。部署时改为实际模型和 tokenizer 目录；没有真实模型或 tokenizer 时，示例不会声称可以生成。

示例把模型工作区预留提高到 4 GiB，以容纳本机 Qwen3-4B 的 F32 KV 与事务副本；这不包括权重。更换模型、增加上下文/并发或启用 MTP 时，须按启动时的预算检查重新设置内存，不能把示例数值直接视为任意设备的推荐配置。

报告配置默认 `max_draft_tokens: 0`。只有确认目标模型带有可加载的 Qwen3.5 MTP 权重、请求满足 greedy 条件并完成验证后，才应在该角色配置中启用草稿预算。视觉配置使用较小的队列和 prefill 分块，仍受服务层四图、`4 × 1024²` 像素和准备内存预算约束。

业务角色和算子后端是两个独立选择。需要原生 C ABI 时，将 `backend` 改为 `native` 并填写相对 `native_library`；需要 Q8 wgpu 线性路径时改为 `wgpu-q8`。两者都不是业务角色示例的默认值，旧顶层调试入口的 resident decode 也不由这些服务配置开启。

TS-Qwen3 这类 tokenizer 目录若没有 `chat_template`，请将 `chat_template_file` 指向仓库模板或经过审查的外部模板，例如：

```json
"chat_template_file": "../../src/frontend/templates/qwen3.jinja"
```

模板内容最多 1 MiB，服务会在加载时覆盖 tokenizer 配置中的模板，不会从请求体接受模板代码。

## HTTP/SSE 子集

路由：

- `GET /health`：返回状态和能力。
- `GET /v1/models`：返回单个已加载模型及能力。
- `GET /metrics`：返回调度、生成、MTP 与后端计数；`mtp_rounds` 是验证轮数，`draft_proposed`/`draft_accepted` 只统计草稿部分，不包含目标模型直接给出的首个 token；原生插件报告句柄与回退数量，wgpu 报告驻留估算和 CPU/GPU 调用数。
- `POST /v1/chat/completions`：非流式 JSON 或 `stream: true` 的 SSE。

请求至少需要 `model`、`messages`。当前支持 `tools`（仅 function）、`stream`、`n=1`、`max_tokens`、`temperature`、`top_p`、`top_k`、`repetition_penalty`、`seed`、`stop`、`enable_thinking`、`thinking_budget`、`template_name`、`timeout_ms` 和 `cache_prefix`。未知字段、`n != 1`、非法工具参数、超出上下文或资源预算都会被拒绝；这不是完整 OpenAI API 兼容层。

非回环监听必须设置 `SAFETYRAISE_API_KEY`。服务端用 `Authorization: Bearer <key>` 校验；回环监听可以不设置 key，但生产部署仍应在反向代理或网络策略中限制访问。

## 算子后端与诊断

### wgpu 后端选择与诊断

`wgpu-q8` 默认使用 wgpu PRIMARY 后端集合，保留 Vulkan/DX12/Metal 的平台适配器选择，不默认初始化 GL。可以在启动进程前通过 `WGPU_BACKEND` 选择 `dx12`、`vulkan`、`metal`、`gl`、`primary`、`all` 或逗号组合；别名 `d3d12`、`vk`、`gles` 也受支持。非法值或没有任何已编译后端的选择会报错，不会静默忽略。`all` 会恢复包含 GL 的多后端初始化，只用于明确需要的兼容性或故障复现。

设置 `RSINFER_GPU_DIAGNOSTICS=1` 可在 stderr 记录初始化阶段、所选适配器、驱动和后端；正常推理 stdout 不受诊断影响。`gpu-tests` 是测试开关，生产程序启用 GPU 不需要该 feature。设备验收命令与托管软件适配器边界见 `docs/validation.md`。

排查软件驱动时，可设置 `RSINFER_GPU_FORCE_FALLBACK=1` 请求软件适配器；这不是性能优化选项，没有可用软件适配器会报错。取消该环境变量即可恢复默认适配器选择。

### 原生插件边界

`native/CMakeLists.txt` 生成 CPU 动态库，并可在检测到 CUDA 编译器时尝试构建可选 CUDA 插件。当前仓库没有 CUDA 编译验收证据；配置示例不默认选择 CUDA，不能把插件存在或 `wgpu` adapter 存在解释为目标硬件已经通过验证。原生插件只通过固定 ABI 接收受约束的线性输入输出，能力不足时必须报错或走明确的 CPU 路径。
