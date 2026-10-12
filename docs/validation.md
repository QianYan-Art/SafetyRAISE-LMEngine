# 验证证据与实验边界

## 本轮范围

2026-10-12 在 Windows 本机重构，起点为 `9e8873b`。使用 Rust 1.96.0，默认不设置 `target-cpu=native`。验证不修改模型源目录；以下结果限定本轮代码、设备及列出的测试资产。

本轮验证包括独立小尺寸数学参考、实际解析器与微型合成权重执行器、请求调度和 HTTP/SSE 管线、本机原生 CPU 插件、实际 wgpu 设备、真实 tokenizer 与模型文件检查。没有完整大模型生成、工作站或 Orin 测试，也没有新版本吞吐、模型回答质量或稳定性成绩。旧 V1.1 的性能记录不能用于证明本轮性能。

## 已执行的检查

| 检查 | 结果与范围 |
| --- | --- |
| `cargo fmt --all`、`cargo clippy --all-targets -- -D warnings` | 格式化完成，所有目标静态检查通过 |
| `cargo test --release --quiet` | 138 项通过，4 项依赖设备或本地资产的测试默认忽略；bin 和 doc-test 无失败 |
| `cargo build --release --quiet` | 发布二进制构建通过，并重新执行 `serve --help` 与真实文件 `inspect` |
| 显式运行全部 4 项 ignored 测试 | 4 项通过；实际运行，没有将设备不可用当作成功跳过 |
| CMake 原生 CPU 动态库 | Windows Release 编译通过，Rust 动态加载、矩阵结果、句柄复用与错误传播通过 |
| wgpu Q8 适配器 | 实际设备执行单行 Q8，与使用同一派生 Q8 权重的 CPU 多行路径一致；预算不足回退通过 |
| DeepSeek-R1-Distill-Qwen-1.5B tokenizer | 使用本机真实 tokenizer 与内置模板，Python 字符串方法、工具消息及初始 thinking 状态通过 |
| TS-Qwen3 tokenizer | 使用本机真实 tokenizer 与外部固定版本 Qwen3 模板，thinking 开关、中文逐 token 解码通过 |
| 统一 CLI | `serve --help`、`run --help` 与真实 TS-Qwen3 `inspect` 通过；未通过该 CLI 运行完整模型生成或性能基准 |

CI 配置 Linux/Windows 的格式、Clippy、CPU/协议 release 测试及原生 CPU 插件动态加载。上表记录本机结果；每个提交的跨平台结果以对应 GitHub Actions 运行记录为准，不由本机测试推定。

托管 CI 使用 `cargo test --release -- --skip gpu::tests:: --skip _when_available`，明确过滤旧 GPU 模块的 23 项设备测试及模型层的 5 项设备测试；其余 4 项资产/设备测试保持默认 ignored，CPU 插件测试在构建插件后单独显式运行。过滤项不计作通过。首次 Windows CI（运行 `38301260686`）在执行旧 GPU 测试期间以 `STATUS_ACCESS_VIOLATION` 异常退出，日志没有足够信息定位具体适配器或底层原因；本机完整测试通过不能消除此风险。GPU 验收需在指定设备上运行下面的完整测试与显式 wgpu 测试，不由托管 CPU CI 代替。

### 承重回归

- 权重资源：safetensors/GGUF 字节解析、分片索引、非法范围、量化块与共享 mmap 生命周期；F32/F16/BF16、Q4_0/Q8_0/Q4_K/Q6_K 的表示与必要反量化路径。
- 模型数学：Qwen2/Qwen3/Qwen3.5 微型合成权重执行；整段与分块 prefill、跨 KV 页、状态快照重放、跨模型状态拒绝、失败不提交。独立手算覆盖 online attention、部分 RoPE、卷积与先衰减后更新的 GDN。
- GGUF 混合布局：使用 K=2/V=6 不对称 GDN，验证 qkv V 行、z/a/b、A_log/dt、卷积及 out projection 列的 tiled→grouped 还原；GGUF 与等价 grouped HF 权重经完整执行器连续分块所得 logits/hidden 一致。行列重排保留底层字节映射。
- MTP：后继 token 与前一 target hidden/位置配对，整段/分块一致；目标实例绑定和跨模型状态拒绝；目标全词表验证，拒绝后从完整 KV/GDN 快照重放；调度器 MTP 输出与普通 greedy 输出一致。只统计验证后接受的 token，不发布未经目标确认的草稿。
- 视觉：静帧 temporal 复制、patch 顺序、位置插值、轴向位置和归一化/merger 的小尺寸参考；媒体嵌入与多轴位置交接；像素硬上限、patch 计算量和准备内存提前拒绝。
- 调度与协议：前缀复用一致、慢消费者不阻塞同伴、队列超时、资源拒绝、取消覆盖缓冲正常终态、思考预算为零及晚进入后的完整收束。微型合成模型进入 HTTP JSON/SSE，输出一致；未知模型/字段/认证错误、stop 和首个 token 前的 SSE 断连经过回归。
- 文本输出：跨 token 不完整 UTF-8 暂存，跨片停止串不泄漏；正文、思考和完整工具 JSON 分流。工具仅返回给调用方，本引擎不执行工具。
- 收尾复审：普通 greedy 与 MTP 的并列 logits 均选择较小 token ID；重复惩罚对每个历史 token 只应用一次；用户文本中的 thinking 标记不打开助手初始思考状态；工具定义/调用拒绝非法名称、空 ID、非对象参数及未支持字段。

这些小尺寸执行对照可以发现布局、位置与生命周期错误，但尚未形成覆盖全部层输出的 Transformers/官方量化工具差分基准，不能推导完整模型质量等价。

## 真实文件检查

本机 `D:\MCP_Server\root\autodl-tmp\TS-Qwen3` 的 `inspect --context 4096` 返回：

| 项目 | 结果 |
| --- | --- |
| 架构 | Qwen3，36 层，hidden 2560，32 Q / 8 KV，head_dim 128 |
| 张量数 | 398 |
| 逻辑权重字节 | 8,044,936,192 |
| 4096 上下文单请求状态估算 | 1,207,959,552 字节 |
| 证据等级 | 配置/权重容器解析，不是完整模型执行 |

`inspect` 不执行各层前向，也不验证所有模型张量能否成功组装；逻辑权重字节和状态估算不是进程峰值内存。TS-Qwen3 本地 tokenizer_config 没有 chat_template，因此通过 `chat_template_file` 显式覆盖，不修改原模型文件。

## 实现能力与明确限制

| 范围 | 当前实现 | 本轮没有证明或没有支持的内容 |
| --- | --- | --- |
| Qwen 文本 | Qwen2/Qwen3 dense、Qwen3.5 dense hybrid | 不承诺所有 Qwen 发行版本；MoE、滑窗、非 default RoPE scaling 明确拒绝 |
| GGUF | 已列格式及 Qwen 名称/数学语义桥接 | 不支持全部 GGUF 类型、Bonsai 特殊三元格式；未加载 27B 最终报告资产 |
| 视觉 | 指定 Qwen3.5 视觉合同、静态 PNG/JPEG/RGB、媒体交接 | 无真实视觉权重质量验收；不支持视频、远程图片下载或客户端任意文件路径 |
| MTP | 单 MTP block、greedy 全词表目标验证、可选草稿词表与自适应草稿长度 | 随机采样/重复惩罚/媒体/思考预算请求走普通生成；无 MTP 加速证据 |
| CPU | 完整模型参考路径，权重按行读取/反量化 | 未实现面向大模型的全面融合与调优，不能视作高吞吐 CPU 后端 |
| 新 wgpu-q8 | 预算内派生行 Q8、共享权重、decode 线性 GPU 执行 | KV/attention/GDN 仍在 CPU；不是旧版完整 resident decode 链，Q4_K/Q6_K 保持原格式走 CPU |
| 原生 ABI | v1 插件加载、能力检查、权重句柄与线性算子调用；CPU 插件已运行 | CUDA F16 内核已有源码，但本机无 nvcc，本轮未编译/运行 CUDA、SM86 或 Orin |
| 状态与缓存 | F32 分页 KV、COW、GDN 完整快照，同 Engine 内纯文本前缀 LRU | 无跨模型/跨进程缓存、KV 量化或持久缓存；MTP 与媒体请求禁用前缀复用 |
| 调度 | 单模型工作线程，chunk prefill 和逐 token/投机 round 轮转 | 不是多请求合并 GEMM 的 continuous batching；不能在单个算子内部抢占 |
| HTTP | 有限 OpenAI 风格 JSON/SSE，模型/健康/指标、可选 Bearer 认证 | 不承诺完整 OpenAI 兼容，无 JSON grammar、tool choice 强约束、多模型路由或工具执行 |
| benchmark | 指定真实请求的加载与端到端计时 | 本轮未执行大模型 benchmark；没有独立 TTFT/prefill/decode 指标，也没有相对 llama.cpp 的性能结论 |

报告模型发布名中的 “Qwen3.8-27B” 不作为架构依据。历史交付记录指向 Qwen3.5 dense hybrid（48 GDN + 16 full attention，GDN K16/V48）；该完整资产及重训 MTP 未在本轮实机验收。工作站 Bonsai、笔记本 RTX 4060 Laptop、工作站 RTX 3050 Laptop 与 Jetson AGX Orin 是不同环境，不能相互替代实验。

### 性能与资源语义

MTP 目前在验证后重新执行实际提交序列，以保证混合状态事务正确；全部接受时仍有重放成本，可能比普通 greedy 更慢。`draft_proposed`/`draft_accepted` 不包含目标直接给出的首 token。

取消和超时在 chunk/round 及事件发布边界生效；正在运行的单个 CPU/GPU 算子不能被中断。已进入消费者通道的 token 无法撤回，HTTP 停止后会忽略后续已缓冲文本；usage 记录实际生成数量，可能包含同一轮已验证但未展示的 token。背压超过上限会关闭通道，客户端须将缺少完整终态视为失败。

思考强制结束标记计入总输出 token 预算。预算不足以再打开并关闭思考段时结束生成，可能在到达数字上限前返回 length；用户停止串、取消或超时可以终止当前思考段。

`engine.working_memory_bytes` 覆盖状态、事务副本、队列输入、前缀及保守激活估算；`preparation_memory_bytes` 单独覆盖并发 JSON/模板/媒体准备。权重 mmap、操作系统页缓存、插件权重副本、wgpu 主机 Q8 副本、驱动和分配器开销不等于这两个预算。所有估算都不是操作系统级 RSS 硬限额，部署应另留内存余量并观测峰值。媒体预算采用保守预留，可能拒绝实际占用更小的请求。

## 复现命令

在仓库根目录执行常规检查：

```powershell
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --release
cargo build --release
cmake -S native -B target/native-cpu -DSAFETYRAISE_BUILD_CUDA=OFF
cmake --build target/native-cpu --config Release
```

在具有对应资产和 wgpu 设备的机器显式执行补充测试；环境变量仅对当前终端有效，路径按实际位置调整：

```powershell
$env:RSINFER_NATIVE_CPU_PLUGIN = "$PWD/target/native-cpu/bin/Release/safetyraise_cpu.dll"
$env:RSINFER_QWEN_TOKENIZER_DIR = 'C:/Users/lenovo/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B'
$env:RSINFER_QWEN3_TOKENIZER_DIR = 'D:/MCP_Server/root/autodl-tmp/TS-Qwen3'
cargo test --release --lib -- --ignored
cargo run --release -- inspect --model-path 'D:/MCP_Server/root/autodl-tmp/TS-Qwen3' --context 4096
```

Linux 的 CMake CPU 插件输出为 `target/native-cpu/bin/libsafetyraise_cpu.so`。只验证动态库而不要求模型资产或 GPU 时，可执行：

```bash
RSINFER_NATIVE_CPU_PLUGIN="$PWD/target/native-cpu/bin/libsafetyraise_cpu.so" \
  cargo test --release --lib backend::tests::native_cpu_plugin_roundtrip_caches_weight_and_propagates_error -- --ignored --exact
```

## 后续实机验收

取得目标设备和最终模型后，先与固定版本 Transformers/量化参考核对逐层 hidden/logits、视觉 embedding 和 MTP 接受/拒绝，再测同一模型、量化、KV 格式、输入/输出长度与并发下的峰值内存、TTFT、prefill/decode、端到端和长时间稳定性。27B 40K 窗口、6 槽、MTP 词表和 Orin CUDA 分别验收，不能直接搬用历史配置。

本轮构建和原生库保留在已忽略的 `target/`；任务恢复资料在已忽略的 `.mission/`，文档检查在自忽略的 `.doc-maintenance/`。测试 fixture 位于模块内，临时权重测试自行回收文件，不在项目根目录遗留模型、截图、备份或测速产物。
