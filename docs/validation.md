# 验证证据与实验边界

## 本轮范围

2026-10-12 在 Windows 本机重构，起点为 `9e8873b`。使用 Rust 1.96.0，默认不设置 `target-cpu=native`。验证不修改模型源目录；以下结果限定本轮代码、设备及列出的测试资产。

本轮验证包括独立小尺寸数学参考、实际解析器与微型合成权重执行器、请求调度和 HTTP/SSE 管线、本机原生 CPU 插件、实际 wgpu 设备、真实 tokenizer 与模型文件检查。没有完整大模型生成、工作站或 Orin 测试，也没有新版本吞吐、模型回答质量或稳定性成绩。旧 V1.1 的性能记录不能用于证明本轮性能。

## 重构初次发布检查（GPU 验证修复前）

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

首次 Windows CI（运行 `38301260686`）在执行旧 GPU 测试期间以 `STATUS_ACCESS_VIOLATION` 异常退出，日志只记录了 59 项通过、4 项忽略，没有适配器信息或 dump；不能据此确定崩溃测试或底层原因。后续运行 `38303310043` 通过过滤 28 项设备测试恢复 CPU CI，不能视为 GPU 已修复。旧设备测试还会把初始化失败直接返回为通过，因此上表历史测试数本身不能证明所有设备算子实际执行。

## GPU 验证修复

`gpu-tests` Cargo feature 仅决定设备测试是否默认执行，不改变生产 GPU 支持。31 项设备测试默认明确标为 ignored；启用 feature 后必须完成设备初始化与算子断言，初始化失败保留完整错误并使测试失败。CPU/协议任务直接执行 `cargo test --release --lib`，不再依赖名称过滤。另有 CPU 插件和两个真实 tokenizer 资产测试保持独立 ignored。

独立 GPU CI 每次执行 Linux Mesa Vulkan 软件适配器和 Windows DX12，两者均严格传递失败退出码。软件适配器验证算子正确性，不证明物理设备性能。手动触发时 `inputs.gpu_backend` 可选 `dx12`、`vulkan`、`all`；Linux 始终用 Vulkan，Windows 将选项原样传入 `WGPU_BACKEND`。其中 `all` 初始化全部可用后端，供重现旧行为使用，不代表逐个后端都得到测试。

测试日志包含 `test_begin`、`instance_begin`、`instance_ready`、`adapter_ready`、`device_ready` 和 `pipelines_ready`。设备信息包含后端、驱动及设备类型。Windows 的 `scripts/gpu-ci-diagnostics.ps1` 仅在 GitHub Actions 临时运行器设置 WER dump 并在收集后恢复，另采集系统事件和显卡信息；诊断 artifact 保留 `adapter-environment.txt`、`cargo-test.log`，Linux 还保留 `vulkaninfo-summary.txt`。无 dump 时不宣称已有崩溃栈。

修复还覆盖 Q8 argmax 的 1–3 个尾部输入元素，以及 raw 批量 matvec 的输入宽度和共享上下文检查。尾部回归逐个使非对齐元素决定排名，防止旧测试数据碰巧得到相同 argmax。这两项修复不作为原 Windows 访问冲突的因果证据。

提交 `9e8380b` 本机最终 Vulkan 套件为 143 通过、3 资产忽略，显式 3 项资产测试全部通过；无效后端负向测试退出 101。托管运行 `38341821022` 的 Linux llvmpipe/Mesa 25.2.8 套件同样 143 通过、3 忽略，CPU/协议及 CPU 插件在两平台通过。Windows DX12 与 `all` 对照运行 `38341882386` 均在首次 Q8 执行期间访问冲突，所选设备为 Microsoft Basic Render Driver 10.0.26100.33438。两个运行都完成了管线初始化，但没有 WER dump，不能根据这些证据断言具体底层根因。

为定位软件驱动路径，`RSINFER_GPU_FORCE_FALLBACK=1` 会显式请求 wgpu 的 fallback 软件适配器；平台没有可用软件适配器时仍报错。Windows CI 失败后使用官方 Sysinternals ProcDump 重现首个 Q8 测试并收集异常 dump，保留原测试失败状态，不以诊断重试替代套件验收。诊断包含 Q8 输入写入、提交和等待阶段。

提交 `13a1ea4` 的运行 `38343684659` 成功通过 ProcDump 捕获崩溃 dump，CDB 栈为 `d3d10warp!ComputeShaderTransformer::ConstructLoadStoreSets` → `TransformComputeShader` → `PixelJitOptimizer` → `Task_CompileShader`，确认故障发生在 Microsoft WARP 的 compute shader JIT 工作线程。该运行 Windows GPU 失败，Linux GPU 和两平台 CPU 作业通过。本机软件适配器驱动 `10.0.26100.9278` 同样复现 Q8 标量和 vec4 路径崩溃，而 RMSNorm 和 Q8 argmax 通过。

单变量复现中，扩大输出缓冲区到 16 字节、将动态归约展开为固定六步都未消除崩溃。仅把两个 Q8 shader 的提前返回改为受 `row_valid` 保护的计算，让所有 lane 统一到达工作组 barrier，标量及 vec4 路径即通过，最小样例与 CPU 结果完全一致。正式修复保留原缓冲区大小、动态归约顺序和数值算法，输出也受行范围保护；没有切换编译器、升级驱动或绕过 DX12。该证据定位了这两段 shader 与 WARP JIT 的触发组合，不代表所有含提前返回的 WGSL 都非法或所有驱动均有缺陷。

原 Q8 matvec 回归扩展为输入宽度 `1/4/5/255/256/257/512` 与输出行数 `1/2/3/65` 的 28 组 CPU 数值对照，覆盖标量、vec4、非对齐尾部、空闲 lane、多轮累加及小输出缓冲区。完整平台验收使用上面的严格 GPU 作业和下文复现命令，实际结果以对应提交的 GitHub Actions 运行记录为准。

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

在具有 wgpu 设备的机器显式执行设备测试；环境变量仅对当前终端有效：

```powershell
$env:WGPU_BACKEND = 'dx12'
$env:RSINFER_GPU_DIAGNOSTICS = '1'
cargo test --release --features gpu-tests --lib -- --test-threads=1 --nocapture
```

生产默认 PRIMARY（Vulkan/DX12/Metal 等主后端，不默认初始化 GL），保留平台适配器自动选择；Windows CI 单独固定 DX12。本机 Vulkan 初始化明显快于 DX12，不能把托管验证后端直接强制为所有机器的生产默认。`WGPU_BACKEND` 可显式选择 `dx12`、`vulkan`、`metal`、`gl`、`primary`、`all` 或逗号组合；空值、拼写错误及没有任何已编译后端的选择均报错。生产诊断由 `RSINFER_GPU_DIAGNOSTICS=1` 开启，输出到 stderr；测试总是记录诊断。换后端成功只能证明该后端通过，不能证明旧崩溃原因。

具有对应本地资产时，再执行资产测试；启用 feature 后 `--ignored` 只选取三个外部资产测试，不会重复执行 GPU 测试。路径按实际位置调整：

```powershell
$env:RSINFER_NATIVE_CPU_PLUGIN = "$PWD/target/native-cpu/bin/Release/safetyraise_cpu.dll"
$env:RSINFER_QWEN_TOKENIZER_DIR = 'C:/Users/lenovo/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B'
$env:RSINFER_QWEN3_TOKENIZER_DIR = 'D:/MCP_Server/root/autodl-tmp/TS-Qwen3'
cargo test --release --features gpu-tests --lib -- --ignored
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
