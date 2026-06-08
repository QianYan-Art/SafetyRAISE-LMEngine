# SafetyRAISE-LMEngine

[![CI](https://github.com/QianYan-Art/SafetyRAISE-LMEngine/actions/workflows/ci.yml/badge.svg)](https://github.com/QianYan-Art/SafetyRAISE-LMEngine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

用 Rust 从零写的极简大模型推理引擎（crate 名 `rsinfer`），默认 CPU 执行，专门适配微调的 **Qwen3-4B-Thinking** 模型，服务于 SafetyRAISE 系统。

定位是学习/练手项目：把 Transformer 推理的每一环（权重加载、张量算子、注意力、KV cache、采样、生成循环）都用可读的 Rust 实现一遍。要追求生产级速度请用 llama.cpp。

## 支持的模型

HuggingFace `safetensors` 格式的 Qwen3（`Qwen3ForCausalLM`）。已验证：微调版 Qwen3-4B-Thinking-2507。

关键架构特性均已实现：GQA、Qwen3 的 QK-Norm、RoPE、SwiGLU、RMSNorm、权重绑定（tie embeddings）、f16 权重存储。

## 构建

```powershell
cargo build --release
```

`.cargo/config.toml` 已开启 `target-cpu=native`（AVX2/FMA），二进制仅保证在本机架构上运行。

## 使用

单次生成：

```powershell
cargo run --release -- --model-path <模型目录> --chat -p "用一句话介绍杭州。"
```

交互式多轮对话（保留历史）：

```powershell
cargo run --release -- --model-path <模型目录> --chat --interactive
```

交互命令：`reset` 清空历史，`exit`/`quit` 退出。

### 修改模型路径

命令行用 `--model-path` 指定模型目录即可。若用双击的 `启动对话.bat`，里面的路径是写死的，换机器或换模型时用记事本改这一行：

```bat
"%~dp0target\release\rsinfer.exe" --model-path "你的模型目录" --chat -i -n 1024
```

模型目录需包含 `config.json`、`tokenizer.json` 和 `model.safetensors`（或分片 + index.json）。

### 主要参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--model-path` | 模型目录 | 必填 |
| `-p, --prompt` | 提示词，省略则进交互模式 | — |
| `--chat` | 套用 Qwen3 chat 模板 | 关 |
| `--system` | system 提示词（chat 模式） | — |
| `-n, --max-tokens` | 最大生成长度 | 256 |
| `-t, --temperature` | 温度，0 为贪心 | 0.6 |
| `--top-p` / `--top-k` | 核采样 / top-k | 0.95 / 20 |
| `-i, --interactive` | 交互模式 | 关 |
| `-v, --verbose` | 打印 prompt 与耗时 | 关 |
| `--device` | 运行时设备规划：`cpu` / `auto` / `hybrid` | `cpu` |
| `--gpu-layers` | 计划放到 GPU 的 Transformer 层数 | 自动估计或 0 |

采样默认值对齐 Qwen3-Thinking 官方推荐（temp 0.6 / top-k 20 / top-p 0.95）。

`--device auto` / `--device hybrid` 会在 Windows 上通过 `nvidia-smi` 探测 NVIDIA GPU，并在 `--verbose` 模式打印 CPU/GPU 分层计划。`--gpu-layers` 会让前 N 个 Transformer 层的 decode 单 token 线性层通过共享 wgpu context 尝试 GPU matvec：`q/k/v` 同输入批量提交，MLP decode 在可用时把 `gate/up -> SwiGLU -> down_proj` 留在 GPU 路径中，只回读最终 MLP 输出；prefill 多 token 仍回退 CPU。`lm_head` 也复用同一个 wgpu context 做 GPU matvec。输出中的 `runtime.transformer_decode_gpu_layers` 和 `runtime.lm_head` 会标明实际 GPU/fallback 状态，尚未接入的 attention/KV/prefill 不会被误报成加速。

## 代码结构

```
src/
├── tensor/    张量与数学算子 (matmul, rmsnorm, rope, attention, silu)
├── model/     config 解析 / safetensors 加载 / Transformer 层 / 模型组装
├── engine/    KV cache / 采样器 / 生成循环
├── runtime.rs 设备探测与 CPU/GPU 分层规划
└── main.rs    CLI 与交互式对话
```

## 已知限制与后续方向

- 当前 prefill、attention 与 KV cache 仍走 CPU；选中 Transformer 层的 decode 线性 matvec、MLP fused SwiGLU/down 路径与 `lm_head` 可在 `auto`/`hybrid` 模式下尝试 wgpu GPU offload。
- 无 batch、无 prompt 缓存复用、无量化（int8/int4）。
- KV cache 用简单拼接（短序列下非瓶颈）。
- 已有 GPU/CPU 运行时规划入口、decode 线性层 GPU matvec 和 `lm_head` GPU matvec；还没有 attention/KV/prefill GPU kernel。要生产级 GPU 推理仍建议用 llama.cpp + 量化 GGUF 作为参考基线。

## 许可证

[Apache-2.0](LICENSE)
