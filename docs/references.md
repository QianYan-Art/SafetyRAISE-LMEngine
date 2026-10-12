# 设计参考与证据来源

本项目保持独立的 Rust 运行时实现。参考项目提供设计经验、格式定义和算法核对入口；其性能声明不转用为本项目成绩。复制或改编源码时逐处记录来源与许可证，思想参考和代码移植分别标注。

## 固定参考版本

| 来源 | 固定提交 | 许可证 | 实际用途 |
| --- | --- | --- | --- |
| Niko1221/Strata | `7695859aa13afb248bdb4d36f9b2e95707db079d` | MIT | 分层执行、GGUF、驻留与缓存、草稿/验证、按设备限定的内核 |
| Neroued/ninfer | `81c8ce093b2c1646a87566a8e59d807fcf0ec95c` | Apache-2.0 | Engine/Frontend/Program 所有权、资源调度、混合状态、分块 prefill、MTP 与原生算子合同 |
| Qwen/Qwen3-0.6B `tokenizer_config.json` | `c916fa4defd319b7d4e4da17604ca7338f4d99f5` | Apache-2.0 | `src/frontend/templates/qwen3.jinja` 的 chat template 来源；模板中的 Transformers `generation` 标记在本地渲染前规范化 |
| Hugging Face Transformers Qwen3.5 实现 | `812a950d7037ba93bb624b154030e9068afdce4d` | Apache-2.0 | Qwen3.5 文本的 GDN、RMSNorm、旋转位置编码，以及 `src/vision` 的 3D patch、视觉 attention 和几何核对 |
| Hugging Face Qwen2VLImageProcessor | `a12a224f759d4457e1a2fd4ba7ced46f9a6cc430` | Apache-2.0 | Qwen2VL 图像预处理、patchify 和多模态位置几何核对 |
| llama.cpp GGUF 映射参考 | `8bb3cf0d216a5ae48360179f2931daf284be4d01` | MIT | `src/model/gguf.rs` 的 Qwen2/Qwen3/Qwen3.5 metadata、张量命名和 GDN V-head tiled 映射 |
| ggml 量化布局参考 | `3612c0af93f2cfc8f048295300a61187723b12ae` | MIT | `src/artifact/quant.rs` 与 GGUF 量化字节布局的局部算法核对 |

参考路径：

- `https://github.com/Niko1221/Strata/tree/7695859aa13afb248bdb4d36f9b2e95707db079d`
- `https://github.com/Neroued/ninfer/blob/81c8ce093b2c1646a87566a8e59d807fcf0ec95c/docs/maintainer/engine-architecture.md`
- `https://github.com/Neroued/ninfer/blob/81c8ce093b2c1646a87566a8e59d807fcf0ec95c/docs/maintainer/qwen3_5-model.md`

## 数学、模型与格式边界

- Qwen2、Qwen3、Qwen3.5 文本模型的参数名、层合同和计算图由本项目 `ModelSpec`、`native` 执行器及测试 fixture 约束。Qwen3.5 文本数学核对使用上表固定 Transformers 提交；Qwen3 文本当前只保留模型配置和本地实现证据，未在源码中另行固定 Transformers 提交，不虚构版本。
- Qwen3.5 的 GDN、混合注意力、MTP 与视觉路径分别记录状态和验证边界；模型发行名不能替代 `model_type`、架构和层类型判断。
- MTP hidden 配对与逐步递归参考固定 NInfer 提交的 `src/models/qwen3_5/execution/text.cpp`、`src/models/qwen3_5/program/speculative/mtp.cpp`；上表 Transformers Qwen3.5 文件没有实现 MTP，不能用它作为 MTP 数值等价证据。
- GGUF/GGML 参考用于解释容器、张量命名和量化字节，不引入 llama.cpp 服务作为本项目执行后端。llama.cpp 只作为格式/基线参考，不能把它的速度或 CUDA 能力算作 `rsinfer` 的证据。
- 旧版源码、工作站 Bonsai 部署记录和 Orin 训练/交付记录用于提取实际约束；历史测速只作为设计动机，不能代替当前协议或真实权重验证。

## 许可证与移植记录

本项目主代码使用 Apache-2.0。格式算法、模板和数学参考的文件范围与许可声明保留在 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)；新增移植代码应在源文件注释中保留固定提交和许可证，并补充文件范围。性能声明、硬件验收和实测限制集中记录在 [验证证据与实验边界](validation.md)。
