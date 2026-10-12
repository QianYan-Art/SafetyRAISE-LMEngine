# 第三方来源与许可

本项目主体采用 [Apache-2.0](LICENSE)。以下列出实际格式算法与模板参考；固定提交、设计借鉴和证据范围见 [docs/references.md](docs/references.md)。下方英文为原始许可及版权声明，保留原文。

## ggml / llama.cpp

`src/artifact/quant.rs` 的 GGML 分块反量化布局与 `src/model/gguf.rs` 的 GGUF 张量映射参考这些 MIT 许可项目。Rust 实现包含布局还原、检查和视图适配；不使用它们的推理服务。

- ggml：`3612c0af93f2cfc8f048295300a61187723b12ae`
- llama.cpp：`8bb3cf0d216a5ae48360179f2931daf284be4d01`

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Qwen / Hugging Face Transformers

- `src/frontend/templates/qwen3.jinja` 取自 Qwen/Qwen3-0.6B 的 `tokenizer_config.json`，固定提交 `c916fa4defd319b7d4e4da17604ca7338f4d99f5`，Apache-2.0。本地加载器会规范化 Transformers 的 `generation` 标记；模板覆盖须由部署者显式选择。
- Qwen3.5 文本/视觉数学参考 Transformers `812a950d7037ba93bb624b154030e9068afdce4d`。原始文件版权声明：`Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.`，Apache-2.0。
- 图像预处理参考 Qwen2VLImageProcessor `a12a224f759d4457e1a2fd4ba7ced46f9a6cc430`，Apache-2.0。

本仓库按相应计算合同实现 Rust CPU 版本并加入本地预算、权重视图和状态事务；这些来源不代表上游对本项目的背书或实机性能保证。Apache-2.0 许可全文随本仓库 [LICENSE](LICENSE) 提供。

## 设计参考

Strata（MIT）和 NInfer（Apache-2.0）用于分层、调度、缓存、投机解码与算子接口设计取证，未将它们作为运行时依赖。其固定版本及参考范围见参考文档；依赖库许可由各 Cargo 包随包提供。
