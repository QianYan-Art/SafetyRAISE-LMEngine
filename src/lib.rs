//! rsinfer - 一个简易的 LLM 推理引擎
//!
//! 使用 Rust 实现，支持 LLaMA 架构，可加载 HuggingFace SafeTensors 模型进行文本生成。

pub mod engine;
pub mod error;
pub mod model;
pub mod runtime;
pub mod tensor;

pub use error::{Result, RsinferError};
