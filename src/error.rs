//! 统一错误类型定义
//!
//! 使用 `thiserror` 提供类型安全的错误处理，避免 panic。

use thiserror::Error;

/// 推理引擎的统一错误类型
#[derive(Error, Debug)]
pub enum RsinferError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensors(String),

    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Tensor dimension error: {0}")]
    DimensionError(String),

    #[error("Model config error: {0}")]
    ConfigError(String),

    #[error("Weight loading error: {0}")]
    WeightError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}

/// 推理引擎的 Result 类型别名
pub type Result<T> = std::result::Result<T, RsinferError>;
