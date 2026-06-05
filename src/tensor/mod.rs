//! 张量模块
//!
//! 提供张量数据结构、基础操作和数学算子。

mod dtypes;
mod ops;
#[allow(clippy::module_inception)]
mod tensor;

pub use dtypes::*;
pub use ops::*;
pub use tensor::*;
