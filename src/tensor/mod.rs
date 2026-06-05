//! 张量模块
//!
//! 提供张量数据结构、基础操作和数学算子。

mod dtypes;
#[allow(clippy::module_inception)]
mod tensor;
mod ops;

pub use dtypes::*;
pub use tensor::*;
pub use ops::*;
