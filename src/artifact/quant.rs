//! GGML 量化块的只读解码。
//!
//! 算法改编自 ggml（MIT License），固定参考提交：
//! `3612c0af93f2cfc8f048295300a61187723b12ae`，来源为
//! `src/ggml-quants.c` 与 `src/ggml-common.h`。本模块只保留 Q4_0、Q8_0、
//! Q4_K、Q6_K 的反量化逻辑；没有链接或调用 llama.cpp/ggml 运行时。

use crate::error::{Result, RsinferError};

const QK_K: usize = 256;

fn malformed(msg: impl Into<String>) -> RsinferError {
    RsinferError::WeightError(msg.into())
}

fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

/// 解码一个或多个 Q4_0 行，单行宽度必须是 32 的整数倍。
pub(crate) fn q4_0_row(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    if out.is_empty() || !out.len().is_multiple_of(32) {
        return Err(malformed("Q4_0 行宽必须是 32 的正整数倍"));
    }
    let blocks = out.len() / 32;
    let expected = blocks
        .checked_mul(18)
        .ok_or_else(|| malformed("Q4_0 行字节长度溢出"))?;
    if bytes.len() != expected {
        return Err(malformed(format!(
            "Q4_0 行字节长度错误：期望 {expected}，实际 {}",
            bytes.len()
        )));
    }
    for block in 0..blocks {
        let src = &bytes[block * 18..block * 18 + 18];
        let d = read_f16(src);
        for j in 0..16 {
            let packed = src[2 + j];
            out[block * 32 + j] = (i32::from(packed & 0x0f) - 8) as f32 * d;
            out[block * 32 + j + 16] = (i32::from(packed >> 4) - 8) as f32 * d;
        }
    }
    Ok(())
}

/// 解码一个或多个 Q8_0 行，单行宽度必须是 32 的整数倍。
pub(crate) fn q8_0_row(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    if out.is_empty() || !out.len().is_multiple_of(32) {
        return Err(malformed("Q8_0 行宽必须是 32 的正整数倍"));
    }
    let blocks = out.len() / 32;
    let expected = blocks
        .checked_mul(34)
        .ok_or_else(|| malformed("Q8_0 行字节长度溢出"))?;
    if bytes.len() != expected {
        return Err(malformed(format!(
            "Q8_0 行字节长度错误：期望 {expected}，实际 {}",
            bytes.len()
        )));
    }
    for block in 0..blocks {
        let src = &bytes[block * 34..block * 34 + 34];
        let d = read_f16(src);
        for j in 0..32 {
            out[block * 32 + j] = i8::from_le_bytes([src[2 + j]]) as f32 * d;
        }
    }
    Ok(())
}

fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 0x3f, scales[index + 4] & 0x3f)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

/// 解码一个或多个 Q4_K 行，单行宽度必须是 256 的整数倍。
pub(crate) fn q4_k_row(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    if out.is_empty() || !out.len().is_multiple_of(QK_K) {
        return Err(malformed("Q4_K 行宽必须是 256 的正整数倍"));
    }
    let blocks = out.len() / QK_K;
    let expected = blocks
        .checked_mul(144)
        .ok_or_else(|| malformed("Q4_K 行字节长度溢出"))?;
    if bytes.len() != expected {
        return Err(malformed(format!(
            "Q4_K 行字节长度错误：期望 {expected}，实际 {}",
            bytes.len()
        )));
    }
    for block in 0..blocks {
        let src = &bytes[block * 144..block * 144 + 144];
        let d = read_f16(src);
        let min = read_f16(&src[2..]);
        let scales = &src[4..16];
        let qs = &src[16..];
        let base = block * QK_K;
        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let d2 = d * f32::from(sc2);
            let m1 = min * f32::from(m1);
            let m2 = min * f32::from(m2);
            let q = &qs[(j / 2)..(j / 2 + 32)];
            for l in 0..32 {
                out[base + j + l] = d1 * f32::from(q[l] & 0x0f) - m1;
                out[base + j + l + 32] = d2 * f32::from(q[l] >> 4) - m2;
            }
            is += 2;
        }
    }
    Ok(())
}

/// 解码一个或多个 Q6_K 行，单行宽度必须是 256 的整数倍。
pub(crate) fn q6_k_row(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    if out.is_empty() || !out.len().is_multiple_of(QK_K) {
        return Err(malformed("Q6_K 行宽必须是 256 的正整数倍"));
    }
    let blocks = out.len() / QK_K;
    let expected = blocks
        .checked_mul(210)
        .ok_or_else(|| malformed("Q6_K 行字节长度溢出"))?;
    if bytes.len() != expected {
        return Err(malformed(format!(
            "Q6_K 行字节长度错误：期望 {expected}，实际 {}",
            bytes.len()
        )));
    }
    for block in 0..blocks {
        let src = &bytes[block * 210..block * 210 + 210];
        let ql = &src[..128];
        let qh = &src[128..192];
        let scales = &src[192..208];
        let d = read_f16(&src[208..]);
        let base = block * QK_K;
        for n in (0..QK_K).step_by(128) {
            let ql_chunk = &ql[(n / 2)..(n / 2 + 64)];
            let qh_chunk = &qh[(n / 4)..(n / 4 + 32)];
            let sc = &scales[(n / 16)..(n / 16 + 8)];
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql_chunk[l] & 0x0f) | ((qh_chunk[l] & 0x03) << 4)) - 32;
                let q2 =
                    i32::from((ql_chunk[l + 32] & 0x0f) | (((qh_chunk[l] >> 2) & 0x03) << 4)) - 32;
                let q3 = i32::from((ql_chunk[l] >> 4) | (((qh_chunk[l] >> 4) & 0x03) << 4)) - 32;
                let q4 =
                    i32::from((ql_chunk[l + 32] >> 4) | (((qh_chunk[l] >> 6) & 0x03) << 4)) - 32;
                out[base + n + l] = d * i8::from_le_bytes([sc[is]]) as f32 * q1 as f32;
                out[base + n + l + 32] = d * i8::from_le_bytes([sc[is + 2]]) as f32 * q2 as f32;
                out[base + n + l + 64] = d * i8::from_le_bytes([sc[is + 4]]) as f32 * q3 as f32;
                out[base + n + l + 96] = d * i8::from_le_bytes([sc[is + 6]]) as f32 * q4 as f32;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_known_block() {
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 的 1.0
        for b in &mut block[2..] {
            *b = 0x87; // 低半字节为 -1，高半字节为 0
        }
        let mut out = [0.0f32; 32];
        q4_0_row(&block, &mut out).unwrap();
        assert_eq!(out[0], -1.0);
        assert_eq!(out[16], 0.0);
    }

    #[test]
    fn q8_0_known_block() {
        let mut block = vec![0u8; 34];
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = 0xff;
        let mut out = [0.0f32; 32];
        q8_0_row(&block, &mut out).unwrap();
        assert_eq!(out[0], -1.0);
    }

    #[test]
    fn q4_k_known_sub_blocks() {
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3c; // d = 1
                         // 前 8 个低 6 位 scale，后 4 个字节保存高位 scale/min nibble。
        for value in &mut block[4..12] {
            *value = 1;
        }
        for value in &mut block[12..16] {
            *value = 1;
        }
        for value in &mut block[16..48] {
            *value = 0x11;
        }
        let mut out = [0.0f32; 256];
        q4_k_row(&block, &mut out).unwrap();
        assert!(out[..64].iter().all(|value| *value == 1.0));
        assert!(out[64..].iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q6_k_known_sub_block() {
        let mut block = vec![0u8; 210];
        for value in &mut block[192..208] {
            *value = 1;
        }
        block[208] = 0x00;
        block[209] = 0x3c; // d = 1
                           // 低半字节 1 加上高两位 2 编码 q=33，因此 q-32=1。
        for value in &mut block[0..32] {
            *value = 1;
        }
        for value in &mut block[128..192] {
            *value = 0xaa; // q1 到 q4 的高两位都编码为 32
        }
        let mut out = [0.0f32; 256];
        q6_k_row(&block, &mut out).unwrap();
        assert!(out[..32].iter().all(|value| *value == 1.0));
        assert!(out[32..].iter().all(|value| *value == 0.0));
    }
}
