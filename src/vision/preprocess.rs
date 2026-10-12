//! Qwen3.5 静态图像前处理。
//!
//! 预处理排列与 Hugging Face Transformers 的 Qwen2VLImageProcessor 对齐：
//! 先按 `patch_size * spatial_merge_size` 约束动态缩放，再按
//! `[merge_block_h, merge_block_w, in_block_h, in_block_w]` 展开 patch。
//! Qwen3.5 的静态帧复制到 `temporal_patch_size` 个时间槽，以匹配 Conv3d
//! patch projection 的输入宽度；这不是视频支持。

use std::io::Cursor;

use image::{imageops::FilterType, ImageReader, RgbImage};

use crate::{Result, RsinferError};

use super::VisionLimits;

const IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGE_STD: [f32; 3] = [0.2686295, 0.2613026, 0.275_777_1];

#[derive(Debug)]
pub(crate) struct PreparedImage {
    pub(crate) patches: Vec<f32>,
    pub(crate) grid_h: usize,
    pub(crate) grid_w: usize,
    pub(crate) patch_width: usize,
}

pub(crate) fn prepare_bytes(
    bytes: &[u8],
    limits: &VisionLimits,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
) -> Result<PreparedImage> {
    validate_limits(limits)?;
    if bytes.len() > limits.max_bytes {
        return Err(RsinferError::DimensionError(format!(
            "图像字节数 {} 超过上限 {}",
            bytes.len(),
            limits.max_bytes
        )));
    }

    let decode_limits = image_decode_limits(limits)?;
    let mut dimensions_reader = ImageReader::new(Cursor::new(bytes));
    dimensions_reader.limits(decode_limits.clone());
    let dimensions = dimensions_reader
        .with_guessed_format()
        .map_err(reader_error)?
        .into_dimensions()
        .map_err(image_error)?;
    let width = usize::try_from(dimensions.0)
        .map_err(|_| RsinferError::DimensionError("图像宽度超出平台范围".into()))?;
    let height = usize::try_from(dimensions.1)
        .map_err(|_| RsinferError::DimensionError("图像高度超出平台范围".into()))?;
    check_decoded_dimensions(width, height, limits)?;

    let mut image_reader = ImageReader::new(Cursor::new(bytes));
    image_reader.limits(decode_limits);
    let image = image_reader
        .with_guessed_format()
        .map_err(reader_error)?
        .decode()
        .map_err(image_error)?
        .to_rgb8();
    prepare_rgb(
        image.width() as usize,
        image.height() as usize,
        image.as_raw(),
        limits,
        patch_size,
        merge_size,
        temporal_patch_size,
    )
}

pub(crate) fn prepare_rgb(
    width: usize,
    height: usize,
    rgb: &[u8],
    limits: &VisionLimits,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
) -> Result<PreparedImage> {
    validate_limits(limits)?;
    if width == 0 || height == 0 {
        return Err(RsinferError::DimensionError("图像宽高必须大于零".into()));
    }
    if patch_size == 0 || merge_size == 0 || temporal_patch_size == 0 {
        return Err(RsinferError::DimensionError(
            "视觉 patch、merge 和 temporal 维度必须大于零".into(),
        ));
    }
    let input_pixels = width
        .checked_mul(height)
        .ok_or_else(|| RsinferError::DimensionError("图像像素数溢出".into()))?;
    let input_bytes = input_pixels
        .checked_mul(3)
        .ok_or_else(|| RsinferError::DimensionError("图像字节数溢出".into()))?;
    if rgb.len() != input_bytes {
        return Err(RsinferError::DimensionError(format!(
            "RGB 缓冲长度不匹配：期望 {input_bytes}，实际 {}",
            rgb.len()
        )));
    }
    if input_bytes > limits.max_bytes {
        return Err(RsinferError::DimensionError(format!(
            "RGB 缓冲字节数 {} 超过上限 {}",
            input_bytes, limits.max_bytes
        )));
    }
    check_decoded_dimensions(width, height, limits)?;

    let factor = patch_size
        .checked_mul(merge_size)
        .ok_or_else(|| RsinferError::DimensionError("缩放因子溢出".into()))?;
    let patch_cap = limits.max_patches.min(
        limits
            .max_tokens
            .checked_mul(merge_size)
            .and_then(|n| n.checked_mul(merge_size))
            .ok_or_else(|| RsinferError::DimensionError("视觉 token 上限溢出".into()))?,
    );
    let patch_pixel_cap = patch_cap
        .checked_mul(patch_size)
        .and_then(|n| n.checked_mul(patch_size))
        .ok_or_else(|| RsinferError::DimensionError("视觉 patch 像素上限溢出".into()))?;
    let resize_pixel_cap = limits.max_pixels.min(patch_pixel_cap);
    if resize_pixel_cap == 0 {
        return Err(RsinferError::DimensionError(
            "视觉像素上限必须大于零".into(),
        ));
    }
    let (resized_height, resized_width) = smart_resize(height, width, factor, resize_pixel_cap)?;
    let resized_bytes = resized_height
        .checked_mul(resized_width)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| RsinferError::DimensionError("缩放后图像字节数溢出".into()))?;
    if resized_bytes > limits.max_bytes {
        return Err(RsinferError::DimensionError(format!(
            "缩放后 RGB 缓冲字节数 {} 超过上限 {}",
            resized_bytes, limits.max_bytes
        )));
    }

    let source = RgbImage::from_raw(width as u32, height as u32, rgb.to_vec())
        .ok_or_else(|| RsinferError::DimensionError("RGB 图像缓冲无法建立".into()))?;
    let resized = if resized_width == width && resized_height == height {
        source
    } else {
        image::imageops::resize(
            &source,
            resized_width as u32,
            resized_height as u32,
            FilterType::CatmullRom,
        )
    };

    let grid_h = resized_height / patch_size;
    let grid_w = resized_width / patch_size;
    let raw_patches = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| RsinferError::DimensionError("视觉 patch 数量溢出".into()))?;
    let merged_h = grid_h / merge_size;
    let merged_w = grid_w / merge_size;
    let merged_tokens = merged_h
        .checked_mul(merged_w)
        .ok_or_else(|| RsinferError::DimensionError("视觉 token 数量溢出".into()))?;
    if raw_patches == 0 || raw_patches > limits.max_patches {
        return Err(RsinferError::DimensionError(format!(
            "视觉 patch 数量 {} 超过上限 {}",
            raw_patches, limits.max_patches
        )));
    }
    if merged_tokens == 0 || merged_tokens > limits.max_tokens {
        return Err(RsinferError::DimensionError(format!(
            "合并后视觉 token 数量 {} 超过上限 {}",
            merged_tokens, limits.max_tokens
        )));
    }

    let patch_width = 3usize
        .checked_mul(temporal_patch_size)
        .and_then(|n| n.checked_mul(patch_size))
        .and_then(|n| n.checked_mul(patch_size))
        .ok_or_else(|| RsinferError::DimensionError("patch 输入宽度溢出".into()))?;
    let patches = patchify(
        &resized,
        grid_h,
        grid_w,
        patch_size,
        merge_size,
        temporal_patch_size,
    )?;
    Ok(PreparedImage {
        patches,
        grid_h,
        grid_w,
        patch_width,
    })
}

fn validate_limits(limits: &VisionLimits) -> Result<()> {
    if limits.max_pixels == 0
        || limits.max_patches == 0
        || limits.max_tokens == 0
        || limits.max_bytes == 0
    {
        return Err(RsinferError::DimensionError(
            "VisionLimits 的所有上限必须大于零".into(),
        ));
    }
    Ok(())
}

fn check_decoded_dimensions(width: usize, height: usize, limits: &VisionLimits) -> Result<()> {
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| RsinferError::DimensionError("解码图像像素数溢出".into()))?;
    if pixels > limits.max_pixels {
        return Err(RsinferError::DimensionError(format!(
            "解码图像像素数 {} 超过上限 {}",
            pixels, limits.max_pixels
        )));
    }
    Ok(())
}

fn image_decode_limits(limits: &VisionLimits) -> Result<image::Limits> {
    let max_alloc = limits
        .max_pixels
        .checked_mul(16)
        .ok_or_else(|| RsinferError::DimensionError("解码分配上限溢出".into()))?;
    let max_alloc = u64::try_from(max_alloc)
        .map_err(|_| RsinferError::DimensionError("解码分配上限超出平台范围".into()))?;
    let mut decode_limits = image::Limits::default();
    decode_limits.max_alloc = Some(max_alloc);
    Ok(decode_limits)
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if factor == 0 || max_pixels == 0 {
        return Err(RsinferError::DimensionError(
            "缩放因子和像素上限必须大于零".into(),
        ));
    }
    let short = height.min(width);
    let long = height.max(width);
    if short == 0 || (long as f64 / short as f64) > 200.0 {
        return Err(RsinferError::DimensionError(
            "图像宽高比必须不超过 200".into(),
        ));
    }
    let round_factor = |n: usize| -> usize {
        let q = (n as f64 / factor as f64).round() as usize;
        q.max(1).saturating_mul(factor)
    };
    let mut h_bar = round_factor(height);
    let mut w_bar = round_factor(width);
    let rounded_pixels = h_bar
        .checked_mul(w_bar)
        .ok_or_else(|| RsinferError::DimensionError("缩放像素数溢出".into()))?;
    let source_pixels = height
        .checked_mul(width)
        .ok_or_else(|| RsinferError::DimensionError("原图像素数溢出".into()))?;
    if rounded_pixels > max_pixels {
        let beta = (source_pixels as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / factor as f64).floor() as usize)
            .max(1)
            .saturating_mul(factor);
        w_bar = ((width as f64 / beta / factor as f64).floor() as usize)
            .max(1)
            .saturating_mul(factor);
    }
    let resized_pixels = h_bar
        .checked_mul(w_bar)
        .ok_or_else(|| RsinferError::DimensionError("缩放像素数溢出".into()))?;
    if resized_pixels > max_pixels {
        return Err(RsinferError::DimensionError(format!(
            "按 patch 因子缩放后仍需 {} 像素，超过上限 {}",
            resized_pixels, max_pixels
        )));
    }
    Ok((h_bar, w_bar))
}

fn patchify(
    image: &RgbImage,
    grid_h: usize,
    grid_w: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
) -> Result<Vec<f32>> {
    if !grid_h.is_multiple_of(merge_size) || !grid_w.is_multiple_of(merge_size) {
        return Err(RsinferError::DimensionError(
            "视觉网格必须能整除 spatial_merge_size".into(),
        ));
    }
    let patch_width = 3usize
        .checked_mul(temporal_patch_size)
        .and_then(|n| n.checked_mul(patch_size))
        .and_then(|n| n.checked_mul(patch_size))
        .ok_or_else(|| RsinferError::DimensionError("patch 宽度溢出".into()))?;
    let count = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| RsinferError::DimensionError("patch 数量溢出".into()))?;
    let capacity = count
        .checked_mul(patch_width)
        .ok_or_else(|| RsinferError::DimensionError("patch 缓冲大小溢出".into()))?;
    let blocks_w = grid_w / merge_size;
    let mut out = Vec::with_capacity(capacity);

    // 与 Qwen2VLImageProcessor.patchify 的 permute(0,2,5,3,6,1,4,7) 对齐。
    for block_h in 0..(grid_h / merge_size) {
        for block_w in 0..blocks_w {
            for inner_h in 0..merge_size {
                for inner_w in 0..merge_size {
                    let origin_y = (block_h * merge_size + inner_h) * patch_size;
                    let origin_x = (block_w * merge_size + inner_w) * patch_size;
                    for channel in 0..3 {
                        for _time in 0..temporal_patch_size {
                            for dy in 0..patch_size {
                                for dx in 0..patch_size {
                                    let pixel = image
                                        .get_pixel((origin_x + dx) as u32, (origin_y + dy) as u32);
                                    let value = pixel[channel] as f32 / 255.0;
                                    out.push((value - IMAGE_MEAN[channel]) / IMAGE_STD[channel]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn image_error(error: image::ImageError) -> RsinferError {
    RsinferError::Unsupported(format!("无法读取 PNG/JPEG 图像：{error}"))
}

fn reader_error(error: std::io::Error) -> RsinferError {
    RsinferError::Unsupported(format!("无法识别 PNG/JPEG 图像格式：{error}"))
}

pub(crate) fn patch_coordinates(
    token: usize,
    _grid_h: usize,
    grid_w: usize,
    merge: usize,
) -> (usize, usize) {
    let block_width = grid_w / merge;
    let block = token / (merge * merge);
    let inner = token % (merge * merge);
    let block_h = block / block_width;
    let block_w = block % block_width;
    (
        block_h * merge + inner / merge,
        block_w * merge + inner % merge,
    )
}

pub(crate) fn add_interpolated_positions(
    hidden: &mut [f32],
    position_embedding: &[f32],
    num_position_embeddings: usize,
    hidden_size: usize,
    grid_h: usize,
    grid_w: usize,
    merge: usize,
) -> Result<()> {
    let side = (num_position_embeddings as f64).sqrt() as usize;
    if side == 0 || side * side != num_position_embeddings {
        return Err(RsinferError::DimensionError(
            "num_position_embeddings 必须是平方数".into(),
        ));
    }
    let tokens = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| RsinferError::DimensionError("位置 token 数量溢出".into()))?;
    if hidden.len() != tokens * hidden_size
        || position_embedding.len() != num_position_embeddings * hidden_size
    {
        return Err(RsinferError::DimensionError(
            "位置嵌入表或隐藏状态长度不匹配".into(),
        ));
    }
    for token in 0..tokens {
        let (row, col) = patch_coordinates(token, grid_h, grid_w, merge);
        let h_taps = interpolation_taps(row, grid_h, side);
        let w_taps = interpolation_taps(col, grid_w, side);
        let dst = &mut hidden[token * hidden_size..(token + 1) * hidden_size];
        for (h_index, h_weight) in h_taps {
            for (w_index, w_weight) in w_taps {
                let weight = h_weight * w_weight;
                let source = &position_embedding[(h_index * side + w_index) * hidden_size
                    ..(h_index * side + w_index + 1) * hidden_size];
                for (out, value) in dst.iter_mut().zip(source) {
                    *out += weight * value;
                }
            }
        }
    }
    Ok(())
}

fn interpolation_taps(index: usize, size: usize, side: usize) -> [(usize, f32); 2] {
    if side <= 1 || size <= 1 {
        return [(0, 1.0), (0, 0.0)];
    }
    // align_corners=True: source = index * (side - 1) / (size - 1).
    let source = index as f32 * (side - 1) as f32 / (size - 1) as f32;
    let floor = source.floor() as usize;
    let next = (floor + 1).min(side - 1);
    let fraction = source - floor as f32;
    [(floor.min(side - 1), 1.0 - fraction), (next, fraction)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> VisionLimits {
        VisionLimits {
            max_pixels: 1024,
            max_patches: 16,
            max_tokens: 4,
            max_bytes: 4096,
        }
    }

    #[test]
    fn prepare_bytes_rejects_oversized_dimensions_before_decode() {
        use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};

        let raw = vec![0_u8; 33 * 33 * 3];
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes)
            .write_image(&raw, 33, 33, ColorType::Rgb8.into())
            .unwrap();

        let error = prepare_bytes(&bytes, &limits(), 2, 2, 2).unwrap_err();
        assert!(matches!(
            error,
            RsinferError::DimensionError(message)
                if message == "解码图像像素数 1089 超过上限 1024"
        ));
    }

    #[test]
    fn patch_order_is_merge_major_and_temporal_repeats() {
        let mut rgb = Vec::new();
        for y in 0..4u8 {
            for x in 0..4u8 {
                rgb.extend([x, y, x + y]);
            }
        }
        let prepared = prepare_rgb(4, 4, &rgb, &limits(), 2, 2, 2).unwrap();
        assert_eq!((prepared.grid_h, prepared.grid_w), (2, 2));
        assert_eq!(prepared.patch_width, 24);
        // First patch: channel R, temporal frame 0 then frame 1, 2x2 pixels.
        let first = &prepared.patches[..8];
        assert_eq!(first[0], first[4]);
        assert_eq!(first[1], first[5]);
        // The second patch is the right-hand patch in the same 2x2 merge block.
        let second = &prepared.patches[24..32];
        assert_ne!(first[0], second[0]);
    }

    #[test]
    fn interpolation_align_corners_hits_table_corners() {
        let mut hidden = vec![0.0; 4];
        let position = (0..4).map(|x| x as f32).collect::<Vec<_>>();
        add_interpolated_positions(&mut hidden, &position, 4, 1, 2, 2, 1).unwrap();
        assert_eq!(hidden, position);
    }

    #[test]
    fn merge_coordinates_follow_processor_layout() {
        let coords = (0..16)
            .map(|i| patch_coordinates(i, 4, 4, 2))
            .collect::<Vec<_>>();
        assert_eq!(coords[..4], [(0, 0), (0, 1), (1, 0), (1, 1)]);
        assert_eq!(coords[4..8], [(0, 2), (0, 3), (1, 2), (1, 3)]);
        assert_eq!(coords[8..12], [(2, 0), (2, 1), (3, 0), (3, 1)]);
    }
}
