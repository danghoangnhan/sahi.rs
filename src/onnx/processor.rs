//! Image preprocessing utilities for ONNX models.
//!
//! Provides traits and implementations for preprocessing images before
//! model inference and postprocessing model outputs.

use ndarray::{Array3, Array4, ArrayView3, Axis};

use crate::error::{Error, Result};
use crate::inference::ImageData;

/// Information about letterbox preprocessing.
///
/// Used to map detection coordinates back to original image space.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    /// Original image width.
    pub orig_width: u32,
    /// Original image height.
    pub orig_height: u32,
    /// Target width after letterbox.
    pub target_width: u32,
    /// Target height after letterbox.
    pub target_height: u32,
    /// Padding on the left.
    pub pad_left: u32,
    /// Padding on the top.
    pub pad_top: u32,
    /// Scale factor applied to the image.
    pub scale: f32,
}

impl LetterboxInfo {
    /// Map x coordinate from letterboxed space to original image space.
    #[inline]
    pub fn map_x(&self, x: f32) -> f32 {
        (x - self.pad_left as f32) / self.scale
    }

    /// Map y coordinate from letterboxed space to original image space.
    #[inline]
    pub fn map_y(&self, y: f32) -> f32 {
        (y - self.pad_top as f32) / self.scale
    }

    /// Map width from letterboxed space to original image space.
    #[inline]
    pub fn map_width(&self, w: f32) -> f32 {
        w / self.scale
    }

    /// Map height from letterboxed space to original image space.
    #[inline]
    pub fn map_height(&self, h: f32) -> f32 {
        h / self.scale
    }

    /// Map a bounding box (x, y, w, h) to original image space.
    pub fn map_bbox(&self, x: f32, y: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
        let mapped_x = self.map_x(x);
        let mapped_y = self.map_y(y);
        let mapped_w = self.map_width(w);
        let mapped_h = self.map_height(h);

        // Clip to image bounds
        let clipped_x = mapped_x.max(0.0).min(self.orig_width as f32);
        let clipped_y = mapped_y.max(0.0).min(self.orig_height as f32);
        let clipped_w = mapped_w.min(self.orig_width as f32 - clipped_x);
        let clipped_h = mapped_h.min(self.orig_height as f32 - clipped_y);

        (clipped_x, clipped_y, clipped_w, clipped_h)
    }
}

/// Trait for image preprocessing.
pub trait ImageProcessor: Send + Sync {
    /// Preprocess an image for model inference.
    ///
    /// Returns a 4D tensor in NCHW format and letterbox info for coordinate mapping.
    fn preprocess(&self, image: &ImageData) -> Result<(Array4<f32>, LetterboxInfo)>;
}

/// Standard preprocessor with configurable options.
#[derive(Debug, Clone)]
pub struct Preprocessor {
    /// Target width for the model input.
    pub target_width: u32,
    /// Target height for the model input.
    pub target_height: u32,
    /// Whether to normalize pixel values (0-255 -> 0-1).
    pub normalize: bool,
    /// Padding value for letterbox (0-255).
    pub pad_value: u8,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self {
            target_width: 640,
            target_height: 640,
            normalize: true,
            pad_value: 114,
        }
    }
}

impl Preprocessor {
    /// Create a new preprocessor with the specified target size.
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width,
            target_height,
            ..Default::default()
        }
    }

    /// Set the target size.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.target_width = width;
        self.target_height = height;
        self
    }

    /// Enable or disable normalization.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the padding value for letterbox.
    pub fn with_pad_value(mut self, value: u8) -> Self {
        self.pad_value = value;
        self
    }

    /// Apply letterbox resize to an image.
    ///
    /// Resizes the image to fit within target dimensions while maintaining
    /// aspect ratio, padding with the pad_value as needed.
    pub fn letterbox(&self, image: &ImageData) -> Result<(Array3<u8>, LetterboxInfo)> {
        let orig_h = image.height;
        let orig_w = image.width;
        let target_h = self.target_height;
        let target_w = self.target_width;

        // Calculate scale to fit image in target while maintaining aspect ratio
        let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);

        // Calculate new dimensions
        let new_w = (orig_w as f32 * scale).round() as u32;
        let new_h = (orig_h as f32 * scale).round() as u32;

        // Calculate padding
        let pad_w = target_w - new_w;
        let pad_h = target_h - new_h;
        let pad_left = pad_w / 2;
        let pad_top = pad_h / 2;

        let info = LetterboxInfo {
            orig_width: orig_w,
            orig_height: orig_h,
            target_width: target_w,
            target_height: target_h,
            pad_left,
            pad_top,
            scale,
        };

        // Create output array filled with padding value
        let mut output = Array3::<u8>::from_elem(
            (
                target_h as usize,
                target_w as usize,
                image.channels as usize,
            ),
            self.pad_value,
        );

        // Bilinear resize into the (possibly padded) target region.
        let src_view = ArrayView3::from_shape(
            (orig_h as usize, orig_w as usize, image.channels as usize),
            &image.data,
        )
        .map_err(|e| Error::preprocessing(format!("Failed to create array view: {}", e)))?;

        let max_x = orig_w as usize - 1;
        let max_y = orig_h as usize - 1;
        for y in 0..new_h as usize {
            let src_yf = y as f32 / scale;
            let y0 = (src_yf.floor() as usize).min(max_y);
            let y1 = (y0 + 1).min(max_y);
            let wy = src_yf - y0 as f32;

            for x in 0..new_w as usize {
                let src_xf = x as f32 / scale;
                let x0 = (src_xf.floor() as usize).min(max_x);
                let x1 = (x0 + 1).min(max_x);
                let wx = src_xf - x0 as f32;

                let dst_y = y + pad_top as usize;
                let dst_x = x + pad_left as usize;

                for c in 0..image.channels as usize {
                    let v00 = src_view[[y0, x0, c]] as f32;
                    let v01 = src_view[[y0, x1, c]] as f32;
                    let v10 = src_view[[y1, x0, c]] as f32;
                    let v11 = src_view[[y1, x1, c]] as f32;
                    let top = v00 + (v01 - v00) * wx;
                    let bottom = v10 + (v11 - v10) * wx;
                    output[[dst_y, dst_x, c]] = (top + (bottom - top) * wy).round() as u8;
                }
            }
        }

        Ok((output, info))
    }

    /// Convert HWC u8 array to NCHW f32 tensor.
    pub fn hwc_to_nchw(&self, image: &Array3<u8>) -> Array4<f32> {
        let normalize = self.normalize;
        // HWC -> CHW, convert to f32 (optionally normalized), then add the batch axis.
        image
            .view()
            .permuted_axes([2, 0, 1])
            .mapv(|v| {
                let value = v as f32;
                if normalize {
                    value / 255.0
                } else {
                    value
                }
            })
            .insert_axis(Axis(0))
    }
}

impl ImageProcessor for Preprocessor {
    fn preprocess(&self, image: &ImageData) -> Result<(Array4<f32>, LetterboxInfo)> {
        // Apply letterbox resize
        let (resized, info) = self.letterbox(image)?;

        // Convert to NCHW tensor
        let tensor = self.hwc_to_nchw(&resized);

        Ok((tensor, info))
    }
}

/// Non-Maximum Suppression for detection outputs.
pub fn nms(
    boxes: &[(f32, f32, f32, f32, f32, u32)], // (x, y, w, h, confidence, class_id)
    iou_threshold: f32,
) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Sort by confidence (descending)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].4.partial_cmp(&boxes[a].4).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }

        keep.push(i);
        let (x1, y1, w1, h1, _, class1) = boxes[i];

        for &j in &indices {
            if suppressed[j] || i == j {
                continue;
            }

            let (x2, y2, w2, h2, _, class2) = boxes[j];

            // Only suppress same class
            if class1 != class2 {
                continue;
            }

            let iou = compute_iou(x1, y1, w1, h1, x2, y2, w2, h2);
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// Compute IoU between two boxes in XYWH format.
#[allow(clippy::too_many_arguments)]
fn compute_iou(x1: f32, y1: f32, w1: f32, h1: f32, x2: f32, y2: f32, w2: f32, h2: f32) -> f32 {
    // Convert to corner format
    let x1_min = x1;
    let y1_min = y1;
    let x1_max = x1 + w1;
    let y1_max = y1 + h1;

    let x2_min = x2;
    let y2_min = y2;
    let x2_max = x2 + w2;
    let y2_max = y2 + h2;

    // Intersection
    let inter_x1 = x1_min.max(x2_min);
    let inter_y1 = y1_min.max(y2_min);
    let inter_x2 = x1_max.min(x2_max);
    let inter_y2 = y1_max.min(y2_max);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    // Union
    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let union_area = area1 + area2 - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_info_mapping() {
        let info = LetterboxInfo {
            orig_width: 1920,
            orig_height: 1080,
            target_width: 640,
            target_height: 640,
            pad_left: 0,
            pad_top: 140,
            scale: 640.0 / 1920.0,
        };

        // Test mapping
        let (x, _y, w, _h) = info.map_bbox(320.0, 320.0, 100.0, 100.0);
        assert!((x - 960.0).abs() < 1.0);
        assert!((w - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_nms() {
        let boxes = vec![
            (10.0, 10.0, 50.0, 50.0, 0.9, 0),
            (15.0, 15.0, 50.0, 50.0, 0.8, 0), // Overlaps with first
            (100.0, 100.0, 50.0, 50.0, 0.7, 0), // No overlap
        ];

        let keep = nms(&boxes, 0.5);
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&0));
        assert!(keep.contains(&2));
    }

    #[test]
    fn test_compute_iou() {
        // Same box
        let iou = compute_iou(0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 100.0, 100.0);
        assert!((iou - 1.0).abs() < 0.001);

        // No overlap
        let iou = compute_iou(0.0, 0.0, 100.0, 100.0, 200.0, 200.0, 100.0, 100.0);
        assert!(iou < 0.001);

        // 50% overlap
        let iou = compute_iou(0.0, 0.0, 100.0, 100.0, 50.0, 0.0, 100.0, 100.0);
        // Intersection: 50x100 = 5000, Union: 10000 + 10000 - 5000 = 15000
        assert!((iou - 5000.0 / 15000.0).abs() < 0.001);
    }

    #[test]
    fn test_hwc_to_nchw_reorders_and_normalizes() {
        // 2x2 RGB in HWC: [y][x][c].
        let data = vec![
            0u8, 1, 2, 10, 11, 12, // row y=0: (x0), (x1)
            20, 21, 22, 30, 31, 32, // row y=1: (x0), (x1)
        ];
        let hwc = Array3::from_shape_vec((2, 2, 3), data).unwrap();

        // normalize = true: NCHW[0, c, y, x] == HWC[y, x, c] / 255
        let t = Preprocessor::default().hwc_to_nchw(&hwc);
        assert_eq!(t.dim(), (1, 3, 2, 2));
        assert!((t[[0, 0, 0, 0]] - 0.0 / 255.0).abs() < 1e-7); // r @ (0,0)
        assert!((t[[0, 1, 0, 1]] - 11.0 / 255.0).abs() < 1e-7); // g @ (0,1)
        assert!((t[[0, 2, 1, 0]] - 22.0 / 255.0).abs() < 1e-7); // b @ (1,0)
        assert!((t[[0, 0, 1, 1]] - 30.0 / 255.0).abs() < 1e-7); // r @ (1,1)

        // normalize = false: raw values.
        let t2 = Preprocessor::default()
            .with_normalize(false)
            .hwc_to_nchw(&hwc);
        assert_eq!(t2[[0, 1, 0, 1]], 11.0);
        assert_eq!(t2[[0, 2, 1, 0]], 22.0);
    }

    #[test]
    fn test_letterbox_bilinear_interpolates() {
        // 2x2 single-channel image: column 0 = 0, column 1 = 100 (a horizontal step).
        let image = ImageData::new(vec![0, 100, 0, 100], 2, 2, 1);
        // Upscale 2x2 -> 4x4 (scale = 2, no padding).
        let (out, info) = Preprocessor::new(4, 4).letterbox(&image).unwrap();

        assert!((info.scale - 2.0).abs() < 1e-6);
        // Exact at both impls: dst x=0 -> src col 0 (=0); dst x=2 -> src col 1 (=100).
        assert_eq!(out[[0, 0, 0]], 0);
        assert_eq!(out[[0, 2, 0]], 100);
        // dst x=1 maps to src_x = 0.5 -> bilinear average of 0 and 100 = 50.
        // (Nearest-neighbor would give 0.)
        assert_eq!(out[[0, 1, 0]], 50);
    }
}
