//! Instance-segmentation pipeline (sub-project 9a of the segmentation epic).
//!
//! Adds a mask-carrying path alongside the bbox detection pipeline. A
//! [`SegmentationCallback`] returns slice-relative [`MaskedDetection`]s; the
//! [`predict_instances`](crate::Sahi::predict_instances) method slices the image,
//! runs the callback per slice, rebases results to image coordinates, and stitches
//! them with NMS (each survivor keeps its own mask).
//!
//! Mask-aware merging (NMM/GREEDYNMM mask union) and Python bindings are separate
//! sub-projects (9b, 9c).

use crate::annotation::Mask;
use crate::detection::Detection;
use crate::error::Result;
use crate::inference::ImageData;
use crate::postprocess::{MatchMetric, PostprocessConfig};

/// A detection paired with an optional segmentation mask.
///
/// Score, class, and bounding box live in `detection` (so NMS has a score to sort
/// and threshold on); `mask` is `None` for detection-only results.
#[derive(Debug, Clone)]
pub struct MaskedDetection {
    /// The underlying detection (bbox, class, confidence).
    pub detection: Detection,
    /// Optional instance mask, in the same coordinate space as `detection.bbox`.
    pub mask: Option<Mask>,
}

impl MaskedDetection {
    /// Create a new masked detection.
    pub fn new(detection: Detection, mask: Option<Mask>) -> Self {
        Self { detection, mask }
    }

    /// Detection confidence (convenience accessor).
    #[inline]
    pub fn confidence(&self) -> f32 {
        self.detection.confidence
    }

    /// Whether this result carries a mask.
    #[inline]
    pub fn has_mask(&self) -> bool {
        self.mask.is_some()
    }
}

/// Trait for segmentation inference callbacks.
///
/// Returns detections with masks in coordinates relative to the given image slice.
pub trait SegmentationCallback: Send + Sync {
    /// Run segmentation inference on a single image (slice).
    fn infer(&self, image: &ImageData) -> Result<Vec<MaskedDetection>>;

    /// Run inference on a batch of images. Defaults to calling `infer` per image.
    fn infer_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<MaskedDetection>>> {
        images.iter().map(|img| self.infer(img)).collect()
    }
}

/// Closure-backed segmentation callback.
pub struct FnSegCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<MaskedDetection>> + Send + Sync,
{
    func: F,
}

impl<F> FnSegCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<MaskedDetection>> + Send + Sync,
{
    /// Create a callback from a function.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> SegmentationCallback for FnSegCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<MaskedDetection>> + Send + Sync,
{
    fn infer(&self, image: &ImageData) -> Result<Vec<MaskedDetection>> {
        (self.func)(image)
    }
}

/// Create a [`SegmentationCallback`] from a closure.
pub fn seg_callback<F>(f: F) -> FnSegCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<MaskedDetection>> + Send + Sync,
{
    FnSegCallback::new(f)
}

/// Rebase a slice-relative mask into image coordinates: shift every polygon by
/// `(dx, dy)` and reset `full_shape` to the full image dimensions.
///
/// A dedicated rebase is needed because `Mask::get_shifted` clips to the slice's
/// `full_shape` and cannot move a mask into image space.
pub fn rebase_mask(mask: &Mask, dx: f32, dy: f32, image_w: u32, image_h: u32) -> Mask {
    let shifted: Vec<Vec<f32>> = mask
        .segmentation()
        .iter()
        .map(|p| p.shift(dx, dy).points)
        .collect();
    Mask::new(shifted, [image_h, image_w], None)
}

/// Stitch image-space masked detections with NMS, keeping each survivor's mask.
///
/// 9a is NMS-only regardless of `config.postprocess_type`; mask-union merging for
/// NMM/GREEDYNMM is sub-project 9b.
pub fn seg_stitch(
    mut items: Vec<MaskedDetection>,
    config: &PostprocessConfig,
) -> Vec<MaskedDetection> {
    // Drop low-confidence detections, then sort by confidence (descending).
    items.retain(|m| m.detection.confidence >= config.confidence_threshold);
    items.sort_by(|a, b| {
        b.detection
            .confidence
            .partial_cmp(&a.detection.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Greedy NMS on bounding boxes; survivors keep their own mask.
    let n = items.len();
    let mut keep = vec![true; n];
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..n {
            if !keep[j] {
                continue;
            }
            if config.class_aware && items[i].detection.class_id != items[j].detection.class_id {
                continue;
            }
            let (a, b) = (&items[i].detection.bbox, &items[j].detection.bbox);
            let score = match config.match_metric {
                MatchMetric::IOU => a.iou(b),
                MatchMetric::IOS => a.ios(b),
            };
            if score > config.match_threshold {
                keep[j] = false;
            }
        }
    }

    items
        .into_iter()
        .zip(keep)
        .filter_map(|(m, k)| k.then_some(m))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::FullShape;
    use crate::detection::BoundingBox;
    use crate::Sahi;

    fn masked(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        conf: f32,
        class: u32,
        with_mask: bool,
    ) -> MaskedDetection {
        let mask = with_mask.then(|| {
            Mask::new(
                vec![vec![x, y, x + w, y, x + w, y + h, x, y + h]],
                [100u32, 100u32],
                None,
            )
        });
        MaskedDetection::new(
            Detection::new(BoundingBox::new(x, y, w, h), class, conf, None),
            mask,
        )
    }

    #[test]
    fn test_rebase_mask_shifts_and_resets_shape() {
        let m = Mask::new(
            vec![vec![10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]],
            [50u32, 50u32],
            None,
        );
        let r = rebase_mask(&m, 100.0, 5.0, 200, 100);
        let pts = &r.segmentation()[0].points;
        assert_eq!(pts[0], 110.0);
        assert_eq!(pts[1], 15.0);
        assert_eq!(r.full_shape(), FullShape::new(100, 200));
    }

    #[test]
    fn test_seg_stitch_nms_keeps_top_and_mask() {
        let cfg = PostprocessConfig::new(0.5, 0.0);
        let items = vec![
            masked(0.0, 0.0, 100.0, 100.0, 0.9, 0, true),
            masked(10.0, 10.0, 100.0, 100.0, 0.8, 0, true), // overlaps the first
            masked(500.0, 500.0, 20.0, 20.0, 0.7, 0, true), // far away
        ];
        let out = seg_stitch(items, &cfg);
        assert_eq!(out.len(), 2);
        assert!((out[0].detection.confidence - 0.9).abs() < 1e-6);
        assert!(out[0].mask.is_some());
    }

    #[test]
    fn test_seg_stitch_confidence_filter() {
        let cfg = PostprocessConfig::new(0.5, 0.5);
        let items = vec![
            masked(0.0, 0.0, 10.0, 10.0, 0.6, 0, true),
            masked(50.0, 50.0, 10.0, 10.0, 0.3, 0, true), // below the conf threshold
        ];
        let out = seg_stitch(items, &cfg);
        assert_eq!(out.len(), 1);
        assert!((out[0].detection.confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_predict_instances_rebases_to_image_coords() {
        // 200x100 image, 100x100 slices, no overlap -> two slices at x=0 and x=100.
        let image = ImageData::from_rgb(vec![0u8; 200 * 100 * 3], 200, 100);
        let sahi = Sahi::builder()
            .slice_size(100, 100)
            .overlap(0.0, 0.0)
            .build();

        let cb = seg_callback(|_img: &ImageData| {
            Ok(vec![MaskedDetection::new(
                Detection::new(BoundingBox::new(10.0, 10.0, 10.0, 10.0), 0, 0.9, None),
                Some(Mask::new(
                    vec![vec![10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]],
                    [100u32, 100u32],
                    None,
                )),
            )])
        });

        let out = sahi.predict_instances(&image, &cb).unwrap();
        assert_eq!(out.len(), 2);

        // The second slice (origin x=100) -> detection at x=110, mask shifted, full image shape.
        let far = out
            .iter()
            .find(|m| m.detection.bbox.x > 100.0)
            .expect("slice-1 detection present");
        let pts = &far.mask.as_ref().unwrap().segmentation()[0].points;
        assert_eq!(pts[0], 110.0);
        assert_eq!(pts[1], 10.0);
        assert_eq!(
            far.mask.as_ref().unwrap().full_shape(),
            FullShape::new(100, 200)
        );
    }

    #[test]
    fn test_predict_instances_detection_only_passthrough() {
        let image = ImageData::from_rgb(vec![0u8; 100 * 100 * 3], 100, 100);
        let sahi = Sahi::builder().slice_size(100, 100).build();
        let cb = seg_callback(|_img: &ImageData| {
            Ok(vec![MaskedDetection::new(
                Detection::new(BoundingBox::new(5.0, 5.0, 10.0, 10.0), 0, 0.9, None),
                None,
            )])
        });
        let out = sahi.predict_instances(&image, &cb).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].mask.is_none());
    }
}
