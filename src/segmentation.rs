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
use crate::detection::{BoundingBox, Detection};
use crate::error::Result;
use crate::inference::ImageData;
use crate::postprocess::{MatchMetric, PostprocessConfig, PostprocessType};

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

    match config.postprocess_type {
        PostprocessType::NMS => seg_nms(items, config),
        PostprocessType::NMM => seg_nmm(items, config),
        PostprocessType::GREEDYNMM => seg_greedy_nmm(items, config),
    }
}

/// Bounding-box match score for the configured metric.
fn match_score(a: &BoundingBox, b: &BoundingBox, metric: MatchMetric) -> f32 {
    match metric {
        MatchMetric::IOU => a.iou(b),
        MatchMetric::IOS => a.ios(b),
    }
}

/// NMS: keep the highest-confidence box and suppress overlapping same-class boxes;
/// survivors keep their own mask. Assumes `items` is already sorted by descending
/// confidence.
fn seg_nms(items: Vec<MaskedDetection>, config: &PostprocessConfig) -> Vec<MaskedDetection> {
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
            let score = match_score(
                &items[i].detection.bbox,
                &items[j].detection.bbox,
                config.match_metric,
            );
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

/// NMM: group each anchor's matches (compared against the anchor) and merge each
/// group into one detection (unioned bbox, anchor's max confidence, unioned mask).
fn seg_nmm(items: Vec<MaskedDetection>, config: &PostprocessConfig) -> Vec<MaskedDetection> {
    let n = items.len();
    let mut used = vec![false; n];
    let mut result = Vec::new();
    for i in 0..n {
        if used[i] {
            continue;
        }
        let mut group = vec![i];
        for j in (i + 1)..n {
            if used[j] {
                continue;
            }
            if config.class_aware && items[i].detection.class_id != items[j].detection.class_id {
                continue;
            }
            let score = match_score(
                &items[i].detection.bbox,
                &items[j].detection.bbox,
                config.match_metric,
            );
            if score > config.match_threshold {
                group.push(j);
            }
        }
        for &k in &group {
            used[k] = true;
        }
        result.push(merge_group(&items, &group));
    }
    result
}

/// GREEDYNMM: like NMM, but the comparison box grows as it absorbs matches.
fn seg_greedy_nmm(items: Vec<MaskedDetection>, config: &PostprocessConfig) -> Vec<MaskedDetection> {
    let n = items.len();
    let mut used = vec![false; n];
    let mut result = Vec::new();
    for i in 0..n {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut group = vec![i];
        let mut bbox = items[i].detection.bbox;
        for j in (i + 1)..n {
            if used[j] {
                continue;
            }
            if config.class_aware && items[i].detection.class_id != items[j].detection.class_id {
                continue;
            }
            let score = match_score(&bbox, &items[j].detection.bbox, config.match_metric);
            if score > config.match_threshold {
                bbox = bbox.union_box(&items[j].detection.bbox);
                used[j] = true;
                group.push(j);
            }
        }
        result.push(merge_group(&items, &group));
    }
    result
}

/// Merge a group of detections (the first is the highest-confidence anchor) into
/// one: union bbox, the anchor's confidence/class, and the union of present masks.
fn merge_group(items: &[MaskedDetection], group: &[usize]) -> MaskedDetection {
    let anchor = &items[group[0]];
    let mut bbox = anchor.detection.bbox;
    for &k in &group[1..] {
        bbox = bbox.union_box(&items[k].detection.bbox);
    }
    let mut detection = anchor.detection.clone();
    detection.bbox = bbox;

    let masks: Vec<Mask> = group
        .iter()
        .filter_map(|&k| items[k].mask.clone())
        .collect();
    MaskedDetection::new(detection, union_masks(&masks))
}

/// Union several masks into one by rasterizing each into a shared bitmap (OR),
/// then re-tracing a single clean polygon set with the contour tracer. Counting
/// overlap once keeps `Mask::area` accurate and bounds the polygon count, unlike
/// concatenating polygon sets (which double-counts overlap and grows unbounded
/// across merges). Members are rasterized at the first mask's `full_shape`, which
/// the result also adopts. Returns `None` when no masks are given.
fn union_masks(masks: &[Mask]) -> Option<Mask> {
    let first = masks.first()?;
    let shape = first.full_shape();
    let (w, h) = (shape.width as usize, shape.height as usize);

    let mut acc = vec![false; w * h];
    for m in masks {
        // Rasterize this member's polygons at the shared `shape`, then OR it in.
        let bm = Mask::new(m.to_coco_segmentation(), shape, None).to_bool_mask();
        for (slot, &on) in acc.iter_mut().zip(bm.iter()) {
            *slot |= on;
        }
    }

    Some(Mask::from_bool_mask(
        &acc,
        shape.width,
        shape.height,
        shape,
        None,
    ))
}

// ============================================================================
// Python bindings (sub-project 9c)
// ============================================================================

#[cfg(feature = "python")]
pub(crate) mod python {
    use numpy::{PyArray, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    use crate::annotation::Mask;
    use crate::detection::Detection;
    use crate::error::{Error, Result};
    use crate::inference::ImageData;

    use super::{MaskedDetection, SegmentationCallback};

    /// Python wrapper for a detection with an optional polygon mask (COCO format).
    #[pyclass(name = "MaskedDetection")]
    #[derive(Clone)]
    pub struct PyMaskedDetection {
        /// The underlying detection (bbox, class, confidence).
        #[pyo3(get)]
        detection: Detection,
        /// Optional segmentation polygons, COCO `[[x1, y1, x2, y2, ...], ...]`.
        polygons: Option<Vec<Vec<f32>>>,
    }

    #[pymethods]
    impl PyMaskedDetection {
        #[new]
        #[pyo3(signature = (detection, mask=None))]
        fn new(detection: Detection, mask: Option<Vec<Vec<f32>>>) -> Self {
            Self {
                detection,
                polygons: mask,
            }
        }

        /// Segmentation polygons (COCO format), or `None`.
        #[getter]
        fn mask(&self) -> Option<Vec<Vec<f32>>> {
            self.polygons.clone()
        }

        /// Rasterize the polygon mask to a boolean numpy array of shape `(height, width)`.
        fn mask_array<'py>(
            &self,
            py: Python<'py>,
            height: u32,
            width: u32,
        ) -> PyResult<Bound<'py, PyArray2<bool>>> {
            let polys = self.polygons.clone().unwrap_or_default();
            let mask = Mask::new(polys, [height, width], None);
            let flat = mask.to_bool_mask();
            let arr = PyArray::from_vec(py, flat).reshape([height as usize, width as usize])?;
            Ok(arr)
        }

        fn __repr__(&self) -> String {
            format!(
                "MaskedDetection(class_id={}, confidence={:.3}, has_mask={})",
                self.detection.class_id,
                self.detection.confidence,
                self.polygons.is_some()
            )
        }
    }

    /// Adapter exposing a Python callback as a Rust [`SegmentationCallback`].
    struct PySegCallback {
        callback: PyObject,
    }

    // Safety: GIL-bound; only dereferenced within `Python::with_gil`.
    unsafe impl Send for PySegCallback {}
    unsafe impl Sync for PySegCallback {}

    impl SegmentationCallback for PySegCallback {
        fn infer(&self, image: &ImageData) -> Result<Vec<MaskedDetection>> {
            Python::with_gil(|py| {
                let array = PyArray::from_slice(py, &image.data);
                let reshaped = array
                    .reshape([
                        image.height as usize,
                        image.width as usize,
                        image.channels as usize,
                    ])
                    .map_err(|e| Error::Inference(e.to_string()))?;

                let result = self
                    .callback
                    .call1(py, (reshaped,))
                    .map_err(|e| Error::Inference(e.to_string()))?;
                let list = result.downcast_bound::<PyList>(py).map_err(|e| {
                    Error::Inference(format!("Expected list of MaskedDetection: {}", e))
                })?;

                let mut out = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let md: PyMaskedDetection = item
                        .extract()
                        .map_err(|e| Error::Inference(format!("Invalid MaskedDetection: {}", e)))?;
                    // The callback's polygons are in slice coordinates; full_shape is the slice.
                    let mask = md
                        .polygons
                        .map(|p| Mask::new(p, [image.height, image.width], None));
                    out.push(MaskedDetection::new(md.detection, mask));
                }
                Ok(out)
            })
        }
    }

    /// Drive `Sahi::predict_instances` with a Python callback and wrap the results.
    pub fn run_predict_instances(
        inner: &crate::Sahi,
        image: &Bound<'_, PyArray3<u8>>,
        callback: PyObject,
    ) -> PyResult<Vec<PyMaskedDetection>> {
        let shape = image.shape();
        let height = shape[0] as u32;
        let width = shape[1] as u32;
        let channels = shape[2] as u32;
        // Safety: only read within this scope.
        let arr = unsafe { image.as_array() };
        let data: Vec<u8> = arr.iter().copied().collect();
        let image_data = ImageData::new(data, width, height, channels);

        let cb = PySegCallback { callback };
        let results = inner
            .predict_instances(&image_data, &cb)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|md| PyMaskedDetection {
                detection: md.detection,
                polygons: md.mask.map(|m| m.to_coco_segmentation()),
            })
            .collect())
    }
}

#[cfg(feature = "python")]
pub use python::PyMaskedDetection;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::FullShape;
    use crate::detection::BoundingBox;
    use crate::postprocess::PostprocessType;
    use crate::Sahi;

    fn masked_poly(
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        conf: f32,
        class: u32,
        poly: Vec<f32>,
    ) -> MaskedDetection {
        MaskedDetection::new(
            Detection::new(BoundingBox::new(x, y, w, h), class, conf, None),
            Some(Mask::new(vec![poly], [200u32, 200u32], None)),
        )
    }

    fn masked_nomask(x: f32, y: f32, w: f32, h: f32, conf: f32, class: u32) -> MaskedDetection {
        MaskedDetection::new(
            Detection::new(BoundingBox::new(x, y, w, h), class, conf, None),
            None,
        )
    }

    fn in_mask(m: &Mask, x: usize, y: usize) -> bool {
        m.to_bool_mask()[y * (m.full_shape().width as usize) + x]
    }

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
        let cfg = PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::NMS);
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

    #[test]
    fn test_union_masks_keeps_disjoint_regions_separate() {
        // Two non-overlapping squares remain two separate components after the
        // rasterize/OR/re-trace union; the first mask's shape is preserved.
        let m1 = Mask::new(
            vec![vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]],
            [50u32, 50u32],
            None,
        );
        let m2 = Mask::new(
            vec![vec![20.0, 20.0, 30.0, 20.0, 30.0, 30.0, 20.0, 30.0]],
            [50u32, 50u32],
            None,
        );
        let u = union_masks(&[m1, m2]).unwrap();
        assert_eq!(u.segmentation().len(), 2);
        assert_eq!(u.full_shape(), FullShape::new(50, 50));
        assert!(union_masks(&[]).is_none());
    }

    /// Records how the pipeline invokes the callback.
    struct CountingSeg {
        infer_calls: std::sync::atomic::AtomicUsize,
        batch_calls: std::sync::atomic::AtomicUsize,
    }

    impl SegmentationCallback for CountingSeg {
        fn infer(&self, _image: &ImageData) -> Result<Vec<MaskedDetection>> {
            self.infer_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Vec::new())
        }
        fn infer_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<MaskedDetection>>> {
            self.batch_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(images.iter().map(|_| Vec::new()).collect())
        }
    }

    #[test]
    fn test_predict_instances_extracts_and_infers_through_backend_batch() {
        use std::sync::atomic::Ordering;
        // 200x100 image, 100x100 slices, no overlap -> two slices.
        let image = ImageData::from_rgb(vec![0u8; 200 * 100 * 3], 200, 100);
        let sahi = Sahi::builder()
            .slice_size(100, 100)
            .overlap(0.0, 0.0)
            .build();
        let cb = CountingSeg {
            infer_calls: std::sync::atomic::AtomicUsize::new(0),
            batch_calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let _ = sahi.predict_instances(&image, &cb).unwrap();
        assert_eq!(
            cb.batch_calls.load(Ordering::SeqCst),
            1,
            "slices should be inferred through a single infer_batch call"
        );
        assert_eq!(
            cb.infer_calls.load(Ordering::SeqCst),
            0,
            "slices should not be inferred one-by-one via infer"
        );
    }

    #[test]
    fn test_union_masks_dedups_overlapping_into_single_polygon() {
        // Two overlapping 50x50 squares on a 100x100 canvas form one connected region.
        let a = Mask::new(
            vec![vec![10.0, 10.0, 60.0, 10.0, 60.0, 60.0, 10.0, 60.0]],
            [100u32, 100u32],
            None,
        );
        let b = Mask::new(
            vec![vec![40.0, 40.0, 90.0, 40.0, 90.0, 90.0, 40.0, 90.0]],
            [100u32, 100u32],
            None,
        );
        let u = union_masks(&[a, b]).unwrap();
        assert_eq!(
            u.segmentation().len(),
            1,
            "overlapping masks should re-trace into a single polygon"
        );
    }

    #[test]
    fn test_union_masks_overlapping_area_is_union_not_sum() {
        let a = Mask::new(
            vec![vec![10.0, 10.0, 60.0, 10.0, 60.0, 60.0, 10.0, 60.0]],
            [100u32, 100u32],
            None,
        );
        let b = Mask::new(
            vec![vec![40.0, 40.0, 90.0, 40.0, 90.0, 90.0, 40.0, 90.0]],
            [100u32, 100u32],
            None,
        );
        // 50x50 + 50x50 polygons overlapping in a 20x20 corner; the overlap must
        // not be double-counted, so the union area is well below their sum.
        let summed = a.area() + b.area();
        let u = union_masks(&[a, b]).unwrap();
        assert!(
            u.area() < summed - 100.0,
            "union area {} should be well below the summed area {} (overlap double-counted)",
            u.area(),
            summed
        );
        assert!(
            u.area() > 4000.0,
            "union area {} should still cover the merged region",
            u.area()
        );
    }

    #[test]
    fn test_nmm_unions_masks() {
        let cfg = PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::NMM);
        // A and B overlap (bbox IoU ~0.68 > 0.5); their masks cover disjoint regions.
        let a = masked_poly(
            0.0,
            0.0,
            100.0,
            100.0,
            0.9,
            0,
            vec![5.0, 5.0, 35.0, 5.0, 35.0, 35.0, 5.0, 35.0],
        );
        let b = masked_poly(
            10.0,
            10.0,
            100.0,
            100.0,
            0.8,
            0,
            vec![60.0, 60.0, 90.0, 60.0, 90.0, 90.0, 60.0, 90.0],
        );
        let out = seg_stitch(vec![a, b], &cfg);
        assert_eq!(out.len(), 1);
        let m = &out[0];
        assert!((m.detection.confidence - 0.9).abs() < 1e-6); // max of the group
        assert_eq!(m.detection.bbox.width, 110.0); // union of (0..100) and (10..110)
        let mask = m.mask.as_ref().unwrap();
        assert!(in_mask(mask, 20, 20)); // A's region
        assert!(in_mask(mask, 75, 75)); // B's region -> proves the union
    }

    #[test]
    fn test_greedy_nmm_chain_merges_to_one() {
        let cfg =
            PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::GREEDYNMM);
        let a = masked_poly(
            0.0,
            0.0,
            100.0,
            100.0,
            0.9,
            0,
            vec![5.0, 5.0, 15.0, 5.0, 15.0, 15.0, 5.0, 15.0],
        );
        let b = masked_poly(
            20.0,
            0.0,
            100.0,
            100.0,
            0.8,
            0,
            vec![60.0, 5.0, 70.0, 5.0, 70.0, 15.0, 60.0, 15.0],
        );
        let c = masked_poly(
            40.0,
            0.0,
            100.0,
            100.0,
            0.7,
            0,
            vec![125.0, 5.0, 135.0, 5.0, 135.0, 15.0, 125.0, 15.0],
        );
        let out = seg_stitch(vec![a, b, c], &cfg);
        assert_eq!(out.len(), 1);
        let m = &out[0];
        assert!((m.detection.confidence - 0.9).abs() < 1e-6);
        assert_eq!(m.detection.bbox.width, 140.0); // (0..140)
        let mask = m.mask.as_ref().unwrap();
        assert!(in_mask(mask, 10, 10)); // A's region
        assert!(in_mask(mask, 130, 10)); // C's region -> whole chain merged
    }

    #[test]
    fn test_merge_uses_present_mask_when_anchor_has_none() {
        let cfg = PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::NMM);
        // The higher-confidence anchor has no mask; the lower-conf member carries one.
        let anchor = masked_nomask(0.0, 0.0, 100.0, 100.0, 0.9, 0);
        let member = masked_poly(
            10.0,
            10.0,
            100.0,
            100.0,
            0.8,
            0,
            vec![20.0, 20.0, 40.0, 20.0, 40.0, 40.0, 20.0, 40.0],
        );
        let out = seg_stitch(vec![anchor, member], &cfg);
        assert_eq!(out.len(), 1);
        let mask = out[0].mask.as_ref().expect("merged mask from the member");
        assert!(in_mask(mask, 30, 30));
    }

    #[test]
    fn test_nmm_keeps_disjoint_detections() {
        let cfg = PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::NMM);
        let a = masked_poly(
            0.0,
            0.0,
            20.0,
            20.0,
            0.9,
            0,
            vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0],
        );
        let b = masked_poly(
            150.0,
            150.0,
            20.0,
            20.0,
            0.8,
            0,
            vec![150.0, 150.0, 160.0, 150.0, 160.0, 160.0, 150.0, 160.0],
        );
        let out = seg_stitch(vec![a, b], &cfg);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|m| m.mask.is_some()));
    }
}
