//! YOLOv8-seg instance segmentation model.
//!
//! A built-in segmenter that implements [`SegmentationCallback`], so sliced
//! instance segmentation works without a hand-written callback. Decodes the two
//! YOLOv8-seg ONNX outputs — detections `(1, 4+num_classes+num_masks, N)` (or the
//! transposed `(1, N, 4+num_classes+num_masks)`) and mask prototypes
//! `(1, num_masks, mh, mw)` — into slice-relative [`MaskedDetection`]s. The
//! [`predict_instances`](crate::Sahi::predict_instances) pipeline rebases the
//! slice-relative masks into image space.

use std::path::PathBuf;
use std::sync::Mutex;

use ndarray::ArrayView2;

use crate::annotation::Mask;
use crate::detection::{BoundingBox, Detection};
use crate::error::{Error, Result};
use crate::inference::ImageData;
use crate::onnx::processor::{nms, ImageProcessor, LetterboxInfo, Preprocessor};
use crate::onnx::session::{ExecutionProvider, OnnxSession, OnnxSessionBuilder};
use crate::segmentation::{MaskedDetection, SegmentationCallback};

use super::processor::YOLOv8OutputFormat;

/// Number of mask prototypes/coefficients in a standard YOLOv8-seg export.
pub const DEFAULT_NUM_MASKS: u32 = 32;

/// Maximum number of pixels (`orig_width * orig_height`) a single decoded mask
/// buffer may cover. `decode_mask` allocates one `f32` per pixel per surviving
/// detection at full slice resolution, so an absurd image size (e.g. 60000²)
/// would request many gigabytes. 64M pixels (~256 MiB per `f32` buffer) is a sane
/// upper bound that still comfortably covers real images.
const MAX_MASK_PIXELS: u64 = 64 * 1024 * 1024;

/// Configuration for [`YOLOv8SegDetector`].
#[derive(Debug, Clone)]
pub struct YOLOv8SegConfig {
    /// Model path (ONNX file).
    pub model_path: PathBuf,
    /// Number of classes.
    pub num_classes: u32,
    /// Number of mask prototypes / per-box mask coefficients.
    pub num_masks: u32,
    /// Input image size (assumes square input).
    pub input_size: u32,
    /// Confidence threshold.
    pub confidence_threshold: f32,
    /// IoU threshold for NMS.
    pub iou_threshold: f32,
    /// Threshold applied to the (sigmoid) mask before tracing polygons.
    pub mask_threshold: f32,
    /// Execution provider.
    pub execution_provider: ExecutionProvider,
}

impl Default for YOLOv8SegConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            num_classes: 80,
            num_masks: DEFAULT_NUM_MASKS,
            input_size: 640,
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            mask_threshold: 0.5,
            execution_provider: ExecutionProvider::Cpu,
        }
    }
}

impl YOLOv8SegConfig {
    /// Create a config for the given model path (other fields defaulted).
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            ..Default::default()
        }
    }

    /// Set the number of classes.
    pub fn with_num_classes(mut self, num_classes: u32) -> Self {
        self.num_classes = num_classes;
        self
    }

    /// Set the number of mask prototypes/coefficients.
    pub fn with_num_masks(mut self, num_masks: u32) -> Self {
        self.num_masks = num_masks;
        self
    }

    /// Set the input size.
    pub fn with_input_size(mut self, size: u32) -> Self {
        self.input_size = size;
        self
    }

    /// Set the confidence threshold.
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the IoU threshold.
    pub fn with_iou_threshold(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Set the mask threshold.
    pub fn with_mask_threshold(mut self, threshold: f32) -> Self {
        self.mask_threshold = threshold;
        self
    }

    /// Set the execution provider.
    pub fn with_execution_provider(mut self, ep: ExecutionProvider) -> Self {
        self.execution_provider = ep;
        self
    }
}

/// Pre/post-processing for YOLOv8-seg.
#[derive(Debug, Clone)]
pub struct YOLOv8SegProcessor {
    /// Number of classes the model was trained on.
    pub num_classes: u32,
    /// Number of mask prototypes / per-box mask coefficients.
    pub num_masks: u32,
    /// Image preprocessor (letterbox + NCHW).
    pub preprocessor: Preprocessor,
    /// Confidence threshold for filtering detections.
    pub confidence_threshold: f32,
    /// IoU threshold for NMS.
    pub iou_threshold: f32,
    /// Threshold applied to the (sigmoid) mask before tracing polygons.
    pub mask_threshold: f32,
}

impl YOLOv8SegProcessor {
    /// Create a new processor.
    pub fn new(input_size: u32, num_classes: u32, num_masks: u32) -> Self {
        Self {
            num_classes,
            num_masks,
            preprocessor: Preprocessor::new(input_size, input_size),
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            mask_threshold: 0.5,
        }
    }

    /// Set the confidence threshold.
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the IoU threshold.
    pub fn with_iou_threshold(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Set the mask threshold.
    pub fn with_mask_threshold(mut self, threshold: f32) -> Self {
        self.mask_threshold = threshold;
        self
    }

    /// Per-box feature dimension: 4 bbox values + `num_classes` + `num_masks`.
    fn box_dim(&self) -> usize {
        4 + self.num_classes as usize + self.num_masks as usize
    }

    /// Auto-detect the detection-tensor layout from its shape: Standard when
    /// `shape[1] == box_dim`, Transposed when `shape[2] == box_dim`.
    pub fn detect_seg_output_format(&self, out0_shape: &[i64]) -> YOLOv8OutputFormat {
        if out0_shape.len() < 3 {
            return YOLOv8OutputFormat::Standard;
        }
        let box_dim = self.box_dim() as i64;
        if out0_shape[1] == box_dim {
            YOLOv8OutputFormat::Standard
        } else if out0_shape[2] == box_dim {
            YOLOv8OutputFormat::Transposed
        } else {
            YOLOv8OutputFormat::Standard
        }
    }

    /// Decode YOLOv8-seg outputs into slice-relative masked detections.
    ///
    /// * `out0` / `out0_shape` — detections `(1, box_dim, N)` or `(1, N, box_dim)`.
    /// * `proto` / `proto_shape` — mask prototypes `(1, num_masks, mh, mw)`.
    /// * `info` — letterbox info for mapping coordinates back to slice space.
    pub fn process_seg_output(
        &self,
        out0: &[f32],
        out0_shape: &[i64],
        proto: &[f32],
        proto_shape: &[i64],
        info: &LetterboxInfo,
    ) -> Result<Vec<MaskedDetection>> {
        if out0_shape.len() < 3 {
            return Err(Error::invalid_output(format!(
                "expected 3D detection output, got shape {:?}",
                out0_shape
            )));
        }
        if out0_shape[0] != 1 {
            return Err(Error::invalid_output(format!(
                "expected batch size 1, got {}",
                out0_shape[0]
            )));
        }
        if proto_shape.len() != 4 || proto_shape[0] != 1 {
            return Err(Error::invalid_output(format!(
                "expected prototypes shape (1, num_masks, mh, mw), got {:?}",
                proto_shape
            )));
        }

        let dim1 = out0_shape[1] as usize;
        let dim2 = out0_shape[2] as usize;
        if out0.len() != dim1 * dim2 {
            return Err(Error::invalid_output(format!(
                "detection output length {} != dim1*dim2 = {}",
                out0.len(),
                dim1 * dim2
            )));
        }
        let format = self.detect_seg_output_format(out0_shape);
        let (num_boxes, found_box_dim) = match format {
            YOLOv8OutputFormat::Standard => (dim2, dim1),
            YOLOv8OutputFormat::Transposed => (dim1, dim2),
        };

        // Validate the box dimension. `detect_seg_output_format` silently falls back to
        // Standard when neither dim matches `box_dim`, so without this guard a model
        // whose box-dim differs (e.g. (1,5,8400) with a 116-dim config) would drive the
        // `get(b, 4+c)` reads below out of bounds and panic. Mirror the detection path.
        let expected_box_dim = self.box_dim();
        if found_box_dim != expected_box_dim {
            return Err(Error::invalid_output(format!(
                "expected {} features per box (4 + {} classes + {} masks), got {}",
                expected_box_dim, self.num_classes, self.num_masks, found_box_dim
            )));
        }

        let array = ArrayView2::from_shape((dim1, dim2), out0).map_err(|e| {
            Error::invalid_output(format!("failed to view detection output: {}", e))
        })?;
        // Read feature `feat` of box `b`, honoring the detected layout.
        let get = |b: usize, feat: usize| -> f32 {
            match format {
                YOLOv8OutputFormat::Standard => array[[feat, b]],
                YOLOv8OutputFormat::Transposed => array[[b, feat]],
            }
        };

        let nm = proto_shape[1] as usize;
        let mh = proto_shape[2] as usize;
        let mw = proto_shape[3] as usize;
        if nm != self.num_masks as usize {
            return Err(Error::invalid_output(format!(
                "prototype channels {} != num_masks {}",
                nm, self.num_masks
            )));
        }
        if proto.len() != nm * mh * mw {
            return Err(Error::invalid_output(format!(
                "prototype length {} != num_masks*mh*mw = {}",
                proto.len(),
                nm * mh * mw
            )));
        }

        let nc = self.num_classes as usize;
        let mut candidates: Vec<Candidate> = Vec::new();
        for b in 0..num_boxes {
            // Best class score.
            let mut best_c = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            for c in 0..nc {
                let s = get(b, 4 + c);
                if s > best_s {
                    best_s = s;
                    best_c = c;
                }
            }
            if best_s < self.confidence_threshold {
                continue;
            }

            // Center -> corner, then map to slice (original) coordinates.
            let (cx, cy, w, h) = (get(b, 0), get(b, 1), get(b, 2), get(b, 3));
            let (ox, oy, ow, oh) = info.map_bbox(cx - w / 2.0, cy - h / 2.0, w, h);
            if ow <= 0.0 || oh <= 0.0 {
                continue;
            }

            let mut coeffs = Vec::with_capacity(nm);
            for k in 0..nm {
                coeffs.push(get(b, 4 + nc + k));
            }
            candidates.push(Candidate {
                bbox: BoundingBox::new(ox, oy, ow, oh),
                class_id: best_c as u32,
                confidence: best_s,
                coeffs,
            });
        }

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Box NMS in slice coordinates.
        let nms_boxes: Vec<(f32, f32, f32, f32, f32, u32)> = candidates
            .iter()
            .map(|c| {
                (
                    c.bbox.x,
                    c.bbox.y,
                    c.bbox.width,
                    c.bbox.height,
                    c.confidence,
                    c.class_id,
                )
            })
            .collect();
        let keep = nms(&nms_boxes, self.iou_threshold);

        // Decode a mask only for the survivors.
        let mut results = Vec::with_capacity(keep.len());
        for &i in &keep {
            let c = &candidates[i];
            let mask = self.decode_mask(&c.coeffs, &c.bbox, proto, nm, mh, mw, info)?;
            results.push(MaskedDetection::new(
                Detection::new(c.bbox, c.class_id, c.confidence, None),
                Some(mask),
            ));
        }
        Ok(results)
    }

    /// Build a slice-relative mask for one detection from its mask coefficients
    /// and the prototype tensor. The (sigmoid) mask is sampled only within the
    /// box (cropped) at slice resolution, then thresholded and traced to polygons
    /// via [`Mask::from_float_mask`]. Sampling maps each slice pixel through the
    /// letterbox into the prototype grid (nearest-neighbor).
    #[allow(clippy::too_many_arguments)]
    fn decode_mask(
        &self,
        coeffs: &[f32],
        bbox: &BoundingBox,
        proto: &[f32],
        nm: usize,
        mh: usize,
        mw: usize,
        info: &LetterboxInfo,
    ) -> Result<Mask> {
        // Bound the per-mask allocation: this buffer is `orig_width * orig_height`
        // f32s, so an absurd image size would request many GB and OOM. Compute the
        // product in u64 to avoid the multiply itself overflowing.
        let pixels = info.orig_width as u64 * info.orig_height as u64;
        if pixels > MAX_MASK_PIXELS {
            return Err(Error::invalid_output(format!(
                "mask resolution {}x{} = {} pixels exceeds cap {}",
                info.orig_width, info.orig_height, pixels, MAX_MASK_PIXELS
            )));
        }

        let ow = info.orig_width as usize;
        let oh = info.orig_height as usize;

        // Box pixel range in slice coordinates, clamped to the slice bounds.
        let bx0 = (bbox.x.floor().max(0.0) as usize).min(ow);
        let by0 = (bbox.y.floor().max(0.0) as usize).min(oh);
        let bx1 = (((bbox.x + bbox.width).ceil()).max(0.0) as usize).min(ow);
        let by1 = (((bbox.y + bbox.height).ceil()).max(0.0) as usize).min(oh);

        // Prototype grid covers the letterboxed target; map slice -> target -> proto.
        let sx = mw as f32 / info.target_width as f32;
        let sy = mh as f32 / info.target_height as f32;

        let mut mask = vec![0.0f32; ow * oh];
        for oy in by0..by1 {
            let lb_y = oy as f32 * info.scale + info.pad_top as f32;
            let py = ((lb_y * sy) as usize).min(mh.saturating_sub(1));
            for ox in bx0..bx1 {
                let lb_x = ox as f32 * info.scale + info.pad_left as f32;
                let px = ((lb_x * sx) as usize).min(mw.saturating_sub(1));
                let mut acc = 0.0f32;
                for (k, &coeff) in coeffs.iter().enumerate().take(nm) {
                    acc += coeff * proto[k * mh * mw + py * mw + px];
                }
                mask[oy * ow + ox] = sigmoid(acc);
            }
        }

        Ok(Mask::from_float_mask(
            &mask,
            ow as u32,
            oh as u32,
            self.mask_threshold,
            [oh as u32, ow as u32],
            None,
        ))
    }
}

/// A decoded candidate detection awaiting NMS + mask decoding.
struct Candidate {
    bbox: BoundingBox,
    class_id: u32,
    confidence: f32,
    coeffs: Vec<f32>,
}

/// Numerically-stable-enough logistic sigmoid.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Built-in YOLOv8-seg instance segmenter.
pub struct YOLOv8SegDetector {
    config: YOLOv8SegConfig,
    session: Option<Mutex<OnnxSession>>,
    processor: YOLOv8SegProcessor,
}

impl std::fmt::Debug for YOLOv8SegDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOv8SegDetector")
            .field("config", &self.config)
            .field("is_loaded", &self.session.is_some())
            .finish()
    }
}

impl YOLOv8SegDetector {
    /// Create a detector from a full config.
    pub fn from_config(config: YOLOv8SegConfig) -> Self {
        let processor =
            YOLOv8SegProcessor::new(config.input_size, config.num_classes, config.num_masks)
                .with_confidence_threshold(config.confidence_threshold)
                .with_iou_threshold(config.iou_threshold)
                .with_mask_threshold(config.mask_threshold);
        Self {
            config,
            session: None,
            processor,
        }
    }

    /// Create a detector from just a model path.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self::from_config(YOLOv8SegConfig::new(model_path))
    }

    /// Get the configuration.
    pub fn config(&self) -> &YOLOv8SegConfig {
        &self.config
    }

    /// Get the processor.
    pub fn processor(&self) -> &YOLOv8SegProcessor {
        &self.processor
    }

    /// Whether the ONNX session is loaded.
    pub fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// Drop the ONNX session.
    pub fn unload(&mut self) {
        self.session = None;
    }

    /// Load the ONNX session from the configured model path.
    pub fn load(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }
        let session = OnnxSessionBuilder::new()
            .execution_provider(self.config.execution_provider)
            .build_from_file(&self.config.model_path)?;
        self.session = Some(Mutex::new(session));
        Ok(())
    }
}

impl SegmentationCallback for YOLOv8SegDetector {
    fn infer(&self, image: &ImageData) -> Result<Vec<MaskedDetection>> {
        let session_mutex = self.session.as_ref().ok_or(Error::ModelNotLoaded)?;
        let mut session = session_mutex
            .lock()
            .map_err(|_| Error::inference("Session lock poisoned"))?;

        let (input_tensor, info) = self.processor.preprocessor.preprocess(image)?;

        let input_name = session
            .input_name()
            .ok_or_else(|| Error::invalid_output("Model has no input".to_string()))?
            .to_string();
        let output_names: Vec<String> = session
            .output_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        if output_names.len() < 2 {
            return Err(Error::invalid_output(format!(
                "YOLOv8-seg expects 2 outputs (detections + prototypes), got {}",
                output_names.len()
            )));
        }

        let ort_input = ort::value::Tensor::from_array(input_tensor.into_dyn())?;
        let outputs = session
            .session_mut()
            .run(ort::inputs![&input_name => ort_input])?;

        // Disambiguate the detection tensor (rank 3) from the prototypes (rank 4).
        let mut det: Option<(Vec<i64>, Vec<f32>)> = None;
        let mut proto: Option<(Vec<i64>, Vec<f32>)> = None;
        for name in &output_names {
            let Some(value) = outputs.get(name) else {
                continue;
            };
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            let shape_v: Vec<i64> = shape.iter().copied().collect();
            match shape_v.len() {
                3 => det = Some((shape_v, data.to_vec())),
                4 => proto = Some((shape_v, data.to_vec())),
                _ => {}
            }
        }

        let (det_shape, det_data) =
            det.ok_or_else(|| Error::invalid_output("missing 3D detection output".to_string()))?;
        let (proto_shape, proto_data) = proto
            .ok_or_else(|| Error::invalid_output("missing 4D prototype output".to_string()))?;

        self.processor
            .process_seg_output(&det_data, &det_shape, &proto_data, &proto_shape, &info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_info(size: u32) -> LetterboxInfo {
        LetterboxInfo {
            orig_width: size,
            orig_height: size,
            target_width: size,
            target_height: size,
            pad_left: 0,
            pad_top: 0,
            scale: 1.0,
        }
    }

    // proto[0] is high (1.0) inside [2,6)x[2,6), low (-1.0) elsewhere, on an 8x8 grid.
    fn proto_block_8x8() -> Vec<f32> {
        let mut p = vec![-1.0f32; 8 * 8];
        for y in 2..6 {
            for x in 2..6 {
                p[y * 8 + x] = 1.0;
            }
        }
        p
    }

    fn pixel(mask: &Mask, x: usize, y: usize) -> bool {
        let w = mask.full_shape().width as usize;
        mask.to_bool_mask()[y * w + x]
    }

    #[test]
    fn test_detect_seg_output_format_standard_and_transposed() {
        // num_classes=1, num_masks=1 -> box_dim = 4 + 1 + 1 = 6
        let p = YOLOv8SegProcessor::new(8, 1, 1);
        assert_eq!(
            p.detect_seg_output_format(&[1, 6, 7]),
            YOLOv8OutputFormat::Standard
        );
        assert_eq!(
            p.detect_seg_output_format(&[1, 7, 6]),
            YOLOv8OutputFormat::Transposed
        );
    }

    #[test]
    fn test_process_seg_output_decodes_box_and_mask() {
        let p = YOLOv8SegProcessor::new(8, 1, 1).with_confidence_threshold(0.1);
        let info = identity_info(8);

        // One box: center (4,4), size 4x4, class score 0.9, mask coeff 10.0.
        // Standard layout (1, box_dim=6, N=1): features in order for the single box.
        let out0 = vec![4.0, 4.0, 4.0, 4.0, 0.9, 10.0];
        let proto = proto_block_8x8();

        let dets = p
            .process_seg_output(&out0, &[1, 6, 1], &proto, &[1, 1, 8, 8], &info)
            .expect("decode");
        assert_eq!(dets.len(), 1, "one detection expected");

        let d = &dets[0];
        assert_eq!(d.detection.class_id, 0);
        assert!((d.detection.bbox.x - 2.0).abs() < 1e-3);
        assert!((d.detection.bbox.y - 2.0).abs() < 1e-3);
        assert!((d.detection.bbox.width - 4.0).abs() < 1e-3);

        let mask = d.mask.as_ref().expect("mask present");
        assert!(pixel(mask, 3, 3), "mask should cover the box interior");
        assert!(!pixel(mask, 0, 0), "mask should not cover outside the box");
    }

    #[test]
    fn test_process_seg_output_transposed_matches_standard() {
        let p = YOLOv8SegProcessor::new(8, 1, 1).with_confidence_threshold(0.1);
        let info = identity_info(8);
        let proto = proto_block_8x8();
        // Same single box, transposed layout (1, N=1, box_dim=6) — same flat data.
        let out0 = vec![4.0, 4.0, 4.0, 4.0, 0.9, 10.0];

        let dets = p
            .process_seg_output(&out0, &[1, 1, 6], &proto, &[1, 1, 8, 8], &info)
            .expect("decode transposed");
        assert_eq!(dets.len(), 1);
        let mask = dets[0].mask.as_ref().expect("mask present");
        assert!(pixel(mask, 3, 3));
    }

    #[test]
    fn test_process_seg_output_mismatched_box_dim_returns_err() {
        // num_classes=1, num_masks=1 -> box_dim = 6. Feed an output whose box-dim
        // matches neither dim1 (5) nor dim2 (3). detect_seg_output_format silently
        // returns Standard, so the decode loop would index get(b, 4+c) out of bounds.
        // The processor must validate the box dimension and return Err instead.
        let p = YOLOv8SegProcessor::new(8, 1, 1).with_confidence_threshold(0.1);
        let info = identity_info(8);
        // Shape (1, 5, 3): dim1=5 (not 6), dim2=3 (not 6). 5*3 = 15 elements.
        let out0 = vec![0.5f32; 15];
        let proto = proto_block_8x8();
        let res = p.process_seg_output(&out0, &[1, 5, 3], &proto, &[1, 1, 8, 8], &info);
        assert!(
            res.is_err(),
            "mismatched box-dim must return Err, not panic/OOB"
        );
    }

    #[test]
    fn test_process_seg_output_absurd_orig_dims_returns_err() {
        // A surviving detection with absurd original image dimensions would drive
        // decode_mask to allocate ow*oh f32s (~many GB). The processor must reject
        // such dimensions and return Err rather than attempting the allocation.
        // box_dim = 6 here, matching dim1, so we reach the decode path with one box.
        let p = YOLOv8SegProcessor::new(8, 1, 1).with_confidence_threshold(0.1);
        // orig_width * orig_height = 100_000 * 100_000 = 1e10 pixels, far over the cap.
        let info = LetterboxInfo {
            orig_width: 100_000,
            orig_height: 100_000,
            target_width: 8,
            target_height: 8,
            pad_left: 0,
            pad_top: 0,
            scale: 8.0 / 100_000.0,
        };
        // One box covering a small region; mask coeff high so it passes threshold.
        let out0 = vec![4.0, 4.0, 4.0, 4.0, 0.9, 10.0];
        let proto = proto_block_8x8();
        let res = p.process_seg_output(&out0, &[1, 6, 1], &proto, &[1, 1, 8, 8], &info);
        assert!(
            res.is_err(),
            "absurd orig dims must return Err, not attempt a huge allocation"
        );
    }

    #[test]
    fn test_process_seg_output_subthreshold_yields_no_detections() {
        let p = YOLOv8SegProcessor::new(8, 1, 1).with_confidence_threshold(0.5);
        let info = identity_info(8);
        let proto = proto_block_8x8();
        // class score 0.05 < confidence threshold 0.5 -> filtered out.
        let out0 = vec![4.0, 4.0, 4.0, 4.0, 0.05, 10.0];
        let dets = p
            .process_seg_output(&out0, &[1, 6, 1], &proto, &[1, 1, 8, 8], &info)
            .expect("decode");
        assert!(dets.is_empty());
    }

    #[test]
    fn test_detector_usable_as_callback_and_requires_load() {
        let det = YOLOv8SegDetector::new("nonexistent.onnx");
        assert!(!det.is_loaded());
        // Coerces to the trait object that `predict_instances` accepts.
        let cb: &dyn SegmentationCallback = &det;
        let img = ImageData::from_rgb(vec![0u8; 8 * 8 * 3], 8, 8);
        assert!(cb.infer(&img).is_err(), "infer before load should error");
    }

    #[test]
    #[ignore = "requires a real YOLOv8-seg ONNX model via SAHI_TEST_YOLOV8_SEG_MODEL"]
    fn integration_predict_with_real_model() {
        let path = match std::env::var("SAHI_TEST_YOLOV8_SEG_MODEL") {
            Ok(p) => p,
            Err(_) => return,
        };
        let mut det = YOLOv8SegDetector::new(path);
        det.load().expect("load model");
        let img = ImageData::from_rgb(vec![0u8; 640 * 640 * 3], 640, 640);
        let out = det.infer(&img).expect("infer");
        // Smoke test: the call runs end-to-end and returns a (possibly empty) vec.
        let _ = out.len();
    }
}
