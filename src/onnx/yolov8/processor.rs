//! YOLOv8-specific preprocessing and postprocessing.
//!
//! Handles the specific input/output formats for YOLOv8 models.

use ndarray::{ArrayView1, ArrayView2};

use crate::detection::{BoundingBox, Detection};
use crate::error::{Error, Result};
use crate::onnx::processor::{nms, LetterboxInfo, Preprocessor};

/// YOLOv8 output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum YOLOv8OutputFormat {
    /// Standard YOLOv8 format: (1, 84, 8400) or (1, num_classes+4, num_boxes)
    /// Each column is [x_center, y_center, w, h, class_scores...]
    #[default]
    Standard,
    /// Transposed format: (1, 8400, 84) or (1, num_boxes, num_classes+4)
    Transposed,
}

/// YOLOv8-specific processor configuration.
#[derive(Debug, Clone)]
pub struct YOLOv8Processor {
    /// Number of classes the model was trained on.
    pub num_classes: u32,
    /// Image preprocessor.
    pub preprocessor: Preprocessor,
    /// Confidence threshold for filtering detections.
    pub confidence_threshold: f32,
    /// IoU threshold for NMS.
    pub iou_threshold: f32,
    /// Expected output format.
    pub output_format: YOLOv8OutputFormat,
}

impl Default for YOLOv8Processor {
    fn default() -> Self {
        Self {
            num_classes: 80, // COCO classes
            preprocessor: Preprocessor::default(),
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            output_format: YOLOv8OutputFormat::Standard,
        }
    }
}

impl YOLOv8Processor {
    /// Create a new processor with the specified settings.
    pub fn new(input_size: u32, num_classes: u32) -> Self {
        Self {
            num_classes,
            preprocessor: Preprocessor::new(input_size, input_size),
            ..Default::default()
        }
    }

    /// Set the confidence threshold.
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set the IoU threshold for NMS.
    pub fn with_iou_threshold(mut self, threshold: f32) -> Self {
        self.iou_threshold = threshold;
        self
    }

    /// Set the output format.
    pub fn with_output_format(mut self, format: YOLOv8OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Process YOLOv8 output tensor into detections.
    ///
    /// # Arguments
    ///
    /// * `output` - Raw model output tensor (flattened)
    /// * `output_shape` - Shape of the output tensor [batch, dim1, dim2]
    /// * `letterbox_info` - Information about the preprocessing
    ///
    /// # Returns
    ///
    /// Vector of detections in original image coordinates.
    pub fn process_output(
        &self,
        output: &[f32],
        output_shape: &[i64],
        letterbox_info: &LetterboxInfo,
    ) -> Result<Vec<Detection>> {
        // Validate shape
        if output_shape.len() < 3 {
            return Err(Error::invalid_output(format!(
                "Expected 3D output, got shape: {:?}",
                output_shape
            )));
        }

        let batch_size = output_shape[0] as usize;
        let dim1 = output_shape[1] as usize;
        let dim2 = output_shape[2] as usize;

        if batch_size != 1 {
            return Err(Error::invalid_output(format!(
                "Expected batch size 1, got {}",
                batch_size
            )));
        }

        // Auto-detect the output layout from the actual tensor shape. YOLOv8 exports vary
        // between Standard (1, box_dim, num_boxes) and Transposed (1, num_boxes, box_dim).
        let format = Self::detect_output_format(output_shape, self.num_classes);

        let (_num_boxes, box_dim) = match format {
            YOLOv8OutputFormat::Standard => {
                // Standard: (1, 84, 8400) -> dim1 is box_dim, dim2 is num_boxes
                (dim2, dim1)
            }
            YOLOv8OutputFormat::Transposed => {
                // Transposed: (1, 8400, 84) -> dim1 is num_boxes, dim2 is box_dim
                (dim1, dim2)
            }
        };

        // Validate dimensions
        let expected_dim = 4 + self.num_classes as usize;
        if box_dim != expected_dim {
            return Err(Error::invalid_output(format!(
                "Expected {} dimensions per box (4 + {} classes), got {}",
                expected_dim, self.num_classes, box_dim
            )));
        }

        // Create array view
        let array = ArrayView2::from_shape((dim1, dim2), output)
            .map_err(|e| Error::invalid_output(format!("Failed to create array view: {}", e)))?;

        // Process based on the detected format
        let detections = match format {
            YOLOv8OutputFormat::Standard => self.process_standard_output(&array, letterbox_info),
            YOLOv8OutputFormat::Transposed => {
                self.process_transposed_output(&array, letterbox_info)
            }
        };

        Ok(detections)
    }

    /// Process standard format: (84, 8400) where columns are boxes
    fn process_standard_output(
        &self,
        array: &ArrayView2<f32>,
        letterbox_info: &LetterboxInfo,
    ) -> Vec<Detection> {
        let num_boxes = array.ncols();
        let mut candidates = Vec::with_capacity(num_boxes / 10); // Rough estimate

        for box_idx in 0..num_boxes {
            let box_data = array.column(box_idx);
            if let Some(det) = self.decode_box(&box_data, letterbox_info) {
                candidates.push(det);
            }
        }

        self.apply_nms(candidates)
    }

    /// Process transposed format: (8400, 84) where rows are boxes
    fn process_transposed_output(
        &self,
        array: &ArrayView2<f32>,
        letterbox_info: &LetterboxInfo,
    ) -> Vec<Detection> {
        let num_boxes = array.nrows();
        let mut candidates = Vec::with_capacity(num_boxes / 10);

        for box_idx in 0..num_boxes {
            let box_data = array.row(box_idx);
            if let Some(det) = self.decode_box(&box_data, letterbox_info) {
                candidates.push(det);
            }
        }

        self.apply_nms(candidates)
    }

    /// Decode a single box from YOLOv8 output.
    fn decode_box(
        &self,
        box_data: &ArrayView1<f32>,
        letterbox_info: &LetterboxInfo,
    ) -> Option<Detection> {
        // YOLOv8 format: [x_center, y_center, width, height, class_scores...]
        let x_center = box_data[0];
        let y_center = box_data[1];
        let width = box_data[2];
        let height = box_data[3];

        // Find best class
        let class_scores = box_data.slice(ndarray::s![4..]);
        let (class_id, confidence) = class_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        // Apply confidence threshold
        if *confidence < self.confidence_threshold {
            return None;
        }

        // Convert from center format to corner format
        let x = x_center - width / 2.0;
        let y = y_center - height / 2.0;

        // Map back to original image coordinates
        let (orig_x, orig_y, orig_w, orig_h) = letterbox_info.map_bbox(x, y, width, height);

        // Skip invalid boxes
        if orig_w <= 0.0 || orig_h <= 0.0 {
            return None;
        }

        Some(Detection::new(
            BoundingBox::new(orig_x, orig_y, orig_w, orig_h),
            class_id as u32,
            *confidence,
            None,
        ))
    }

    /// Apply NMS to candidate detections.
    fn apply_nms(&self, candidates: Vec<Detection>) -> Vec<Detection> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // Convert to format expected by NMS
        let boxes: Vec<_> = candidates
            .iter()
            .map(|d| {
                (
                    d.bbox.x,
                    d.bbox.y,
                    d.bbox.width,
                    d.bbox.height,
                    d.confidence,
                    d.class_id,
                )
            })
            .collect();

        // Run NMS
        let keep_indices = nms(&boxes, self.iou_threshold);

        // Return kept detections
        keep_indices
            .into_iter()
            .map(|i| candidates[i].clone())
            .collect()
    }

    /// Auto-detect output format from tensor shape.
    pub fn detect_output_format(output_shape: &[i64], num_classes: u32) -> YOLOv8OutputFormat {
        if output_shape.len() < 3 {
            return YOLOv8OutputFormat::Standard;
        }

        let dim1 = output_shape[1] as u32;
        let dim2 = output_shape[2] as u32;
        let expected_box_dim = 4 + num_classes;

        if dim1 == expected_box_dim {
            YOLOv8OutputFormat::Standard
        } else if dim2 == expected_box_dim {
            YOLOv8OutputFormat::Transposed
        } else {
            // Default to standard
            YOLOv8OutputFormat::Standard
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_output_format() {
        // Standard: (1, 84, 8400)
        assert_eq!(
            YOLOv8Processor::detect_output_format(&[1, 84, 8400], 80),
            YOLOv8OutputFormat::Standard
        );

        // Transposed: (1, 8400, 84)
        assert_eq!(
            YOLOv8Processor::detect_output_format(&[1, 8400, 84], 80),
            YOLOv8OutputFormat::Transposed
        );
    }

    #[test]
    fn test_default_processor() {
        let processor = YOLOv8Processor::default();
        assert_eq!(processor.num_classes, 80);
        assert_eq!(processor.confidence_threshold, 0.25);
        assert_eq!(processor.iou_threshold, 0.45);
    }

    #[test]
    fn test_process_output_handles_standard_and_transposed() {
        let num_classes = 2;
        let processor = YOLOv8Processor::new(64, num_classes).with_confidence_threshold(0.1);

        // Identity letterbox: scale 1, no padding, so decoded coords == model coords.
        let info = crate::onnx::processor::LetterboxInfo {
            orig_width: 100,
            orig_height: 100,
            target_width: 64,
            target_height: 64,
            pad_left: 0,
            pad_top: 0,
            scale: 1.0,
        };

        // One box: center (50, 50), size 20x20, class 0 score 0.1, class 1 score 0.9.
        // Per-box layout: [x_center, y_center, w, h, score_c0, score_c1]
        let data = vec![50.0, 50.0, 20.0, 20.0, 0.1, 0.9];

        // Standard layout: (1, box_dim=6, num_boxes=1)
        let standard = processor
            .process_output(&data, &[1, 6, 1], &info)
            .expect("standard layout should decode");
        assert_eq!(standard.len(), 1);
        assert_eq!(standard[0].class_id, 1);
        assert!((standard[0].bbox.x - 40.0).abs() < 1e-3); // 50 - 20/2
        assert!((standard[0].bbox.width - 20.0).abs() < 1e-3);

        // Transposed layout: (1, num_boxes=1, box_dim=6) — same flat data, must auto-detect.
        let transposed = processor
            .process_output(&data, &[1, 1, 6], &info)
            .expect("transposed layout should auto-detect and decode");
        assert_eq!(transposed.len(), 1);
        assert_eq!(transposed[0].class_id, 1);
        assert!((transposed[0].bbox.x - 40.0).abs() < 1e-3);
    }
}
