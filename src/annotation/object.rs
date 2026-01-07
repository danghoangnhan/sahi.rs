//! Object annotation combining bounding box, mask, and category.
//!
//! This module provides the `ObjectAnnotation` type which represents a complete
//! detection/annotation with optional segmentation mask support.

use super::bbox::AnnotationBoundingBox;
use super::detection::Detection;
use super::mask::Mask;
use super::FullShape;
use crate::model::Category;

// ============================================================================
// Object Annotation
// ============================================================================

/// A complete object annotation with bounding box, optional mask, and category.
///
/// This is the primary type for representing detection/annotation results
/// in SAHI, supporting both object detection and instance segmentation.
///
/// # Example
///
/// ```rust,ignore
/// use sahi::annotation::ObjectAnnotation;
///
/// // Create from COCO bbox format
/// let ann = ObjectAnnotation::from_coco_bbox(
///     [100.0, 100.0, 50.0, 50.0],
///     0,
///     Some("person"),
///     [640, 480],
///     None,
/// );
///
/// // Get shifted annotation
/// let shifted = ann.get_shifted();
///
/// // Convert to detection
/// let detection = shifted.to_detection(0.95);
/// ```
#[derive(Debug, Clone)]
pub struct ObjectAnnotation {
    /// Bounding box with shift tracking.
    pub bbox: AnnotationBoundingBox,
    /// Optional segmentation mask.
    pub mask: Option<Mask>,
    /// Category (id and name).
    pub category: Category,
    /// Whether this annotation was merged from multiple sources.
    pub merged: bool,
}

impl ObjectAnnotation {
    /// Create a new object annotation.
    ///
    /// # Arguments
    ///
    /// * `bbox` - Bounding box [minx, miny, maxx, maxy]
    /// * `category_id` - Category ID
    /// * `category_name` - Optional category name
    /// * `segmentation` - Optional segmentation polygons (COCO format)
    /// * `full_shape` - Full image shape [height, width]
    /// * `shift_amount` - Optional shift [shift_x, shift_y]
    pub fn new(
        bbox: [f32; 4],
        category_id: u32,
        category_name: Option<&str>,
        segmentation: Option<Vec<Vec<f32>>>,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        let full_shape = full_shape.into();
        let shift = shift_amount.unwrap_or([0.0, 0.0]);

        let mask = segmentation.map(|seg| Mask::new(seg, full_shape, shift_amount));

        // Use mask bbox if available, otherwise use provided bbox
        let final_bbox = if let Some(ref m) = mask {
            m.bounding_box()
                .map(|b| [b.x, b.y, b.x2(), b.y2()])
                .unwrap_or(bbox)
        } else {
            bbox
        };

        // Clip bbox to image boundaries
        let clipped = [
            final_bbox[0].max(0.0),
            final_bbox[1].max(0.0),
            final_bbox[2].min(full_shape.width as f32),
            final_bbox[3].min(full_shape.height as f32),
        ];

        let ann_bbox = AnnotationBoundingBox::from_voc(clipped).with_shift(shift[0], shift[1]);

        let name = category_name
            .map(|n| n.to_string())
            .unwrap_or_else(|| category_id.to_string());

        Self {
            bbox: ann_bbox,
            mask,
            category: Category::new(category_id, name),
            merged: false,
        }
    }

    /// Create from COCO bbox format [x, y, width, height].
    pub fn from_coco_bbox(
        bbox: [f32; 4],
        category_id: u32,
        category_name: Option<&str>,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        // Convert COCO to VOC format
        let voc = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]];
        Self::new(
            voc,
            category_id,
            category_name,
            None,
            full_shape,
            shift_amount,
        )
    }

    /// Create from COCO segmentation format.
    pub fn from_coco_segmentation(
        segmentation: Vec<Vec<f32>>,
        category_id: u32,
        category_name: Option<&str>,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        // Compute bbox from segmentation
        let bbox = compute_bbox_from_segmentation(&segmentation).unwrap_or([0.0, 0.0, 0.0, 0.0]);
        Self::new(
            bbox,
            category_id,
            category_name,
            Some(segmentation),
            full_shape,
            shift_amount,
        )
    }

    /// Create from a boolean mask.
    pub fn from_bool_mask(
        mask: &[bool],
        width: u32,
        height: u32,
        category_id: u32,
        category_name: Option<&str>,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        let full_shape = full_shape.into();
        let mask_obj = Mask::from_bool_mask(mask, width, height, full_shape, shift_amount);
        let bbox = mask_obj
            .bounding_box()
            .map(|b| [b.x, b.y, b.x2(), b.y2()])
            .unwrap_or([0.0, 0.0, 0.0, 0.0]);

        let segmentation = mask_obj.to_coco_segmentation();
        Self::new(
            bbox,
            category_id,
            category_name,
            Some(segmentation),
            full_shape,
            shift_amount,
        )
    }

    /// Create from a core Detection (without mask).
    pub fn from_detection(
        detection: &Detection,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        let bbox = detection.bbox;
        Self::new(
            [bbox.x, bbox.y, bbox.x2(), bbox.y2()],
            detection.class_id,
            detection.class_name.as_deref(),
            None,
            full_shape,
            shift_amount,
        )
    }

    /// Get shifted object annotation (applies shift to bbox and mask).
    pub fn get_shifted(&self) -> Self {
        Self {
            bbox: self.bbox.get_shifted(),
            mask: self.mask.as_ref().map(|m| m.get_shifted()),
            category: self.category.clone(),
            merged: self.merged,
        }
    }

    /// Convert to core Detection type.
    ///
    /// Note: This loses mask information.
    pub fn to_detection(&self, confidence: f32) -> Detection {
        Detection::new(
            self.bbox.to_bbox(),
            self.category.id,
            confidence,
            Some(self.category.name().to_string()),
        )
    }

    /// Get the bounding box in COCO format [x, y, width, height].
    pub fn to_coco_bbox(&self) -> [f32; 4] {
        self.bbox.to_coco_bbox()
    }

    /// Get the bounding box in VOC format [xmin, ymin, xmax, ymax].
    pub fn to_voc_bbox(&self) -> [f32; 4] {
        self.bbox.to_voc_bbox()
    }

    /// Check if this annotation has a segmentation mask.
    pub fn has_mask(&self) -> bool {
        self.mask.is_some()
    }

    /// Get the segmentation in COCO format (if available).
    pub fn segmentation(&self) -> Option<Vec<Vec<f32>>> {
        self.mask.as_ref().map(|m| m.to_coco_segmentation())
    }

    /// Get the area of this annotation.
    ///
    /// Uses mask area if available, otherwise bbox area.
    pub fn area(&self) -> f32 {
        if let Some(ref mask) = self.mask {
            mask.area()
        } else {
            self.bbox.area()
        }
    }

    /// Create a deep copy of this annotation.
    pub fn deepcopy(&self) -> Self {
        self.clone()
    }

    /// Mark this annotation as merged.
    pub fn mark_merged(&mut self) {
        self.merged = true;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute bounding box from COCO segmentation polygons.
fn compute_bbox_from_segmentation(segmentation: &[Vec<f32>]) -> Option<[f32; 4]> {
    if segmentation.is_empty() {
        return None;
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for polygon in segmentation {
        for chunk in polygon.chunks_exact(2) {
            let x = chunk[0];
            let y = chunk[1];
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    if min_x == f32::MAX {
        None
    } else {
        Some([min_x, min_y, max_x, max_y])
    }
}

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(name = "ObjectAnnotation")]
    #[derive(Clone)]
    pub struct PyObjectAnnotation {
        inner: ObjectAnnotation,
    }

    #[pymethods]
    impl PyObjectAnnotation {
        #[new]
        #[pyo3(signature = (
            bbox,
            category_id,
            category_name=None,
            segmentation=None,
            full_shape=[0, 0],
            shift_amount=None
        ))]
        fn new(
            bbox: [f32; 4],
            category_id: u32,
            category_name: Option<&str>,
            segmentation: Option<Vec<Vec<f32>>>,
            full_shape: [u32; 2],
            shift_amount: Option<[f32; 2]>,
        ) -> Self {
            Self {
                inner: ObjectAnnotation::new(
                    bbox,
                    category_id,
                    category_name,
                    segmentation,
                    full_shape,
                    shift_amount,
                ),
            }
        }

        #[staticmethod]
        #[pyo3(signature = (bbox, category_id, category_name=None, full_shape=[0, 0], shift_amount=None))]
        fn from_coco_bbox(
            bbox: [f32; 4],
            category_id: u32,
            category_name: Option<&str>,
            full_shape: [u32; 2],
            shift_amount: Option<[f32; 2]>,
        ) -> Self {
            Self {
                inner: ObjectAnnotation::from_coco_bbox(
                    bbox,
                    category_id,
                    category_name,
                    full_shape,
                    shift_amount,
                ),
            }
        }

        #[staticmethod]
        #[pyo3(signature = (segmentation, category_id, category_name=None, full_shape=[0, 0], shift_amount=None))]
        fn from_coco_segmentation(
            segmentation: Vec<Vec<f32>>,
            category_id: u32,
            category_name: Option<&str>,
            full_shape: [u32; 2],
            shift_amount: Option<[f32; 2]>,
        ) -> Self {
            Self {
                inner: ObjectAnnotation::from_coco_segmentation(
                    segmentation,
                    category_id,
                    category_name,
                    full_shape,
                    shift_amount,
                ),
            }
        }

        fn get_shifted(&self) -> Self {
            Self {
                inner: self.inner.get_shifted(),
            }
        }

        #[getter]
        fn bbox(&self) -> AnnotationBoundingBox {
            self.inner.bbox
        }

        #[getter]
        fn category_id(&self) -> u32 {
            self.inner.category.id
        }

        #[getter]
        fn category_name(&self) -> &str {
            self.inner.category.name()
        }

        #[getter]
        fn has_mask(&self) -> bool {
            self.inner.has_mask()
        }

        #[getter]
        fn merged(&self) -> bool {
            self.inner.merged
        }

        fn to_coco_bbox(&self) -> [f32; 4] {
            self.inner.to_coco_bbox()
        }

        fn to_voc_bbox(&self) -> [f32; 4] {
            self.inner.to_voc_bbox()
        }

        fn segmentation(&self) -> Option<Vec<Vec<f32>>> {
            self.inner.segmentation()
        }

        fn area(&self) -> f32 {
            self.inner.area()
        }

        fn __repr__(&self) -> String {
            format!(
                "ObjectAnnotation(bbox={:?}, category={}:{}, has_mask={})",
                self.inner.to_voc_bbox(),
                self.inner.category.id,
                self.inner.category.name(),
                self.inner.has_mask()
            )
        }
    }
}


// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annotation_new() {
        let ann = ObjectAnnotation::new(
            [10.0, 20.0, 60.0, 80.0],
            0,
            Some("person"),
            None,
            [480, 640],
            None,
        );

        assert_eq!(ann.category.id, 0);
        assert_eq!(ann.category.name(), "person");
        assert!(!ann.has_mask());
        assert!(!ann.merged);
    }

    #[test]
    fn test_annotation_from_coco_bbox() {
        let ann = ObjectAnnotation::from_coco_bbox(
            [10.0, 20.0, 50.0, 60.0], // x, y, w, h
            1,
            Some("car"),
            [480, 640],
            None,
        );

        let voc = ann.to_voc_bbox();
        assert_eq!(voc[0], 10.0);
        assert_eq!(voc[1], 20.0);
        assert_eq!(voc[2], 60.0); // x + w
        assert_eq!(voc[3], 80.0); // y + h
    }

    #[test]
    fn test_annotation_from_segmentation() {
        let segmentation = vec![vec![10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]];
        let ann = ObjectAnnotation::from_coco_segmentation(
            segmentation.clone(),
            0,
            Some("person"),
            [100, 100],
            None,
        );

        assert!(ann.has_mask());
        assert_eq!(ann.segmentation(), Some(segmentation));
    }

    #[test]
    fn test_annotation_shift() {
        let ann = ObjectAnnotation::new(
            [10.0, 20.0, 60.0, 80.0],
            0,
            Some("person"),
            None,
            [480, 640],
            Some([100.0, 200.0]),
        );

        let shifted = ann.get_shifted();
        let bbox = shifted.to_voc_bbox();

        assert_eq!(bbox[0], 110.0);
        assert_eq!(bbox[1], 220.0);
        assert_eq!(bbox[2], 160.0);
        assert_eq!(bbox[3], 280.0);
    }

    #[test]
    fn test_annotation_to_detection() {
        let ann = ObjectAnnotation::new(
            [10.0, 20.0, 60.0, 80.0],
            0,
            Some("person"),
            None,
            [480, 640],
            None,
        );

        let detection = ann.to_detection(0.95);
        assert_eq!(detection.class_id, 0);
        assert_eq!(detection.confidence, 0.95);
        assert_eq!(detection.class_name.as_deref(), Some("person"));
    }

    #[test]
    fn test_annotation_area() {
        let ann = ObjectAnnotation::new([0.0, 0.0, 10.0, 10.0], 0, None, None, [100, 100], None);

        // Without mask, uses bbox area
        assert_eq!(ann.area(), 100.0);
    }

    #[test]
    fn test_annotation_category_name_default() {
        let ann = ObjectAnnotation::new(
            [0.0, 0.0, 10.0, 10.0],
            42,
            None, // No name provided
            None,
            [100, 100],
            None,
        );

        // Should default to category ID as string
        assert_eq!(ann.category.name(), "42");
    }

    #[test]
    fn test_compute_bbox_from_segmentation() {
        let seg = vec![vec![10.0, 20.0, 50.0, 20.0, 50.0, 60.0, 10.0, 60.0]];
        let bbox = compute_bbox_from_segmentation(&seg).unwrap();

        assert_eq!(bbox, [10.0, 20.0, 50.0, 60.0]);
    }

    #[test]
    fn test_annotation_clipping() {
        // Bbox extends beyond image boundaries
        let ann = ObjectAnnotation::new(
            [-10.0, -20.0, 700.0, 500.0],
            0,
            None,
            None,
            [480, 640],
            None,
        );

        let bbox = ann.to_voc_bbox();
        assert_eq!(bbox[0], 0.0); // Clipped to 0
        assert_eq!(bbox[1], 0.0); // Clipped to 0
        assert_eq!(bbox[2], 640.0); // Clipped to width
        assert_eq!(bbox[3], 480.0); // Clipped to height
    }
}
