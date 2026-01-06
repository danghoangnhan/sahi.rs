//! Bounding box types for object detection and annotations.
//!
//! This module provides the core `BoundingBox` type and the annotation-specific
//! `AnnotationBoundingBox` with shift amount tracking.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::model::ShiftAmount;

// ============================================================================
// Core Bounding Box
// ============================================================================

/// A bounding box in pixel coordinates.
///
/// Stored in (x, y, width, height) format where (x, y) is the top-left corner.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct BoundingBox {
    /// Top-left x coordinate
    pub x: f32,
    /// Top-left y coordinate
    pub y: f32,
    /// Width of the box
    pub width: f32,
    /// Height of the box
    pub height: f32,
}

impl BoundingBox {
    /// Create a new bounding box.
    #[inline]
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Create from corner coordinates (x1, y1, x2, y2) / VOC format.
    #[inline]
    pub fn from_corners(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        }
    }

    /// Create from COCO format [x, y, width, height].
    #[inline]
    pub fn from_coco(bbox: [f32; 4]) -> Self {
        Self::new(bbox[0], bbox[1], bbox[2], bbox[3])
    }

    /// Create from VOC format [xmin, ymin, xmax, ymax].
    #[inline]
    pub fn from_voc(bbox: [f32; 4]) -> Self {
        Self::from_corners(bbox[0], bbox[1], bbox[2], bbox[3])
    }

    /// Get the area of the bounding box.
    #[inline]
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Get the right edge x coordinate (xmax).
    #[inline]
    pub fn x2(&self) -> f32 {
        self.x + self.width
    }

    /// Get the bottom edge y coordinate (ymax).
    #[inline]
    pub fn y2(&self) -> f32 {
        self.y + self.height
    }

    /// Get the center point of the bounding box.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Calculate Intersection over Union (IoU) with another box.
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = self.x2().min(other.x2());
        let y2 = self.y2().min(other.y2());

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + other.area() - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Calculate Intersection over Smaller area (IoS).
    ///
    /// Useful for detecting when a smaller box is mostly inside a larger one.
    pub fn ios(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = self.x2().min(other.x2());
        let y2 = self.y2().min(other.y2());

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let smaller = self.area().min(other.area());

        if smaller > 0.0 {
            intersection / smaller
        } else {
            0.0
        }
    }

    /// Translate the bounding box by an offset.
    #[inline]
    pub fn translate(&self, dx: f32, dy: f32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            width: self.width,
            height: self.height,
        }
    }

    /// Convert to COCO format [x, y, width, height].
    #[inline]
    pub fn to_coco(&self) -> [f32; 4] {
        [self.x, self.y, self.width, self.height]
    }

    /// Alias for to_coco().
    #[inline]
    pub fn to_xywh(&self) -> [f32; 4] {
        self.to_coco()
    }

    /// Convert to VOC format [xmin, ymin, xmax, ymax].
    #[inline]
    pub fn to_voc(&self) -> [f32; 4] {
        [self.x, self.y, self.x2(), self.y2()]
    }

    /// Alias for to_voc().
    #[inline]
    pub fn to_xyxy(&self) -> [f32; 4] {
        self.to_voc()
    }

    /// Get an expanded bounding box by a ratio.
    ///
    /// The expansion is applied equally in all directions.
    /// Optionally clip to max boundaries.
    pub fn get_expanded(&self, ratio: f32, max_x: Option<f32>, max_y: Option<f32>) -> Self {
        let x_margin = self.width * ratio;
        let y_margin = self.height * ratio;

        let minx = (self.x - x_margin).max(0.0);
        let miny = (self.y - y_margin).max(0.0);
        let maxx = if let Some(max) = max_x {
            (self.x2() + x_margin).min(max)
        } else {
            self.x2() + x_margin
        };
        let maxy = if let Some(max) = max_y {
            (self.y2() + y_margin).min(max)
        } else {
            self.y2() + y_margin
        };

        Self::from_corners(minx, miny, maxx, maxy)
    }

    /// Check if this box contains a point.
    #[inline]
    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        x >= self.x && x <= self.x2() && y >= self.y && y <= self.y2()
    }

    /// Check if this box fully contains another box.
    #[inline]
    pub fn contains_box(&self, other: &BoundingBox) -> bool {
        self.x <= other.x
            && self.y <= other.y
            && self.x2() >= other.x2()
            && self.y2() >= other.y2()
    }

    /// Get the intersection with another box, if any.
    pub fn intersection(&self, other: &BoundingBox) -> Option<BoundingBox> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = self.x2().min(other.x2());
        let y2 = self.y2().min(other.y2());

        if x2 > x1 && y2 > y1 {
            Some(BoundingBox::from_corners(x1, y1, x2, y2))
        } else {
            None
        }
    }

    /// Get the union bounding box (smallest box containing both).
    pub fn union_box(&self, other: &BoundingBox) -> BoundingBox {
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = self.x2().max(other.x2());
        let y2 = self.y2().max(other.y2());
        BoundingBox::from_corners(x1, y1, x2, y2)
    }

    /// Clip the bounding box to image boundaries.
    pub fn clip(&self, width: f32, height: f32) -> BoundingBox {
        let x1 = self.x.max(0.0).min(width);
        let y1 = self.y.max(0.0).min(height);
        let x2 = self.x2().max(0.0).min(width);
        let y2 = self.y2().max(0.0).min(height);
        BoundingBox::from_corners(x1, y1, x2, y2)
    }

    /// Scale the bounding box by a factor.
    pub fn scale(&self, factor: f32) -> BoundingBox {
        BoundingBox::new(
            self.x * factor,
            self.y * factor,
            self.width * factor,
            self.height * factor,
        )
    }

    /// Scale with separate x and y factors.
    pub fn scale_xy(&self, factor_x: f32, factor_y: f32) -> BoundingBox {
        BoundingBox::new(
            self.x * factor_x,
            self.y * factor_y,
            self.width * factor_x,
            self.height * factor_y,
        )
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl BoundingBox {
    #[new]
    fn py_new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self::new(x, y, width, height)
    }

    #[staticmethod]
    #[pyo3(name = "from_corners")]
    fn py_from_corners(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self::from_corners(x1, y1, x2, y2)
    }

    #[staticmethod]
    #[pyo3(name = "from_coco")]
    fn py_from_coco(bbox: [f32; 4]) -> Self {
        Self::from_coco(bbox)
    }

    #[staticmethod]
    #[pyo3(name = "from_voc")]
    fn py_from_voc(bbox: [f32; 4]) -> Self {
        Self::from_voc(bbox)
    }

    #[getter]
    fn get_x2(&self) -> f32 {
        self.x2()
    }

    #[getter]
    fn get_y2(&self) -> f32 {
        self.y2()
    }

    #[getter]
    fn get_area(&self) -> f32 {
        self.area()
    }

    #[getter]
    fn get_center(&self) -> (f32, f32) {
        self.center()
    }

    fn __repr__(&self) -> String {
        format!(
            "BoundingBox(x={}, y={}, width={}, height={})",
            self.x, self.y, self.width, self.height
        )
    }
}

// ============================================================================
// Annotation Bounding Box (with shift tracking)
// ============================================================================

/// A bounding box with shift amount for annotation coordinate translation.
///
/// This extends the core `BoundingBox` type with shift tracking, which is
/// essential for SAHI's sliced inference where detections need to be
/// translated from slice coordinates to full image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct AnnotationBoundingBox {
    /// Minimum x coordinate (left edge).
    pub minx: f32,
    /// Minimum y coordinate (top edge).
    pub miny: f32,
    /// Maximum x coordinate (right edge).
    pub maxx: f32,
    /// Maximum y coordinate (bottom edge).
    pub maxy: f32,
    /// X shift amount for coordinate translation.
    pub shift_x: f32,
    /// Y shift amount for coordinate translation.
    pub shift_y: f32,
}

impl AnnotationBoundingBox {
    /// Create a new annotation bounding box from corner coordinates.
    pub fn new(minx: f32, miny: f32, maxx: f32, maxy: f32) -> Self {
        Self {
            minx,
            miny,
            maxx,
            maxy,
            shift_x: 0.0,
            shift_y: 0.0,
        }
    }

    /// Create with shift amount.
    pub fn with_shift(mut self, shift_x: f32, shift_y: f32) -> Self {
        self.shift_x = shift_x;
        self.shift_y = shift_y;
        self
    }

    /// Create from COCO format [x, y, width, height].
    pub fn from_coco(bbox: [f32; 4]) -> Self {
        Self::new(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    }

    /// Create from COCO format with shift.
    pub fn from_coco_with_shift(bbox: [f32; 4], shift: [f32; 2]) -> Self {
        Self::from_coco(bbox).with_shift(shift[0], shift[1])
    }

    /// Create from VOC format [xmin, ymin, xmax, ymax].
    #[inline]
    pub fn from_voc(bbox: [f32; 4]) -> Self {
        Self::new(bbox[0], bbox[1], bbox[2], bbox[3])
    }

    /// Create from core BoundingBox.
    pub fn from_bbox(bbox: &BoundingBox) -> Self {
        Self::new(bbox.x, bbox.y, bbox.x2(), bbox.y2())
    }

    /// Create from core BoundingBox with shift.
    pub fn from_bbox_with_shift(bbox: &BoundingBox, shift: ShiftAmount) -> Self {
        Self::from_bbox(bbox).with_shift(shift.x, shift.y)
    }

    /// Convert to core BoundingBox (without shift).
    pub fn to_bbox(&self) -> BoundingBox {
        BoundingBox::from_corners(self.minx, self.miny, self.maxx, self.maxy)
    }

    /// Get width.
    #[inline]
    pub fn width(&self) -> f32 {
        self.maxx - self.minx
    }

    /// Get height.
    #[inline]
    pub fn height(&self) -> f32 {
        self.maxy - self.miny
    }

    /// Get area.
    #[inline]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Get shift amount as tuple.
    #[inline]
    pub fn shift_amount(&self) -> (f32, f32) {
        (self.shift_x, self.shift_y)
    }

    /// Convert to COCO format [x, y, width, height].
    #[inline]
    pub fn to_coco_bbox(&self) -> [f32; 4] {
        self.to_xywh()
    }

    /// Convert to VOC format [xmin, ymin, xmax, ymax].
    #[inline]
    pub fn to_voc_bbox(&self) -> [f32; 4] {
        self.to_xyxy()
    }

    /// Convert to [x, y, width, height] format.
    #[inline]
    pub fn to_xywh(&self) -> [f32; 4] {
        [self.minx, self.miny, self.width(), self.height()]
    }

    /// Convert to [xmin, ymin, xmax, ymax] format.
    #[inline]
    pub fn to_xyxy(&self) -> [f32; 4] {
        [self.minx, self.miny, self.maxx, self.maxy]
    }

    /// Get shifted bounding box (applies shift to coordinates).
    pub fn get_shifted(&self) -> Self {
        Self {
            minx: self.minx + self.shift_x,
            miny: self.miny + self.shift_y,
            maxx: self.maxx + self.shift_x,
            maxy: self.maxy + self.shift_y,
            shift_x: 0.0,
            shift_y: 0.0,
        }
    }

    /// Get expanded bounding box by a ratio.
    pub fn get_expanded(&self, ratio: f32, max_x: Option<f32>, max_y: Option<f32>) -> Self {
        let w = self.width();
        let h = self.height();
        let x_margin = w * ratio;
        let y_margin = h * ratio;

        let minx = (self.minx - x_margin).max(0.0);
        let miny = (self.miny - y_margin).max(0.0);
        let maxx = if let Some(max) = max_x {
            (self.maxx + x_margin).min(max)
        } else {
            self.maxx + x_margin
        };
        let maxy = if let Some(max) = max_y {
            (self.maxy + y_margin).min(max)
        } else {
            self.maxy + y_margin
        };

        Self {
            minx,
            miny,
            maxx,
            maxy,
            shift_x: self.shift_x,
            shift_y: self.shift_y,
        }
    }

    /// Get the center point.
    #[inline]
    pub fn center(&self) -> (f32, f32) {
        (
            self.minx + self.width() / 2.0,
            self.miny + self.height() / 2.0,
        )
    }

    /// Clip to image boundaries.
    pub fn clip(&self, width: f32, height: f32) -> Self {
        Self {
            minx: self.minx.max(0.0).min(width),
            miny: self.miny.max(0.0).min(height),
            maxx: self.maxx.max(0.0).min(width),
            maxy: self.maxy.max(0.0).min(height),
            shift_x: self.shift_x,
            shift_y: self.shift_y,
        }
    }
}

impl Default for AnnotationBoundingBox {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

impl From<BoundingBox> for AnnotationBoundingBox {
    fn from(bbox: BoundingBox) -> Self {
        Self::from_bbox(&bbox)
    }
}

impl From<AnnotationBoundingBox> for BoundingBox {
    fn from(bbox: AnnotationBoundingBox) -> Self {
        bbox.to_bbox()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl AnnotationBoundingBox {
    #[new]
    #[pyo3(signature = (minx, miny, maxx, maxy, shift_x=0.0, shift_y=0.0))]
    fn py_new(minx: f32, miny: f32, maxx: f32, maxy: f32, shift_x: f32, shift_y: f32) -> Self {
        Self {
            minx,
            miny,
            maxx,
            maxy,
            shift_x,
            shift_y,
        }
    }

    #[getter]
    fn get_width(&self) -> f32 {
        self.width()
    }

    #[getter]
    fn get_height(&self) -> f32 {
        self.height()
    }

    #[getter]
    fn get_area(&self) -> f32 {
        self.area()
    }

    fn __repr__(&self) -> String {
        format!(
            "AnnotationBoundingBox(({}, {}, {}, {}), shift=({}, {}))",
            self.minx, self.miny, self.maxx, self.maxy, self.shift_x, self.shift_y
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Core BoundingBox tests
    #[test]
    fn test_bbox_new() {
        let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 30.0);
        assert_eq!(bbox.height, 40.0);
    }

    #[test]
    fn test_bbox_from_corners() {
        let bbox = BoundingBox::from_corners(10.0, 20.0, 40.0, 60.0);
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 30.0);
        assert_eq!(bbox.height, 40.0);
    }

    #[test]
    fn test_bbox_iou() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(5.0, 5.0, 10.0, 10.0);

        // Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        let iou = a.iou(&b);
        assert!((iou - 25.0 / 175.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_no_overlap() {
        let a = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BoundingBox::new(20.0, 20.0, 10.0, 10.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_bbox_translate() {
        let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        let translated = bbox.translate(5.0, 10.0);
        assert_eq!(translated.x, 15.0);
        assert_eq!(translated.y, 30.0);
    }

    #[test]
    fn test_bbox_formats() {
        let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(bbox.to_coco(), [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(bbox.to_xywh(), [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(bbox.to_voc(), [10.0, 20.0, 40.0, 60.0]);
        assert_eq!(bbox.to_xyxy(), [10.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_bbox_from_formats() {
        let coco = BoundingBox::from_coco([10.0, 20.0, 30.0, 40.0]);
        assert_eq!(coco.x, 10.0);
        assert_eq!(coco.width, 30.0);

        let voc = BoundingBox::from_voc([10.0, 20.0, 40.0, 60.0]);
        assert_eq!(voc.x, 10.0);
        assert_eq!(voc.width, 30.0);
    }

    #[test]
    fn test_bbox_expanded() {
        let bbox = BoundingBox::new(100.0, 100.0, 100.0, 100.0);
        let expanded = bbox.get_expanded(0.1, None, None);

        assert_eq!(expanded.x, 90.0);
        assert_eq!(expanded.y, 90.0);
        assert_eq!(expanded.width, 120.0);
        assert_eq!(expanded.height, 120.0);
    }

    #[test]
    fn test_bbox_center() {
        let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        let (cx, cy) = bbox.center();
        assert_eq!(cx, 25.0);
        assert_eq!(cy, 40.0);
    }

    #[test]
    fn test_bbox_contains() {
        let outer = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let inner = BoundingBox::new(20.0, 20.0, 30.0, 30.0);

        assert!(outer.contains_point(50.0, 50.0));
        assert!(!outer.contains_point(150.0, 50.0));
        assert!(outer.contains_box(&inner));
        assert!(!inner.contains_box(&outer));
    }

    #[test]
    fn test_bbox_intersection() {
        let a = BoundingBox::new(0.0, 0.0, 50.0, 50.0);
        let b = BoundingBox::new(25.0, 25.0, 50.0, 50.0);

        let intersection = a.intersection(&b).unwrap();
        assert_eq!(intersection.x, 25.0);
        assert_eq!(intersection.y, 25.0);
        assert_eq!(intersection.width, 25.0);
        assert_eq!(intersection.height, 25.0);

        let c = BoundingBox::new(100.0, 100.0, 10.0, 10.0);
        assert!(a.intersection(&c).is_none());
    }

    #[test]
    fn test_bbox_ios() {
        let large = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let small = BoundingBox::new(25.0, 25.0, 50.0, 50.0);

        let ios = small.ios(&large);
        assert!((ios - 1.0).abs() < 1e-6);
    }

    // AnnotationBoundingBox tests
    #[test]
    fn test_annotation_bbox_new() {
        let bbox = AnnotationBoundingBox::new(10.0, 20.0, 40.0, 60.0);
        assert_eq!(bbox.minx, 10.0);
        assert_eq!(bbox.miny, 20.0);
        assert_eq!(bbox.maxx, 40.0);
        assert_eq!(bbox.maxy, 60.0);
        assert_eq!(bbox.width(), 30.0);
        assert_eq!(bbox.height(), 40.0);
    }

    #[test]
    fn test_annotation_bbox_shift() {
        let bbox = AnnotationBoundingBox::new(10.0, 20.0, 40.0, 60.0).with_shift(100.0, 200.0);

        let shifted = bbox.get_shifted();
        assert_eq!(shifted.minx, 110.0);
        assert_eq!(shifted.miny, 220.0);
        assert_eq!(shifted.maxx, 140.0);
        assert_eq!(shifted.maxy, 260.0);
        assert_eq!(shifted.shift_x, 0.0);
        assert_eq!(shifted.shift_y, 0.0);
    }

    #[test]
    fn test_annotation_bbox_from_coco() {
        let bbox = AnnotationBoundingBox::from_coco([10.0, 20.0, 30.0, 40.0]);
        assert_eq!(bbox.minx, 10.0);
        assert_eq!(bbox.miny, 20.0);
        assert_eq!(bbox.width(), 30.0);
        assert_eq!(bbox.height(), 40.0);
    }

    #[test]
    fn test_annotation_bbox_formats() {
        let bbox = AnnotationBoundingBox::new(10.0, 20.0, 40.0, 60.0);
        assert_eq!(bbox.to_coco_bbox(), [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(bbox.to_voc_bbox(), [10.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_annotation_bbox_conversion() {
        let core = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
        let ann: AnnotationBoundingBox = core.into();
        assert_eq!(ann.minx, 10.0);
        assert_eq!(ann.width(), 30.0);

        let back: BoundingBox = ann.into();
        assert_eq!(back.x, 10.0);
        assert_eq!(back.width, 30.0);
    }
}
