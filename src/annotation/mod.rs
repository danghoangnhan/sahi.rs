//! Annotation types for object detection and instance segmentation.
//!
//! This module provides types for representing detection annotations including
//! bounding boxes, segmentation masks, and full object annotations.
//!
//! # Example
//!
//! ```rust,ignore
//! use sahi::annotation::{ObjectAnnotation, Mask, BoundingBox};
//!
//! // Create annotation from COCO format
//! let ann = ObjectAnnotation::from_coco_bbox(
//!     [100.0, 100.0, 50.0, 50.0],  // x, y, w, h
//!     0,
//!     Some("person"),
//!     [640, 480],
//! );
//!
//! // Shift to full image coordinates
//! let shifted = ann.get_shifted();
//! ```

mod bbox;
mod detection;
mod mask;
mod object;

pub use bbox::{AnnotationBoundingBox, BoundingBox};
pub use detection::Detection;
pub use mask::{Mask, MaskFormat, Polygon, RleData};
pub use object::ObjectAnnotation;

// ============================================================================
// Shift Amount (re-export from model for convenience)
// ============================================================================

/// Re-export ShiftAmount for annotation coordinate translation.
pub use crate::model::ShiftAmount;

// ============================================================================
// Full Shape
// ============================================================================

/// Represents the full image dimensions [height, width].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
pub struct FullShape {
    /// Image height in pixels.
    pub height: u32,
    /// Image width in pixels.
    pub width: u32,
}

impl FullShape {
    /// Create a new full shape.
    #[inline]
    pub const fn new(height: u32, width: u32) -> Self {
        Self { height, width }
    }

    /// Create from [height, width] array.
    #[inline]
    pub const fn from_hw(hw: [u32; 2]) -> Self {
        Self {
            height: hw[0],
            width: hw[1],
        }
    }

    /// Convert to [height, width] array.
    #[inline]
    pub const fn to_hw(&self) -> [u32; 2] {
        [self.height, self.width]
    }

    /// Get as (width, height) tuple.
    #[inline]
    pub const fn to_wh(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl From<[u32; 2]> for FullShape {
    fn from(hw: [u32; 2]) -> Self {
        Self::from_hw(hw)
    }
}

impl From<(u32, u32)> for FullShape {
    fn from((height, width): (u32, u32)) -> Self {
        Self { height, width }
    }
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl FullShape {
    #[new]
    fn py_new(height: u32, width: u32) -> Self {
        Self::new(height, width)
    }

    fn __repr__(&self) -> String {
        format!("FullShape(height={}, width={})", self.height, self.width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_shape() {
        let shape = FullShape::new(480, 640);
        assert_eq!(shape.to_hw(), [480, 640]);
        assert_eq!(shape.to_wh(), (640, 480));

        let from_arr: FullShape = [480, 640].into();
        assert_eq!(from_arr, shape);
    }
}
