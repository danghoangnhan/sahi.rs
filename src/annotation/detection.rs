//! Detection types for SAHI.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use super::BoundingBox;

/// A detection result with bounding box, class, and confidence.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
pub struct Detection {
    /// The bounding box.
    pub bbox: BoundingBox,
    /// Class ID or label index.
    pub class_id: u32,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    /// Optional class name.
    pub class_name: Option<String>,
}

impl Detection {
    /// Create a new detection.
    pub fn new(
        bbox: BoundingBox,
        class_id: u32,
        confidence: f32,
        class_name: Option<String>,
    ) -> Self {
        Self {
            bbox,
            class_id,
            confidence,
            class_name,
        }
    }

    /// Create a detection with a class name.
    pub fn with_class_name(mut self, name: impl Into<String>) -> Self {
        self.class_name = Some(name.into());
        self
    }

    /// Translate the detection by an offset (used when mapping slice coords to image coords).
    pub fn translate(&self, dx: f32, dy: f32) -> Self {
        Self {
            bbox: self.bbox.translate(dx, dy),
            class_id: self.class_id,
            confidence: self.confidence,
            class_name: self.class_name.clone(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Detection {
    #[new]
    #[pyo3(signature = (bbox, class_id, confidence, class_name=None))]
    fn py_new(
        bbox: BoundingBox,
        class_id: u32,
        confidence: f32,
        class_name: Option<String>,
    ) -> Self {
        Self::new(bbox, class_id, confidence, class_name)
    }

    fn __repr__(&self) -> String {
        format!(
            "Detection(bbox=BoundingBox({}, {}, {}, {}), class_id={}, confidence={:.3}, class_name={:?})",
            self.bbox.x, self.bbox.y, self.bbox.width, self.bbox.height,
            self.class_id, self.confidence, self.class_name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_translate() {
        let det = Detection::new(BoundingBox::new(10.0, 10.0, 50.0, 50.0), 0, 0.9, None);
        let translated = det.translate(100.0, 200.0);

        assert_eq!(translated.bbox.x, 110.0);
        assert_eq!(translated.bbox.y, 210.0);
    }

    #[test]
    fn test_detection_with_class_name() {
        let det = Detection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0), 0, 0.9, None)
            .with_class_name("person");

        assert_eq!(det.class_name.as_deref(), Some("person"));
    }
}
