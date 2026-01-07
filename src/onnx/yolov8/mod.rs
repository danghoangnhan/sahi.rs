//! YOLOv8 detection model implementation.
//!
//! Provides a complete YOLOv8 detector that implements the `DetectionModel` trait
//! for seamless integration with SAHI.
//!
//! # Supported Variants
//!
//! - YOLOv8n (nano)
//! - YOLOv8s (small)
//! - YOLOv8m (medium)
//! - YOLOv8l (large)
//! - YOLOv8x (extra large)
//!
//! # Output Format
//!
//! The detector auto-detects the output format:
//! - Standard: (1, 84, 8400) - [x_center, y_center, w, h, class_scores...]
//! - Transposed: (1, 8400, 84) - same data, different layout
//!
//! # Example
//!
//! ```rust,ignore
//! use sahi::onnx::YOLOv8Detector;
//! use sahi::model::DetectionModel;
//!
//! // Create and load detector
//! let mut detector = YOLOv8Detector::new("yolov8n.onnx");
//! detector.load()?;
//!
//! // Run inference
//! let detections = detector.predict(&image)?;
//!
//! for det in detections {
//!     println!("Class {}: {:.2}% at {:?}",
//!         det.class_id, det.confidence * 100.0, det.bbox);
//! }
//! ```

mod model;
mod processor;

pub use model::{YOLOv8Config, YOLOv8Detector};
pub use processor::{YOLOv8OutputFormat, YOLOv8Processor};

// ============================================================================
// Python Bindings
// ============================================================================

#[cfg(feature = "python")]
mod python {
    use numpy::{PyArray3, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::prelude::*;

    use crate::detection::Detection;
    use crate::inference::ImageData;
    use crate::model::DetectionModel;
    use crate::onnx::session::ExecutionProvider;

    use super::{YOLOv8Config, YOLOv8Detector};

    /// Python wrapper for YOLOv8Detector.
    #[pyclass(name = "YOLOv8Detector")]
    pub struct PyYOLOv8Detector {
        inner: YOLOv8Detector,
    }

    #[pymethods]
    impl PyYOLOv8Detector {
        /// Create a new YOLOv8 detector.
        ///
        /// Args:
        ///     model_path: Path to the ONNX model file
        ///     num_classes: Number of classes (default: 80 for COCO)
        ///     input_size: Model input size (default: 640)
        ///     confidence_threshold: Minimum confidence (default: 0.25)
        ///     iou_threshold: NMS IoU threshold (default: 0.45)
        ///     device: Device to use - "cpu", "cuda", or "cuda:N" (default: "cpu")
        #[new]
        #[pyo3(signature = (
            model_path,
            num_classes=80,
            input_size=640,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device="cpu"
        ))]
        fn new(
            model_path: &str,
            num_classes: u32,
            input_size: u32,
            confidence_threshold: f32,
            iou_threshold: f32,
            device: &str,
        ) -> PyResult<Self> {
            let ep = parse_device(device)?;

            let config = YOLOv8Config::new(model_path)
                .with_num_classes(num_classes)
                .with_input_size(input_size)
                .with_confidence_threshold(confidence_threshold)
                .with_iou_threshold(iou_threshold)
                .with_execution_provider(ep);

            Ok(Self {
                inner: YOLOv8Detector::from_config(config),
            })
        }

        /// Load the model.
        fn load(&mut self) -> PyResult<()> {
            self.inner
                .load()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        /// Unload the model.
        fn unload(&mut self) {
            self.inner.unload();
        }

        /// Check if the model is loaded.
        fn is_loaded(&self) -> bool {
            self.inner.is_loaded()
        }

        /// Run inference on an image.
        ///
        /// Args:
        ///     image: numpy array (H, W, C) uint8 RGB
        ///
        /// Returns:
        ///     List of Detection objects
        fn predict(
            &self,
            _py: Python<'_>,
            image: &Bound<'_, PyArray3<u8>>,
        ) -> PyResult<Vec<Detection>> {
            // Convert numpy array to ImageData
            let shape = image.shape();
            let height = shape[0] as u32;
            let width = shape[1] as u32;
            let channels = shape[2] as u32;

            let array = unsafe { image.as_array() };
            let data: Vec<u8> = array.iter().copied().collect();
            let image_data = ImageData::new(data, width, height, channels);

            self.inner
                .predict(&image_data)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        /// Set the confidence threshold.
        fn set_confidence_threshold(&mut self, threshold: f32) {
            self.inner.set_confidence_threshold(threshold);
        }

        /// Set the IoU threshold.
        fn set_iou_threshold(&mut self, threshold: f32) {
            self.inner.set_iou_threshold(threshold);
        }

        /// Get current confidence threshold.
        #[getter]
        fn confidence_threshold(&self) -> f32 {
            self.inner.yolo_config().confidence_threshold
        }

        /// Get current IoU threshold.
        #[getter]
        fn iou_threshold(&self) -> f32 {
            self.inner.yolo_config().iou_threshold
        }

        /// Get number of classes.
        #[getter]
        fn num_classes(&self) -> u32 {
            self.inner.yolo_config().num_classes
        }

        /// Get input size.
        #[getter]
        fn input_size(&self) -> u32 {
            self.inner.yolo_config().input_size
        }

        fn __repr__(&self) -> String {
            format!(
                "YOLOv8Detector(model='{}', num_classes={}, input_size={}, loaded={})",
                self.inner.yolo_config().model_path.display(),
                self.inner.yolo_config().num_classes,
                self.inner.yolo_config().input_size,
                self.inner.is_loaded()
            )
        }
    }

    /// Parse device string to ExecutionProvider.
    fn parse_device(device: &str) -> PyResult<ExecutionProvider> {
        let device = device.to_lowercase();
        match device.as_str() {
            "cpu" => Ok(ExecutionProvider::Cpu),
            #[cfg(feature = "onnx-cuda")]
            "cuda" => Ok(ExecutionProvider::cuda()),
            #[cfg(feature = "onnx-cuda")]
            _ if device.starts_with("cuda:") => {
                let idx: i32 = device[5..].parse().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid CUDA device index: {}",
                        device
                    ))
                })?;
                Ok(ExecutionProvider::cuda_with_device(idx))
            }
            #[cfg(not(feature = "onnx-cuda"))]
            _ if device == "cuda" || device.starts_with("cuda:") => {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "CUDA support not enabled. Rebuild with --features onnx-cuda",
                ))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown device: {}",
                device
            ))),
        }
    }
}

#[cfg(feature = "python")]
pub use python::PyYOLOv8Detector;
