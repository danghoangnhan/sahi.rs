//! # SAHI - Slicing Aided Hyper Inference
//!
//! A Rust implementation of [SAHI](https://github.com/obss/sahi) for improved
//! small object detection through image slicing.
//!
//! ## Overview
//!
//! SAHI works by:
//! 1. Slicing a large image into overlapping tiles
//! 2. Running object detection on each tile
//! 3. Stitching results back together using NMS to remove duplicates
//!
//! ## Features
//!
//! - **CPU/GPU backends**: Use `--features cuda` for GPU acceleration
//! - **Model-agnostic**: Bring your own detection model via callbacks
//! - **Configurable**: Tune slice size, overlap, NMS thresholds
//!
//! ## Example
//!
//! ```rust,ignore
//! use sahi::{Sahi, SlicerConfig, callback};
//!
//! // Create SAHI with custom slice config
//! let sahi = Sahi::builder()
//!     .slice_size(640, 640)
//!     .overlap(0.2, 0.2)
//!     .nms_threshold(0.5)
//!     .build();
//!
//! // Define your inference callback
//! let model = callback(|image| {
//!     // Run your model here
//!     Ok(vec![])
//! });
//!
//! // Run sliced inference
//! let detections = sahi.predict(&image_data, &model)?;
//! ```

pub mod annotation;
pub mod backend;
pub mod detection;
pub mod error;
pub mod inference;
pub mod model;
pub mod postprocess;
pub mod segmentation;
pub mod slicer;

// ONNX Runtime support (feature-gated)
#[cfg(feature = "onnx")]
pub mod onnx;

pub use backend::{Backend, BoxedBackend, CpuBackend, CpuBackendConfig};
#[cfg(feature = "cuda")]
pub use backend::{CudaBackend, CudaBackendConfig};

// ONNX exports
#[cfg(feature = "onnx")]
pub use onnx::{ExecutionProvider, OnnxSession, OnnxSessionBuilder, YOLOv8Config, YOLOv8Detector};

// Core detection types
pub use detection::{BoundingBox, Detection};
pub use error::{Error, Result};
pub use inference::{callback, ImageData, InferenceCallback};

// Model types
pub use model::{
    Category, CategoryMapping, CategoryRemapper, DetectionModel, Device, ModelCallback,
    ModelConfig, ModelConfigBuilder, PredictionBuffer, ShiftAmount,
};

// Annotation types
pub use annotation::{
    AnnotationBoundingBox, FullShape, Mask, MaskFormat, ObjectAnnotation, Polygon, RleData,
};

pub use postprocess::{MatchMetric, PostprocessConfig, PostprocessType, Postprocessor};
pub use segmentation::{seg_callback, MaskedDetection, SegmentationCallback};
pub use slicer::{Slice, Slicer, SlicerConfig};

/// High-level SAHI interface.
///
/// Combines slicing, inference, and postprocessing into a single API.
#[derive(Debug)]
pub struct Sahi {
    slicer: Slicer,
    postprocessor: Postprocessor,
    backend: BoxedBackend,
    /// Whether to also run inference on the full image
    pub include_full_image: bool,
}

impl Sahi {
    /// Create a new SAHI instance with default configuration.
    pub fn new() -> Self {
        Self {
            slicer: Slicer::with_defaults(),
            postprocessor: Postprocessor::with_defaults(),
            backend: backend::default_backend(),
            include_full_image: false,
        }
    }

    /// Create a builder for custom configuration.
    pub fn builder() -> SahiBuilder {
        SahiBuilder::new()
    }

    /// Run sliced inference on an image.
    ///
    /// Returns all detections after NMS, with coordinates in image space.
    pub fn predict(
        &self,
        image: &ImageData,
        callback: &dyn InferenceCallback,
    ) -> Result<Vec<Detection>> {
        // Generate slices
        let slices = self.slicer.slice(image.width, image.height);

        // Process slices using the backend
        let mut slice_detections = self.backend.process_slices(image, &slices, callback)?;

        // Optionally include full image inference
        if self.include_full_image {
            let full_slice = Slice::new(0, 0, image.width, image.height, slices.len());
            let full_detections = callback.infer(image)?;
            slice_detections.push((full_slice, full_detections));
        }

        // Postprocess results
        Ok(self.postprocessor.stitch(&slice_detections))
    }

    /// Run sliced instance-segmentation inference on an image.
    ///
    /// Mirrors [`predict`](Self::predict) for masks: slices the image, runs the
    /// segmentation callback per slice, rebases each detection and mask into image
    /// coordinates, and stitches with NMS (each survivor keeps its own mask).
    pub fn predict_instances(
        &self,
        image: &ImageData,
        callback: &dyn SegmentationCallback,
    ) -> Result<Vec<MaskedDetection>> {
        let slices = self.slicer.slice(image.width, image.height);
        let mut all: Vec<MaskedDetection> = Vec::new();

        for slice in &slices {
            let slice_img = image.extract_slice(slice.x, slice.y, slice.width, slice.height);
            for mut pred in callback.infer(&slice_img)? {
                pred.detection = pred.detection.translate(slice.x as f32, slice.y as f32);
                if let Some(m) = pred.mask.take() {
                    pred.mask = Some(segmentation::rebase_mask(
                        &m,
                        slice.x as f32,
                        slice.y as f32,
                        image.width,
                        image.height,
                    ));
                }
                all.push(pred);
            }
        }

        // Optionally include full-image inference (already in image coordinates).
        if self.include_full_image {
            all.extend(callback.infer(image)?);
        }

        Ok(segmentation::seg_stitch(all, self.postprocessor.config()))
    }

    /// Get the current slicer configuration.
    pub fn slicer(&self) -> &Slicer {
        &self.slicer
    }

    /// Get the current postprocessor configuration.
    pub fn postprocessor(&self) -> &Postprocessor {
        &self.postprocessor
    }

    /// Get the backend name.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// Check if the backend is available.
    pub fn backend_available(&self) -> bool {
        self.backend.is_available()
    }
}

impl Default for Sahi {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring SAHI.
#[derive(Debug)]
pub struct SahiBuilder {
    slicer_config: SlicerConfig,
    postprocess_config: PostprocessConfig,
    backend: Option<BoxedBackend>,
    include_full_image: bool,
    /// Number of CPU threads (0 = auto). Only effective with `parallel` feature.
    num_threads: usize,
    /// Enable parallel inference calls (not safe with Python/GIL callbacks).
    parallel_inference: bool,
}

impl SahiBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            slicer_config: SlicerConfig::default(),
            postprocess_config: PostprocessConfig::default(),
            backend: None,
            include_full_image: false,
            num_threads: 0,
            parallel_inference: false,
        }
    }

    /// Set the slice dimensions.
    pub fn slice_size(mut self, width: u32, height: u32) -> Self {
        self.slicer_config.slice_width = width;
        self.slicer_config.slice_height = height;
        self
    }

    /// Set the overlap ratios.
    pub fn overlap(mut self, width_ratio: f32, height_ratio: f32) -> Self {
        self.slicer_config.overlap_width_ratio = width_ratio.clamp(0.0, 0.9);
        self.slicer_config.overlap_height_ratio = height_ratio.clamp(0.0, 0.9);
        self
    }

    /// Set the match threshold for postprocessing.
    ///
    /// Predictions with match score (IOU or IOS) above this threshold
    /// will be merged or suppressed depending on the postprocess type.
    pub fn match_threshold(mut self, threshold: f32) -> Self {
        self.postprocess_config.match_threshold = threshold;
        self
    }

    /// Set the confidence threshold.
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.postprocess_config.confidence_threshold = threshold;
        self
    }

    /// Enable class-aware postprocessing.
    pub fn class_aware(mut self, enabled: bool) -> Self {
        self.postprocess_config.class_aware = enabled;
        self
    }

    /// Set the postprocessing type.
    ///
    /// - `NMS`: Non-Maximum Suppression (removes overlapping boxes)
    /// - `NMM`: Non-Maximum Merging (merges overlapping boxes)
    /// - `GREEDYNMM`: Greedy Non-Maximum Merging (default, like original SAHI)
    pub fn postprocess_type(mut self, postprocess_type: PostprocessType) -> Self {
        self.postprocess_config.postprocess_type = postprocess_type;
        self
    }

    /// Set the match metric for comparing predictions.
    ///
    /// - `IOU`: Intersection over Union
    /// - `IOS`: Intersection over Smaller (better for nested boxes)
    pub fn match_metric(mut self, match_metric: MatchMetric) -> Self {
        self.postprocess_config.match_metric = match_metric;
        self
    }

    /// Also run inference on the full image.
    pub fn include_full_image(mut self, enabled: bool) -> Self {
        self.include_full_image = enabled;
        self
    }

    /// Set the number of CPU threads for parallel slice extraction.
    ///
    /// Only effective when the `parallel` feature is enabled.
    /// A value of 0 (default) uses all available cores.
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Enable parallel inference calls across slices.
    ///
    /// **Warning:** Not safe with Python/GIL-bound callbacks.
    /// Only enable for pure-Rust callbacks. Requires the `parallel` feature.
    pub fn parallel_inference(mut self, enabled: bool) -> Self {
        self.parallel_inference = enabled;
        self
    }

    /// Use a specific backend.
    pub fn backend(mut self, backend: BoxedBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Force CPU backend with the builder's parallelism settings.
    pub fn cpu(mut self) -> Self {
        let config = CpuBackendConfig {
            num_threads: self.num_threads,
            parallel_inference: self.parallel_inference,
        };
        self.backend = Some(Box::new(CpuBackend::with_config(config)));
        self
    }

    /// Force CUDA backend (requires `cuda` feature).
    #[cfg(feature = "cuda")]
    pub fn cuda(mut self) -> Self {
        self.backend = Some(Box::new(CudaBackend::new()));
        self
    }

    /// Build the SAHI instance.
    pub fn build(self) -> Sahi {
        let backend = self.backend.unwrap_or_else(|| {
            // Apply parallelism settings to the default CPU backend
            let config = CpuBackendConfig {
                num_threads: self.num_threads,
                parallel_inference: self.parallel_inference,
            };
            #[cfg(feature = "cuda")]
            {
                let cuda = CudaBackend::new();
                if cuda.is_available() {
                    return Box::new(cuda);
                }
            }
            Box::new(CpuBackend::with_config(config))
        });

        Sahi {
            slicer: Slicer::new(self.slicer_config),
            postprocessor: Postprocessor::new(self.postprocess_config),
            backend,
            include_full_image: self.include_full_image,
        }
    }
}

impl Default for SahiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_callback() -> impl InferenceCallback {
        callback(|_img: &ImageData| {
            Ok(vec![Detection::new(
                BoundingBox::new(10.0, 10.0, 50.0, 50.0),
                0,
                0.9,
                None,
            )])
        })
    }

    #[test]
    fn test_sahi_default() {
        let sahi = Sahi::new();
        assert_eq!(sahi.backend_name(), "cpu");
    }

    #[test]
    fn test_sahi_builder() {
        let sahi = Sahi::builder()
            .slice_size(512, 512)
            .overlap(0.3, 0.3)
            .match_threshold(0.45)
            .confidence_threshold(0.3)
            .build();

        assert_eq!(sahi.slicer().config().slice_width, 512);
        assert_eq!(sahi.slicer().config().slice_height, 512);
    }

    #[test]
    fn test_sahi_predict() {
        let sahi = Sahi::builder().slice_size(50, 50).overlap(0.0, 0.0).build();

        // 100x100 image = 4 slices with 50x50 slice size
        let image = ImageData::from_rgb(vec![0; 30000], 100, 100);
        let cb = dummy_callback();

        let detections: Vec<Detection> = sahi.predict(&image, &cb).unwrap();

        // Each slice returns 1 detection, but after NMS some may be merged
        assert!(!detections.is_empty());
    }

    #[test]
    fn test_include_full_image() {
        let sahi = Sahi::builder()
            .slice_size(100, 100)
            .include_full_image(true)
            .build();

        // Small image = 1 slice + 1 full image = potentially 2 detections before NMS
        let image: ImageData = ImageData::from_rgb(vec![0; 300], 10, 10);
        let cb = dummy_callback();

        let detections: Vec<Detection> = sahi.predict(&image, &cb).unwrap();
        assert!(!detections.is_empty());
    }
}

// ============================================================================
// Python bindings
// ============================================================================

#[cfg(feature = "python")]
mod python {
    use super::*;
    use numpy::{PyArray3, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    /// Python wrapper for SAHI.
    #[pyclass(name = "Sahi")]
    pub struct PySahi {
        inner: Sahi,
    }

    #[pymethods]
    impl PySahi {
        /// Create a new SAHI instance with configuration.
        ///
        /// Args:
        ///     slice_width: Width of each slice (default: 640)
        ///     slice_height: Height of each slice (default: 640)
        ///     overlap_width: Overlap ratio in width direction (default: 0.2)
        ///     overlap_height: Overlap ratio in height direction (default: 0.2)
        ///     postprocess_type: Type of postprocessing - PostprocessType.NMS, .NMM, or .GREEDYNMM (default: GREEDYNMM)
        ///     postprocess_match_metric: Match metric - MatchMetric.IOU or .IOS (default: IOU)
        ///     postprocess_match_threshold: Threshold for matching predictions (default: 0.5)
        ///     confidence_threshold: Minimum confidence threshold (default: 0.25)
        ///     class_aware: Whether to do class-aware postprocessing (default: true)
        ///     include_full_image: Also run inference on full image (default: false)
        #[new]
        #[pyo3(signature = (
            slice_width=640,
            slice_height=640,
            overlap_width=0.2,
            overlap_height=0.2,
            postprocess_type=PostprocessType::GREEDYNMM,
            postprocess_match_metric=MatchMetric::IOU,
            postprocess_match_threshold=0.5,
            confidence_threshold=0.25,
            class_aware=true,
            include_full_image=false
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            slice_width: u32,
            slice_height: u32,
            overlap_width: f32,
            overlap_height: f32,
            postprocess_type: PostprocessType,
            postprocess_match_metric: MatchMetric,
            postprocess_match_threshold: f32,
            confidence_threshold: f32,
            class_aware: bool,
            include_full_image: bool,
        ) -> Self {
            let inner = Sahi::builder()
                .slice_size(slice_width, slice_height)
                .overlap(overlap_width, overlap_height)
                .postprocess_type(postprocess_type)
                .match_metric(postprocess_match_metric)
                .match_threshold(postprocess_match_threshold)
                .confidence_threshold(confidence_threshold)
                .class_aware(class_aware)
                .include_full_image(include_full_image)
                .build();

            Self { inner }
        }

        /// Run sliced inference on an image.
        ///
        /// Args:
        ///     image: numpy array (H, W, C) uint8 RGB
        ///     callback: callable that takes numpy array and returns list of Detection
        ///
        /// Returns:
        ///     List of Detection objects with coordinates in original image space
        fn predict(
            &self,
            _py: Python<'_>,
            image: &Bound<'_, PyArray3<u8>>,
            callback: PyObject,
        ) -> PyResult<Vec<Detection>> {
            // Convert numpy array to ImageData
            let shape = image.shape();
            let height = shape[0] as u32;
            let width = shape[1] as u32;
            let channels = shape[2] as u32;

            // Safety: We only read the array data within this scope
            let array = unsafe { image.as_array() };
            let data: Vec<u8> = array.iter().copied().collect();
            let image_data = ImageData::new(data, width, height, channels);

            // Create a Python callback adapter
            let py_callback = PyCallback { callback };

            // Run inference
            self.inner
                .predict(&image_data, &py_callback)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        /// Run sliced instance-segmentation inference on an image.
        ///
        /// Args:
        ///     image: numpy array (H, W, C) uint8 RGB
        ///     callback: callable taking a numpy array and returning a list of MaskedDetection
        ///
        /// Returns:
        ///     List of MaskedDetection objects with coordinates in original image space.
        fn predict_instances(
            &self,
            _py: Python<'_>,
            image: &Bound<'_, PyArray3<u8>>,
            callback: PyObject,
        ) -> PyResult<Vec<crate::segmentation::PyMaskedDetection>> {
            crate::segmentation::python::run_predict_instances(&self.inner, image, callback)
        }

        /// Get the backend name.
        fn backend_name(&self) -> &'static str {
            self.inner.backend_name()
        }

        /// Get slice configuration as a dict.
        fn get_config(&self, py: Python<'_>) -> PyResult<PyObject> {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            let config = self.inner.slicer().config();
            dict.set_item("slice_width", config.slice_width)?;
            dict.set_item("slice_height", config.slice_height)?;
            dict.set_item("overlap_width_ratio", config.overlap_width_ratio)?;
            dict.set_item("overlap_height_ratio", config.overlap_height_ratio)?;
            Ok(dict.into())
        }

        fn __repr__(&self) -> String {
            let config = self.inner.slicer().config();
            format!(
                "Sahi(slice={}x{}, overlap=({:.1}, {:.1}), backend='{}')",
                config.slice_width,
                config.slice_height,
                config.overlap_width_ratio,
                config.overlap_height_ratio,
                self.inner.backend_name()
            )
        }
    }

    /// Adapter to call Python functions as InferenceCallback.
    struct PyCallback {
        callback: PyObject,
    }

    // Safety: PyObject is GIL-bound, and we only use it within Python::with_gil
    unsafe impl Send for PyCallback {}
    unsafe impl Sync for PyCallback {}

    impl InferenceCallback for PyCallback {
        fn infer(&self, image: &ImageData) -> Result<Vec<Detection>> {
            Python::with_gil(|py| {
                // Convert ImageData to numpy array
                let array = numpy::PyArray::from_slice(py, &image.data);
                let reshaped = array
                    .reshape([
                        image.height as usize,
                        image.width as usize,
                        image.channels as usize,
                    ])
                    .map_err(|e| Error::Inference(e.to_string()))?;

                // Call the Python callback
                let result = self
                    .callback
                    .call1(py, (reshaped,))
                    .map_err(|e| Error::Inference(e.to_string()))?;

                // Convert result to Vec<Detection>
                let list = result
                    .downcast_bound::<PyList>(py)
                    .map_err(|e| Error::Inference(format!("Expected list of Detection: {}", e)))?;

                let mut detections = Vec::with_capacity(list.len());
                for item in list.iter() {
                    let det: Detection = item
                        .extract()
                        .map_err(|e| Error::Inference(format!("Invalid Detection: {}", e)))?;
                    detections.push(det);
                }

                Ok(detections)
            })
        }
    }

    /// Python module initialization.
    #[pymodule]
    #[pyo3(name = "sahi_rs")]
    pub fn sahi_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PySahi>()?;
        m.add_class::<BoundingBox>()?;
        m.add_class::<Detection>()?;
        m.add_class::<ImageData>()?;
        m.add_class::<PostprocessType>()?;
        m.add_class::<MatchMetric>()?;
        m.add_class::<model::Category>()?;
        m.add_class::<crate::segmentation::PyMaskedDetection>()?;

        // ONNX types (feature-gated)
        #[cfg(feature = "onnx")]
        m.add_class::<onnx::PyYOLOv8Detector>()?;

        Ok(())
    }
}
