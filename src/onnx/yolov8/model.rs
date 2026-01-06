//! YOLOv8 detector implementation.
//!
//! Provides a complete YOLOv8 detector that implements the `DetectionModel` trait.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::detection::Detection;
use crate::error::{Error, Result};
use crate::inference::ImageData;
use crate::model::{DetectionModel, ModelConfig};
use crate::onnx::processor::ImageProcessor;
use crate::onnx::session::{ExecutionProvider, OnnxSession, OnnxSessionBuilder};

use super::processor::{YOLOv8OutputFormat, YOLOv8Processor};

/// Configuration for YOLOv8 detector.
#[derive(Debug, Clone)]
pub struct YOLOv8Config {
    /// Model path (ONNX file).
    pub model_path: PathBuf,
    /// Number of classes.
    pub num_classes: u32,
    /// Input image size (assumes square input).
    pub input_size: u32,
    /// Confidence threshold.
    pub confidence_threshold: f32,
    /// IoU threshold for NMS.
    pub iou_threshold: f32,
    /// Execution provider.
    pub execution_provider: ExecutionProvider,
}

impl Default for YOLOv8Config {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            num_classes: 80,
            input_size: 640,
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            execution_provider: ExecutionProvider::Cpu,
        }
    }
}

impl YOLOv8Config {
    /// Create a new config with the specified model path.
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

    /// Set the execution provider.
    pub fn with_execution_provider(mut self, ep: ExecutionProvider) -> Self {
        self.execution_provider = ep;
        self
    }

    /// Convert to ModelConfig.
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig::builder()
            .model_path(self.model_path.display().to_string())
            .confidence_threshold(self.confidence_threshold)
            .image_size(self.input_size)
            .build()
    }
}

/// YOLOv8 object detector.
///
/// Implements the `DetectionModel` trait for integration with SAHI.
///
/// # Example
///
/// ```rust,ignore
/// use sahi::onnx::YOLOv8Detector;
/// use sahi::onnx::yolov8::YOLOv8Config;
///
/// let mut detector = YOLOv8Detector::from_config(
///     YOLOv8Config::new("yolov8n.onnx")
///         .with_confidence_threshold(0.5)
/// );
///
/// detector.load()?;
/// let detections = detector.predict(&image)?;
/// ```
pub struct YOLOv8Detector {
    /// Model configuration.
    config: ModelConfig,
    /// YOLOv8-specific configuration.
    yolo_config: YOLOv8Config,
    /// ONNX session (None until loaded, wrapped in Mutex for interior mutability).
    session: Option<Mutex<OnnxSession>>,
    /// YOLOv8 processor for pre/postprocessing.
    processor: YOLOv8Processor,
    /// Detected output format.
    output_format: YOLOv8OutputFormat,
}

impl std::fmt::Debug for YOLOv8Detector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("YOLOv8Detector")
            .field("config", &self.config)
            .field("yolo_config", &self.yolo_config)
            .field("is_loaded", &self.session.is_some())
            .field("output_format", &self.output_format)
            .finish()
    }
}

impl YOLOv8Detector {
    /// Create a new detector with the given config.
    pub fn from_config(yolo_config: YOLOv8Config) -> Self {
        let config = yolo_config.to_model_config();
        let processor = YOLOv8Processor::new(yolo_config.input_size, yolo_config.num_classes)
            .with_confidence_threshold(yolo_config.confidence_threshold)
            .with_iou_threshold(yolo_config.iou_threshold);

        Self {
            config,
            yolo_config,
            session: None,
            processor,
            output_format: YOLOv8OutputFormat::Standard,
        }
    }

    /// Create a new detector with just a model path.
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self::from_config(YOLOv8Config::new(model_path))
    }

    /// Create a detector with model path and execution provider.
    pub fn with_provider(model_path: impl Into<PathBuf>, ep: ExecutionProvider) -> Self {
        Self::from_config(YOLOv8Config::new(model_path).with_execution_provider(ep))
    }

    /// Get the YOLOv8 configuration.
    pub fn yolo_config(&self) -> &YOLOv8Config {
        &self.yolo_config
    }

    /// Get the processor.
    pub fn processor(&self) -> &YOLOv8Processor {
        &self.processor
    }

    /// Update the confidence threshold.
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.yolo_config.confidence_threshold = threshold;
        self.processor.confidence_threshold = threshold;
    }

    /// Update the IoU threshold.
    pub fn set_iou_threshold(&mut self, threshold: f32) {
        self.yolo_config.iou_threshold = threshold;
        self.processor.iou_threshold = threshold;
    }
}

impl DetectionModel for YOLOv8Detector {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn load(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        // Build session
        let session = OnnxSessionBuilder::new()
            .execution_provider(self.yolo_config.execution_provider)
            .build_from_file(&self.yolo_config.model_path)?;

        self.session = Some(Mutex::new(session));
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    fn unload(&mut self) {
        self.session = None;
    }

    fn predict(&self, image: &ImageData) -> Result<Vec<Detection>> {
        // Lock the session for mutable access
        let session_mutex = self.session.as_ref().ok_or(Error::ModelNotLoaded)?;

        let mut session = session_mutex
            .lock()
            .map_err(|_| Error::inference("Session lock poisoned"))?;

        // Preprocess image
        let (input_tensor, letterbox_info) = self.processor.preprocessor.preprocess(image)?;

        // Get input name (from immutable access)
        let input_name = session
            .input_name()
            .ok_or_else(|| Error::invalid_output("Model has no input".to_string()))?
            .to_string();

        let output_name = session
            .output_name()
            .ok_or_else(|| Error::invalid_output("Model has no output".to_string()))?
            .to_string();

        // Create ONNX input - need owned array
        let input_array = input_tensor.into_dyn();

        // Create input value
        let ort_input = ort::value::Tensor::from_array(input_array)?;

        // Run inference using the mutable session
        let outputs = session
            .session_mut()
            .run(ort::inputs![&input_name => ort_input])?;

        // Get output
        let output_value = outputs
            .get(&output_name)
            .ok_or_else(|| Error::invalid_output(format!("Output '{}' not found", output_name)))?;

        // Extract tensor data - try_extract_tensor returns (Shape, &[T])
        let (output_shape, output_data) = output_value.try_extract_tensor::<f32>()?;
        let output_shape_i64: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();

        // Process output
        self.processor
            .process_output(output_data, &output_shape_i64, &letterbox_info)
    }

    fn predict_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<Detection>>> {
        // For now, process sequentially
        // TODO: Implement true batch processing
        images.iter().map(|img| self.predict(img)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = YOLOv8Config::default();
        assert_eq!(config.num_classes, 80);
        assert_eq!(config.input_size, 640);
        assert_eq!(config.confidence_threshold, 0.25);
    }

    #[test]
    fn test_config_builder() {
        let config = YOLOv8Config::new("model.onnx")
            .with_num_classes(91)
            .with_input_size(320)
            .with_confidence_threshold(0.5);

        assert_eq!(config.num_classes, 91);
        assert_eq!(config.input_size, 320);
        assert_eq!(config.confidence_threshold, 0.5);
    }

    #[test]
    fn test_detector_not_loaded() {
        let detector = YOLOv8Detector::new("model.onnx");
        assert!(!detector.is_loaded());
    }
}
