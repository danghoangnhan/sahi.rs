//! ONNX Runtime support for built-in model inference.
//!
//! This module provides ONNX-based detection models that implement
//! the `DetectionModel` trait for seamless SAHI integration.
//!
//! # Features
//!
//! - `onnx`: Basic ONNX Runtime support (CPU only)
//! - `onnx-cuda`: ONNX Runtime with CUDA execution provider
//! - `onnx-tensorrt`: ONNX Runtime with TensorRT execution provider
//!
//! # Example
//!
//! ```rust,ignore
//! use sahi::onnx::{YOLOv8Detector, ExecutionProvider};
//! use sahi::model::ModelConfig;
//!
//! // Create a YOLOv8 detector
//! let mut detector = YOLOv8Detector::new(
//!     ModelConfig::builder()
//!         .model_path("yolov8n.onnx")
//!         .confidence_threshold(0.5)
//!         .build()
//! );
//!
//! // Load the model
//! detector.load()?;
//!
//! // Run inference (detector implements DetectionModel trait)
//! let detections = detector.predict(&image)?;
//! ```

mod session;
mod processor;
pub mod yolov8;

pub use session::{ExecutionProvider, OnnxSession, OnnxSessionBuilder};
pub use processor::{ImageProcessor, LetterboxInfo, Preprocessor};
pub use yolov8::{YOLOv8Config, YOLOv8Detector};

#[cfg(feature = "python")]
pub use yolov8::PyYOLOv8Detector;
