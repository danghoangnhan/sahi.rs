//! Error types for SAHI.

use thiserror::Error;

/// Result type for SAHI operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for SAHI operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Inference error from the model callback.
    #[error("inference error: {0}")]
    Inference(String),

    /// Invalid image dimensions.
    #[error("invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// GPU/CUDA error.
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Model not loaded.
    #[error("model not loaded: call load() before predict()")]
    ModelNotLoaded,

    /// Model loading failed.
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Model file not found.
    #[error("model file not found: {0}")]
    ModelNotFound(String),

    /// ONNX Runtime error.
    #[cfg(feature = "onnx")]
    #[error("ONNX error: {0}")]
    Onnx(String),

    /// Image preprocessing error.
    #[error("preprocessing error: {0}")]
    Preprocessing(String),

    /// Invalid model output format.
    #[error("invalid model output: {0}")]
    InvalidOutput(String),
}

impl Error {
    /// Create an inference error.
    pub fn inference(msg: impl Into<String>) -> Self {
        Self::Inference(msg.into())
    }

    /// Create a GPU error.
    pub fn gpu(msg: impl Into<String>) -> Self {
        Self::Gpu(msg.into())
    }

    /// Create a configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a model load error.
    pub fn model_load(msg: impl Into<String>) -> Self {
        Self::ModelLoad(msg.into())
    }

    /// Create a model not found error.
    pub fn model_not_found(path: impl Into<String>) -> Self {
        Self::ModelNotFound(path.into())
    }

    /// Create an ONNX error.
    #[cfg(feature = "onnx")]
    pub fn onnx(msg: impl Into<String>) -> Self {
        Self::Onnx(msg.into())
    }

    /// Create a preprocessing error.
    pub fn preprocessing(msg: impl Into<String>) -> Self {
        Self::Preprocessing(msg.into())
    }

    /// Create an invalid output error.
    pub fn invalid_output(msg: impl Into<String>) -> Self {
        Self::InvalidOutput(msg.into())
    }
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for Error {
    fn from(err: ort::Error) -> Self {
        Self::Onnx(err.to_string())
    }
}
