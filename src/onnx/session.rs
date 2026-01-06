//! ONNX Runtime session wrapper.
//!
//! Provides a builder-pattern interface for creating ONNX sessions with
//! configurable execution providers (CPU, CUDA, TensorRT).

use std::path::Path;
use std::sync::Arc;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::error::{Error, Result};
use crate::model::Device;

/// Execution provider for ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// CPU execution (default).
    #[default]
    Cpu,
    /// CUDA GPU execution.
    #[cfg(feature = "onnx-cuda")]
    Cuda {
        /// GPU device ID.
        device_id: i32,
    },
    /// TensorRT execution (requires CUDA).
    #[cfg(feature = "onnx-tensorrt")]
    TensorRT {
        /// GPU device ID.
        device_id: i32,
    },
}

impl ExecutionProvider {
    /// Create a CPU execution provider.
    #[inline]
    pub fn cpu() -> Self {
        Self::Cpu
    }

    /// Create a CUDA execution provider with device 0.
    #[cfg(feature = "onnx-cuda")]
    #[inline]
    pub fn cuda() -> Self {
        Self::Cuda { device_id: 0 }
    }

    /// Create a CUDA execution provider with a specific device.
    #[cfg(feature = "onnx-cuda")]
    #[inline]
    pub fn cuda_with_device(device_id: i32) -> Self {
        Self::Cuda { device_id }
    }

    /// Create a TensorRT execution provider with device 0.
    #[cfg(feature = "onnx-tensorrt")]
    #[inline]
    pub fn tensorrt() -> Self {
        Self::TensorRT { device_id: 0 }
    }

    /// Create a TensorRT execution provider with a specific device.
    #[cfg(feature = "onnx-tensorrt")]
    #[inline]
    pub fn tensorrt_with_device(device_id: i32) -> Self {
        Self::TensorRT { device_id }
    }
}

impl From<Device> for ExecutionProvider {
    fn from(device: Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            #[cfg(feature = "onnx-cuda")]
            Device::Cuda(id) => Self::Cuda {
                device_id: id as i32,
            },
            #[cfg(not(feature = "onnx-cuda"))]
            Device::Cuda(_) => Self::Cpu,
            Device::Mps => Self::Cpu, // MPS not supported by ONNX Runtime
            Device::Auto => {
                #[cfg(feature = "onnx-cuda")]
                {
                    Self::Cuda { device_id: 0 }
                }
                #[cfg(not(feature = "onnx-cuda"))]
                {
                    Self::Cpu
                }
            }
        }
    }
}

/// Wrapper around an ONNX Runtime session.
///
/// Provides convenient access to session inputs/outputs and manages
/// the session lifecycle.
#[derive(Debug)]
pub struct OnnxSession {
    /// The underlying ONNX Runtime session.
    session: Session,
    /// Input names.
    input_names: Vec<Arc<str>>,
    /// Output names.
    output_names: Vec<Arc<str>>,
}

impl OnnxSession {
    /// Create a builder for configuring the session.
    pub fn builder() -> OnnxSessionBuilder {
        OnnxSessionBuilder::new()
    }

    /// Get a reference to the underlying session.
    #[inline]
    pub fn inner(&self) -> &Session {
        &self.session
    }

    /// Get input names.
    #[inline]
    pub fn input_names(&self) -> &[Arc<str>] {
        &self.input_names
    }

    /// Get output names.
    #[inline]
    pub fn output_names(&self) -> &[Arc<str>] {
        &self.output_names
    }

    /// Get the first input name.
    #[inline]
    pub fn input_name(&self) -> Option<&str> {
        self.input_names.first().map(|s| s.as_ref())
    }

    /// Get the first output name.
    #[inline]
    pub fn output_name(&self) -> Option<&str> {
        self.output_names.first().map(|s| s.as_ref())
    }

    /// Run inference using the underlying session.
    ///
    /// Returns the session for direct access to run methods.
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

/// Builder for creating ONNX sessions.
#[derive(Debug)]
pub struct OnnxSessionBuilder {
    execution_provider: ExecutionProvider,
    optimization_level: GraphOptimizationLevel,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
}

impl Default for OnnxSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxSessionBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            execution_provider: ExecutionProvider::Cpu,
            optimization_level: GraphOptimizationLevel::Level3,
            intra_threads: None,
            inter_threads: None,
        }
    }

    /// Set the execution provider.
    pub fn execution_provider(mut self, ep: ExecutionProvider) -> Self {
        self.execution_provider = ep;
        self
    }

    /// Set the execution provider from a Device.
    pub fn device(mut self, device: Device) -> Self {
        self.execution_provider = device.into();
        self
    }

    /// Set the graph optimization level.
    pub fn optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// Set the number of intra-op threads.
    pub fn intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    /// Set the number of inter-op threads.
    pub fn inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    /// Build the session from a model file.
    pub fn build_from_file(self, path: impl AsRef<Path>) -> Result<OnnxSession> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::model_not_found(path.display().to_string()));
        }

        let mut builder = Session::builder()?;

        // Apply optimization level
        builder = builder.with_optimization_level(self.optimization_level)?;

        // Apply thread settings
        if let Some(threads) = self.intra_threads {
            builder = builder.with_intra_threads(threads)?;
        }
        if let Some(threads) = self.inter_threads {
            builder = builder.with_inter_threads(threads)?;
        }

        // Configure execution providers
        #[cfg(feature = "onnx-cuda")]
        if let ExecutionProvider::Cuda { device_id } = self.execution_provider {
            use ort::execution_providers::CUDAExecutionProvider;
            builder = builder.with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?;
        }

        #[cfg(feature = "onnx-tensorrt")]
        if let ExecutionProvider::TensorRT { device_id } = self.execution_provider {
            use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
            builder = builder.with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
                CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
            ])?;
        }

        // Load the model
        let session = builder.commit_from_file(path)?;

        // Extract input/output names
        let input_names = session
            .inputs
            .iter()
            .map(|i| Arc::from(i.name.as_str()))
            .collect();
        let output_names = session
            .outputs
            .iter()
            .map(|o| Arc::from(o.name.as_str()))
            .collect();

        Ok(OnnxSession {
            session,
            input_names,
            output_names,
        })
    }

    /// Build the session from model bytes.
    pub fn build_from_memory(self, model_bytes: &[u8]) -> Result<OnnxSession> {
        let mut builder = Session::builder()?;

        // Apply optimization level
        builder = builder.with_optimization_level(self.optimization_level)?;

        // Apply thread settings
        if let Some(threads) = self.intra_threads {
            builder = builder.with_intra_threads(threads)?;
        }
        if let Some(threads) = self.inter_threads {
            builder = builder.with_inter_threads(threads)?;
        }

        // Configure execution providers
        #[cfg(feature = "onnx-cuda")]
        if let ExecutionProvider::Cuda { device_id } = self.execution_provider {
            use ort::execution_providers::CUDAExecutionProvider;
            builder = builder.with_execution_providers([CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()])?;
        }

        #[cfg(feature = "onnx-tensorrt")]
        if let ExecutionProvider::TensorRT { device_id } = self.execution_provider {
            use ort::execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider};
            builder = builder.with_execution_providers([
                TensorRTExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
                CUDAExecutionProvider::default()
                    .with_device_id(device_id)
                    .build(),
            ])?;
        }

        // Load the model from memory
        let session = builder.commit_from_memory(model_bytes)?;

        // Extract input/output names
        let input_names = session
            .inputs
            .iter()
            .map(|i| Arc::from(i.name.as_str()))
            .collect();
        let output_names = session
            .outputs
            .iter()
            .map(|o| Arc::from(o.name.as_str()))
            .collect();

        Ok(OnnxSession {
            session,
            input_names,
            output_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_provider_from_device() {
        assert_eq!(
            ExecutionProvider::from(Device::Cpu),
            ExecutionProvider::Cpu
        );
    }

    #[test]
    fn test_builder_defaults() {
        let builder = OnnxSessionBuilder::new();
        assert_eq!(builder.execution_provider, ExecutionProvider::Cpu);
    }
}
