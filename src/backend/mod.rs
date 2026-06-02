//! Backend abstraction for CPU/GPU execution.

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

pub use cpu::{CpuBackend, CpuBackendConfig};

#[cfg(feature = "cuda")]
pub use cuda::{CudaBackend, CudaBackendConfig};

use crate::detection::Detection;
use crate::error::Result;
use crate::inference::{ImageData, InferenceCallback};
use crate::slicer::Slice;

/// Trait for execution backends.
///
/// Backends handle the orchestration of sliced inference,
/// potentially parallelizing on CPU threads or GPU streams.
pub trait Backend: Send + Sync + std::fmt::Debug {
    /// Process all slices and return detections for each.
    fn process_slices(
        &self,
        image: &ImageData,
        slices: &[Slice],
        callback: &dyn InferenceCallback,
    ) -> Result<Vec<(Slice, Vec<Detection>)>>;

    /// Extract every slice image from `image`, in the same order as `slices`.
    ///
    /// Backends may parallelize (CPU rayon) or accelerate (CUDA) extraction; the
    /// default is sequential. Both the detection and segmentation pipelines build
    /// on this so masks get the same acceleration as boxes.
    fn extract_slices(&self, image: &ImageData, slices: &[Slice]) -> Result<Vec<ImageData>> {
        Ok(slices
            .iter()
            .map(|s| image.extract_slice(s.x, s.y, s.width, s.height))
            .collect())
    }

    /// Get the backend name.
    fn name(&self) -> &'static str;

    /// Check if the backend is available.
    fn is_available(&self) -> bool;
}

/// Boxed backend for dynamic dispatch.
pub type BoxedBackend = Box<dyn Backend>;

/// Create the default backend based on available features.
pub fn default_backend() -> BoxedBackend {
    #[cfg(feature = "cuda")]
    {
        let cuda = CudaBackend::new();
        if cuda.is_available() {
            return Box::new(cuda);
        }
    }

    Box::new(CpuBackend::new())
}

/// Explicitly create a CPU backend.
pub fn cpu_backend() -> CpuBackend {
    CpuBackend::new()
}

/// Explicitly create a CUDA backend (requires `cuda` feature).
#[cfg(feature = "cuda")]
pub fn cuda_backend() -> CudaBackend {
    CudaBackend::new()
}
