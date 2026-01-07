//! CPU backend for sequential/parallel slice processing.

use crate::backend::Backend;
use crate::detection::Detection;
use crate::error::Result;
use crate::inference::{ImageData, InferenceCallback};
use crate::slicer::Slice;

/// CPU backend configuration.
#[derive(Debug, Clone, Default)]
pub struct CpuBackendConfig {
    /// Number of threads for parallel processing (0 = auto)
    pub num_threads: usize,
}

/// CPU backend for slice processing.
#[derive(Debug)]
pub struct CpuBackend {
    config: CpuBackendConfig,
}

impl CpuBackend {
    /// Create a new CPU backend with default configuration.
    pub fn new() -> Self {
        Self {
            config: CpuBackendConfig::default(),
        }
    }

    /// Create a CPU backend with custom configuration.
    pub fn with_config(config: CpuBackendConfig) -> Self {
        Self { config }
    }

    /// Set the number of threads.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = num_threads;
        self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn process_slices(
        &self,
        image: &ImageData,
        slices: &[Slice],
        callback: &dyn InferenceCallback,
    ) -> Result<Vec<(Slice, Vec<Detection>)>> {
        // Extract slice images
        let slice_images: Vec<ImageData> = slices
            .iter()
            .map(|s| image.extract_slice(s.x, s.y, s.width, s.height))
            .collect();

        // Try batch inference first
        let detections = callback.infer_batch(&slice_images)?;

        // Pair slices with their detections
        Ok(slices.iter().copied().zip(detections).collect())
    }

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::BoundingBox;
    use crate::inference::callback;

    #[test]
    fn test_cpu_backend() {
        let backend = CpuBackend::new();

        assert!(backend.is_available());
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_process_slices() {
        let backend = CpuBackend::new();
        let image = ImageData::from_rgb(vec![0; 300], 10, 10);
        let slices = vec![Slice::new(0, 0, 5, 5, 0), Slice::new(5, 5, 5, 5, 1)];

        let cb = callback(|_img: &ImageData| {
            Ok(vec![Detection::new(
                BoundingBox::new(1.0, 1.0, 2.0, 2.0),
                0,
                0.9,
                None,
            )])
        });

        let result = backend.process_slices(&image, &slices, &cb).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1.len(), 1);
        assert_eq!(result[1].1.len(), 1);
    }
}
