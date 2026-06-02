//! CPU backend for sequential/parallel slice processing.

use crate::backend::Backend;
use crate::detection::Detection;
use crate::error::Result;
use crate::inference::{ImageData, InferenceCallback};
use crate::slicer::Slice;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// CPU backend configuration.
#[derive(Debug, Clone, Default)]
pub struct CpuBackendConfig {
    /// Number of threads for parallel processing (0 = auto).
    /// Only used when the `parallel` feature is enabled.
    pub num_threads: usize,
    /// Enable parallel inference calls across slices.
    ///
    /// **Warning:** Not safe with Python/GIL-bound callbacks, as they will
    /// contend for the GIL and may serialize with extra overhead.
    /// Only enable this for pure-Rust callbacks.
    pub parallel_inference: bool,
}

/// CPU backend for slice processing.
///
/// When the `parallel` feature is enabled, slice extraction is parallelized
/// using rayon. Inference parallelism is opt-in via `parallel_inference`.
#[derive(Debug)]
pub struct CpuBackend {
    config: CpuBackendConfig,
    #[cfg(feature = "parallel")]
    thread_pool: Option<rayon::ThreadPool>,
}

impl CpuBackend {
    /// Create a new CPU backend with default configuration.
    pub fn new() -> Self {
        let config = CpuBackendConfig::default();
        Self {
            #[cfg(feature = "parallel")]
            thread_pool: Self::build_thread_pool(config.num_threads),
            config,
        }
    }

    /// Create a CPU backend with custom configuration.
    pub fn with_config(config: CpuBackendConfig) -> Self {
        Self {
            #[cfg(feature = "parallel")]
            thread_pool: Self::build_thread_pool(config.num_threads),
            config,
        }
    }

    /// Set the number of threads.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = num_threads;
        #[cfg(feature = "parallel")]
        {
            self.thread_pool = Self::build_thread_pool(num_threads);
        }
        self
    }

    /// Enable or disable parallel inference.
    pub fn with_parallel_inference(mut self, enabled: bool) -> Self {
        self.config.parallel_inference = enabled;
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &CpuBackendConfig {
        &self.config
    }

    /// Build the configured rayon pool, falling back to `None` (rayon's global
    /// pool) if it cannot be constructed — rather than panicking.
    #[cfg(feature = "parallel")]
    fn build_thread_pool(num_threads: usize) -> Option<rayon::ThreadPool> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if num_threads > 0 {
            builder = builder.num_threads(num_threads);
        }
        builder.build().ok()
    }

    /// Run `op` on the configured pool, or on rayon's global pool when no custom
    /// pool is available.
    #[cfg(feature = "parallel")]
    fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match &self.thread_pool {
            Some(pool) => pool.install(op),
            None => op(),
        }
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
        #[cfg(feature = "parallel")]
        {
            self.process_slices_parallel(image, slices, callback)
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.process_slices_sequential(image, slices, callback)
        }
    }

    fn extract_slices(&self, image: &ImageData, slices: &[Slice]) -> Result<Vec<ImageData>> {
        #[cfg(feature = "parallel")]
        {
            Ok(self.install(|| {
                slices
                    .par_iter()
                    .map(|s| image.extract_slice(s.x, s.y, s.width, s.height))
                    .collect()
            }))
        }

        #[cfg(not(feature = "parallel"))]
        {
            Ok(slices
                .iter()
                .map(|s| image.extract_slice(s.x, s.y, s.width, s.height))
                .collect())
        }
    }

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true
    }
}

impl CpuBackend {
    /// Sequential processing (no rayon).
    #[cfg_attr(feature = "parallel", allow(dead_code))]
    fn process_slices_sequential(
        &self,
        image: &ImageData,
        slices: &[Slice],
        callback: &dyn InferenceCallback,
    ) -> Result<Vec<(Slice, Vec<Detection>)>> {
        let slice_images = self.extract_slices(image, slices)?;

        let detections = callback.infer_batch(&slice_images)?;
        Ok(slices.iter().copied().zip(detections).collect())
    }

    /// Parallel processing using rayon thread pool.
    #[cfg(feature = "parallel")]
    fn process_slices_parallel(
        &self,
        image: &ImageData,
        slices: &[Slice],
        callback: &dyn InferenceCallback,
    ) -> Result<Vec<(Slice, Vec<Detection>)>> {
        // Parallel slice extraction (shared with the segmentation path).
        let slice_images = self.extract_slices(image, slices)?;

        if self.config.parallel_inference {
            // Parallel inference (opt-in, NOT safe for Python/GIL callbacks)
            let results: Result<Vec<(Slice, Vec<Detection>)>> = self.install(|| {
                slices
                    .par_iter()
                    .zip(slice_images.par_iter())
                    .map(|(s, img)| {
                        let dets = callback.infer(img)?;
                        Ok((*s, dets))
                    })
                    .collect()
            });
            results
        } else {
            // Sequential inference (safe for all callbacks)
            let detections = callback.infer_batch(&slice_images)?;
            Ok(slices.iter().copied().zip(detections).collect())
        }
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

    #[test]
    fn test_extract_slices_matches_direct_extraction() {
        let backend = CpuBackend::new();
        let image = ImageData::from_rgb((0..300).map(|i| (i % 256) as u8).collect(), 10, 10);
        let slices = vec![
            Slice::new(0, 0, 5, 5, 0),
            Slice::new(5, 5, 5, 5, 1),
            Slice::new(2, 3, 4, 4, 2),
        ];
        let got = backend.extract_slices(&image, &slices).unwrap();
        assert_eq!(got.len(), slices.len());
        for (s, img) in slices.iter().zip(&got) {
            let expected = image.extract_slice(s.x, s.y, s.width, s.height);
            assert_eq!(img.data, expected.data);
            assert_eq!((img.width, img.height), (s.width, s.height));
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_extract_slices_falls_back_to_global_pool() {
        // A `None` custom pool (e.g. if building one failed) must not panic;
        // extraction falls back to rayon's global pool and stays correct.
        let mut backend = CpuBackend::new();
        backend.thread_pool = None;
        let image = ImageData::from_rgb((0..300).map(|i| (i % 256) as u8).collect(), 10, 10);
        let slices = vec![Slice::new(0, 0, 5, 5, 0), Slice::new(5, 5, 5, 5, 1)];
        let got = backend.extract_slices(&image, &slices).unwrap();
        assert_eq!(got.len(), 2);
        for (s, img) in slices.iter().zip(&got) {
            assert_eq!(
                img.data,
                image.extract_slice(s.x, s.y, s.width, s.height).data
            );
        }
    }

    #[test]
    fn test_default_config() {
        let config = CpuBackendConfig::default();
        assert_eq!(config.num_threads, 0);
        assert!(!config.parallel_inference);
    }

    #[test]
    fn test_with_threads_config() {
        let backend = CpuBackend::new().with_threads(4);
        assert_eq!(backend.config().num_threads, 4);
    }

    #[test]
    fn test_with_parallel_inference_config() {
        let backend = CpuBackend::new().with_parallel_inference(true);
        assert!(backend.config().parallel_inference);
    }

    #[test]
    fn test_sequential_matches_results() {
        // Verify sequential path produces correct results
        let backend = CpuBackend::new();
        let image = ImageData::from_rgb((0..300).map(|i| (i % 256) as u8).collect(), 10, 10);
        let slices = vec![
            Slice::new(0, 0, 5, 5, 0),
            Slice::new(3, 3, 5, 5, 1),
            Slice::new(5, 5, 5, 5, 2),
        ];

        let cb = callback(|img: &ImageData| {
            // Return detection based on image content for verification
            let sum: u32 = img.data.iter().map(|&b| b as u32).sum();
            Ok(vec![Detection::new(
                BoundingBox::new(0.0, 0.0, 1.0, 1.0),
                0,
                sum as f32,
                None,
            )])
        });

        let result = backend
            .process_slices_sequential(&image, &slices, &cb)
            .unwrap();
        assert_eq!(result.len(), 3);
        // Each slice should produce one detection with a unique confidence based on pixel content
        assert_ne!(result[0].1[0].confidence, result[1].1[0].confidence);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_extraction_matches_sequential() {
        let image = ImageData::from_rgb((0..300).map(|i| (i % 256) as u8).collect(), 10, 10);
        let slices = vec![
            Slice::new(0, 0, 5, 5, 0),
            Slice::new(3, 3, 5, 5, 1),
            Slice::new(5, 5, 5, 5, 2),
        ];

        let cb = callback(|img: &ImageData| {
            let sum: u32 = img.data.iter().map(|&b| b as u32).sum();
            Ok(vec![Detection::new(
                BoundingBox::new(0.0, 0.0, 1.0, 1.0),
                0,
                sum as f32,
                None,
            )])
        });

        // Sequential backend
        let seq_backend = CpuBackend::new();
        let seq_result = seq_backend
            .process_slices_sequential(&image, &slices, &cb)
            .unwrap();

        // Parallel backend
        let par_backend = CpuBackend::new().with_threads(2);
        let par_result = par_backend
            .process_slices_parallel(&image, &slices, &cb)
            .unwrap();

        assert_eq!(seq_result.len(), par_result.len());
        for (seq, par) in seq_result.iter().zip(par_result.iter()) {
            assert_eq!(seq.0, par.0); // same slice
            assert_eq!(seq.1.len(), par.1.len());
            assert_eq!(seq.1[0].confidence, par.1[0].confidence);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_inference_produces_correct_results() {
        let backend = CpuBackend::new()
            .with_threads(2)
            .with_parallel_inference(true);

        let image = ImageData::from_rgb(vec![128; 300], 10, 10);
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
