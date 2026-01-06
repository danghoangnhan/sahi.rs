//! CUDA backend for GPU-accelerated slice processing.

use cudarc::driver::result::DriverError;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use crate::backend::Backend;
use crate::detection::Detection;
use crate::error::{Error, Result};
use crate::inference::{ImageData, InferenceCallback};
use crate::slicer::Slice;

/// CUDA backend configuration.
#[derive(Debug, Clone)]
pub struct CudaBackendConfig {
    /// GPU device index
    pub device_id: usize,
    /// Number of CUDA streams for concurrent kernel execution
    pub num_streams: usize,
    /// Maximum batch size for inference
    pub max_batch_size: usize,
}

impl Default for CudaBackendConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            num_streams: 2,
            max_batch_size: 8,
        }
    }
}

/// CUDA backend for GPU-accelerated slice processing.
pub struct CudaBackend {
    config: CudaBackendConfig,
    device: Option<Arc<CudaDevice>>,
}

impl CudaBackend {
    /// Create a new CUDA backend with default configuration.
    pub fn new() -> Self {
        let config = CudaBackendConfig::default();
        let device = Self::init_device(config.device_id);

        Self { config, device }
    }

    /// Create a CUDA backend with custom configuration.
    pub fn with_config(config: CudaBackendConfig) -> Self {
        let device = Self::init_device(config.device_id);
        Self { config, device }
    }

    /// Initialize the CUDA device.
    fn init_device(device_id: usize) -> Option<Arc<CudaDevice>> {
        CudaDevice::new(device_id).ok()
    }

    /// Get the CUDA device.
    pub fn device(&self) -> Option<&Arc<CudaDevice>> {
        self.device.as_ref()
    }

    /// Upload image data to GPU.
    pub fn upload_image(&self, image: &ImageData) -> Result<CudaSlice<u8>> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| Error::gpu("CUDA not initialized"))?;

        device
            .htod_copy(image.data.clone())
            .map_err(|e| Error::gpu(format!("Failed to upload image: {}", e)))
    }

    /// Download data from GPU.
    pub fn download<T: DeviceRepr + Clone>(&self, slice: &CudaSlice<T>) -> Result<Vec<T>> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| Error::gpu("CUDA not initialized"))?;

        device
            .dtoh_sync_copy(slice)
            .map_err(|e| Error::gpu(format!("Failed to download data: {}", e)))
    }

    /// Extract slice on GPU (placeholder - would use a CUDA kernel).
    fn extract_slice_gpu(
        &self,
        _src: &CudaSlice<u8>,
        _src_width: u32,
        _src_height: u32,
        _slice: &Slice,
    ) -> Result<CudaSlice<u8>> {
        // In a full implementation, this would launch a CUDA kernel
        // to extract the slice directly on the GPU.
        //
        // For now, we fall back to CPU extraction in process_slices.
        Err(Error::gpu("GPU slice extraction not yet implemented"))
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("config", &self.config)
            .field("device_available", &self.device.is_some())
            .finish()
    }
}

impl Backend for CudaBackend {
    fn process_slices(
        &self,
        image: &ImageData,
        slices: &[Slice],
        callback: &dyn InferenceCallback,
    ) -> Result<Vec<(Slice, Vec<Detection>)>> {
        // Check if CUDA is available
        if !self.is_available() {
            return Err(Error::gpu("CUDA device not available"));
        }

        // Process in batches for efficient GPU utilization
        let mut results = Vec::with_capacity(slices.len());

        for chunk in slices.chunks(self.config.max_batch_size) {
            // Extract slice images (CPU for now, GPU kernel would be faster)
            let slice_images: Vec<ImageData> = chunk
                .iter()
                .map(|s| image.extract_slice(s.x, s.y, s.width, s.height))
                .collect();

            // Run batch inference (model should handle GPU internally)
            let detections = callback.infer_batch(&slice_images)?;

            // Pair slices with detections
            for (slice, dets) in chunk.iter().copied().zip(detections) {
                results.push((slice, dets));
            }
        }

        Ok(results)
    }

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        self.device.is_some()
    }
}

/// CUDA kernel utilities (for future expansion).
pub mod kernels {
    use super::*;

    /// PTX source for slice extraction kernel.
    pub const SLICE_EXTRACT_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// extract_slice_kernel(src, dst, src_width, slice_x, slice_y, slice_width, slice_height, channels)
.visible .entry extract_slice_kernel(
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u32 src_width,
    .param .u32 slice_x,
    .param .u32 slice_y,
    .param .u32 slice_width,
    .param .u32 slice_height,
    .param .u32 channels
) {
    .reg .pred %p<2>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .b32 %f<2>;

    // Thread coordinates
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;  // dst_x

    mov.u32 %r5, %ctaid.y;
    mov.u32 %r6, %ntid.y;
    mov.u32 %r7, %tid.y;
    mad.lo.u32 %r8, %r5, %r6, %r7;  // dst_y

    // Load parameters
    ld.param.u32 %r9, [slice_width];
    ld.param.u32 %r10, [slice_height];

    // Bounds check
    setp.ge.u32 %p0, %r4, %r9;
    setp.ge.u32 %p1, %r8, %r10;
    or.pred %p0, %p0, %p1;
    @%p0 bra done;

    // Calculate source and destination offsets
    ld.param.u32 %r11, [slice_x];
    ld.param.u32 %r12, [slice_y];
    ld.param.u32 %r13, [src_width];
    ld.param.u32 %r14, [channels];

    // src_offset = ((slice_y + dst_y) * src_width + (slice_x + dst_x)) * channels
    add.u32 %r15, %r12, %r8;        // slice_y + dst_y
    mad.lo.u32 %r15, %r15, %r13, %r11;  // * src_width + slice_x
    add.u32 %r15, %r15, %r4;        // + dst_x
    mul.lo.u32 %r15, %r15, %r14;    // * channels

    // dst_offset = (dst_y * slice_width + dst_x) * channels
    mad.lo.u32 %r16, %r8, %r9, %r4;
    mul.lo.u32 %r16, %r16, %r14;

    // Copy pixel (simplified: single byte, would loop for all channels)
    ld.param.u64 %rd1, [src_ptr];
    ld.param.u64 %rd2, [dst_ptr];
    cvt.u64.u32 %rd3, %r15;
    cvt.u64.u32 %rd4, %r16;
    add.u64 %rd5, %rd1, %rd3;
    add.u64 %rd6, %rd2, %rd4;

    ld.global.u8 %r0, [%rd5];
    st.global.u8 [%rd6], %r0;

done:
    ret;
}
"#;

    /// Load the slice extraction kernel.
    pub fn load_slice_kernel(device: &Arc<CudaDevice>) -> std::result::Result<(), DriverError> {
        device.load_ptx(
            cudarc::nvrtc::Ptx::from_src(SLICE_EXTRACT_PTX),
            "sahi_kernels",
            &["extract_slice_kernel"],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_creation() {
        let backend = CudaBackend::new();
        // May or may not be available depending on system
        println!("CUDA available: {}", backend.is_available());
        assert_eq!(backend.name(), "cuda");
    }
}
