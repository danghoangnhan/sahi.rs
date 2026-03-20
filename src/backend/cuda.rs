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
    kernel_loaded: bool,
}

impl CudaBackend {
    /// Create a new CUDA backend with default configuration.
    pub fn new() -> Self {
        let config = CudaBackendConfig::default();
        let device = Self::init_device(config.device_id);
        let kernel_loaded = device
            .as_ref()
            .map(|d| kernels::load_slice_kernel(d).is_ok())
            .unwrap_or(false);

        Self {
            config,
            device,
            kernel_loaded,
        }
    }

    /// Create a CUDA backend with custom configuration.
    pub fn with_config(config: CudaBackendConfig) -> Self {
        let device = Self::init_device(config.device_id);
        let kernel_loaded = device
            .as_ref()
            .map(|d| kernels::load_slice_kernel(d).is_ok())
            .unwrap_or(false);

        Self {
            config,
            device,
            kernel_loaded,
        }
    }

    /// Initialize the CUDA device.
    fn init_device(device_id: usize) -> Option<Arc<CudaDevice>> {
        CudaDevice::new(device_id).ok()
    }

    /// Get the CUDA device.
    pub fn device(&self) -> Option<&Arc<CudaDevice>> {
        self.device.as_ref()
    }

    /// Check if the slice extraction kernel is loaded.
    pub fn kernel_loaded(&self) -> bool {
        self.kernel_loaded
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

    /// Extract a slice from a source image on GPU using the CUDA kernel.
    fn extract_slice_gpu(
        &self,
        src: &CudaSlice<u8>,
        src_width: u32,
        _src_height: u32,
        channels: u32,
        slice: &Slice,
    ) -> Result<CudaSlice<u8>> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| Error::gpu("CUDA not initialized"))?;

        if !self.kernel_loaded {
            return Err(Error::gpu("Slice extraction kernel not loaded"));
        }

        let output_size = (slice.width * slice.height * channels) as usize;

        // Allocate zeroed output buffer on GPU
        let dst: CudaSlice<u8> = device
            .alloc_zeros(output_size)
            .map_err(|e| Error::gpu(format!("Failed to allocate GPU output: {}", e)))?;

        // Configure launch: one thread per pixel, 16x16 thread blocks
        let block_dim = (16u32, 16u32, 1u32);
        let grid_dim = (
            slice.width.div_ceil(block_dim.0),
            slice.height.div_ceil(block_dim.1),
            1u32,
        );
        let launch_config = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };

        // Get the loaded kernel function
        let func = device
            .get_func("sahi_kernels", "extract_slice_kernel")
            .ok_or_else(|| Error::gpu("Kernel function not found"))?;

        // Launch kernel: (src, dst, src_width, slice_x, slice_y, slice_width, slice_height, channels)
        unsafe {
            func.launch(
                launch_config,
                (
                    src,
                    &dst,
                    src_width,
                    slice.x,
                    slice.y,
                    slice.width,
                    slice.height,
                    channels,
                ),
            )
        }
        .map_err(|e| Error::gpu(format!("Kernel launch failed: {}", e)))?;

        Ok(dst)
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
            .field("kernel_loaded", &self.kernel_loaded)
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
        if !self.is_available() {
            return Err(Error::gpu("CUDA device not available"));
        }

        let mut results = Vec::with_capacity(slices.len());

        if self.kernel_loaded {
            // GPU path: upload image once, extract slices on GPU
            let gpu_image = self.upload_image(image)?;

            for chunk in slices.chunks(self.config.max_batch_size) {
                let slice_images: Vec<ImageData> = chunk
                    .iter()
                    .map(|s| {
                        let gpu_slice = self.extract_slice_gpu(
                            &gpu_image,
                            image.width,
                            image.height,
                            image.channels,
                            s,
                        )?;
                        let data = self.download(&gpu_slice)?;
                        Ok(ImageData::new(data, s.width, s.height, image.channels))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let detections = callback.infer_batch(&slice_images)?;

                for (slice, dets) in chunk.iter().copied().zip(detections) {
                    results.push((slice, dets));
                }
            }
        } else {
            // CPU fallback: kernel not loaded, extract slices on CPU
            for chunk in slices.chunks(self.config.max_batch_size) {
                let slice_images: Vec<ImageData> = chunk
                    .iter()
                    .map(|s| image.extract_slice(s.x, s.y, s.width, s.height))
                    .collect();

                let detections = callback.infer_batch(&slice_images)?;

                for (slice, dets) in chunk.iter().copied().zip(detections) {
                    results.push((slice, dets));
                }
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

/// CUDA kernel utilities.
pub mod kernels {
    use super::*;

    /// PTX source for slice extraction kernel.
    ///
    /// Extracts a rectangular slice from a source image, copying all channels per pixel.
    /// Each thread handles one pixel (all channels).
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
    .reg .b32 %r<17>;
    .reg .b64 %rd<9>;

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
    add.u32 %r15, %r12, %r8;             // slice_y + dst_y
    mad.lo.u32 %r15, %r15, %r13, %r11;   // * src_width + slice_x
    add.u32 %r15, %r15, %r4;             // + dst_x
    mul.lo.u32 %r15, %r15, %r14;         // * channels

    // dst_offset = (dst_y * slice_width + dst_x) * channels
    mad.lo.u32 %r16, %r8, %r9, %r4;
    mul.lo.u32 %r16, %r16, %r14;

    // Compute base pointers for this pixel
    ld.param.u64 %rd1, [src_ptr];
    ld.param.u64 %rd2, [dst_ptr];
    cvt.u64.u32 %rd3, %r15;
    cvt.u64.u32 %rd4, %r16;
    add.u64 %rd5, %rd1, %rd3;   // src base for this pixel
    add.u64 %rd6, %rd2, %rd4;   // dst base for this pixel

    // Loop over all channels
    mov.u32 %r0, 0;
channel_loop:
    setp.ge.u32 %p0, %r0, %r14;
    @%p0 bra done;

    cvt.u64.u32 %rd7, %r0;
    add.u64 %rd3, %rd5, %rd7;   // src + channel offset
    add.u64 %rd4, %rd6, %rd7;   // dst + channel offset

    ld.global.u8 %r1, [%rd3];
    st.global.u8 [%rd4], %r1;

    add.u32 %r0, %r0, 1;
    bra channel_loop;

done:
    ret;
}
"#;

    /// Load the slice extraction kernel onto the device.
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
    use crate::detection::BoundingBox;
    use crate::inference::callback;

    #[test]
    fn test_cuda_backend_creation() {
        let backend = CudaBackend::new();
        println!("CUDA available: {}", backend.is_available());
        println!("Kernel loaded: {}", backend.kernel_loaded());
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn test_ptx_source_validity() {
        assert!(kernels::SLICE_EXTRACT_PTX.contains("extract_slice_kernel"));
        assert!(kernels::SLICE_EXTRACT_PTX.contains("channel_loop"));
        assert!(kernels::SLICE_EXTRACT_PTX.contains(".entry"));
        assert!(kernels::SLICE_EXTRACT_PTX.contains("ld.global.u8"));
        assert!(kernels::SLICE_EXTRACT_PTX.contains("st.global.u8"));
    }

    #[test]
    fn test_extract_slice_gpu() {
        let backend = CudaBackend::new();
        if !backend.is_available() || !backend.kernel_loaded() {
            println!("Skipping GPU test: CUDA not available or kernel not loaded");
            return;
        }

        // Create a 4x4 RGB image with known values
        let data: Vec<u8> = (0..48).collect();
        let image = ImageData::from_rgb(data, 4, 4);

        let gpu_image = backend.upload_image(&image).unwrap();
        let slice = Slice::new(1, 1, 2, 2, 0);
        let gpu_result = backend
            .extract_slice_gpu(&gpu_image, 4, 4, 3, &slice)
            .unwrap();
        let result = backend.download(&gpu_result).unwrap();

        // Compare with CPU extraction
        let cpu_result = image.extract_slice(1, 1, 2, 2);
        assert_eq!(result, cpu_result.data);
    }

    #[test]
    fn test_process_slices_cpu_fallback() {
        let backend = CudaBackend::new();
        if !backend.is_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        // Force CPU fallback by creating a backend without kernel
        let backend = CudaBackend {
            config: CudaBackendConfig::default(),
            device: backend.device.clone(),
            kernel_loaded: false,
        };

        let image = ImageData::from_rgb(vec![0; 300], 10, 10);
        let slices = vec![Slice::new(0, 0, 5, 5, 0), Slice::new(5, 5, 5, 5, 1)];
        let cb = callback(|_img: &ImageData| Ok(vec![]));

        let result = backend.process_slices(&image, &slices, &cb).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_process_slices_gpu_path() {
        let backend = CudaBackend::new();
        if !backend.is_available() || !backend.kernel_loaded() {
            println!("Skipping GPU test: CUDA not available or kernel not loaded");
            return;
        }

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
