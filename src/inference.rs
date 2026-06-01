//! Inference callback pattern for model-agnostic detection.

use crate::detection::Detection;
use crate::error::Result;

#[cfg(feature = "python")]
use numpy::{PyArray3, PyArrayMethods, PyUntypedArrayMethods};
#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Raw image data for inference.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all))]
pub struct ImageData {
    /// Raw pixel data (RGB, HWC format)
    pub data: Vec<u8>,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Number of channels (typically 3 for RGB)
    pub channels: u32,
}

impl ImageData {
    /// Create new image data.
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u32) -> Self {
        Self {
            data,
            width,
            height,
            channels,
        }
    }

    /// Create from raw RGB bytes.
    pub fn from_rgb(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self::new(data, width, height, 3)
    }

    /// Extract a slice of the image.
    ///
    /// The result is always `w * h * channels` bytes. Any region outside the source
    /// image — past the right or bottom edge, or starting beyond it — is zero-padded,
    /// so the returned dimensions and buffer length are always consistent.
    pub fn extract_slice(&self, x: u32, y: u32, w: u32, h: u32) -> ImageData {
        let c = self.channels as usize;
        let row_len = w as usize * c;
        let mut slice_data = vec![0u8; row_len * h as usize];

        // Source columns available from `x` (0 if `x` is past the right edge).
        let avail_cols = if x < self.width {
            (self.width - x).min(w) as usize
        } else {
            0
        };

        if avail_cols > 0 {
            let copy_len = avail_cols * c;
            for out_row in 0..h as usize {
                let src_row = y as usize + out_row;
                if src_row >= self.height as usize {
                    break; // rows past the bottom edge stay zero-padded
                }
                let src_start = (src_row * self.width as usize + x as usize) * c;
                if src_start >= self.data.len() {
                    break;
                }
                let src_end = (src_start + copy_len).min(self.data.len());
                let dst_start = out_row * row_len;
                let n = src_end - src_start;
                slice_data[dst_start..dst_start + n]
                    .copy_from_slice(&self.data[src_start..src_end]);
            }
        }

        ImageData::new(slice_data, w, h, self.channels)
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl ImageData {
    /// Create ImageData from a numpy array (H, W, C) uint8.
    #[new]
    pub fn from_numpy(array: &Bound<'_, PyArray3<u8>>) -> PyResult<Self> {
        let shape = array.shape();
        let height = shape[0] as u32;
        let width = shape[1] as u32;
        let channels = shape[2] as u32;

        // Get contiguous data
        // Safety: We only read the array data, which is valid for the lifetime of the Bound reference
        let array = unsafe { array.as_array() };
        let data: Vec<u8> = array.iter().copied().collect();

        Ok(Self {
            data,
            width,
            height,
            channels,
        })
    }

    /// Convert to numpy array (H, W, C).
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        use numpy::PyArray;
        let array = PyArray::from_slice(py, &self.data);
        let reshaped = array.reshape([
            self.height as usize,
            self.width as usize,
            self.channels as usize,
        ])?;
        Ok(reshaped)
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageData(width={}, height={}, channels={})",
            self.width, self.height, self.channels
        )
    }
}

/// Trait for inference callbacks.
///
/// Implement this trait to integrate your detection model with SAHI.
///
/// # Example
///
/// ```ignore
/// struct MyYoloModel {
///     // your model state
/// }
///
/// impl InferenceCallback for MyYoloModel {
///     fn infer(&self, image: &ImageData) -> Result<Vec<Detection>> {
///         // Run your model inference here
///         // Return detections in slice-relative coordinates
///         Ok(vec![])
///     }
/// }
/// ```
pub trait InferenceCallback: Send + Sync {
    /// Run inference on an image slice.
    ///
    /// Returns detections with coordinates relative to the slice (not the original image).
    fn infer(&self, image: &ImageData) -> Result<Vec<Detection>>;

    /// Optional: Run batch inference on multiple slices.
    ///
    /// Default implementation calls `infer` for each slice sequentially.
    /// Override for GPU batch processing.
    fn infer_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<Detection>>> {
        images.iter().map(|img| self.infer(img)).collect()
    }
}

/// A boxed inference callback for dynamic dispatch.
pub type BoxedCallback = Box<dyn InferenceCallback>;

/// Function pointer callback for simple use cases.
pub struct FnCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<Detection>> + Send + Sync,
{
    func: F,
}

impl<F> FnCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<Detection>> + Send + Sync,
{
    /// Create a callback from a function.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> InferenceCallback for FnCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<Detection>> + Send + Sync,
{
    fn infer(&self, image: &ImageData) -> Result<Vec<Detection>> {
        (self.func)(image)
    }
}

/// Create an inference callback from a closure.
pub fn callback<F>(f: F) -> FnCallback<F>
where
    F: Fn(&ImageData) -> Result<Vec<Detection>> + Send + Sync,
{
    FnCallback::new(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::BoundingBox;

    #[test]
    fn test_image_extract_slice() {
        // 4x4 RGB image (48 bytes)
        let data: Vec<u8> = (0..48).collect();
        let img: ImageData = ImageData::from_rgb(data, 4, 4);

        let slice: ImageData = img.extract_slice(1, 1, 2, 2);

        assert_eq!(slice.width, 2);
        assert_eq!(slice.height, 2);
        assert_eq!(slice.data.len(), 12); // 2x2x3
    }

    #[test]
    fn test_extract_slice_pads_out_of_bounds_rows() {
        // 4x4 RGB (values 0..48). A 4x4 slice at (2,2) overruns both edges; only a
        // 2x2 region is in-bounds.
        let data: Vec<u8> = (0..48).collect();
        let img = ImageData::from_rgb(data, 4, 4);
        let slice = img.extract_slice(2, 2, 4, 4);

        assert_eq!(slice.width, 4);
        assert_eq!(slice.height, 4);
        assert_eq!(slice.data.len(), 4 * 4 * 3); // always w*h*channels (zero-padded)

        // Top-left of the slice is source pixel (2,2) = byte offset (2*4+2)*3 = 30.
        assert_eq!(&slice.data[0..3], &[30, 31, 32]);
        // A row beyond the source bottom stays zero.
        let row3 = 3 * 4 * 3;
        assert_eq!(&slice.data[row3..row3 + 3], &[0, 0, 0]);
    }

    #[test]
    fn test_extract_slice_x_beyond_width_is_zero() {
        let data: Vec<u8> = (0..48).collect();
        let img = ImageData::from_rgb(data, 4, 4);
        // x past the right edge: no source columns -> all zeros, and no panic.
        let slice = img.extract_slice(5, 0, 2, 2);
        assert_eq!(slice.data.len(), 2 * 2 * 3);
        assert!(slice.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_fn_callback() {
        let cb = callback(|_img: &ImageData| {
            Ok(vec![Detection::new(
                BoundingBox::new(0.0, 0.0, 10.0, 10.0),
                0,
                0.9,
                None,
            )])
        });

        let img: ImageData = ImageData::from_rgb(vec![0; 12], 2, 2);
        let result: Vec<Detection> = cb.infer(&img).unwrap();

        assert_eq!(result.len(), 1);
    }
}
