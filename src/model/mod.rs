//! Detection model abstraction for SAHI.
//!
//! This module provides a high-level abstraction for detection models,
//! similar to the Python SAHI DetectionModel class but optimized for Rust.
//!
//! # Example
//!
//! ```rust,ignore
//! use sahi::model::{DetectionModel, ModelConfig, Device};
//!
//! struct MyYoloModel {
//!     // model state
//! }
//!
//! impl DetectionModel for MyYoloModel {
//!     fn load(&mut self) -> Result<()> {
//!         // Load model weights
//!         Ok(())
//!     }
//!
//!     fn predict(&self, image: &ImageData) -> Result<Vec<Detection>> {
//!         // Run inference
//!         Ok(vec![])
//!     }
//! }
//! ```

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::detection::{BoundingBox, Detection};
use crate::error::{Error, Result};
use crate::inference::ImageData;

// ============================================================================
// Category
// ============================================================================

/// A detection category with ID and name.
///
/// Uses `Arc<str>` for the name to enable cheap cloning when the same
/// category appears in many detections.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", pyclass)]
pub struct Category {
    /// Numeric category ID.
    pub id: u32,
    /// Category name (Arc for cheap cloning).
    name: Arc<str>,
}

impl Category {
    /// Create a new category.
    #[inline]
    pub fn new(id: u32, name: impl Into<Arc<str>>) -> Self {
        Self {
            id,
            name: name.into(),
        }
    }

    /// Get the category name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create from static string (zero allocation).
    #[inline]
    pub fn from_static(id: u32, name: &'static str) -> Self {
        Self {
            id,
            name: Arc::from(name),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Category {
    #[new]
    fn py_new(id: u32, name: String) -> Self {
        Self::new(id, name)
    }

    #[getter]
    fn get_id(&self) -> u32 {
        self.id
    }

    #[getter]
    fn get_name(&self) -> &str {
        self.name()
    }

    fn __repr__(&self) -> String {
        format!("Category(id={}, name='{}')", self.id, self.name())
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

// ============================================================================
// Device
// ============================================================================

/// Execution device for the model.
///
/// For Python, use string device names like "cpu", "cuda:0", "mps", "auto".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU execution.
    #[default]
    Cpu,
    /// CUDA GPU execution with device index.
    Cuda(u32),
    /// Apple Metal GPU (MPS).
    Mps,
    /// Automatic device selection (prefers GPU if available).
    Auto,
}

impl Device {
    /// Create a CUDA device with index 0.
    #[inline]
    pub fn cuda() -> Self {
        Self::Cuda(0)
    }

    /// Create a CUDA device with specific index.
    #[inline]
    pub fn cuda_with_index(index: u32) -> Self {
        Self::Cuda(index)
    }

    /// Check if this is a GPU device.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Mps)
    }

    /// Check if this is a CPU device.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Get device string representation (e.g., "cuda:0", "cpu", "mps").
    pub fn as_str(&self) -> Cow<'static, str> {
        match self {
            Self::Cpu => Cow::Borrowed("cpu"),
            Self::Cuda(0) => Cow::Borrowed("cuda:0"),
            Self::Cuda(idx) => Cow::Owned(format!("cuda:{}", idx)),
            Self::Mps => Cow::Borrowed("mps"),
            Self::Auto => Cow::Borrowed("auto"),
        }
    }

    /// Resolve Auto to an actual device.
    #[inline]
    pub fn resolve(&self) -> Self {
        match self {
            Self::Auto => {
                #[cfg(feature = "cuda")]
                {
                    // Try CUDA first
                    Self::Cuda(0)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Self::Cpu
                }
            }
            other => *other,
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Device {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.to_lowercase();
        match s.as_str() {
            "cpu" => Ok(Self::Cpu),
            "mps" => Ok(Self::Mps),
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda(0)),
            _ if s.starts_with("cuda:") => {
                let idx = s[5..]
                    .parse()
                    .map_err(|_| Error::config(format!("invalid CUDA device index: {}", s)))?;
                Ok(Self::Cuda(idx))
            }
            _ => Err(Error::config(format!("unknown device: {}", s))),
        }
    }
}


// ============================================================================
// Category Mapping
// ============================================================================

/// Mapping from category IDs to category information.
///
/// Optimized for fast lookup by ID and supports remapping.
#[derive(Debug, Clone, Default)]
pub struct CategoryMapping {
    /// ID to category mapping.
    by_id: HashMap<u32, Category>,
    /// Name to ID mapping for remapping.
    name_to_id: HashMap<Arc<str>, u32>,
}

impl CategoryMapping {
    /// Create an empty mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a list of (id, name) pairs.
    pub fn from_pairs<I, S>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (u32, S)>,
        S: Into<Arc<str>>,
    {
        let mut mapping = Self::new();
        for (id, name) in pairs {
            mapping.insert(id, name);
        }
        mapping
    }

    /// Insert a category.
    pub fn insert(&mut self, id: u32, name: impl Into<Arc<str>>) -> Option<Category> {
        let name: Arc<str> = name.into();
        self.name_to_id.insert(Arc::clone(&name), id);
        let category = Category { id, name };
        self.by_id.insert(id, category)
    }

    /// Get category by ID.
    #[inline]
    pub fn get(&self, id: u32) -> Option<&Category> {
        self.by_id.get(&id)
    }

    /// Get category name by ID.
    #[inline]
    pub fn get_name(&self, id: u32) -> Option<&str> {
        self.by_id.get(&id).map(|c| c.name())
    }

    /// Get ID by name.
    #[inline]
    pub fn get_id(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Get number of categories.
    #[inline]
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Iterate over categories.
    pub fn iter(&self) -> impl Iterator<Item = &Category> {
        self.by_id.values()
    }

    /// Create a remapping from one set of IDs to another.
    ///
    /// Given a mapping like {"car": 3, "person": 1}, creates a function
    /// that transforms detections to use the new IDs.
    pub fn create_remapper(&self, remap: &HashMap<String, u32>) -> CategoryRemapper {
        let mut id_remap = HashMap::with_capacity(remap.len());

        for (name, new_id) in remap {
            if let Some(&old_id) = self.name_to_id.get(name.as_str()) {
                id_remap.insert(old_id, *new_id);
            }
        }

        CategoryRemapper { id_remap }
    }
}

/// Remaps category IDs in detections.
#[derive(Debug, Clone, Default)]
pub struct CategoryRemapper {
    id_remap: HashMap<u32, u32>,
}

impl CategoryRemapper {
    /// Create an empty remapper (no-op).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a direct ID mapping.
    pub fn from_id_map(id_remap: HashMap<u32, u32>) -> Self {
        Self { id_remap }
    }

    /// Remap a single detection in place.
    #[inline]
    pub fn remap_detection(&self, detection: &mut Detection) {
        if let Some(&new_id) = self.id_remap.get(&detection.class_id) {
            detection.class_id = new_id;
        }
    }

    /// Remap a list of detections in place.
    pub fn remap_all(&self, detections: &mut [Detection]) {
        if self.id_remap.is_empty() {
            return;
        }
        for det in detections {
            self.remap_detection(det);
        }
    }

    /// Check if remapper is a no-op.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id_remap.is_empty()
    }
}

// ============================================================================
// Model Configuration
// ============================================================================

/// Configuration for detection models.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to model weights file.
    pub model_path: Option<String>,
    /// Path to model config file (for some frameworks).
    pub config_path: Option<String>,
    /// Execution device.
    pub device: Device,
    /// Confidence threshold for filtering detections.
    pub confidence_threshold: f32,
    /// Mask threshold for instance segmentation (0.0-1.0).
    pub mask_threshold: f32,
    /// Input image size (for models that require fixed input).
    pub image_size: Option<u32>,
    /// Category ID to name mapping.
    pub category_mapping: CategoryMapping,
    /// Whether to load model immediately on construction.
    pub load_at_init: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            config_path: None,
            device: Device::Auto,
            confidence_threshold: 0.3,
            mask_threshold: 0.5,
            image_size: None,
            category_mapping: CategoryMapping::new(),
            load_at_init: true,
        }
    }
}

impl ModelConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for configuration.
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }
}

/// Builder for ModelConfig.
#[derive(Debug, Clone, Default)]
pub struct ModelConfigBuilder {
    config: ModelConfig,
}

impl ModelConfigBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model path.
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model_path = Some(path.into());
        self
    }

    /// Set the config path.
    pub fn config_path(mut self, path: impl Into<String>) -> Self {
        self.config.config_path = Some(path.into());
        self
    }

    /// Set the device.
    pub fn device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    /// Set confidence threshold.
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set mask threshold.
    pub fn mask_threshold(mut self, threshold: f32) -> Self {
        self.config.mask_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set input image size.
    pub fn image_size(mut self, size: u32) -> Self {
        self.config.image_size = Some(size);
        self
    }

    /// Set category mapping.
    pub fn category_mapping(mut self, mapping: CategoryMapping) -> Self {
        self.config.category_mapping = mapping;
        self
    }

    /// Add a category.
    pub fn add_category(mut self, id: u32, name: impl Into<Arc<str>>) -> Self {
        self.config.category_mapping.insert(id, name);
        self
    }

    /// Set whether to load at init.
    pub fn load_at_init(mut self, load: bool) -> Self {
        self.config.load_at_init = load;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ModelConfig {
        self.config
    }
}

// ============================================================================
// Detection Model Trait
// ============================================================================

/// Trait for detection models.
///
/// This is the core abstraction for integrating detection models with SAHI.
/// Implement this trait to use your model with sliced inference.
///
/// # Lifecycle
///
/// 1. Create model with configuration
/// 2. Call `load()` to initialize (or set `load_at_init: true`)
/// 3. Call `predict()` for inference
/// 4. Optionally call `unload()` to free resources
///
/// # Example
///
/// ```rust,ignore
/// struct OnnxModel {
///     session: Option<ort::Session>,
///     config: ModelConfig,
/// }
///
/// impl DetectionModel for OnnxModel {
///     fn config(&self) -> &ModelConfig {
///         &self.config
///     }
///
///     fn load(&mut self) -> Result<()> {
///         // Initialize ONNX runtime session
///         self.session = Some(ort::Session::builder()?.with_model_from_file(
///             self.config.model_path.as_ref().unwrap()
///         )?);
///         Ok(())
///     }
///
///     fn predict(&self, image: &ImageData) -> Result<Vec<Detection>> {
///         let session = self.session.as_ref()
///             .ok_or_else(|| Error::inference("model not loaded"))?;
///         // Run inference...
///         Ok(detections)
///     }
/// }
/// ```
pub trait DetectionModel: Send + Sync {
    /// Get the model configuration.
    fn config(&self) -> &ModelConfig;

    /// Load the model weights and initialize for inference.
    ///
    /// Called once before any predictions. May be called again after `unload()`.
    fn load(&mut self) -> Result<()>;

    /// Check if the model is loaded and ready for inference.
    fn is_loaded(&self) -> bool;

    /// Unload the model and free resources.
    ///
    /// After unloading, `load()` must be called again before `predict()`.
    fn unload(&mut self);

    /// Run inference on a single image.
    ///
    /// Returns detections with coordinates in pixel space relative to the input image.
    /// Detections should already be filtered by `confidence_threshold`.
    fn predict(&self, image: &ImageData) -> Result<Vec<Detection>>;

    /// Run batch inference on multiple images.
    ///
    /// Default implementation calls `predict()` for each image.
    /// Override for efficient batched GPU inference.
    fn predict_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<Detection>>> {
        images.iter().map(|img| self.predict(img)).collect()
    }

    /// Get the model's category mapping.
    #[inline]
    fn categories(&self) -> &CategoryMapping {
        &self.config().category_mapping
    }

    /// Get the confidence threshold.
    #[inline]
    fn confidence_threshold(&self) -> f32 {
        self.config().confidence_threshold
    }

    /// Get the device.
    #[inline]
    fn device(&self) -> Device {
        self.config().device
    }
}

// ============================================================================
// Model Wrapper (adapts DetectionModel to InferenceCallback)
// ============================================================================

/// Adapts a DetectionModel to the InferenceCallback trait.
///
/// This allows using any DetectionModel with the SAHI pipeline.
pub struct ModelCallback<M: DetectionModel> {
    model: M,
}

impl<M: DetectionModel> ModelCallback<M> {
    /// Create a new callback wrapper.
    pub fn new(model: M) -> Self {
        Self { model }
    }

    /// Get a reference to the underlying model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get a mutable reference to the underlying model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Consume and return the underlying model.
    pub fn into_inner(self) -> M {
        self.model
    }
}

impl<M: DetectionModel> crate::inference::InferenceCallback for ModelCallback<M> {
    fn infer(&self, image: &ImageData) -> Result<Vec<Detection>> {
        self.model.predict(image)
    }

    fn infer_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<Detection>>> {
        self.model.predict_batch(images)
    }
}

// ============================================================================
// Prediction Buffer (memory optimization)
// ============================================================================

/// Reusable buffer for collecting predictions.
///
/// Reduces allocations when processing many slices by reusing vectors.
#[derive(Debug, Default)]
pub struct PredictionBuffer {
    /// Pre-allocated detection storage.
    detections: Vec<Detection>,
    /// Estimated capacity based on previous predictions.
    estimated_capacity: usize,
}

impl PredictionBuffer {
    /// Create a new buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            detections: Vec::with_capacity(capacity),
            estimated_capacity: capacity,
        }
    }

    /// Take the detections, leaving an empty buffer.
    ///
    /// The buffer retains its capacity estimate for future use.
    pub fn take(&mut self) -> Vec<Detection> {
        // Update estimate based on what we collected
        if !self.detections.is_empty() {
            self.estimated_capacity = self.estimated_capacity.max(self.detections.len());
        }
        std::mem::take(&mut self.detections)
    }

    /// Clear the buffer for reuse.
    pub fn clear(&mut self) {
        self.detections.clear();
    }

    /// Reserve capacity based on estimate.
    pub fn reserve(&mut self) {
        if self.detections.capacity() < self.estimated_capacity {
            self.detections.reserve(self.estimated_capacity - self.detections.len());
        }
    }

    /// Push a detection.
    #[inline]
    pub fn push(&mut self, detection: Detection) {
        self.detections.push(detection);
    }

    /// Extend with detections.
    pub fn extend(&mut self, detections: impl IntoIterator<Item = Detection>) {
        self.detections.extend(detections);
    }

    /// Get current length.
    #[inline]
    pub fn len(&self) -> usize {
        self.detections.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.detections.is_empty()
    }
}

// ============================================================================
// Shift Amount (coordinate translation helper)
// ============================================================================

/// Represents a coordinate shift for translating predictions.
///
/// Used when mapping predictions from slice coordinates to full image coordinates.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShiftAmount {
    /// X offset in pixels.
    pub x: f32,
    /// Y offset in pixels.
    pub y: f32,
}

impl ShiftAmount {
    /// Create a new shift amount.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Zero shift.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Apply shift to a bounding box.
    #[inline]
    pub fn apply_to_bbox(&self, bbox: &BoundingBox) -> BoundingBox {
        bbox.translate(self.x, self.y)
    }

    /// Apply shift to a detection.
    pub fn apply_to_detection(&self, detection: &Detection) -> Detection {
        detection.translate(self.x, self.y)
    }

    /// Apply shift to multiple detections in place.
    pub fn apply_to_all(&self, detections: &mut [Detection]) {
        if self.x == 0.0 && self.y == 0.0 {
            return;
        }
        for det in detections {
            det.bbox = self.apply_to_bbox(&det.bbox);
        }
    }
}

impl From<(f32, f32)> for ShiftAmount {
    fn from((x, y): (f32, f32)) -> Self {
        Self::new(x, y)
    }
}

impl From<(u32, u32)> for ShiftAmount {
    fn from((x, y): (u32, u32)) -> Self {
        Self::new(x as f32, y as f32)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category() {
        let cat = Category::new(0, "person");
        assert_eq!(cat.id, 0);
        assert_eq!(cat.name(), "person");

        // Test cheap cloning
        let cat2 = cat.clone();
        assert!(Arc::ptr_eq(&cat.name, &cat2.name));
    }

    #[test]
    fn test_category_from_static() {
        let cat = Category::from_static(1, "car");
        assert_eq!(cat.id, 1);
        assert_eq!(cat.name(), "car");
    }

    #[test]
    fn test_device_parse() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("cuda:0".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("cuda:1".parse::<Device>().unwrap(), Device::Cuda(1));
        assert_eq!("mps".parse::<Device>().unwrap(), Device::Mps);
        assert_eq!("auto".parse::<Device>().unwrap(), Device::Auto);
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cuda(1).to_string(), "cuda:1");
        assert_eq!(Device::Mps.to_string(), "mps");
    }

    #[test]
    fn test_category_mapping() {
        let mut mapping = CategoryMapping::new();
        mapping.insert(0, "person");
        mapping.insert(1, "car");
        mapping.insert(2, "bicycle");

        assert_eq!(mapping.len(), 3);
        assert_eq!(mapping.get_name(0), Some("person"));
        assert_eq!(mapping.get_id("car"), Some(1));
        assert!(mapping.get(3).is_none());
    }

    #[test]
    fn test_category_remapper() {
        let mapping = CategoryMapping::from_pairs([(0, "person"), (1, "car"), (2, "bicycle")]);

        let remap: HashMap<String, u32> =
            [("person".to_string(), 10), ("car".to_string(), 20)].into();

        let remapper = mapping.create_remapper(&remap);

        let mut det = Detection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0), 0, 0.9, None);
        remapper.remap_detection(&mut det);
        assert_eq!(det.class_id, 10);

        let mut det2 = Detection::new(BoundingBox::new(0.0, 0.0, 10.0, 10.0), 2, 0.8, None);
        remapper.remap_detection(&mut det2);
        assert_eq!(det2.class_id, 2); // Not in remap, unchanged
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::builder()
            .model_path("/path/to/model.onnx")
            .device(Device::Cuda(0))
            .confidence_threshold(0.5)
            .image_size(640)
            .add_category(0, "person")
            .add_category(1, "car")
            .build();

        assert_eq!(config.model_path.as_deref(), Some("/path/to/model.onnx"));
        assert_eq!(config.device, Device::Cuda(0));
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.image_size, Some(640));
        assert_eq!(config.category_mapping.len(), 2);
    }

    #[test]
    fn test_shift_amount() {
        let shift = ShiftAmount::new(100.0, 200.0);
        let bbox = BoundingBox::new(10.0, 20.0, 50.0, 50.0);
        let shifted = shift.apply_to_bbox(&bbox);

        assert_eq!(shifted.x, 110.0);
        assert_eq!(shifted.y, 220.0);
        assert_eq!(shifted.width, 50.0);
        assert_eq!(shifted.height, 50.0);
    }

    #[test]
    fn test_prediction_buffer() {
        let mut buffer = PredictionBuffer::with_capacity(10);
        buffer.push(Detection::new(
            BoundingBox::new(0.0, 0.0, 10.0, 10.0),
            0,
            0.9,
            None,
        ));
        buffer.push(Detection::new(
            BoundingBox::new(20.0, 20.0, 10.0, 10.0),
            1,
            0.8,
            None,
        ));

        assert_eq!(buffer.len(), 2);

        let detections = buffer.take();
        assert_eq!(detections.len(), 2);
        assert!(buffer.is_empty());
    }
}
