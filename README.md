# sahi.rs

A Rust implementation of [SAHI (Slicing Aided Hyper Inference)](https://github.com/obss/sahi) for improved small object detection through image slicing.

## Architecture

```
src/
├── lib.rs           # Public API, Sahi struct, SahiBuilder
├── detection.rs     # Re-exports Detection/BoundingBox from annotation
├── slicer.rs        # Image slicing with configurable overlap
├── postprocess.rs   # NMS/NMM-based result merging
├── inference.rs     # InferenceCallback trait, ImageData
├── annotation/
│   ├── mod.rs       # Annotation module exports, FullShape
│   ├── bbox.rs      # BoundingBox, AnnotationBoundingBox (COCO/VOC formats)
│   ├── detection.rs # Detection type (bbox, class_id, confidence)
│   ├── mask.rs      # Mask, Polygon, RLE for segmentation
│   └── object.rs    # ObjectAnnotation (bbox + mask + category)
├── model/
│   └── mod.rs       # DetectionModel trait, Category, Device, ModelConfig
├── error.rs         # Error types (thiserror)
└── backend/
    ├── mod.rs       # Backend trait, factory functions
    ├── cpu.rs       # CPU backend (sequential processing)
    └── cuda.rs      # CUDA backend (cudarc, feature-gated)
```

## Build Commands

```bash
cargo check                    # Check compilation (CPU only)
cargo check --features cuda    # Check with CUDA support
cargo test                     # Run all tests
cargo run --example basic      # Run example
```

## Features

- `cuda` - Enable GPU acceleration via cudarc
- `python` - Enable PyO3 Python bindings
- `onnx` - ONNX Runtime support for model inference
- `models` - Includes `python` + `onnx` features
- `models-cuda` - CUDA-accelerated ONNX inference

## Key Patterns

### Callback Pattern for Inference
Models are integrated via the `InferenceCallback` trait:
```rust
let model = callback(|img: &ImageData| -> Result<Vec<Detection>> {
    // Your model inference here
    Ok(vec![])
});
```

### Backend Switching
CPU/GPU backends implement the `Backend` trait. Selection is automatic based on features and availability:
```rust
let sahi = Sahi::builder().cpu().build();   // Force CPU
let sahi = Sahi::builder().cuda().build();  // Force CUDA (requires feature)
```

### Builder Pattern
Use `Sahi::builder()` for configuration:
```rust
Sahi::builder()
    .slice_size(640, 640)
    .overlap(0.2, 0.2)
    .nms_threshold(0.5)
    .confidence_threshold(0.25)
    .build()
```

### DetectionModel Trait
For full model abstraction with lifecycle management:
```rust
impl DetectionModel for MyModel {
    fn config(&self) -> &ModelConfig { &self.config }
    fn load(&mut self) -> Result<()> { /* load weights */ }
    fn is_loaded(&self) -> bool { self.model.is_some() }
    fn unload(&mut self) { self.model = None; }
    fn predict(&self, image: &ImageData) -> Result<Vec<Detection>> {
        // Run inference
    }
}

// Use with SAHI via ModelCallback adapter
let callback = ModelCallback::new(my_model);
sahi.predict(&image, &callback)?;
```

### Model Configuration
```rust
let config = ModelConfig::builder()
    .model_path("/path/to/model.onnx")
    .device(Device::Cuda(0))
    .confidence_threshold(0.5)
    .add_category(0, "person")
    .add_category(1, "car")
    .build();
```

### ObjectAnnotation for Instance Segmentation
```rust
// Create from COCO bbox format
let ann = ObjectAnnotation::from_coco_bbox(
    [100.0, 100.0, 50.0, 50.0],  // x, y, w, h
    0,                            // category_id
    Some("person"),               // category_name
    [480, 640],                   // full_shape [h, w]
    Some([100.0, 200.0]),         // shift_amount
);

// Shift to full image coordinates
let shifted = ann.get_shifted();

// Convert to Detection
let detection = shifted.to_detection(0.95);
```

### BoundingBox Format Conversions
```rust
let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);  // x, y, w, h

// Format conversions
let coco = bbox.to_coco();  // [x, y, w, h]
let voc = bbox.to_voc();    // [xmin, ymin, xmax, ymax]

// Extended operations
let expanded = bbox.get_expanded(0.1, Some(640.0), Some(480.0));
let clipped = bbox.clip(640.0, 480.0);
let ios = bbox.ios(&other);  // Intersection over Smaller
```

## Code Conventions

- All public types derive `Debug`, `Clone` where appropriate
- Use `thiserror` for error types
- Feature-gate optional dependencies with `#[cfg(feature = "...")]`
- Tests are in `#[cfg(test)] mod tests` at bottom of each file
- Coordinates are `f32`, dimensions are `u32`

## Important Types

| Type | Purpose |
|------|---------|
| `Sahi` | Main entry point, orchestrates slicing + inference + postprocessing |
| `Slice` | Defines a region (x, y, width, height, index) |
| `Detection` | Result with bbox, class_id, confidence, optional class_name |
| `BoundingBox` | Rectangle with IoU/IoS, COCO/VOC format conversion |
| `AnnotationBoundingBox` | BoundingBox with shift tracking for coordinate translation |
| `ObjectAnnotation` | Full annotation: bbox + optional mask + category |
| `Mask` | Segmentation mask (polygon/RLE formats) |
| `Polygon` | Single polygon contour [x1,y1,x2,y2,...] |
| `FullShape` | Image dimensions [height, width] |
| `ImageData` | Raw pixel data (RGB, HWC format) |
| `InferenceCallback` | Trait for model integration (closure-based) |
| `DetectionModel` | Trait for full model abstraction (lifecycle, config, batch) |
| `ModelConfig` | Model configuration (device, thresholds, categories) |
| `Category` | Detection category with ID and name (`Arc<str>` optimized) |
| `Device` | Execution device enum (Cpu, Cuda, Mps, Auto) |
| `CategoryMapping` | Bidirectional ID↔name lookup |
| `Postprocessor` | Merges slice detections via NMS/NMM algorithms |
| `PostprocessConfig` | Configuration for postprocessing (thresholds, algorithm type) |
| `Backend` | Trait for CPU/GPU execution backends |

## NMS Algorithm

Located in `postprocess.rs`:
1. Sort detections by confidence (descending)
2. For each detection, suppress lower-confidence overlapping boxes
3. Class-aware by default (only suppress within same class)
4. Configurable IoU threshold (default 0.5)

## Python Development

### Local Development with maturin

```bash
# Install development dependencies
pip install maturin[patchelf] pytest pillow numpy ruff

# Build and install in development mode (editable)
maturin develop --features python,models

# Build with CUDA support (requires CUDA toolkit)
maturin develop --features python,models-cuda

# Run Python tests
pytest tests/python/ -v

# Auto-fix linting issues
ruff check python/ --fix
ruff format python/
```

### Building Wheels Locally

```bash
# Build wheel for current platform
maturin build --release --features python,models -o dist

# Build wheel with CUDA support
maturin build --release --features python,models-cuda -o dist

# Install built wheel
pip install dist/sahi_rs-*.whl
```

### Testing the Package

```bash
# Quick smoke test
python -c "from sahi_rs import Sahi, BoundingBox, Detection; print('OK')"

# List available exports
python -c "import sahi_rs; print(dir(sahi_rs))"
```

### Python Usage Example

```python
import numpy as np
from sahi_rs import Sahi, BoundingBox, Detection

# Create SAHI instance
sahi = Sahi(slice_width=640, slice_height=640)

# Define your detector callback
def my_detector(image):
    # Your model inference here
    return [Detection(
        bbox=BoundingBox(x=100.0, y=100.0, width=50.0, height=50.0),
        class_id=0,
        confidence=0.9,
        class_name="person"
    )]

# Run prediction
image = np.zeros((1920, 1080, 3), dtype=np.uint8)
results = sahi.predict(image, my_detector)
```
