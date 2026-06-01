# Design: Segmentation 9c — Python bindings

- **Date:** 2026-06-01
- **Issue:** [#9](https://github.com/danghoangnhan/sahi.rs/issues/9) (segmentation epic), sub-project **9c**
- **Builds on:** 9a (#18) + 9b (#19). Stacked on `feat/segmentation-9b`.
- **Status:** Approved (brainstorming → spec)

## Goal

Expose the instance-segmentation pipeline to Python: a `MaskedDetection` class and `Sahi.predict_instances(image, callback)`, with **polygon-based** masks (COCO list-of-lists) plus a numpy readout. Independent of #7.

## Non-goals

- numpy mask **input** (numpy → `Mask`) — needs #7's tracer; follow-up after #17.
- Registering the other dormant annotation classes (`ObjectAnnotation`, `AnnotationBoundingBox`, `MaskFormat`, `Polygon`, `RleData`) — separate from the pipeline.
- A built-in YOLOv8-seg model.

## Python surface (feature-gated `#[cfg(feature = "python")]` in `segmentation.rs`)

### `MaskedDetection` pyclass (Rust `PyMaskedDetection`, Python name `MaskedDetection`)

Stored as `{ detection: Detection, polygons: Option<Vec<Vec<f32>>> }` — no Rust `Mask` retained, so `full_shape` is set only where it's known (the adapter / readout).

- `__new__(detection: Detection, mask: Optional[list[list[float]]] = None)`.
- getter `detection -> Detection`.
- getter `mask -> Optional[list[list[float]]]` (COCO polygons).
- method `mask_array(height: int, width: int) -> numpy.ndarray` (bool, shape `(height, width)`): rebuild `Mask::new(polygons, [height, width], None)`, `to_bool_mask()`, reshape to `(H, W)`. Empty/`None` → all-`False`.
- `__repr__`.

### `PySegCallback` adapter (private)

`{ callback: PyObject }` implementing the Rust `SegmentationCallback`, with `unsafe impl Send + Sync` (GIL-bound — mirrors the existing `PyCallback`). `infer(slice_image)`: convert `ImageData` → numpy `(H, W, C)`, call the Python callback, expect `list[MaskedDetection]`, and build each Rust `MaskedDetection { detection, mask: polygons.map(|p| Mask::new(p, [slice_h, slice_w], None)) }` using the slice's dimensions.

### `Sahi.predict_instances` (on `PySahi` in `lib.rs`)

`predict_instances(self, image: numpy.ndarray /* (H,W,C) uint8 */, callback) -> list[MaskedDetection]`. Converts the image to `ImageData`, runs `self.inner.predict_instances(&image_data, &PySegCallback { callback })`, and wraps each result `MaskedDetection` (image coordinates) into a Python `MaskedDetection` with `polygons = mask.map(|m| m.to_coco_segmentation())`.

## Registration

Add `m.add_class::<segmentation::PyMaskedDetection>()?` to `sahi_module` in `lib.rs` (gated on `python`). `PyMaskedDetection` is `pub use`-exported from `segmentation`.

## Error handling

- Python callback errors and wrong return types map to `Error::Inference` (mirrors `PyCallback`).
- A non-`MaskedDetection` list item yields a clear `Inference` error.

## Test plan (TDD; pytest; built with `maturin develop --features python`)

New `tests/python/test_segmentation.py`:

1. `MaskedDetection(det, mask=[[...]])` → `.mask` equals the polygons; `.detection` round-trips (`class_id`/`confidence`/`bbox`).
2. `MaskedDetection(det)` (no mask) → `.mask is None`.
3. `.mask_array(h, w)` → numpy bool array of shape `(h, w)` with the polygon region `True`.
4. `predict_instances(image, callback)` over a 2-slice image (mock callback returns one masked detection per slice) → results are `MaskedDetection`s whose mask polygons are offset into image coordinates.

## Files

- `src/segmentation.rs`: add `#[cfg(feature = "python")] mod python` (`PyMaskedDetection`, `PySegCallback`) + `pub use`.
- `src/lib.rs`: `PySahi::predict_instances` + register `MaskedDetection` in `sahi_module`.
- `tests/python/test_segmentation.py` (new).
