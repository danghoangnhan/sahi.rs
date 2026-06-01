# Design: Segmentation 9a — mask-carrying pipeline

- **Date:** 2026-06-01
- **Issue:** [#9](https://github.com/danghoangnhan/sahi.rs/issues/9) (segmentation epic), sub-project **9a**
- **Relationship to #7 / PR #17:** conceptual. 9a uses the `Mask` type (already on `main`); it does not require the contour fix at the code level (tests use explicit polygons via `Mask::new`).
- **Status:** Approved (brainstorming → spec)

## Goal

Add an end-to-end instance-segmentation path that carries per-detection masks from a slice-level callback through to image-space results, reusing the existing slicer and box-matching. The existing bbox `Detection` / `predict` / `Backend` / Python paths are untouched (zero risk).

## Non-goals (later sub-projects)

- Mask-aware merging (mask **union** in NMM/GREEDYNMM) → 9b.
- Python bindings + annotation class registration → 9c.
- `Backend` (CPU-parallel / CUDA) extraction for the seg path.
- A built-in YOLOv8-seg model.

## Result type — `src/segmentation.rs` (new module)

```rust
#[derive(Debug, Clone)]
pub struct MaskedDetection {
    pub detection: Detection,
    pub mask: Option<Mask>,
}
```

- Score, class id/name, and bbox live in `detection`, so NMS has a score to sort/threshold on. `ObjectAnnotation` (which has no score) stays the annotation/COCO-export type; a conversion helper can be added later (out of 9a scope).
- `mask: Option<Mask>` — `None` for detection-only results, which flow through unchanged.

Helper:

```rust
fn rebase_mask(mask: &Mask, dx: f32, dy: f32, image_w: u32, image_h: u32) -> Mask
```

Shifts every polygon by `(dx, dy)` (reusing `Polygon::shift`) and rebuilds the `Mask` with `full_shape = (image_h, image_w)`, shift `0`. A dedicated rebase is required because the existing `Mask::get_shifted` clips to the *slice's* `full_shape` and so cannot move a mask into image space.

## Callback (separate from the bbox `InferenceCallback`)

```rust
pub trait SegmentationCallback: Send + Sync {
    fn infer(&self, image: &ImageData) -> Result<Vec<MaskedDetection>>;
    fn infer_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<MaskedDetection>>> {
        images.iter().map(|i| self.infer(i)).collect()
    }
}
```

- Returns slice-relative results; each `Mask` is built with `full_shape` = the slice's `(h, w)`.
- `FnSegCallback<F>` + `seg_callback(f)` closure adapter, mirroring the existing `callback()`.

## Entry point — `Sahi::predict_instances`

```rust
pub fn predict_instances(&self, image: &ImageData, callback: &dyn SegmentationCallback)
    -> Result<Vec<MaskedDetection>>
```

Flow:

1. `slices = self.slicer.slice(image.width, image.height)`.
2. Per slice: `slice_img = image.extract_slice(...)`; `preds = callback.infer(&slice_img)?`.
3. Rebase each pred to image space: `detection = detection.translate(slice.x, slice.y)`; `mask = mask.map(|m| rebase_mask(&m, slice.x, slice.y, W, H))`.
4. If `include_full_image`, also run the callback on the full image (shift `0`).
5. `seg_stitch(all)` (input already in image coords).

Extraction is **CPU-sequential** (bypasses `Backend`); noted as a future enhancement.

## `seg_stitch` — 9a: NMS-style, keep top mask

`seg_stitch(items, config)` takes the `PostprocessConfig` explicitly (`predict_instances` passes `self.postprocessor.config()`), using `confidence_threshold`, `match_threshold`, `match_metric`, and `class_aware`:

1. Drop `MaskedDetection`s with `detection.confidence < confidence_threshold`.
2. Sort by `detection.confidence` descending.
3. Greedy NMS: for each surviving detection, suppress later detections whose bbox match score (`BoundingBox::iou` or `ios`, per `match_metric`) exceeds `match_threshold`; `class_aware` gates the same-class check.
4. Survivors keep their own `mask`.

No mask union and no duplication of the bbox matcher beyond the small NMS loop (it calls the already-public `BoundingBox::iou`/`ios`). Mask union for NMM/GREEDYNMM is **9b**; in 9a the seg path is NMS regardless of the configured `postprocess_type` (documented on the method).

## Error handling

- Callback errors propagate via `Result`.
- A `None` mask stays `None`; an empty polygon list yields a `Mask` with no segmentation (area 0), which is acceptable.

## Test plan (TDD; mock callback; default build, no feature flags, no real model)

1. **Image-space masks** — a ≥2-slice image; the mock returns one `MaskedDetection` per slice with a known slice-local polygon; assert each returned mask's vertices are offset by the slice origin and `full_shape == (W, H)`.
2. **NMS across slices** — overlapping detections in adjacent slices collapse to one; the survivor retains its mask.
3. **Confidence filter** — a low-score masked detection is dropped.
4. **Detection-only passthrough** — `mask: None` survives unchanged.
5. **Unit: `rebase_mask`** — polygon shifted by `(dx, dy)`, `full_shape` reset to image dims.

## File / layout

- New `src/segmentation.rs`: `MaskedDetection`, `SegmentationCallback`, `FnSegCallback` + `seg_callback`, `rebase_mask`, `seg_stitch`.
- `Sahi::predict_instances` added in `src/lib.rs`; `pub mod segmentation;` + re-exports (`MaskedDetection`, `SegmentationCallback`, `seg_callback`).
- No changes to `detection.rs`, `backend/`, `inference.rs`, or the Python module.
