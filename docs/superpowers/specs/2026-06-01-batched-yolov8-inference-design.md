# Design: Batched YOLOv8 inference (#8)

- **Date:** 2026-06-01
- **Issue:** [#8](https://github.com/danghoangnhan/sahi.rs/issues/8)
- **Status:** Approved (brainstorming → spec)

## Problem

`YOLOv8Detector::predict_batch` is a stub — `images.iter().map(|i| self.predict(i)).collect()`. The CPU/CUDA backends call `InferenceCallback::infer_batch` → `predict_batch`, so today every tile is a separate `session.run` and a separate lock of the session `Mutex`. No batching.

## Goal

Run all tiles in a single forward pass: preprocess each, stack into one `(N, C, H, W)` input, run once (locking the session a single time), and split the output per image.

## Non-goals

- The `parallel_inference` + `Mutex` interaction (batching is the real win).
- CUDA-side batched extraction beyond what exists.
- Internal `max_batch_size` chunking (callers chunk; the CUDA backend already does — one run otherwise).
- Bundling a test ONNX model.

## Design

### `YOLOv8Detector::predict_batch(&self, images: &[ImageData]) -> Result<Vec<Vec<Detection>>>` (`model.rs`)

1. `images.is_empty()` → `Ok(Vec::new())`.
2. Lock the session `Mutex` once.
3. For each image: `(tensor, info) = self.processor.preprocessor.preprocess(img)?`; collect `tensors` and `infos`. Letterbox resizes every tile to `input_size²`, so the tensors share `(1, C, H, W)`.
4. `let batch = stack_nchw(&tensors)?` → `(N, C, H, W)`.
5. One `session.run(inputs![input_name => Tensor::from_array(batch.into_dyn())?])`.
6. `try_extract_tensor::<f32>()` → `(shape (N, dim1, dim2), data)`.
7. `self.processor.process_batch_output(data, &shape, &infos)` → `Vec<Vec<Detection>>`.

### `stack_nchw(tensors: &[Array4<f32>]) -> Result<Array4<f32>>` (free fn, `model.rs`)

`ndarray::concatenate(Axis(0), &views)`. Empty input → `Err(invalid_output)`; a shape mismatch surfaces as the ndarray error mapped to `Error::invalid_output`.

### `YOLOv8Processor::process_batch_output(&self, data: &[f32], shape: &[i64], infos: &[LetterboxInfo]) -> Result<Vec<Vec<Detection>>>` (`processor.rs`)

Validate `shape.len() == 3`, `infos.len() == N`, and `data.len() == N · dim1 · dim2`. For each image `i`, slice `data[i·per .. (i+1)·per]` (`per = dim1·dim2`) and call the existing `process_output(sub, &[1, dim1, dim2], &infos[i])` — which auto-detects Standard/Transposed (per #5) and applies that image's letterbox. Returns one `Vec<Detection>` per image.

`process_output` and the single-image `predict` are unchanged.

## Verification

- **Unit (no model):**
  - `stack_nchw` — shape and value placement; empty → `Err`.
  - `process_batch_output` — a synthetic `(2, 6, 1)` output (num_classes = 2): image 0 scores class 1 at one box, image 1 scores class 0 at a different box; identity letterboxes; assert two results, each decoding *its own* slice.
- **Integration:** the `preprocess → stack → run → split` glue is compile-/clippy-verified only — the repo ships no test ONNX model and CI exercises a mock callback, not real YOLOv8 (pre-existing limitation).
- **Gates:** `cargo test --features onnx`, `cargo clippy --features "python,models,onnx" --all-targets -- -D warnings`, `cargo fmt --all -- --check`.

## Files

- `src/onnx/yolov8/model.rs`: real `predict_batch` + `stack_nchw` (+ tests).
- `src/onnx/yolov8/processor.rs`: `process_batch_output` (+ tests).
- No changes to `process_output`, the backends, or `predict`.
