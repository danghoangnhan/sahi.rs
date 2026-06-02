# Milestone: Usable instance segmentation — design

- **Date:** 2026-06-02
- **Status:** Proposed (awaiting review)
- **Type:** Epic / milestone (4 stacked sub-projects)
- **Builds on:** #9 (instance segmentation pipeline: 9a mask pipeline, 9b mask-aware merging, 9c Python bindings) and #7 (contour tracing in `bool_mask_to_polygons`).

## 1. Context

The #9 epic wired instance segmentation end-to-end: `MaskedDetection`, `SegmentationCallback`,
`Sahi::predict_instances` (slice → infer → rebase masks to image space → stitch), mask-aware
NMM/GREEDYNMM, and PyO3 bindings. But three gaps keep it from being usable out of the box:

1. **No built-in segmenter.** The only built-in model is the detection-only `YOLOv8Detector`.
   To segment, a user must hand-write a `SegmentationCallback` (Rust) or a Python callback.
2. **Mask merging is geometrically wrong on overlap.** `union_masks` (`src/segmentation.rs:263`)
   concatenates polygon sets, so `Mask::area()` (shoelace) double-counts overlapping regions and the
   polygon list grows unbounded across merges. This was explicitly deferred in the 9b spec.
3. **The mask pipeline bypasses acceleration.** `predict_instances` (`src/lib.rs:141`) extracts and
   infers slices sequentially and never touches the `Backend` trait, so neither the `parallel`
   (rayon) path nor CUDA slice extraction — both delivered for detection — apply to segmentation.

## 2. Goal and success criteria

**Goal:** a user can run sliced instance segmentation with a built-in model, from Rust and Python,
getting geometrically-correct merged masks accelerated by the configured backend — with no
hand-written callback.

**Success criteria:**
- A built-in `YOLOv8SegDetector` implements `SegmentationCallback` and drops directly into
  `Sahi::predict_instances`, returning image-space `MaskedDetection`s.
- Merged masks (NMM/GREEDYNMM) are correct: `area ≈ union area` (not the sum of overlapping parts),
  and a merged mask's polygon count stays bounded.
- `predict_instances` runs through the `Backend` trait, so the `parallel` and `cuda` features
  accelerate mask slice-extraction exactly as they do for detection; the GIL-safe sequential path
  remains the default.
- The built-in segmenter is reachable from Python.
- Every change is gated on `cargo test`, `cargo clippy --features "python,models,onnx" --all-targets
  -- -D warnings`, `cargo fmt --all -- --check`, and a green PR CI run.

## 3. Non-goals (deferred to a later "reference-parity hardening" milestone)

These are real, audit-confirmed issues but are out of scope here to keep the milestone focused:

- The GREEDYNMM **growing-box** matching divergence from upstream SAHI (candidates matched against
  the enlarging union instead of the fixed anchor). It lives in **both** `src/postprocess.rs:313`
  and `src/segmentation.rs:231`. This milestone must **not** make it worse, but does not fix it.
- Unifying the duplicated NMS/NMM/GREEDYNMM control flow between `postprocess.rs` and
  `segmentation.rs`.
- Contour-tracer Jacob's-criterion fix (`src/annotation/mask.rs:535`).
- CUDA kernel source-bounds check (`src/backend/cuda.rs:534`) and exercising the GPU path in CI.
- Panic-proofing (`partial_cmp().unwrap()` on NaN, rayon `build().expect()`, unbounded `mask_array`).

These will be filed as their own issues so they are not lost.

## 4. Sub-projects (4 stacked PRs)

Each PR is based on the previous one's branch. Each lands test-first (failing test → implementation),
updates docs honestly (per the #12 norm), and merges only on green CI. Daniel merges each PR himself.

### PR1 — Mask-union dedup (rasterize → OR → re-trace)

**What:** replace polygon-set concatenation in `union_masks` with a rasterize/OR/re-trace union.

**Design:**
- For the group's masks, union each member's `to_bool_mask()` (OR) into one `width*height` buffer at
  the anchor's `full_shape()`, then rebuild one polygon set via `bool_mask_to_polygons` (the #7
  tracer) and wrap in `Mask::new(polys, shape, None)`.
- Keep the anchor's `full_shape`. Guard the empty/`None` case as today.
- This touches only `union_masks`; `merge_group`, `seg_nmm`, `seg_greedy_nmm` are unchanged.

**Key decisions:**
- Union at the anchor's `full_shape`. Inside `predict_instances` every mask is already rebased to the
  image `full_shape` before stitching, so members share dimensions; the helper still tolerates a
  group whose anchor carries no mask by unioning over the present masks' shared shape.

**Test strategy (TDD):**
- Two overlapping unit squares → merged `Mask::area()` ≈ union area, strictly less than the sum.
- `to_bool_mask(merged)` equals the elementwise OR of the inputs.
- Repeated merges keep the polygon count bounded (no unbounded growth).
- Disjoint masks → both regions preserved; single mask → unchanged; empty group → `None`.

**Acceptance:** `test_nmm_unions_masks` / `test_greedy_nmm_chain_merges_to_one` still pass; new area
and OR-equivalence tests pass.

### PR2 — Built-in `YOLOv8SegDetector`

**What:** a built-in YOLOv8-seg ONNX segmenter that implements `SegmentationCallback`.

**Design:**
- New module `src/onnx/yolov8/seg.rs` with `YOLOv8SegDetector` (config mirrors `YOLOv8Config`, adds
  `num_masks` defaulting to 32 and a mask `threshold` defaulting to 0.5) and a
  `YOLOv8SegProcessor::process_seg_output(out0, out0_shape, proto, proto_shape, &LetterboxInfo)
  -> Vec<MaskedDetection>`.
- YOLOv8-seg has two outputs: `out0` `(1, 4 + num_classes + num_masks, N)` (or transposed) and the
  prototype tensor `proto` `(1, num_masks, mh, mw)`. Box + class decode reuses the existing
  `decode_box` logic; the trailing `num_masks` values are the mask coefficients.
- Mask decode per surviving box (ultralytics `process_mask` semantics):
  `coeffs(num_masks) · proto(num_masks × mh·mw) → reshape (mh, mw) → sigmoid → crop to the box (in
  proto space) → resize to the box's slice-space size → Mask::from_float_mask(.., threshold, ..)`.
  Masks are emitted in **slice-relative** coordinates; `predict_instances` already rebases slice →
  image, so the segmenter needs no image-level knowledge.
- Box NMS first (reuse `nms`/`apply_nms`), then decode masks only for survivors.
- Extend `detect_output_format` (or add a seg-aware variant) to recognize the seg box-dim
  `4 + num_classes + num_masks` for standard/transposed auto-detection.
- Wire `predict`/`predict_batch` onto the ONNX session exactly like `YOLOv8Detector` (locked session,
  two named outputs), reusing `processor.preprocessor.preprocess` and `LetterboxInfo`.

**Key decisions:**
- Output to the existing polygon-based `Mask` (via `from_float_mask`) rather than a new bitmap type —
  keeps one mask representation across the codebase and reuses the #7 tracer.
- Masks decoded in slice space, not image space — minimal interface, reuses existing rebase.

**Test strategy (TDD):** unit-test the decoder with **synthetic** `out0` + `proto` tensors (no real
`.onnx`), matching how the repo already tests detection decode (`process_output`/
`process_batch_output` are tested with hand-built `Vec<f32>`). Assertions:
- A synthetic box + a proto/coeffs pair that yields a known rectangular mask → decoded
  `MaskedDetection` has the expected bbox and a mask whose rasterization covers the expected region.
- Seg layout auto-detect for standard and transposed shapes.
- Sub-threshold mask → empty/no polygons; empty output → no detections.
- The ONNX-session `predict()` glue gets an `#[ignore]`d integration test gated on a
  `SAHI_TEST_YOLOV8_SEG_MODEL` env var (consistent with detection's `predict()` having no real-model
  test today). See §6 for the test-model decision.

**Acceptance:** decoder unit tests pass; `YOLOv8SegDetector` usable as a `SegmentationCallback` in a
`predict_instances` test driven by a stubbed processor output.

### PR3 — Segmentation through the `Backend` trait

**What:** route `predict_instances` slice extraction/inference through `Backend`, so `parallel` and
`cuda` accelerate the mask pipeline.

**Design:**
- The `Backend` trait is `Detection`-only and adding a generic method would break `dyn` object-safety
  (`BoxedBackend = Box<dyn Backend>`). So add **one object-safe primitive**:
  `fn extract_slices(&self, image: &ImageData, slices: &[Slice]) -> Result<Vec<ImageData>>` — CPU
  implements it with the existing rayon parallel extraction; CUDA with GPU extraction (falling back
  to CPU when the kernel is unavailable, as today).
- Reimplement detection's `process_slices` on top of `extract_slices` (extract, then
  `callback.infer_batch`) to remove duplication, and add masked orchestration that `predict_instances`
  calls: `backend.extract_slices(...)` then `SegmentationCallback::infer_batch(...)`, then the
  existing per-slice translate + `rebase_mask` + `seg_stitch`.
- Preserve current semantics: sequential inference is the default (GIL-safe); parallel inference stays
  opt-in via `CpuBackendConfig` exactly as for detection. `include_full_image` handling is unchanged.

**Key decisions:**
- Extraction is the only thing that must live behind the trait (it is what rayon/CUDA accelerate);
  stitching and rebasing stay in `segmentation.rs`. This contains the trait change to one method.

**Test strategy (TDD):**
- `extract_slices` returns the same slice images as the current direct `extract_slice` loop (CPU).
- `predict_instances` results are identical between the sequential and `parallel`-feature paths.
- A batched `SegmentationCallback` is invoked via `infer_batch` (counted), not per-slice `infer`.
- Existing `predict_instances` tests still pass unchanged.

**Acceptance:** seg path produces identical results across backends/feature flags; CUDA build still
compiles (`--features cuda`) with the new `extract_slices` impl.

### PR4 — Python bindings for the built-in segmenter

**What:** expose `YOLOv8SegDetector` to Python so `predict_instances` works with the built-in model.

**Design:**
- Add a PyO3 wrapper (e.g. `PyYOLOv8SegDetector`) constructible from a model path + config, and a
  `Sahi.predict_instances_yolov8(...)` entry (or accept the built-in segmenter object) that runs the
  built-in segmenter instead of a Python callback. Reuse the existing `PyMaskedDetection` and its
  `mask_array` rasterizer for results.
- Keep the existing Python-callback `predict_instances` path working.

**Test strategy:** pytest (`tests/python/test_segmentation.py`) exercises the binding surface —
construct the segmenter, `PyMaskedDetection`/`mask_array` round-trip, and error paths (bad dtype/shape).
The real-model inference path is skipped/env-gated like PR2. Built via `maturin develop --features
python` in the `uv` `.venv` (`python -m ensurepip` + `pip install pytest`).

**Acceptance:** new pytest cases pass; existing Python tests unchanged.

## 5. Sequencing and stacking

```
main
 └─ PR1 dedup        (isolated correctness fix; makes later merges correct)
     └─ PR2 model    (the headline feature; decoder TDD'd on synthetic tensors)
         └─ PR3 backend  (accelerate the seg pipeline; trait change contained to extract_slices)
             └─ PR4 bindings  (expose the built-in segmenter to Python)
```

Rationale: PR1 has no dependencies and fixes a known bug, so it goes first and ensures any masks
later merged are correct. PR4 depends on PR2's Rust API. PR3 is code-independent of PR2 but more
meaningful once a real built-in segmenter exists to accelerate. Each PR is independently reviewable.

## 6. Cross-cutting test strategy and the test-model question

The repo currently tests all decode logic with synthetic tensors and has **no** `.onnx` fixtures or
real-model tests; detection's `predict()` session glue is effectively untested locally. This milestone
follows that convention: the hard logic (mask decode, union, orchestration) is fully unit-tested with
synthetic data, and the thin ONNX-session glue is covered by `#[ignore]`d, env-gated integration tests.

**Decision (default):** keep real-model end-to-end coverage **out of scope / env-gated**. A tiny
synthetic `yolov8-seg`-shaped `.onnx` fixture (or a CI download of `yolov8n-seg.onnx`) for true
end-to-end CI coverage is an **optional** add-on that can be folded into PR2 if desired; not assumed.

## 7. Risks and mitigations

- **Mask coordinate mapping (proto → box → slice) is the trickiest part.** Mitigated by decoding in
  slice space (reusing the existing rebase) and TDD-ing the decoder on synthetic tensors before any
  session wiring.
- **PR3's trait change ripples to the CUDA backend.** Mitigated by adding a single object-safe method
  and keeping orchestration/stitching outside the trait.
- **No real seg model in CI.** Mitigated by env-gated integration tests; optional fixture per §6.

## 8. Decisions recorded (defaults pending review)

- (a) Synthetic ONNX fixture: **out of scope / env-gated** (recommended; see §6). Flip to in-scope for
  PR2 on request.
- (b) PR order: **dedup → model → backend → bindings** (recommended; see §5).

## 9. Milestone and issue plan (created after spec approval)

- GitHub **milestone**: "Usable instance segmentation".
- One **issue per PR** (PR1–PR4) with the acceptance criteria above, assigned to the milestone.
- A **parent tracking issue** linking the four and listing the deferred non-goals (§3) to be filed
  separately.
