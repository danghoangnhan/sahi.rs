# Design: Segmentation 9b — mask-aware merging

- **Date:** 2026-06-01
- **Issue:** [#9](https://github.com/danghoangnhan/sahi.rs/issues/9) (segmentation epic), sub-project **9b**
- **Builds on:** 9a (PR #18 / branch `feat/segmentation-9a`). This branch is stacked on 9a.
- **Status:** Approved (brainstorming → spec)

## Goal

Make `seg_stitch` honor the configured `postprocess_type` for masked detections: **NMM** and **GREEDYNMM** union the masks of merged detections instead of keeping only the top one. **NMS** is unchanged (keep top, own mask).

## Non-goals

- Bitmap-OR mask dedup / re-polygonization (depends on #7's corrected tracer; deferred).
- Mask-based IoU matching (matching stays bbox-based).
- Python bindings (9c).

## Mask union — polygon-set concatenation

`union_masks(&[Mask]) -> Option<Mask>`: concatenate each member's `to_coco_segmentation()` into one multi-polygon `Mask`, keeping the first member's `full_shape`; `None` if there are no masks.

The rasterized union (`to_bool_mask`) is exact (OR of the polygon fills). Documented trade-offs: `Mask::area()` (shoelace sum) over-counts overlapping regions, and the polygon list grows with merges. A dedup pass (rasterize → OR → re-trace via #7's tracer) is deferred. This approach is **independent of #7**, which matters because 9b is stacked on 9a (off `main`, where the old tracer is still present).

## `seg_stitch` dispatch

Shared prelude: drop detections below `config.confidence_threshold`, then sort by confidence descending. Dispatch on `config.postprocess_type`:

- **NMS** — keep top; suppress later same-class detections whose bbox match score (`iou`/`ios` per `match_metric`) exceeds `match_threshold`; survivors keep their own mask. (9a behavior.)
- **NMM** — for each unused anchor `i`, group every unused `j > i` (class-aware) whose match score *against `i`* exceeds `match_threshold`; mark the group used. Merged result: `detection` = union of the group's bboxes (`BoundingBox::union_box`) with the **max** confidence and the class/name of the anchor (`group[0]`, highest after sorting); `mask` = `union_masks` of the group's present masks.
- **GREEDYNMM** — like NMM, but the comparison box grows: start from anchor `i`, and for each matching `j` expand the working bbox via `union_box` before testing the next. Merged result built the same way (max confidence from the anchor, unioned mask).

Reuses public `BoundingBox::iou` / `ios` / `union_box`. The NMM/GREEDYNMM grouping mirrors `postprocess.rs` but operates on `MaskedDetection` and carries masks; the bbox `Postprocessor` is untouched (modest, isolated duplication).

## `predict_instances`

Unchanged — it already passes `self.postprocessor.config()` to `seg_stitch`, so it picks up merging automatically.

## Test plan (TDD; mock data; default build)

1. **NMM unions masks** — two overlapping same-class masked detections (bbox IoU > threshold), config NMM → 1 result; its mask's `to_bool_mask` covers *both* source regions; confidence = max; bbox = union.
2. **GREEDYNMM chain** — three boxes overlapping in a chain merge to 1 with a unioned mask.
3. **None-mask member** — a merged group where only one member has a mask → merged mask = that member's polygons.
4. **`union_masks` unit** — concatenates polygon lists, preserves `full_shape`; empty input → `None`.
5. **Pin the 9a NMS test** — set `PostprocessType::NMS` explicitly on `test_seg_stitch_nms_keeps_top_and_mask` (its default config is GREEDYNMM, which post-9b would merge), keeping it a true NMS regression.
6. **Non-overlapping under NMM** — two disjoint detections → 2 results, masks intact.

## Files

- `src/segmentation.rs`: add `union_masks`, the NMM/GREEDYNMM masked grouping, dispatch in `seg_stitch`, the new tests; pin the existing NMS test.
- No other files change.
