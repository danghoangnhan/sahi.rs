# Unify NMS/NMM/GREEDYNMM grouping — design

- **Date:** 2026-06-02
- **Status:** Approved
- **Issue:** #36 (milestone: Reference-parity hardening, #2)
- **Follows:** #35 (which corrected the NMM/GREEDYNMM semantics in both copies)

## Problem

The match/group/merge control flow is duplicated across the bbox path
(`Postprocessor::nms`/`nmm`/`greedy_nmm` in `src/postprocess.rs`) and the mask path
(`seg_nms`/`seg_nmm`/`seg_greedy_nmm` in `src/segmentation.rs`). #35 had to fix the same
semantics in both, and any future fix risks the two paths drifting. There is also a duplicated IoS:
`postprocess.rs` has a free `ios(a, b)` fn that is byte-for-byte identical to `BoundingBox::ios`
(the segmentation path already uses the method).

## Change (pure refactor — no behavior change)

Add a crate-level `src/combine.rs` exposing the grouping logic once, over abstract scored boxes:

```rust
pub(crate) struct MatchConfig { metric: MatchMetric, threshold: f32, class_aware: bool }

pub(crate) fn greedy_groups<B, C>(n, cfg, bbox, class) -> Vec<Vec<usize>>;        // fixed anchor
pub(crate) fn connected_components<B, C>(n, cfg, bbox, class) -> Vec<Vec<usize>>; // transitive
//   where B: Fn(usize) -> BoundingBox, C: Fn(usize) -> u32
```

Both take per-item `bbox` (BoundingBox is `Copy`) and `class` accessors; matching uses
`BoundingBox::iou` / `BoundingBox::ios` per the metric, with the class-aware gate. Items are assumed
sorted by descending confidence, so each group's lowest index is its anchor.

The three algorithms reduce to grouping + a per-group reducer:

- **NMS** — `greedy_groups` → keep `items[group[0]]` (the anchor) of each group.
- **GREEDYNMM** — `greedy_groups` → merge each group.
- **NMM** — `connected_components` → merge each group.

Callers keep their own type-specific merge (`merge_detection_group` for `Detection`; `merge_group` +
`union_masks` for `MaskedDetection`) — only the matching/grouping is shared.

### Removals
- `Postprocessor::should_compare`, `Postprocessor::match_score`, and the free `ios` in
  `postprocess.rs` (subsumed by `MatchConfig`).
- The free `match_score` and `uf_find` in `segmentation.rs` (subsumed by `combine`).

## Why behavior is preserved

- IoU paths already used `BoundingBox::iou`; the two IoS impls are identical, so consolidating onto
  `BoundingBox::ios` changes nothing.
- NMS-via-greedy-groups keeps exactly the anchors the current suppress loop keeps (a box is claimed/
  suppressed by the first higher-confidence box it overlaps), in the same descending-confidence order.
- GREEDYNMM and NMM grouping are copied verbatim from the (already-correct, #35) implementations.

## Testing

This is the **refactor** phase under a green suite: the existing NMS/NMM/GREEDYNMM tests, the #35 chain
tests (bbox + mask), `test_nmm_unions_masks`, `test_nmm_keeps_disjoint_detections`, and the IoS-metric
test pin the behavior and must stay green unchanged. Gate on `cargo test`,
`cargo clippy --features "python,models,onnx" --all-targets -- -D warnings`, and
`cargo fmt --all -- --check`. A post-refactor adversarial behavior-preservation review (multi-agent)
cross-checks the unified core against the original semantics of all six functions across edge cases.

## Non-goals

Changing any algorithm's behavior; the contour tracer (#37); CUDA (#38); panic-proofing (#39).
