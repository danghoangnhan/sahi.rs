# Design: Correct contour tracing for `bool_mask_to_polygons`

- **Date:** 2026-06-01
- **Issue:** [#7](https://github.com/danghoangnhan/sahi.rs/issues/7) — `bool_mask_to_polygons` does not trace contours, producing invalid polygons
- **Status:** Approved (brainstorming → spec)

## Problem

`src/annotation/mask.rs::bool_mask_to_polygons` flood-fills each connected
foreground region and appends boundary pixels to the polygon in traversal
(stack-pop) order rather than perimeter order. The result is a self-intersecting,
geometrically meaningless ring. Consequences:

- `Mask::area()` (shoelace over the ring) → garbage.
- `Mask::to_bool_mask()` (scanline `fill_polygon`) → garbage raster.
- COCO segmentation output is an invalid polygon.
- Only `Mask::bounding_box()` survives (min/max is order-independent).

Reached via the public `Mask::from_bool_mask` / `Mask::from_float_mask`.

## Goal / non-goals

**Goal:** make `bool_mask_to_polygons` emit a valid, ordered boundary ring per
connected component, so `area()`, round-trip rasterization, and COCO output are
correct.

**Non-goals (deferred):**
- Hole / hierarchy support (Suzuki–Abe). Outer contours only; holes are filled.
- Vertex simplification (Douglas–Peucker).
- Parallel / SIMD acceleration (see *Future acceleration*).

## Approach: hand-rolled Moore-neighbor tracing (dependency-free)

The `annotation` module has no external dependencies and `mask.rs` is in the
default build, so we avoid pulling in a contour crate (`imageproc`, etc.) and
implement an outer-contour tracer directly.

### Algorithm

1. Raster-scan the grid with a `visited` (component-membership) bitmap.
2. On the first unvisited foreground pixel of a component: by scan order it is
   the component's top-/left-most pixel, so it lies on the outer boundary and
   its west neighbor is background (or out-of-bounds). It is a valid Moore start.
3. Trace clockwise: initial backtrack = west; from the backtrack cell, scan the
   8-neighborhood clockwise for the next foreground pixel, emit it, set the
   backtrack to the background cell examined just before it, and advance. Stop on
   **Jacob's criterion** (re-enter the start pixel from the same direction), with
   an iteration cap (`8 * pixel_count`) as an infinite-loop backstop.
4. Flood-fill (8-connectivity) the component into `visited` so the scan does not
   restart inside it.
5. Drop contours with `< 3` points (isolated / 2-px specks).
6. N disconnected components → N polygons (valid COCO multi-polygon).

Out-of-bounds neighbors are treated as background. Foreground connectivity is
8-connected (consistent with Moore tracing).

### Coordinate convention & limitations (documented on the function)

- Vertices are at pixel **centers** `(x, y)`, so a filled W×H block traces as
  roughly `(W-1)×(H-1)`. Area is slightly under-measured for small blobs,
  negligible for large ones. Corner-accurate tracing is a future tweak.
- Outer contour only — holes are filled.
- Sub-3-point components are dropped as noise.

### Integration — no API change

`fn bool_mask_to_polygons(mask: &[bool], width: u32, height: u32) -> Vec<Vec<f32>>`
keeps its signature; only the body changes. The ordered simple ring it returns is
exactly what the existing scanline `fill_polygon` (`to_bool_mask`) and shoelace
`area()` need, and is a valid COCO polygon.

## Future acceleration (out of scope)

The trace is an inherently serial pointer walk — each step depends on the
previous position and backtrack direction — so SIMD/auto-vectorization gives
nothing on the core loop. If a future segmentation pipeline ([#9]) makes this
hot, the realistic lever is `rayon` **across independent connected components**,
gated behind the existing `parallel` feature (consistent with `backend/cpu.rs`);
the per-row scanline fill in `to_bool_mask` is also parallelizable. Deferred as
YAGNI: this runs per-detection on small masks and is not on the pipeline hot path
(segmentation is not wired in yet — [#9]).

[#9]: https://github.com/danghoangnhan/sahi.rs/issues/9

## Test plan (TDD)

Written first, against the current (incorrect) implementation:

1. **Ordered ring** — a filled rectangle yields exactly one polygon whose
   consecutive vertices are 8-adjacent (distance ≤ √2 + ε). The current impl
   jumps around → fails.
2. **Round-trip** — `mask → from_bool_mask → to_bool_mask` recovers the region
   at high overlap. The pixel-center convention costs a ~1-pixel border, so use a
   large rectangle (e.g. 20×20) and assert IoU ≥ 0.7; the broken impl scores
   near 0.
3. **Area** — a filled W×H rectangle's `area()` is within ~15% of `W·H` (the
   pixel-center convention under-measures by ~1px of border; the broken impl is
   wildly off).
4. **Multi-component** — two disjoint blobs → exactly 2 polygons.
5. **Degenerate** — empty mask → no polygons; single pixel → dropped, no panic.

(Final tolerances are chosen during TDD so each test fails on the current impl
and passes on the correct one.)

`bounding_box()` remains correct throughout (it always was).
