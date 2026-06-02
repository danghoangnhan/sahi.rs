# NMM / GREEDYNMM reference-parity fix — design

- **Date:** 2026-06-02
- **Status:** Approved
- **Issue:** #35 (milestone: Reference-parity hardening, #2)

## Problem

Our NMM and GREEDYNMM diverge from reference SAHI — effectively swapped, and both wrong:

- `greedy_nmm` (`src/postprocess.rs:280`, and `seg_greedy_nmm` in `src/segmentation.rs`) matches each
  candidate against a **progressively grown union box**, so it transitively absorbs boxes that overlap
  the union but not the original anchor → over-merges.
- `nmm` (`src/postprocess.rs:227`, and `seg_nmm`) matches each candidate against the **fixed anchor**
  only (non-transitive, one group per anchor).

Reference SAHI (current obss/sahi docstrings):

- **GreedyNMM** — "each kept prediction only merges boxes that **directly overlap with it** (no
  transitive merging)" → tighter merged boxes.
- **NMM** — transitive: "if A merges with B and B merges with C, all three are merged together even
  without direct A–C overlap."

So the correct mapping is the opposite of ours: our current `nmm` body is actually correct GreedyNMM,
and neither function implements transitive NMM.

## Target semantics

- **GREEDYNMM** (greedy, tight, no transitivity): detections sorted by descending confidence; each
  unused anchor `i` claims a group of itself plus every unused `j` that **directly overlaps the fixed
  anchor `i`** (`match_score(i, j) > match_threshold`, honoring class-aware gating); mark the group
  used; merge. A box is claimed by the highest-confidence anchor it directly overlaps. No box growth.
- **NMM** (transitive): build the match graph over all eligible pairs
  (`match_score > match_threshold`, class-aware respected) and merge each **connected component** into
  one detection (A–B + B–C ⇒ {A,B,C}).
- **Merge** (both): union bounding box, the **max-confidence** member as anchor (class + confidence),
  and the union of masks on the segmentation path. Reuses the existing `merge_detection_group` /
  `merge_group` + `union_masks` (the #40 rasterize→OR→re-trace dedup).

## Implementation

- `src/postprocess.rs`:
  - `greedy_nmm` — replace the grown-box loop with fixed-anchor greedy grouping (the current `nmm`
    grouping logic).
  - `nmm` — replace with transitive connected-components grouping (small union-find or BFS helper).
- `src/segmentation.rs`: apply the same correction to `seg_greedy_nmm` and `seg_nmm`.
- The match/group logic stays duplicated across the bbox and mask paths; **unifying it is #36**, a
  separate follow-up. This change is correctness only.

## Tests (TDD)

Chain fixture: `A(0,0,100,100, conf .9)`, `B(20,0,100,100, .8)`, `C(40,0,100,100, .7)`, IoU
threshold 0.5. A–B and B–C overlap (~0.67); A–C do not (~0.43).

- GREEDYNMM(chain) ⇒ **2** groups (`{A,B}`, `{C}`) — C is not pulled into A's group (today: grows → 1).
- NMM(chain) ⇒ **1** group `{A,B,C}` via transitivity (today: non-transitive → 2).
- Mirror both on the segmentation path; repurpose `test_greedy_nmm_chain_merges_to_one` to assert
  **NMM** merges the chain, and add a GREEDYNMM test asserting it stays split.
- Existing 2-box tests (`test_greedy_nmm_merges_boxes`, `test_nmm_keeps_max_confidence`,
  `test_nmm_unions_masks`, `test_nmm_keeps_disjoint_detections`) remain valid: two directly-overlapping
  boxes merge under both algorithms; disjoint boxes stay split.
- Gate on `cargo test`, `cargo clippy --features "python,models,onnx" --all-targets -- -D warnings`,
  `cargo fmt --all -- --check`, and green PR CI.

## Non-goals

Unifying the duplicated NMS/NMM/GREEDYNMM logic (#36); contour tracer (#37); CUDA bounds + GPU test
coverage (#38); panic-proofing (#39).
