//! Shared NMS / NMM / GREEDYNMM grouping over abstract scored boxes.
//!
//! The bbox postprocess path ([`Detection`](crate::detection::Detection)) and the mask path
//! ([`MaskedDetection`](crate::segmentation::MaskedDetection)) ran duplicate match/group loops.
//! This module holds the matching and grouping logic once; callers supply per-item `bbox`/`class`
//! accessors and reduce each returned group of indices with their own type-specific merge.
//!
//! All three algorithms reduce to grouping + a per-group reducer:
//! - **NMS** — [`greedy_groups`] → keep each group's anchor (`group[0]`).
//! - **GREEDYNMM** — [`greedy_groups`] → merge each group.
//! - **NMM** — [`connected_components`] → merge each group.
//!
//! Items are assumed sorted by descending confidence, so each group's lowest index is its anchor
//! (highest-confidence member).

use crate::detection::BoundingBox;
use crate::postprocess::MatchMetric;

/// Matching configuration shared by all grouping algorithms.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatchConfig {
    /// Overlap metric (IoU or IoS).
    pub metric: MatchMetric,
    /// Boxes match when their score strictly exceeds this threshold.
    pub threshold: f32,
    /// When true, only boxes of the same class are eligible to match.
    pub class_aware: bool,
}

impl MatchConfig {
    fn score(&self, a: &BoundingBox, b: &BoundingBox) -> f32 {
        match self.metric {
            MatchMetric::IOU => a.iou(b),
            MatchMetric::IOS => a.ios(b),
        }
    }

    /// Whether items `(bbox, class)` `a` and `b` are eligible and overlap above threshold.
    fn matches(&self, a: (&BoundingBox, u32), b: (&BoundingBox, u32)) -> bool {
        if self.class_aware && a.1 != b.1 {
            return false;
        }
        self.score(a.0, b.0) > self.threshold
    }
}

/// Greedy fixed-anchor grouping: each unused anchor (in index order) claims every still-unused
/// item that directly overlaps the **fixed anchor** (no transitive merging). Returns groups of
/// indices, each with the anchor first.
#[allow(clippy::needless_range_loop)] // indices drive the bbox/class accessors and `used`
pub(crate) fn greedy_groups<B, C>(n: usize, cfg: MatchConfig, bbox: B, class: C) -> Vec<Vec<usize>>
where
    B: Fn(usize) -> BoundingBox,
    C: Fn(usize) -> u32,
{
    let mut used = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut group = vec![i];
        let anchor = (bbox(i), class(i));
        for j in (i + 1)..n {
            if used[j] {
                continue;
            }
            if cfg.matches((&anchor.0, anchor.1), (&bbox(j), class(j))) {
                used[j] = true;
                group.push(j);
            }
        }
        groups.push(group);
    }
    groups
}

/// Transitive connected-component grouping over the match graph: items are linked when their match
/// score exceeds the threshold, and each connected component becomes one group (so A–B + B–C groups
/// {A, B, C} even when A and C do not overlap). Returns groups of indices in ascending order.
#[allow(clippy::needless_range_loop)] // indices drive the bbox/class accessors and union-find
pub(crate) fn connected_components<B, C>(
    n: usize,
    cfg: MatchConfig,
    bbox: B,
    class: C,
) -> Vec<Vec<usize>>
where
    B: Fn(usize) -> BoundingBox,
    C: Fn(usize) -> u32,
{
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            if cfg.matches((&bbox(i), class(i)), (&bbox(j), class(j))) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut group_of: Vec<Option<usize>> = vec![None; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        let gi = match group_of[root] {
            Some(g) => g,
            None => {
                let g = groups.len();
                group_of[root] = Some(g);
                groups.push(Vec::new());
                g
            }
        };
        groups[gi].push(i);
    }
    groups
}

/// Union-find `find` with path compression.
fn find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}
