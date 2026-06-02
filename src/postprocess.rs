//! Postprocessing module for merging slice detections.
//!
//! Supports NMS (Non-Maximum Suppression), NMM (Non-Maximum Merging),
//! and GREEDYNMM (Greedy Non-Maximum Merging) algorithms.

use crate::combine;
use crate::detection::{BoundingBox, Detection};
use crate::slicer::Slice;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Type of postprocessing algorithm to use after sliced inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum PostprocessType {
    /// Non-Maximum Suppression: removes overlapping boxes, keeps highest confidence.
    NMS = 0,
    /// Non-Maximum Merging: merges overlapping boxes into a single box.
    NMM = 1,
    /// Greedy Non-Maximum Merging: greedily merges boxes (default, like original SAHI).
    #[default]
    GREEDYNMM = 2,
}

/// Metric used to match overlapping predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum MatchMetric {
    /// Intersection over Union.
    #[default]
    IOU = 0,
    /// Intersection over Smaller area (better for nested boxes).
    IOS = 1,
}

/// Configuration for postprocessing.
#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    /// Match threshold for postprocessing (detections with metric > threshold are merged/suppressed).
    pub match_threshold: f32,
    /// Confidence threshold (detections below this are discarded).
    pub confidence_threshold: f32,
    /// Whether to perform class-aware postprocessing (only process within same class).
    pub class_aware: bool,
    /// Type of postprocessing algorithm.
    pub postprocess_type: PostprocessType,
    /// Metric used for matching predictions.
    pub match_metric: MatchMetric,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            match_threshold: 0.5,
            confidence_threshold: 0.25,
            class_aware: true,
            postprocess_type: PostprocessType::GREEDYNMM,
            match_metric: MatchMetric::IOU,
        }
    }
}

impl PostprocessConfig {
    /// Create a new postprocess configuration.
    pub fn new(match_threshold: f32, confidence_threshold: f32) -> Self {
        Self {
            match_threshold,
            confidence_threshold,
            ..Default::default()
        }
    }

    /// Set whether postprocessing should be class-aware.
    pub fn with_class_aware(mut self, class_aware: bool) -> Self {
        self.class_aware = class_aware;
        self
    }

    /// Set the postprocess type.
    pub fn with_postprocess_type(mut self, postprocess_type: PostprocessType) -> Self {
        self.postprocess_type = postprocess_type;
        self
    }

    /// Set the match metric.
    pub fn with_match_metric(mut self, match_metric: MatchMetric) -> Self {
        self.match_metric = match_metric;
        self
    }

    // Legacy alias for backward compatibility
    #[doc(hidden)]
    pub fn nms_threshold(&self) -> f32 {
        self.match_threshold
    }
}

/// Calculate Intersection over Smaller area (IOS).
/// Merge two bounding boxes into one (union).
fn merge_boxes(a: &BoundingBox, b: &BoundingBox) -> BoundingBox {
    let x1 = a.x.min(b.x);
    let y1 = a.y.min(b.y);
    let x2 = a.x2().max(b.x2());
    let y2 = a.y2().max(b.y2());
    BoundingBox::from_corners(x1, y1, x2, y2)
}

/// Postprocessor merges detections from slices using NMS/NMM algorithms.
#[derive(Debug)]
pub struct Postprocessor {
    config: PostprocessConfig,
}

impl Postprocessor {
    /// Create a new postprocessor with the given configuration.
    pub fn new(config: PostprocessConfig) -> Self {
        Self { config }
    }

    /// Create a postprocessor with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PostprocessConfig::default())
    }

    /// Calculate the match score between two boxes based on the configured metric.
    /// Build the shared match configuration used by the grouping algorithms.
    fn match_config(&self) -> combine::MatchConfig {
        combine::MatchConfig {
            metric: self.config.match_metric,
            threshold: self.config.match_threshold,
            class_aware: self.config.class_aware,
        }
    }

    /// Stitch detections from multiple slices.
    ///
    /// Takes slice-relative detections and converts them to image coordinates,
    /// then applies the configured postprocessing to remove/merge duplicates.
    pub fn stitch(&self, slice_detections: &[(Slice, Vec<Detection>)]) -> Vec<Detection> {
        // Collect all detections, translating to image coordinates
        let mut all_detections: Vec<Detection> = slice_detections
            .iter()
            .flat_map(|(slice, detections)| {
                detections
                    .iter()
                    .map(|d| d.translate(slice.x as f32, slice.y as f32))
            })
            .filter(|d| d.confidence >= self.config.confidence_threshold)
            .collect();

        // Apply postprocessing
        match self.config.postprocess_type {
            PostprocessType::NMS => self.nms(&mut all_detections),
            PostprocessType::NMM => self.nmm(&mut all_detections),
            PostprocessType::GREEDYNMM => self.greedy_nmm(&mut all_detections),
        }
    }

    /// Apply Non-Maximum Suppression to a list of detections.
    ///
    /// Removes overlapping boxes, keeping only the one with highest confidence.
    pub fn nms(&self, detections: &mut [Detection]) -> Vec<Detection> {
        sort_by_confidence_desc(detections);
        // Greedy fixed-anchor groups; NMS keeps each group's anchor and suppresses the rest.
        let groups = combine::greedy_groups(
            detections.len(),
            self.match_config(),
            |i| detections[i].bbox,
            |i| detections[i].class_id,
        );
        groups.iter().map(|g| detections[g[0]].clone()).collect()
    }

    /// Apply Non-Maximum Merging to a list of detections.
    ///
    /// Transitive: boxes are linked when their match score exceeds the threshold, and each
    /// connected component of that match graph is merged into one detection (so A–B + B–C
    /// merges {A, B, C} even when A and C do not overlap). Matches reference SAHI's NMM —
    /// looser than [`greedy_nmm`](Self::greedy_nmm), which only merges direct overlaps.
    pub fn nmm(&self, detections: &mut [Detection]) -> Vec<Detection> {
        sort_by_confidence_desc(detections);
        let groups = combine::connected_components(
            detections.len(),
            self.match_config(),
            |i| detections[i].bbox,
            |i| detections[i].class_id,
        );
        groups
            .iter()
            .map(|g| self.merge_detection_group(detections, g))
            .collect()
    }

    /// Apply Greedy Non-Maximum Merging to a list of detections.
    ///
    /// Walks detections in descending confidence; each unused anchor claims a group of every
    /// still-unused box that directly overlaps the **fixed anchor** (no transitive merging, no
    /// box growth), then merges the group. Matches reference SAHI's GreedyNMM — tighter than
    /// [`nmm`](Self::nmm), which merges transitively.
    pub fn greedy_nmm(&self, detections: &mut [Detection]) -> Vec<Detection> {
        sort_by_confidence_desc(detections);
        let groups = combine::greedy_groups(
            detections.len(),
            self.match_config(),
            |i| detections[i].bbox,
            |i| detections[i].class_id,
        );
        groups
            .iter()
            .map(|g| self.merge_detection_group(detections, g))
            .collect()
    }

    /// Merge a group of detections into one.
    fn merge_detection_group(&self, detections: &[Detection], indices: &[usize]) -> Detection {
        let first = &detections[indices[0]];

        if indices.len() == 1 {
            return first.clone();
        }

        // Merge all boxes into the union. Detections are sorted by descending confidence and
        // `indices[0]` is the first (highest) match, so `first.confidence` is the group max.
        let mut merged_bbox = first.bbox;
        for &idx in indices.iter().skip(1) {
            merged_bbox = merge_boxes(&merged_bbox, &detections[idx].bbox);
        }

        Detection::new(
            merged_bbox,
            first.class_id,
            first.confidence,
            first.class_name.clone(),
        )
    }

    /// Get the configuration.
    pub fn config(&self) -> &PostprocessConfig {
        &self.config
    }
}

/// Sort detections by descending confidence so each group's anchor comes first.
fn sort_by_confidence_desc(detections: &mut [Detection]) {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_removes_duplicates() {
        let postprocessor = Postprocessor::new(
            PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::NMS),
        );

        let mut detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9, None),
            Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 0, 0.8, None),
            Detection::new(BoundingBox::new(200.0, 200.0, 100.0, 100.0), 0, 0.7, None),
        ];

        let result = postprocessor.nms(&mut detections);

        // First two boxes overlap significantly, should keep only the one with higher confidence
        // Third box doesn't overlap, should be kept
        assert_eq!(result.len(), 2);
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
        assert!((result[1].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_class_aware_nms() {
        let postprocessor = Postprocessor::new(
            PostprocessConfig::new(0.5, 0.0)
                .with_class_aware(true)
                .with_postprocess_type(PostprocessType::NMS),
        );

        let mut detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9, None),
            Detection::new(BoundingBox::new(10.0, 10.0, 100.0, 100.0), 1, 0.8, None), // Different class
        ];

        let result: Vec<Detection> = postprocessor.nms(&mut detections);

        // Different classes should not suppress each other
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_confidence_threshold() {
        let postprocessor = Postprocessor::new(PostprocessConfig::new(0.5, 0.5));

        let slice = Slice::new(0, 0, 100, 100, 0);
        let detections: Vec<Detection> = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 50.0, 50.0), 0, 0.6, None),
            Detection::new(BoundingBox::new(50.0, 50.0, 50.0, 50.0), 0, 0.3, None), // Below threshold
        ];

        let result: Vec<Detection> = postprocessor.stitch(&[(slice, detections)]);

        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_stitch_translates_coordinates() {
        let postprocessor = Postprocessor::with_defaults();

        let slice = Slice::new(100, 200, 640, 640, 0);
        let detections = vec![Detection::new(
            BoundingBox::new(10.0, 20.0, 50.0, 50.0),
            0,
            0.9,
            None,
        )];

        let result = postprocessor.stitch(&[(slice, detections)]);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].bbox.x, 110.0); // 100 + 10
        assert_eq!(result[0].bbox.y, 220.0); // 200 + 20
    }

    #[test]
    fn test_greedy_nmm_merges_boxes() {
        // Use a low threshold to ensure boxes with any overlap get merged
        let postprocessor = Postprocessor::new(
            PostprocessConfig::new(0.1, 0.0).with_postprocess_type(PostprocessType::GREEDYNMM),
        );

        // These boxes have ~14% IOU which is > 0.1 threshold
        let mut detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9, None),
            Detection::new(BoundingBox::new(50.0, 50.0, 100.0, 100.0), 0, 0.8, None),
        ];

        let result = postprocessor.greedy_nmm(&mut detections);

        // Should merge into one box
        assert_eq!(result.len(), 1);
        // Merged box should encompass both
        assert_eq!(result[0].bbox.x, 0.0);
        assert_eq!(result[0].bbox.y, 0.0);
        assert_eq!(result[0].bbox.width, 150.0);
        assert_eq!(result[0].bbox.height, 150.0);
        // Confidence is the max of the merged group (matches reference SAHI), not the average
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_nmm_keeps_max_confidence() {
        let postprocessor = Postprocessor::new(
            PostprocessConfig::new(0.1, 0.0).with_postprocess_type(PostprocessType::NMM),
        );

        let mut detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9, None),
            Detection::new(BoundingBox::new(50.0, 50.0, 100.0, 100.0), 0, 0.8, None),
        ];

        let result = postprocessor.nmm(&mut detections);

        assert_eq!(result.len(), 1);
        // Merged confidence is the max of the group, not the average
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_greedy_nmm_chain_keeps_indirect_box_separate() {
        // A-B overlap (~0.67) and B-C overlap (~0.67), but A-C do not (~0.43).
        // GREEDYNMM matches only the fixed anchor, so C is not pulled into A's group.
        let postprocessor = Postprocessor::new(
            PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::GREEDYNMM),
        );
        let mut detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9, None),
            Detection::new(BoundingBox::new(20.0, 0.0, 100.0, 100.0), 0, 0.8, None),
            Detection::new(BoundingBox::new(40.0, 0.0, 100.0, 100.0), 0, 0.7, None),
        ];
        let result = postprocessor.greedy_nmm(&mut detections);
        assert_eq!(
            result.len(),
            2,
            "C overlaps the A+B union but not A directly; GREEDYNMM must keep it separate"
        );
    }

    #[test]
    fn test_nmm_chain_merges_transitively() {
        // Same chain; NMM is transitive (A-B, B-C) so all three merge into one.
        let postprocessor = Postprocessor::new(
            PostprocessConfig::new(0.5, 0.0).with_postprocess_type(PostprocessType::NMM),
        );
        let mut detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9, None),
            Detection::new(BoundingBox::new(20.0, 0.0, 100.0, 100.0), 0, 0.8, None),
            Detection::new(BoundingBox::new(40.0, 0.0, 100.0, 100.0), 0, 0.7, None),
        ];
        let result = postprocessor.nmm(&mut detections);
        assert_eq!(result.len(), 1, "transitive NMM merges the whole chain");
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_ios_metric() {
        let postprocessor = Postprocessor::new(
            PostprocessConfig::new(0.5, 0.0)
                .with_postprocess_type(PostprocessType::NMS)
                .with_match_metric(MatchMetric::IOS),
        );

        // Small box fully inside large box
        let mut detections = vec![
            Detection::new(BoundingBox::new(0.0, 0.0, 100.0, 100.0), 0, 0.9, None),
            Detection::new(BoundingBox::new(25.0, 25.0, 50.0, 50.0), 0, 0.8, None), // Nested inside
        ];

        let result = postprocessor.nms(&mut detections);

        // With IOS, the nested box should be suppressed (IOS = 1.0 for fully contained)
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_ios_calculation() {
        let large = BoundingBox::new(0.0, 0.0, 100.0, 100.0);
        let small = BoundingBox::new(25.0, 25.0, 50.0, 50.0);

        // Small box fully inside large box: IOS should be 1.0
        let ios_score = large.ios(&small);
        assert!((ios_score - 1.0).abs() < 1e-6);

        // IOU would be much smaller
        let iou_score = large.iou(&small);
        assert!(iou_score < 0.5);
    }
}
