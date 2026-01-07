//! Segmentation mask types for instance segmentation.
//!
//! This module provides the `Mask` type for representing segmentation masks
//! in various formats (polygon, RLE, boolean mask).

use super::FullShape;
use crate::detection::BoundingBox;

// ============================================================================
// Mask Format
// ============================================================================

/// The format of segmentation mask data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq, eq_int))]
pub enum MaskFormat {
    /// Polygon format: list of [x1, y1, x2, y2, ...] coordinate lists.
    #[default]
    Polygon,
    /// Run-length encoding (RLE).
    Rle,
    /// Boolean mask (2D array).
    BoolMask,
}

// ============================================================================
// RLE Data
// ============================================================================

/// Run-length encoded mask data.
///
/// RLE is an efficient way to store binary masks by encoding runs of
/// consecutive pixels with the same value.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RleData {
    /// Run lengths (alternating background/foreground).
    pub counts: Vec<u32>,
    /// Image size [height, width].
    pub size: [u32; 2],
}

impl RleData {
    /// Create new RLE data.
    pub fn new(counts: Vec<u32>, size: [u32; 2]) -> Self {
        Self { counts, size }
    }

    /// Decode RLE to boolean mask.
    pub fn to_bool_mask(&self) -> Vec<bool> {
        let total = (self.size[0] * self.size[1]) as usize;
        let mut mask = vec![false; total];
        let mut pos = 0usize;
        let mut val = false;

        for &count in &self.counts {
            let count = count as usize;
            if val {
                for item in mask.iter_mut().take((pos + count).min(total)).skip(pos) {
                    *item = true;
                }
            }
            pos += count;
            val = !val;
        }

        mask
    }

    /// Get mask width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.size[1]
    }

    /// Get mask height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.size[0]
    }

    /// Calculate the area (number of foreground pixels).
    pub fn area(&self) -> u32 {
        let mut area = 0u32;
        let mut is_foreground = false;
        for &count in &self.counts {
            if is_foreground {
                area += count;
            }
            is_foreground = !is_foreground;
        }
        area
    }
}

// ============================================================================
// Polygon
// ============================================================================

/// A single polygon represented as a list of coordinates.
///
/// Format: [x1, y1, x2, y2, x3, y3, ...]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polygon {
    /// Flattened coordinates [x1, y1, x2, y2, ...].
    pub points: Vec<f32>,
}

impl Polygon {
    /// Create a new polygon from flattened coordinates.
    pub fn new(points: Vec<f32>) -> Self {
        Self { points }
    }

    /// Create from pairs of (x, y) coordinates.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (f32, f32)>) -> Self {
        let points: Vec<f32> = pairs.into_iter().flat_map(|(x, y)| [x, y]).collect();
        Self { points }
    }

    /// Get number of vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.points.len() / 2
    }

    /// Check if polygon is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Iterate over (x, y) coordinate pairs.
    pub fn iter_points(&self) -> impl Iterator<Item = (f32, f32)> + '_ {
        self.points.chunks_exact(2).map(|c| (c[0], c[1]))
    }

    /// Get the bounding box of this polygon.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        if self.points.len() < 4 {
            return None;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for (x, y) in self.iter_points() {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        Some(BoundingBox::from_corners(min_x, min_y, max_x, max_y))
    }

    /// Shift the polygon by an offset.
    pub fn shift(&self, dx: f32, dy: f32) -> Self {
        let points: Vec<f32> = self
            .points
            .chunks_exact(2)
            .flat_map(|c| [c[0] + dx, c[1] + dy])
            .collect();
        Self { points }
    }

    /// Clip coordinates to boundaries.
    pub fn clip(&self, width: f32, height: f32) -> Self {
        let points: Vec<f32> = self
            .points
            .chunks_exact(2)
            .flat_map(|c| [c[0].clamp(0.0, width), c[1].clamp(0.0, height)])
            .collect();
        Self { points }
    }

    /// Scale the polygon by a factor.
    pub fn scale(&self, factor: f32) -> Self {
        let points: Vec<f32> = self.points.iter().map(|&v| v * factor).collect();
        Self { points }
    }
}

// ============================================================================
// Mask
// ============================================================================

/// A segmentation mask with support for multiple formats.
///
/// The mask stores the original format (polygon/RLE) and can lazily
/// compute other representations when needed.
#[derive(Debug, Clone, PartialEq)]
pub struct Mask {
    /// Segmentation polygons (COCO format).
    segmentation: Vec<Polygon>,
    /// Full image shape [height, width].
    full_shape: FullShape,
    /// Shift amount [shift_x, shift_y].
    shift_x: f32,
    shift_y: f32,
}

impl Mask {
    /// Create a new mask from COCO-style polygon segmentation.
    ///
    /// # Arguments
    ///
    /// * `segmentation` - List of polygons, each polygon is [x1, y1, x2, y2, ...]
    /// * `full_shape` - Full image shape [height, width]
    /// * `shift_amount` - Optional shift [shift_x, shift_y]
    pub fn new(
        segmentation: Vec<Vec<f32>>,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        let shift = shift_amount.unwrap_or([0.0, 0.0]);
        Self {
            segmentation: segmentation.into_iter().map(Polygon::new).collect(),
            full_shape: full_shape.into(),
            shift_x: shift[0],
            shift_y: shift[1],
        }
    }

    /// Create from a boolean mask.
    ///
    /// Converts the boolean mask to polygon representation.
    pub fn from_bool_mask(
        mask: &[bool],
        width: u32,
        height: u32,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        // Convert bool mask to polygon using simple contour finding
        let polygons = bool_mask_to_polygons(mask, width, height);
        Self::new(polygons, full_shape, shift_amount)
    }

    /// Create from a float mask with thresholding.
    pub fn from_float_mask(
        mask: &[f32],
        width: u32,
        height: u32,
        threshold: f32,
        full_shape: impl Into<FullShape>,
        shift_amount: Option<[f32; 2]>,
    ) -> Self {
        let bool_mask: Vec<bool> = mask.iter().map(|&v| v > threshold).collect();
        Self::from_bool_mask(&bool_mask, width, height, full_shape, shift_amount)
    }

    /// Get the segmentation polygons.
    pub fn segmentation(&self) -> &[Polygon] {
        &self.segmentation
    }

    /// Get the raw segmentation data as nested vectors (COCO format).
    pub fn to_coco_segmentation(&self) -> Vec<Vec<f32>> {
        self.segmentation.iter().map(|p| p.points.clone()).collect()
    }

    /// Get the full image shape.
    pub fn full_shape(&self) -> FullShape {
        self.full_shape
    }

    /// Get the shift amount.
    pub fn shift_amount(&self) -> [f32; 2] {
        [self.shift_x, self.shift_y]
    }

    /// Convert to boolean mask.
    ///
    /// Returns a flattened boolean array of shape [height * width].
    pub fn to_bool_mask(&self) -> Vec<bool> {
        let width = self.full_shape.width as usize;
        let height = self.full_shape.height as usize;
        let mut mask = vec![false; width * height];

        for polygon in &self.segmentation {
            fill_polygon(&mut mask, width, height, polygon);
        }

        mask
    }

    /// Get the bounding box of this mask.
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        if self.segmentation.is_empty() {
            return None;
        }

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;

        for polygon in &self.segmentation {
            for (x, y) in polygon.iter_points() {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }

        if min_x == f32::MAX {
            None
        } else {
            Some(BoundingBox::from_corners(min_x, min_y, max_x, max_y))
        }
    }

    /// Get shifted mask (applies shift to all polygon coordinates).
    pub fn get_shifted(&self) -> Self {
        let shifted_segmentation: Vec<Polygon> = self
            .segmentation
            .iter()
            .map(|polygon| {
                let shifted = polygon.shift(self.shift_x, self.shift_y);
                shifted.clip(self.full_shape.width as f32, self.full_shape.height as f32)
            })
            .collect();

        Self {
            segmentation: shifted_segmentation,
            full_shape: self.full_shape,
            shift_x: 0.0,
            shift_y: 0.0,
        }
    }

    /// Get mask shape [height, width].
    pub fn shape(&self) -> [u32; 2] {
        [self.full_shape.height, self.full_shape.width]
    }

    /// Check if mask is empty.
    pub fn is_empty(&self) -> bool {
        self.segmentation.is_empty() || self.segmentation.iter().all(|p| p.is_empty())
    }

    /// Calculate the area (number of mask pixels).
    ///
    /// This is computed from the polygon using the shoelace formula.
    pub fn area(&self) -> f32 {
        let mut total_area = 0.0f32;

        for polygon in &self.segmentation {
            if polygon.points.len() < 6 {
                continue; // Need at least 3 vertices
            }

            // Shoelace formula for polygon area
            let n = polygon.num_vertices();
            let mut area = 0.0f32;

            for i in 0..n {
                let j = (i + 1) % n;
                let (x1, y1) = (polygon.points[i * 2], polygon.points[i * 2 + 1]);
                let (x2, y2) = (polygon.points[j * 2], polygon.points[j * 2 + 1]);
                area += x1 * y2 - x2 * y1;
            }

            total_area += area.abs() / 2.0;
        }

        total_area
    }
}

impl Default for Mask {
    fn default() -> Self {
        Self {
            segmentation: Vec::new(),
            full_shape: FullShape::default(),
            shift_x: 0.0,
            shift_y: 0.0,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Fill a polygon in a boolean mask using scanline algorithm.
fn fill_polygon(mask: &mut [bool], width: usize, height: usize, polygon: &Polygon) {
    if polygon.points.len() < 6 {
        return; // Need at least 3 vertices
    }

    let points: Vec<(f32, f32)> = polygon.iter_points().collect();
    let n = points.len();

    // Find bounding box
    let min_y = points.iter().map(|p| p.1).fold(f32::MAX, f32::min) as i32;
    let max_y = points.iter().map(|p| p.1).fold(f32::MIN, f32::max) as i32;

    // Scanline fill
    for y in min_y.max(0)..=max_y.min(height as i32 - 1) {
        let mut intersections: Vec<f32> = Vec::new();

        // Find intersections with polygon edges
        for i in 0..n {
            let j = (i + 1) % n;
            let (x1, y1) = points[i];
            let (x2, y2) = points[j];

            let yf = y as f32;

            if (y1 <= yf && yf < y2) || (y2 <= yf && yf < y1) {
                // Edge crosses scanline
                let x = x1 + (yf - y1) * (x2 - x1) / (y2 - y1);
                intersections.push(x);
            }
        }

        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Fill between pairs of intersections
        for pair in intersections.chunks_exact(2) {
            let x_start = (pair[0].ceil() as i32).max(0) as usize;
            let x_end = (pair[1].floor() as i32).min(width as i32 - 1) as usize;

            for x in x_start..=x_end {
                mask[y as usize * width + x] = true;
            }
        }
    }
}

/// Convert a boolean mask to polygon representation.
///
/// This uses a simple contour tracing algorithm.
fn bool_mask_to_polygons(mask: &[bool], width: u32, height: u32) -> Vec<Vec<f32>> {
    // Simple implementation: find connected regions and trace their contours
    // For production use, consider using a proper contour finding algorithm

    let mut polygons = Vec::new();
    let mut visited = vec![false; mask.len()];

    let width = width as usize;
    let height = height as usize;

    for start_y in 0..height {
        for start_x in 0..width {
            let idx = start_y * width + start_x;

            if mask[idx] && !visited[idx] {
                // Found a new region, trace its contour
                let x = start_x;
                let y = start_y;

                // Simple boundary following
                let mut points = Vec::new();
                let mut queue = vec![(x, y)];

                while let Some((cx, cy)) = queue.pop() {
                    let idx = cy * width + cx;
                    if visited[idx] {
                        continue;
                    }
                    visited[idx] = true;

                    // Check if this is a boundary pixel
                    let is_boundary = cx == 0
                        || cy == 0
                        || cx == width - 1
                        || cy == height - 1
                        || !mask[cy * width + (cx - 1)]
                        || !mask[cy * width + (cx + 1)]
                        || !mask[(cy - 1) * width + cx]
                        || !mask[(cy + 1) * width + cx];

                    if is_boundary {
                        points.push((cx as f32, cy as f32));
                    }

                    // Add neighbors to queue
                    if cx > 0 && mask[cy * width + (cx - 1)] && !visited[cy * width + (cx - 1)] {
                        queue.push((cx - 1, cy));
                    }
                    if cx < width - 1
                        && mask[cy * width + (cx + 1)]
                        && !visited[cy * width + (cx + 1)]
                    {
                        queue.push((cx + 1, cy));
                    }
                    if cy > 0 && mask[(cy - 1) * width + cx] && !visited[(cy - 1) * width + cx] {
                        queue.push((cx, cy - 1));
                    }
                    if cy < height - 1
                        && mask[(cy + 1) * width + cx]
                        && !visited[(cy + 1) * width + cx]
                    {
                        queue.push((cx, cy + 1));
                    }
                }

                // Convert to flat polygon format
                if points.len() >= 3 {
                    let mut polygon = Vec::new();
                    for (px, py) in points {
                        polygon.push(px);
                        polygon.push(py);
                    }
                    polygons.push(polygon);
                }
            }
        }
    }

    polygons
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_new() {
        let polygon = Polygon::new(vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]);
        assert_eq!(polygon.num_vertices(), 4);
    }

    #[test]
    fn test_polygon_from_pairs() {
        let polygon = Polygon::from_pairs([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]);
        assert_eq!(polygon.num_vertices(), 4);
        assert_eq!(
            polygon.points,
            vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
        );
    }

    #[test]
    fn test_polygon_bounding_box() {
        let polygon = Polygon::new(vec![10.0, 20.0, 50.0, 20.0, 50.0, 60.0, 10.0, 60.0]);
        let bbox = polygon.bounding_box().unwrap();
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
        assert_eq!(bbox.width, 40.0);
        assert_eq!(bbox.height, 40.0);
    }

    #[test]
    fn test_polygon_shift() {
        let polygon = Polygon::new(vec![0.0, 0.0, 10.0, 10.0]);
        let shifted = polygon.shift(5.0, 10.0);
        assert_eq!(shifted.points, vec![5.0, 10.0, 15.0, 20.0]);
    }

    #[test]
    fn test_mask_new() {
        let segmentation = vec![vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]];
        let mask = Mask::new(segmentation, [100, 100], None);

        assert!(!mask.is_empty());
        assert_eq!(mask.shape(), [100, 100]);
        assert_eq!(mask.shift_amount(), [0.0, 0.0]);
    }

    #[test]
    fn test_mask_with_shift() {
        let segmentation = vec![vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]];
        let mask = Mask::new(segmentation, [100, 100], Some([50.0, 50.0]));

        let shifted = mask.get_shifted();
        assert_eq!(shifted.shift_amount(), [0.0, 0.0]);

        let poly = &shifted.segmentation()[0];
        let first_point = (poly.points[0], poly.points[1]);
        assert_eq!(first_point, (50.0, 50.0));
    }

    #[test]
    fn test_mask_bounding_box() {
        let segmentation = vec![vec![10.0, 20.0, 50.0, 20.0, 50.0, 60.0, 10.0, 60.0]];
        let mask = Mask::new(segmentation, [100, 100], None);

        let bbox = mask.bounding_box().unwrap();
        assert_eq!(bbox.x, 10.0);
        assert_eq!(bbox.y, 20.0);
    }

    #[test]
    fn test_mask_area() {
        // 10x10 square
        let segmentation = vec![vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]];
        let mask = Mask::new(segmentation, [100, 100], None);

        let area = mask.area();
        assert!((area - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_rle_decode() {
        // Simple RLE: 2 background, 3 foreground, 2 background
        let rle = RleData::new(vec![2, 3, 2], [1, 7]);
        let mask = rle.to_bool_mask();

        assert_eq!(mask, vec![false, false, true, true, true, false, false]);
        assert_eq!(rle.area(), 3);
    }

    #[test]
    fn test_mask_to_coco() {
        let segmentation = vec![vec![0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]];
        let mask = Mask::new(segmentation.clone(), [100, 100], None);

        let coco = mask.to_coco_segmentation();
        assert_eq!(coco, segmentation);
    }
}
