//! Image slicing logic for SAHI.

/// A slice definition representing a region of the source image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// X offset in the source image
    pub x: u32,
    /// Y offset in the source image
    pub y: u32,
    /// Width of the slice
    pub width: u32,
    /// Height of the slice
    pub height: u32,
    /// Index of this slice
    pub index: usize,
}

impl Slice {
    /// Create a new slice.
    pub fn new(x: u32, y: u32, width: u32, height: u32, index: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
            index,
        }
    }
}

/// Configuration for the slicer.
#[derive(Debug, Clone)]
pub struct SlicerConfig {
    /// Width of each slice
    pub slice_width: u32,
    /// Height of each slice
    pub slice_height: u32,
    /// Horizontal overlap ratio (0.0 - 1.0)
    pub overlap_width_ratio: f32,
    /// Vertical overlap ratio (0.0 - 1.0)
    pub overlap_height_ratio: f32,
    /// Keep edge tiles full-size by shifting the last row/column flush to the
    /// image edge, instead of clipping them. Images smaller than a slice still
    /// yield one image-sized tile.
    pub fixed_size: bool,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            slice_width: 640,
            slice_height: 640,
            overlap_width_ratio: 0.2,
            overlap_height_ratio: 0.2,
            fixed_size: false,
        }
    }
}

impl SlicerConfig {
    /// Create a new slicer configuration.
    pub fn new(slice_width: u32, slice_height: u32) -> Self {
        Self {
            slice_width,
            slice_height,
            ..Default::default()
        }
    }

    /// Set the overlap ratios.
    pub fn with_overlap(mut self, width_ratio: f32, height_ratio: f32) -> Self {
        self.overlap_width_ratio = width_ratio.clamp(0.0, 0.9);
        self.overlap_height_ratio = height_ratio.clamp(0.0, 0.9);
        self
    }

    /// Keep edge tiles full-size (shifted flush to the edge) instead of clipping.
    pub fn with_fixed_size(mut self, fixed_size: bool) -> Self {
        self.fixed_size = fixed_size;
        self
    }
}

/// Slicer generates slice definitions for an image.
#[derive(Debug)]
pub struct Slicer {
    config: SlicerConfig,
}

impl Slicer {
    /// Create a new slicer with the given configuration.
    pub fn new(config: SlicerConfig) -> Self {
        Self { config }
    }

    /// Create a slicer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(SlicerConfig::default())
    }

    /// Generate slices for an image of the given dimensions.
    pub fn slice(&self, image_width: u32, image_height: u32) -> Vec<Slice> {
        let step_x = ((self.config.slice_width as f32 * (1.0 - self.config.overlap_width_ratio))
            as u32)
            .max(1);
        let step_y = ((self.config.slice_height as f32 * (1.0 - self.config.overlap_height_ratio))
            as u32)
            .max(1);

        let fixed = self.config.fixed_size;
        let xs = axis_starts(image_width, self.config.slice_width, step_x, fixed);
        let ys = axis_starts(image_height, self.config.slice_height, step_y, fixed);

        let mut slices = Vec::with_capacity(xs.len() * ys.len());
        let mut index = 0;
        for &y in &ys {
            // Full-size in fixed mode (when the image exceeds the slice), else clipped.
            let slice_h = if fixed && image_height > self.config.slice_height {
                self.config.slice_height
            } else {
                self.config.slice_height.min(image_height - y)
            };
            for &x in &xs {
                let slice_w = if fixed && image_width > self.config.slice_width {
                    self.config.slice_width
                } else {
                    self.config.slice_width.min(image_width - x)
                };
                slices.push(Slice::new(x, y, slice_w, slice_h, index));
                index += 1;
            }
        }

        slices
    }

    /// Get the number of slices that will be generated for an image.
    pub fn count_slices(&self, image_width: u32, image_height: u32) -> usize {
        self.slice(image_width, image_height).len()
    }

    /// Get the configuration.
    pub fn config(&self) -> &SlicerConfig {
        &self.config
    }
}

/// Tile start positions along one axis. In fixed-size mode the final start is
/// clamped to `image_dim - slice_dim`, so the last tile stays full-size and flush
/// to the edge. Returns `[0]` when the image is no larger than the slice.
fn axis_starts(image_dim: u32, slice_dim: u32, step: u32, fixed_size: bool) -> Vec<u32> {
    if image_dim <= slice_dim {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut p = 0u32;
    loop {
        starts.push(p);
        if p + slice_dim >= image_dim {
            break;
        }
        p += step;
    }
    if fixed_size {
        if let Some(last) = starts.last_mut() {
            *last = image_dim - slice_dim;
        }
    }
    starts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_slice_small_image() {
        let slicer: Slicer = Slicer::new(SlicerConfig::new(640, 640));
        let slices: Vec<Slice> = slicer.slice(320, 320);

        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].x, 0);
        assert_eq!(slices[0].y, 0);
        assert_eq!(slices[0].width, 320);
        assert_eq!(slices[0].height, 320);
    }

    #[test]
    fn test_multiple_slices_no_overlap() {
        let config: SlicerConfig = SlicerConfig::new(100, 100).with_overlap(0.0, 0.0);
        let slicer: Slicer = Slicer::new(config);
        let slices: Vec<Slice> = slicer.slice(200, 200);

        assert_eq!(slices.len(), 4);
    }

    #[test]
    fn test_overlap_creates_more_slices() {
        let no_overlap: SlicerConfig = SlicerConfig::new(100, 100).with_overlap(0.0, 0.0);
        let with_overlap: SlicerConfig = SlicerConfig::new(100, 100).with_overlap(0.5, 0.5);

        let slicer_no: Slicer = Slicer::new(no_overlap);
        let slicer_yes: Slicer = Slicer::new(with_overlap);

        let count_no: usize = slicer_no.count_slices(200, 200);
        let count_yes: usize = slicer_yes.count_slices(200, 200);

        assert!(count_yes > count_no);
    }

    #[test]
    fn test_slice_indices_sequential() {
        let slicer: Slicer = Slicer::new(SlicerConfig::new(100, 100));
        let slices: Vec<Slice> = slicer.slice(300, 300);

        for (i, slice) in slices.iter().enumerate() {
            assert_eq!(slice.index, i);
        }
    }

    #[test]
    fn test_fixed_size_keeps_full_tiles_and_shifts_last() {
        let config = SlicerConfig::new(100, 100)
            .with_overlap(0.0, 0.0)
            .with_fixed_size(true);
        let slicer = Slicer::new(config);
        // 250x100: clip mode gives widths 100,100,50; fixed gives three 100-wide tiles.
        let slices = slicer.slice(250, 100);

        assert_eq!(slices.len(), 3);
        assert!(slices.iter().all(|s| s.width == 100 && s.height == 100));
        // Last column tile is flush to the right edge: x = 250 - 100 = 150.
        assert_eq!(slices.iter().map(|s| s.x).max().unwrap(), 150);
        // Every tile stays within the image.
        assert!(slices
            .iter()
            .all(|s| s.x + s.width <= 250 && s.y + s.height <= 100));
    }

    #[test]
    fn test_fixed_size_small_image_single_clipped_tile() {
        let config = SlicerConfig::new(100, 100).with_fixed_size(true);
        let slicer = Slicer::new(config);
        // Image smaller than a slice -> can't shift; one image-sized tile.
        let slices = slicer.slice(50, 50);
        assert_eq!(slices.len(), 1);
        assert_eq!((slices[0].width, slices[0].height), (50, 50));
    }

    #[test]
    fn test_default_clips_edge_tiles() {
        // Default (clip) behavior is unchanged: the last tile is smaller.
        let config = SlicerConfig::new(100, 100).with_overlap(0.0, 0.0);
        let slicer = Slicer::new(config);
        let slices = slicer.slice(250, 100);
        assert_eq!(slices.len(), 3);
        assert_eq!(slices.iter().map(|s| s.width).min().unwrap(), 50); // clipped last tile
    }
}
