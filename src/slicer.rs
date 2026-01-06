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
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            slice_width: 640,
            slice_height: 640,
            overlap_width_ratio: 0.2,
            overlap_height_ratio: 0.2,
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
        let mut slices: Vec<Slice> = Vec::new();

        let step_x =
            (self.config.slice_width as f32 * (1.0 - self.config.overlap_width_ratio)) as u32;
        let step_y =
            (self.config.slice_height as f32 * (1.0 - self.config.overlap_height_ratio)) as u32;

        // Ensure we have at least 1 pixel step
        let step_x = step_x.max(1);
        let step_y = step_y.max(1);

        let mut index = 0;
        let mut y = 0u32;

        while y < image_height {
            let mut x = 0u32;

            // Calculate actual slice height (may be smaller at edges)
            let slice_h = self.config.slice_height.min(image_height - y);

            while x < image_width {
                // Calculate actual slice width (may be smaller at edges)
                let slice_w = self.config.slice_width.min(image_width - x);

                slices.push(Slice::new(x, y, slice_w, slice_h, index));
                index += 1;

                // Move to next x position
                if x + self.config.slice_width >= image_width {
                    break;
                }
                x += step_x;
            }

            // Move to next y position
            if y + self.config.slice_height >= image_height {
                break;
            }
            y += step_y;
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
}
