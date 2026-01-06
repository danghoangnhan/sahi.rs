//! Basic example of using SAHI with a mock model.
//!
//! Run with: `cargo run --example basic`

use sahi::{callback, BoundingBox, Detection, ImageData, MatchMetric, PostprocessType, Sahi};

fn main() {
    basic_usage();
    demo_postprocess_types();
}

/// Basic SAHI usage with sliced inference.
fn basic_usage() {
    println!("=== Basic SAHI Usage ===\n");

    // Create a mock image (1920x1080 RGB)
    let width = 1920;
    let height = 1080;
    let image = ImageData::from_rgb(vec![0u8; width * height * 3], width as u32, height as u32);

    println!("Image size: {}x{}", width, height);

    // Configure SAHI
    let sahi = Sahi::builder()
        .slice_size(640, 640) // YOLO-friendly size
        .overlap(0.2, 0.2) // 20% overlap
        .match_threshold(0.5) // Threshold for matching predictions
        .postprocess_type(PostprocessType::GREEDYNMM) // Options: NMS, NMM, GREEDYNMM
        .match_metric(MatchMetric::IOU) // Options: IOU, IOS
        .confidence_threshold(0.25)
        .class_aware(true)
        .include_full_image(false)
        .build();

    // Count slices
    let slice_count = sahi.slicer().count_slices(width as u32, height as u32);
    println!("Number of slices: {}", slice_count);
    println!("Backend: {}", sahi.backend_name());

    // Create a mock detection callback
    // In real usage, this would call your ML model (YOLO, DETR, etc.)
    let mock_model = callback(|img: &ImageData| {
        // Simulate finding a small object in each slice
        let det = Detection::new(
            BoundingBox::new(img.width as f32 * 0.3, img.height as f32 * 0.3, 50.0, 50.0),
            0, // class_id: "person"
            0.85,
            Some("person".to_string()),
        );

        Ok(vec![det])
    });

    // Run sliced inference
    println!("\nRunning sliced inference...");
    let detections = sahi.predict(&image, &mock_model).unwrap();

    println!("Detections after postprocessing: {}", detections.len());

    // Print first few detections
    for (i, det) in detections.iter().take(5).enumerate() {
        println!(
            "  [{}] {} @ ({:.1}, {:.1}, {:.1}x{:.1}) conf={:.2}",
            i,
            det.class_name.as_deref().unwrap_or("unknown"),
            det.bbox.x,
            det.bbox.y,
            det.bbox.width,
            det.bbox.height,
            det.confidence
        );
    }

    if detections.len() > 5 {
        println!("  ... and {} more", detections.len() - 5);
    }
}

/// Demonstrate different postprocessing algorithms.
fn demo_postprocess_types() {
    println!("\n{}", "=".repeat(60));
    println!("Postprocessing Type Comparison");
    println!("{}", "=".repeat(60));

    // Create a small test image
    let image = ImageData::from_rgb(vec![0u8; 640 * 640 * 3], 640, 640);

    // Callback that returns overlapping detections to show postprocessing differences
    let detector_with_overlaps = callback(|_img: &ImageData| {
        Ok(vec![
            Detection::new(
                BoundingBox::new(100.0, 100.0, 200.0, 200.0),
                0,
                0.9,
                Some("obj".to_string()),
            ),
            Detection::new(
                BoundingBox::new(150.0, 150.0, 200.0, 200.0),
                0,
                0.8,
                Some("obj".to_string()),
            ), // Overlapping
        ])
    });

    for pp_type in [
        PostprocessType::NMS,
        PostprocessType::NMM,
        PostprocessType::GREEDYNMM,
    ] {
        let sahi = Sahi::builder()
            .slice_size(640, 640)
            .postprocess_type(pp_type)
            .match_threshold(0.3)
            .build();

        let results = sahi.predict(&image, &detector_with_overlaps).unwrap();
        println!("\n{:?}: {} detection(s)", pp_type, results.len());
        for det in &results {
            println!(
                "  bbox=({:.0}, {:.0}, {:.0}, {:.0}), conf={:.2}",
                det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height, det.confidence
            );
        }
    }
}
