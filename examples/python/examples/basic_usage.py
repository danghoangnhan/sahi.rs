"""
Basic usage example for sahi-rs Python bindings.

This example demonstrates how to use SAHI for sliced inference
with your own object detection model.
"""

import numpy as np
from sahi import BoundingBox, Detection, MatchMetric, PostprocessType, Sahi


def mock_detector(image: np.ndarray) -> list[Detection]:
    """
    Mock detection function that simulates finding objects.

    In real usage, replace this with your actual model inference:
    - YOLO: results = model(image); return convert_yolo_results(results)
    - RT-DETR: results = model(image); return convert_detr_results(results)
    - etc.

    Args:
        image: numpy array (H, W, C) uint8 RGB

    Returns:
        List of Detection objects with coordinates relative to the input slice
    """
    h, w, c = image.shape

    # Simulate finding a small object in each slice
    detections = [
        Detection(
            bbox=BoundingBox(
                x=w * 0.3,
                y=h * 0.3,
                width=50.0,
                height=50.0,
            ),
            class_id=0,
            confidence=0.85,
            class_name="person",
        )
    ]

    return detections


def main():
    # Create SAHI instance with custom configuration
    sahi = Sahi(
        slice_width=640,
        slice_height=640,
        overlap_width=0.2,
        overlap_height=0.2,
        # Postprocessing configuration (like original SAHI)
        postprocess_type=PostprocessType.GREEDYNMM,  # Options: NMS, NMM, GREEDYNMM
        postprocess_match_metric=MatchMetric.IOU,  # Options: IOU, IOS
        postprocess_match_threshold=0.5,
        confidence_threshold=0.25,
        class_aware=True,
        include_full_image=False,
    )

    print(f"SAHI configuration: {sahi}")
    print(f"Backend: {sahi.backend_name()}")
    print(f"Config: {sahi.get_config()}")

    # Create a test image (1920x1080 RGB)
    # In real usage: image = np.array(Image.open("your_image.jpg"))
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Add some fake content
    image[100:200, 100:200] = [255, 0, 0]  # Red square
    image[500:600, 800:900] = [0, 255, 0]  # Green square

    print(f"\nImage shape: {image.shape}")

    # Run sliced inference
    detections = sahi.predict(image, mock_detector)

    print(f"\nFound {len(detections)} detections after postprocessing:")
    for i, det in enumerate(detections):
        print(f"  [{i}] {det}")
        print(f"       Class: {det.class_id} ({det.class_name})")
        print(f"       Confidence: {det.confidence:.3f}")
        print(
            f"       BBox: x={det.bbox.x:.1f}, y={det.bbox.y:.1f}, "
            f"w={det.bbox.width:.1f}, h={det.bbox.height:.1f}"
        )


def demo_postprocess_types():
    """Demonstrate different postprocessing algorithms."""
    print("\n" + "=" * 60)
    print("Postprocessing Type Comparison")
    print("=" * 60)

    image = np.zeros((640, 640, 3), dtype=np.uint8)

    def detector_with_overlaps(img):
        """Return overlapping detections to show postprocessing differences."""
        return [
            Detection(BoundingBox(100, 100, 200, 200), 0, 0.9, "obj"),
            Detection(BoundingBox(150, 150, 200, 200), 0, 0.8, "obj"),  # Overlapping
        ]

    for pp_type in [
        PostprocessType.NMS,
        PostprocessType.NMM,
        PostprocessType.GREEDYNMM,
    ]:
        sahi = Sahi(
            slice_width=640,
            slice_height=640,
            postprocess_type=pp_type,
            postprocess_match_threshold=0.3,
        )
        results = sahi.predict(image, detector_with_overlaps)
        print(f"\n{pp_type}: {len(results)} detection(s)")
        for det in results:
            print(
                f"  bbox=({det.bbox.x:.0f}, {det.bbox.y:.0f}, "
                f"{det.bbox.width:.0f}, {det.bbox.height:.0f}), "
                f"conf={det.confidence:.2f}"
            )


if __name__ == "__main__":
    main()
    demo_postprocess_types()
