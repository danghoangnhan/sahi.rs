"""Instance-segmentation Python bindings tests (sub-project 9c)."""

import os

import numpy as np
import pytest


def test_masked_detection_with_mask():
    from sahi_rs import BoundingBox, Detection, MaskedDetection

    det = Detection(
        bbox=BoundingBox(x=10.0, y=10.0, width=20.0, height=20.0),
        class_id=0,
        confidence=0.9,
    )
    poly = [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]]
    md = MaskedDetection(det, poly)

    assert md.mask == poly
    assert md.detection.class_id == 0
    assert md.detection.confidence == pytest.approx(0.9, rel=1e-5)


def test_masked_detection_without_mask():
    from sahi_rs import BoundingBox, Detection, MaskedDetection

    det = Detection(
        bbox=BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0),
        class_id=1,
        confidence=0.8,
    )
    md = MaskedDetection(det)
    assert md.mask is None


def test_mask_array_rasterizes_polygon():
    from sahi_rs import BoundingBox, Detection, MaskedDetection

    det = Detection(
        bbox=BoundingBox(x=0.0, y=0.0, width=40.0, height=40.0),
        class_id=0,
        confidence=0.9,
    )
    poly = [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]]
    md = MaskedDetection(det, poly)

    arr = md.mask_array(50, 50)
    assert arr.shape == (50, 50)
    assert arr.dtype == np.bool_
    assert bool(arr[20, 20])  # inside the (10,10)-(30,30) square
    assert not bool(arr[45, 45])  # outside


def test_mask_array_rejects_huge_dimensions():
    from sahi_rs import BoundingBox, Detection, MaskedDetection

    det = Detection(
        bbox=BoundingBox(x=0.0, y=0.0, width=10.0, height=10.0),
        class_id=0,
        confidence=0.9,
    )
    md = MaskedDetection(det, [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])
    # ~1e10 pixels would be a multi-GB allocation; must raise, not OOM.
    with pytest.raises(ValueError):
        md.mask_array(100000, 100000)


def test_predict_instances_returns_image_space_masks():
    from sahi_rs import BoundingBox, Detection, MaskedDetection, Sahi

    sahi = Sahi(
        slice_width=100,
        slice_height=100,
        overlap_width=0.0,
        overlap_height=0.0,
    )
    # H=100, W=200 -> two 100x100 slices at x=0 and x=100.
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    def detector(_image):
        return [
            MaskedDetection(
                Detection(
                    bbox=BoundingBox(x=10.0, y=10.0, width=10.0, height=10.0),
                    class_id=0,
                    confidence=0.9,
                ),
                [[10.0, 10.0, 20.0, 10.0, 20.0, 20.0, 10.0, 20.0]],
            )
        ]

    results = sahi.predict_instances(image, detector)
    assert isinstance(results, list)
    assert len(results) == 2

    # One detection comes from the second slice (origin x=100): its mask polygon's
    # first vertex must be offset into image coordinates (x >= 100).
    first_xs = [r.mask[0][0] for r in results if r.mask is not None]
    assert any(x >= 100.0 for x in first_xs)


def test_yolov8_seg_detector_construction_and_attrs():
    from sahi_rs import YOLOv8SegDetector

    det = YOLOv8SegDetector(
        "nonexistent.onnx", num_classes=80, num_masks=32, input_size=640
    )
    assert det.is_loaded() is False
    assert det.num_classes == 80
    assert det.num_masks == 32
    assert det.input_size == 640
    assert "YOLOv8SegDetector" in repr(det)


def test_yolov8_seg_detector_load_missing_model_errors():
    from sahi_rs import YOLOv8SegDetector

    det = YOLOv8SegDetector("definitely_missing_model.onnx")
    with pytest.raises(RuntimeError):
        det.load()


def test_predict_instances_yolov8_requires_loaded_model():
    from sahi_rs import Sahi, YOLOv8SegDetector

    sahi = Sahi(slice_width=64, slice_height=64)
    det = YOLOv8SegDetector("nonexistent.onnx")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        sahi.predict_instances_yolov8(img, det)


@pytest.mark.skipif(
    "SAHI_TEST_YOLOV8_SEG_MODEL" not in os.environ,
    reason="requires a real YOLOv8-seg ONNX model",
)
def test_predict_instances_yolov8_real_model():
    from sahi_rs import Sahi, YOLOv8SegDetector

    det = YOLOv8SegDetector(os.environ["SAHI_TEST_YOLOV8_SEG_MODEL"])
    det.load()
    sahi = Sahi(slice_width=640, slice_height=640)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    out = sahi.predict_instances_yolov8(img, det)
    assert isinstance(out, list)
