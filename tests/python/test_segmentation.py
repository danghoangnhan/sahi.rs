"""Instance-segmentation Python bindings tests (sub-project 9c)."""

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
