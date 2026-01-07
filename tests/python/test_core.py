"""Core functionality tests for sahi-rs."""

import numpy as np
import pytest


def test_import_core_types():
    """Test that core types can be imported."""
    from sahi_rs import BoundingBox, Detection, Sahi

    assert Sahi is not None
    assert BoundingBox is not None
    assert Detection is not None


def test_sahi_creation():
    """Test Sahi instance creation with default parameters."""
    from sahi_rs import Sahi

    sahi = Sahi(slice_width=640, slice_height=640)
    assert sahi is not None


def test_sahi_creation_with_options():
    """Test Sahi instance creation with custom parameters."""
    from sahi_rs import Sahi

    sahi = Sahi(
        slice_width=512,
        slice_height=512,
        overlap_width=0.25,
        overlap_height=0.25,
        confidence_threshold=0.3,
        postprocess_match_threshold=0.45,
    )
    assert sahi is not None


def test_bounding_box_creation():
    """Test BoundingBox creation and properties."""
    from sahi_rs import BoundingBox

    bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
    assert bbox.x == 10.0
    assert bbox.y == 20.0
    assert bbox.width == 100.0
    assert bbox.height == 50.0


def test_bounding_box_repr():
    """Test BoundingBox string representation."""
    from sahi_rs import BoundingBox

    bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
    repr_str = repr(bbox)
    assert "BoundingBox" in repr_str


def test_detection_creation():
    """Test Detection creation with all fields."""
    from sahi_rs import BoundingBox, Detection

    bbox = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
    det = Detection(bbox=bbox, class_id=0, confidence=0.9, class_name="person")
    assert det.class_id == 0
    assert det.confidence == pytest.approx(0.9, rel=1e-5)
    assert det.class_name == "person"
    assert det.bbox.width == pytest.approx(100.0)


def test_detection_without_class_name():
    """Test Detection creation without optional class_name."""
    from sahi_rs import BoundingBox, Detection

    bbox = BoundingBox(x=0.0, y=0.0, width=50.0, height=50.0)
    det = Detection(bbox=bbox, class_id=1, confidence=0.85)
    assert det.class_id == 1
    assert det.confidence == pytest.approx(0.85, rel=1e-5)


def test_predict_with_callback(sample_image):
    """Test prediction with a callback function."""
    from sahi_rs import BoundingBox, Detection, Sahi

    def mock_detector(_image):
        # Return a single detection for testing
        return [
            Detection(
                bbox=BoundingBox(x=10.0, y=10.0, width=50.0, height=50.0),
                class_id=0,
                confidence=0.9,
            )
        ]

    sahi = Sahi(slice_width=320, slice_height=320)
    results = sahi.predict(sample_image, mock_detector)
    assert isinstance(results, list)


def test_predict_empty_results(sample_image):
    """Test prediction with callback returning no detections."""
    from sahi_rs import Sahi

    def empty_detector(_image):
        return []

    sahi = Sahi(slice_width=320, slice_height=320)
    results = sahi.predict(sample_image, empty_detector)
    assert isinstance(results, list)
    assert len(results) == 0


def test_category_creation():
    """Test Category type."""
    from sahi_rs import Category

    cat = Category(id=0, name="person")
    assert cat.id == 0
    assert cat.name == "person"


def test_image_data_types(sample_image):
    """Test that various numpy dtypes work as image input."""
    from sahi_rs import Sahi

    def dummy_detector(_image):
        return []

    sahi = Sahi(slice_width=320, slice_height=320)

    # Test uint8
    results = sahi.predict(sample_image.astype(np.uint8), dummy_detector)
    assert isinstance(results, list)
