"""Pytest configuration and fixtures for sahi-rs tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_large():
    """Create a larger sample image for slice testing."""
    return np.zeros((1920, 1080, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_with_content():
    """Create a sample image with some content for more realistic testing."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Add a white rectangle
    img[100:200, 100:200] = 255
    return img
