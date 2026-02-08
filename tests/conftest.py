"""
Pytest fixtures for image generation and report directories.
"""

import io
import os
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def rgb_image() -> Image.Image:
    """Return a small RGB image for tests."""
    return Image.new("RGB", (64, 64), color=(120, 80, 200))


@pytest.fixture()
def rgb_jpeg_bytes(
    rgb_image: Image.Image,  # pylint: disable=redefined-outer-name
) -> bytes:
    """Return JPEG-encoded bytes for the RGB test image."""
    buf = io.BytesIO()
    rgb_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def tmp_reports_dir(tmp_path: Path) -> Path:
    """Create and return a temporary reports directory."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


@pytest.fixture(autouse=True, scope="session")
def force_matplotlib_agg() -> None:
    """Force matplotlib to use the Agg backend for the entire test session."""
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel

        matplotlib.use("Agg", force=True)
    except (ImportError, RuntimeError):
        # Matplotlib may not be installed or backend may already be set
        pass
